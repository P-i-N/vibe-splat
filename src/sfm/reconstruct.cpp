#include "ingest/ingest.hpp"
#include "sfm/bundle.hpp"
#include "sfm/features.hpp"
#include "sfm/sfm.hpp"
#include "splat/gaussian_cloud.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <stdexcept>
#include <thread>

namespace splat {

namespace fs = std::filesystem;

namespace {

constexpr int kMinPairInliers = 15;
constexpr int kMinInitialPoints = 20;
constexpr int kMinRegistrationInliers = 12;
constexpr double kPairThresholdPixels = 3.0;
constexpr double kPoseThresholdPixels = 4.0;
constexpr double kPointThresholdPixels = 4.0;
constexpr double kFinalThresholdPixels = 1.5; // Once the model has settled
constexpr double kMinTriangulationAngleDeg = 1.5;
constexpr std::size_t kExhaustiveLimit = 150;

template <class Job> void parallel_for(std::size_t count, int threads, const Job &job) {
  const unsigned wanted = threads > 0 ? static_cast<unsigned>(threads) : std::max(1u, std::thread::hardware_concurrency());
  std::atomic<std::size_t> next{0};
  std::vector<std::jthread> workers;
  for (unsigned t = 0; t < std::min<std::size_t>(wanted, count); ++t) {
    workers.emplace_back([&] {
      for (std::size_t i = next++; i < count; i = next++) job(i);
    });
  }
}

double degrees_between(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  return std::atan2(a.cross(b).norm(), a.dot(b)) * 180.0 / std::numbers::pi;
}

struct VerifiedPair {
  int a = 0;
  int b = 0;
  std::vector<std::pair<int, int>> inliers;
  Pose pose; // b relative to a
  double medianAngleDeg = 0.0;
};

struct Observation {
  int image = 0;
  int feature = 0;
  bool active = false; // Part of the 3D point
};

struct Track {
  std::vector<Observation> observations;
  bool hasPoint = false;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

class DisjointSets {
public:
  explicit DisjointSets(std::size_t size) : parent_(size) { std::iota(parent_.begin(), parent_.end(), 0); }
  int find(int x) {
    while (parent_[static_cast<std::size_t>(x)] != x) {
      parent_[static_cast<std::size_t>(x)] = parent_[static_cast<std::size_t>(parent_[static_cast<std::size_t>(x)])];
      x = parent_[static_cast<std::size_t>(x)];
    }
    return x;
  }
  void join(int a, int b) { parent_[static_cast<std::size_t>(find(a))] = find(b); }

private:
  std::vector<int> parent_;
};

// Holds a growing reconstruction: which images have poses and which tracks have points.
class Mapper {
public:
  Mapper(std::vector<Lens> lenses, std::vector<int> lensOfImage, const std::vector<ImageFeatures> &features, std::vector<Track> tracks)
      : initialLenses_(lenses), lenses_(std::move(lenses)), lensOfImage_(std::move(lensOfImage)), features_(features),
        tracks_(std::move(tracks)), poses_(features.size()), tracksOfImage_(features.size()), attempts_(features.size(), 0) {
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
      for (const Observation &o : tracks_[t].observations) tracksOfImage_[static_cast<std::size_t>(o.image)].push_back(static_cast<int>(t));
    }
  }

  // Starts from a pair of images. False if it yields too few points.
  bool initialize(const VerifiedPair &pair) {
    poses_[static_cast<std::size_t>(pair.a)] = Pose{};
    poses_[static_cast<std::size_t>(pair.b)] = pair.pose;
    anchor_ = pair.a;
    triangulate_new(pair.b);
    bundle(false, 20);
    filter();
    if (point_count() >= kMinInitialPoints) return true;
    reset();
    return false;
  }

  void reset() {
    lenses_ = initialLenses_;
    std::fill(poses_.begin(), poses_.end(), std::nullopt);
    for (Track &track : tracks_) {
      track.hasPoint = false;
      for (Observation &o : track.observations) o.active = false;
    }
  }

  void grow() {
    std::size_t lastGlobal = registered_count();
    for (;;) {
      const auto image = next_image();
      if (!image) break;
      if (!register_image(*image)) continue;
      extend(*image);
      triangulate_new(*image);
      const std::size_t count = registered_count();
      // Bundle adjustment gets slower as the model grows, but matters less
      // often: once per 10% of growth.
      if (count <= 10 || static_cast<double>(count) >= 1.1 * static_cast<double>(lastGlobal)) {
        bundle(count >= 3, 8);
        filter();
        lastGlobal = count;
      }
      std::println("  registered {} images, {} points", count, point_count());
    }
  }

  void finish() {
    // A generous threshold is right while poses are rough. Once they are good,
    // what still misses by a few pixels is wrong: typically a false corner,
    // where edges at different depths cross. Left in, those bias everything.
    for (const double threshold : {kPointThresholdPixels, kFinalThresholdPixels, kFinalThresholdPixels}) {
      const BundleResult result = bundle(true, 40);
      filter(threshold);
      std::println("  final bundle adjustment: {:.3f} -> {:.3f} px RMS in {} iterations", result.initialRms, result.finalRms, result.iterations);
    }
  }

  [[nodiscard]] std::size_t registered_count() const {
    return static_cast<std::size_t>(std::count_if(poses_.begin(), poses_.end(), [](const auto &p) { return p.has_value(); }));
  }
  [[nodiscard]] std::size_t point_count() const {
    return static_cast<std::size_t>(std::count_if(tracks_.begin(), tracks_.end(), [](const Track &t) { return t.hasPoint; }));
  }

  [[nodiscard]] const std::vector<Lens> &lenses() const { return lenses_; }
  [[nodiscard]] const std::vector<std::optional<Pose>> &poses() const { return poses_; }
  [[nodiscard]] const std::vector<Track> &tracks() const { return tracks_; }

private:
  [[nodiscard]] const Lens &lens_of(int image) const { return lenses_[static_cast<std::size_t>(lensOfImage_[static_cast<std::size_t>(image)])]; }

  [[nodiscard]] Eigen::Vector2d pixel_of(const Observation &o) const {
    const Keypoint &k = features_[static_cast<std::size_t>(o.image)].keypoints[static_cast<std::size_t>(o.feature)];
    return {k.x, k.y};
  }

  [[nodiscard]] double error_pixels(const Observation &o, const Eigen::Vector3d &point) const {
    const Pose &pose = *poses_[static_cast<std::size_t>(o.image)];
    Eigen::Vector2d pixel;
    if (!project_point(lens_of(o.image), pose.R * point + pose.t, pixel)) return std::numeric_limits<double>::infinity();
    return (pixel - pixel_of(o)).norm();
  }

  // The unregistered image seeing the most known points and not tried too often.
  [[nodiscard]] std::optional<int> next_image() {
    std::optional<int> best;
    int bestCount = kMinRegistrationInliers - 1;
    for (std::size_t i = 0; i < poses_.size(); ++i) {
      if (poses_[i] || attempts_[i] >= 2) continue;
      int count = 0;
      for (const int t : tracksOfImage_[i]) count += tracks_[static_cast<std::size_t>(t)].hasPoint ? 1 : 0;
      // A failed image is only worth another try once the model has grown.
      if (count > bestCount && count > attemptedAt_[i]) {
        bestCount = count;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  bool register_image(int image) {
    std::vector<Eigen::Vector3d> bearings, points;
    std::vector<Eigen::Vector2d> pixels;
    for (const int t : tracksOfImage_[static_cast<std::size_t>(image)]) {
      const Track &track = tracks_[static_cast<std::size_t>(t)];
      if (!track.hasPoint) continue;
      for (const Observation &o : track.observations) {
        if (o.image != image) continue;
        pixels.push_back(pixel_of(o));
        bearings.push_back(unproject_pixel(lens_of(image), pixels.back()));
        points.push_back(track.point);
      }
    }
    attemptedAt_[static_cast<std::size_t>(image)] = static_cast<int>(points.size()) * 5 / 4;
    ++attempts_[static_cast<std::size_t>(image)];

    std::mt19937 rng(static_cast<std::uint32_t>(image) * 7919u + 13u);
    const auto result = estimate_absolute_pose(bearings, points, kPoseThresholdPixels / lens_of(image).f, rng);
    if (!result || result->inliers.size() < kMinRegistrationInliers ||
        result->inliers.size() * 4 < points.size()) {
      return false;
    }
    std::vector<Eigen::Vector3d> inlierPoints;
    std::vector<Eigen::Vector2d> inlierPixels;
    for (const int i : result->inliers) {
      inlierPoints.push_back(points[static_cast<std::size_t>(i)]);
      inlierPixels.push_back(pixels[static_cast<std::size_t>(i)]);
    }
    Pose pose = result->pose;
    refine_pose(lens_of(image), pose, inlierPoints, inlierPixels);
    poses_[static_cast<std::size_t>(image)] = pose;
    return true;
  }

  // Lets existing points pick up their observation in a new image.
  void extend(int image) {
    for (const int t : tracksOfImage_[static_cast<std::size_t>(image)]) {
      Track &track = tracks_[static_cast<std::size_t>(t)];
      if (!track.hasPoint) continue;
      for (Observation &o : track.observations) {
        if (o.image == image && error_pixels(o, track.point) < kPointThresholdPixels) o.active = true;
      }
    }
  }

  // Creates points for tracks the new image can now triangulate.
  void triangulate_new(int image) {
    for (const int t : tracksOfImage_[static_cast<std::size_t>(image)]) {
      Track &track = tracks_[static_cast<std::size_t>(t)];
      if (track.hasPoint) continue;
      const auto own = std::find_if(track.observations.begin(), track.observations.end(), [&](const Observation &o) { return o.image == image; });
      const Pose &ownPose = *poses_[static_cast<std::size_t>(image)];
      const Eigen::Vector3d ownBearing = unproject_pixel(lens_of(image), pixel_of(*own));
      const Eigen::Vector3d ownRay = ownPose.R.transpose() * ownBearing;

      // Pair the new view with whichever registered view gives the widest angle.
      const Observation *partner = nullptr;
      double bestAngle = kMinTriangulationAngleDeg;
      Eigen::Vector3d partnerBearing;
      for (const Observation &o : track.observations) {
        if (o.image == image || !poses_[static_cast<std::size_t>(o.image)]) continue;
        const Eigen::Vector3d bearing = unproject_pixel(lens_of(o.image), pixel_of(o));
        const double angle = degrees_between(ownRay, poses_[static_cast<std::size_t>(o.image)]->R.transpose() * bearing);
        if (angle > bestAngle) {
          bestAngle = angle;
          partner = &o;
          partnerBearing = bearing;
        }
      }
      if (partner == nullptr) continue;

      const Eigen::Vector3d point = triangulate({ownPose, *poses_[static_cast<std::size_t>(partner->image)]}, {ownBearing, partnerBearing});
      if (!point.allFinite()) continue;
      int active = 0;
      for (const Observation &o : track.observations) {
        if (poses_[static_cast<std::size_t>(o.image)] && error_pixels(o, point) < kPointThresholdPixels) ++active;
      }
      if (active < 2) continue;
      for (Observation &o : track.observations) {
        o.active = poses_[static_cast<std::size_t>(o.image)] && error_pixels(o, point) < kPointThresholdPixels;
      }
      track.point = point;
      track.hasPoint = true;
    }
  }

  // Drops observations that no longer fit, and points that lost their support.
  void filter(double threshold = kPointThresholdPixels) {
    for (Track &track : tracks_) {
      if (!track.hasPoint) continue;
      int active = 0;
      double widest = 0.0;
      std::vector<Eigen::Vector3d> rays;
      for (Observation &o : track.observations) {
        if (!o.active) continue;
        if (error_pixels(o, track.point) >= threshold) {
          o.active = false;
          continue;
        }
        ++active;
        rays.push_back(track.point - poses_[static_cast<std::size_t>(o.image)]->center());
      }
      for (std::size_t a = 0; a < rays.size(); ++a) {
        for (std::size_t b = a + 1; b < rays.size(); ++b) widest = std::max(widest, degrees_between(rays[a], rays[b]));
      }
      if (active < 2 || widest < 1.0) {
        track.hasPoint = false;
        for (Observation &o : track.observations) o.active = false;
      }
    }
  }

  BundleResult bundle(bool refineLenses, int iterations) {
    std::vector<BundleView> views;
    std::vector<int> viewImages;
    std::map<int, int> viewOfImage;
    for (std::size_t i = 0; i < poses_.size(); ++i) {
      if (!poses_[i]) continue;
      viewOfImage[static_cast<int>(i)] = static_cast<int>(views.size());
      viewImages.push_back(static_cast<int>(i));
      // One fixed view anchors position and rotation. The damping handles scale.
      views.push_back({*poses_[i], lensOfImage_[i], static_cast<int>(i) == anchor_});
    }
    std::vector<Eigen::Vector3d> points;
    std::vector<int> pointTracks;
    std::vector<BundleObservation> observations;
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
      const Track &track = tracks_[t];
      if (!track.hasPoint) continue;
      const int index = static_cast<int>(points.size());
      points.push_back(track.point);
      pointTracks.push_back(static_cast<int>(t));
      for (const Observation &o : track.observations) {
        if (o.active) observations.push_back({viewOfImage.at(o.image), index, pixel_of(o)});
      }
    }
    BundleOptions options;
    options.iterations = iterations;
    options.refineLenses = refineLenses;
    const BundleResult result = bundle_adjust(lenses_, views, points, observations, options);
    for (std::size_t v = 0; v < views.size(); ++v) poses_[static_cast<std::size_t>(viewImages[v])] = views[v].pose;
    for (std::size_t p = 0; p < points.size(); ++p) tracks_[static_cast<std::size_t>(pointTracks[p])].point = points[p];
    return result;
  }

  std::vector<Lens> initialLenses_;
  std::vector<Lens> lenses_;
  std::vector<int> lensOfImage_;
  const std::vector<ImageFeatures> &features_;
  std::vector<Track> tracks_;
  std::vector<std::optional<Pose>> poses_;
  std::vector<std::vector<int>> tracksOfImage_;
  std::vector<int> attempts_;
  std::map<std::size_t, int> attemptedAt_;
  int anchor_ = 0;
};

std::vector<std::pair<int, int>> choose_pairs(std::size_t count, const IngestManifest &manifest, const SfmOptions &options) {
  const bool anyVideo = std::any_of(manifest.images.begin(), manifest.images.end(), [](const IngestImage &i) { return i.videoSeconds.has_value(); });
  const bool exhaustive = options.matcher == Matcher::Exhaustive ||
                          (options.matcher == Matcher::Auto && !anyVideo && count <= kExhaustiveLimit);
  std::vector<std::pair<int, int>> pairs;
  const int n = static_cast<int>(count);
  if (exhaustive) {
    for (int a = 0; a < n; ++a) {
      for (int b = a + 1; b < n; ++b) pairs.emplace_back(a, b);
    }
  } else {
    if (options.matcher == Matcher::Auto && !anyVideo) {
      std::println("  {} photos: matching neighbours only, assuming they were shot in order (--matcher exhaustive overrides this)", count);
    }
    for (int a = 0; a < n; ++a) {
      for (int d = 1; d <= options.window && a + d < n; ++d) pairs.emplace_back(a, a + d);
      // Distant frames at growing steps, so a camera returning to a spot can link up.
      for (int d = options.window * 2; a + d < n; d *= 2) pairs.emplace_back(a, a + d);
    }
  }
  return pairs;
}

std::vector<Track> build_tracks(const std::vector<ImageFeatures> &features, const std::vector<VerifiedPair> &pairs) {
  std::vector<int> offsets(features.size() + 1, 0);
  for (std::size_t i = 0; i < features.size(); ++i) offsets[i + 1] = offsets[i] + static_cast<int>(features[i].keypoints.size());

  DisjointSets sets(static_cast<std::size_t>(offsets.back()));
  std::vector<char> used(static_cast<std::size_t>(offsets.back()), 0);
  for (const VerifiedPair &pair : pairs) {
    for (const auto &[fa, fb] : pair.inliers) {
      const int a = offsets[static_cast<std::size_t>(pair.a)] + fa, b = offsets[static_cast<std::size_t>(pair.b)] + fb;
      sets.join(a, b);
      used[static_cast<std::size_t>(a)] = used[static_cast<std::size_t>(b)] = 1;
    }
  }

  std::map<int, Track> byRoot;
  for (std::size_t image = 0; image < features.size(); ++image) {
    for (int f = offsets[image]; f < offsets[image + 1]; ++f) {
      if (!used[static_cast<std::size_t>(f)]) continue;
      Track &track = byRoot[sets.find(f)];
      // Two features of one image in a track means a mismatch somewhere. Keep
      // the first; the reprojection check sorts out the rest.
      if (!track.observations.empty() && track.observations.back().image == static_cast<int>(image)) continue;
      track.observations.push_back({static_cast<int>(image), f - offsets[image], false});
    }
  }
  std::vector<Track> tracks;
  for (auto &[root, track] : byRoot) {
    if (track.observations.size() >= 2) tracks.push_back(std::move(track));
  }
  return tracks;
}

std::vector<std::uint8_t> read_all(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A small round Gaussian per point, to look at the result with `splat view`.
GaussianCloud preview_cloud(const ColmapModel &model) {
  GaussianCloud cloud;
  cloud.resize(model.points.size(), 0);
  std::vector<float> distances;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const ColmapPoint &p : model.points) mean += Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  mean /= std::max<double>(1.0, static_cast<double>(model.points.size()));
  for (const ColmapPoint &p : model.points) {
    distances.push_back(static_cast<float>((Eigen::Vector3d(p.position.x, p.position.y, p.position.z) - mean).norm()));
  }
  float size = 0.01f;
  if (!distances.empty()) {
    const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), middle, distances.end());
    size = std::max(*middle * 0.006f, 1e-6f);
  }
  for (std::size_t i = 0; i < model.points.size(); ++i) {
    const ColmapPoint &p = model.points[i];
    for (std::size_t c = 0; c < 3; ++c) {
      cloud.positions[i * 3 + c] = static_cast<float>(c == 0 ? p.position.x : c == 1 ? p.position.y : p.position.z);
      cloud.scales[i * 3 + c] = std::log(size);
      cloud.sh[i * 3 + c] = (static_cast<float>(p.color[c]) / 255.f - 0.5f) / 0.28209479f;
    }
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = 4.f;
  }
  return cloud;
}

} // namespace

SfmReport run_sfm(const SfmOptions &options) {
  const IngestManifest manifest = load_manifest(options.project / "ingest.json");
  const std::size_t count = manifest.images.size();
  if (count < 2) throw std::runtime_error("structure from motion needs at least two images");

  std::map<int, int> lensOfCamera;
  std::vector<Lens> lenses;
  for (const IngestCamera &camera : manifest.cameras) {
    Lens lens;
    lens.f = lens.fPrior = 0.5 * (camera.fx + camera.fy);
    lens.cx = camera.cx;
    lens.cy = camera.cy;
    lens.fixedF = camera.focalSource == FocalSource::Manual;
    lens.fPriorSigma = lens.f * (camera.focalSource == FocalSource::Default ? 0.3 : 0.1);
    lensOfCamera[camera.id] = static_cast<int>(lenses.size());
    lenses.push_back(lens);
  }
  std::vector<int> lensOfImage;
  for (const IngestImage &image : manifest.images) lensOfImage.push_back(lensOfCamera.at(image.cameraId));

  std::println("Extracting features from {} images...", count);
  std::vector<ImageFeatures> features(count);
  std::mutex errorMutex;
  std::string firstError;
  parallel_for(count, options.threads, [&](std::size_t i) {
    try {
      features[i] = extract_features(decode_image(read_all(options.project / manifest.images[i].file)), options.maxFeatures);
    } catch (const std::exception &e) {
      const std::lock_guard lock(errorMutex);
      if (firstError.empty()) firstError = std::format("{}: {}", manifest.images[i].file, e.what());
    }
  });
  if (!firstError.empty()) throw std::runtime_error(firstError);
  std::size_t totalFeatures = 0;
  for (const ImageFeatures &f : features) totalFeatures += f.keypoints.size();
  std::println("  {:.0f} features per image", static_cast<double>(totalFeatures) / static_cast<double>(count));

  const std::vector<std::pair<int, int>> pairs = choose_pairs(count, manifest, options);
  std::println("Matching {} image pairs...", pairs.size());
  std::vector<std::optional<VerifiedPair>> verified(pairs.size());
  parallel_for(pairs.size(), options.threads, [&](std::size_t p) {
    const auto [a, b] = pairs[p];
    const auto matches = match_features(features[static_cast<std::size_t>(a)], features[static_cast<std::size_t>(b)]);
    if (matches.size() < kMinPairInliers) return;
    const Lens &lensA = lenses[static_cast<std::size_t>(lensOfImage[static_cast<std::size_t>(a)])];
    const Lens &lensB = lenses[static_cast<std::size_t>(lensOfImage[static_cast<std::size_t>(b)])];
    std::vector<Eigen::Vector3d> x1, x2;
    for (const auto &[fa, fb] : matches) {
      const Keypoint &ka = features[static_cast<std::size_t>(a)].keypoints[static_cast<std::size_t>(fa)];
      const Keypoint &kb = features[static_cast<std::size_t>(b)].keypoints[static_cast<std::size_t>(fb)];
      x1.push_back(unproject_pixel(lensA, {ka.x, ka.y}));
      x2.push_back(unproject_pixel(lensB, {kb.x, kb.y}));
    }
    std::mt19937 rng(static_cast<std::uint32_t>(p) * 2654435761u + 1u);
    const auto result = estimate_relative_pose(x1, x2, kPairThresholdPixels / (0.5 * (lensA.f + lensB.f)), rng);
    if (!result || result->inliers.size() < kMinPairInliers) return;
    VerifiedPair pair;
    pair.a = a;
    pair.b = b;
    pair.pose = result->pose;
    pair.medianAngleDeg = result->medianAngleDeg;
    for (const int i : result->inliers) pair.inliers.push_back(matches[static_cast<std::size_t>(i)]);
    verified[p] = std::move(pair);
  });
  std::vector<VerifiedPair> goodPairs;
  for (auto &pair : verified) {
    if (pair) goodPairs.push_back(std::move(*pair));
  }
  std::println("  {} pairs passed geometric verification", goodPairs.size());
  if (goodPairs.empty()) throw std::runtime_error("no image pair matched well enough; do the images overlap?");

  std::vector<Track> tracks = build_tracks(features, goodPairs);
  std::println("  {} tracks", tracks.size());

  // Start from the pair with the most matches among those with a wide baseline,
  // accepting a narrower one only if nothing better works.
  Mapper mapper(lenses, lensOfImage, features, std::move(tracks));
  std::vector<const VerifiedPair *> byInliers;
  for (const VerifiedPair &pair : goodPairs) byInliers.push_back(&pair);
  std::sort(byInliers.begin(), byInliers.end(), [](const VerifiedPair *a, const VerifiedPair *b) { return a->inliers.size() > b->inliers.size(); });

  bool started = false;
  for (const double minAngle : {8.0, 4.0, 2.0}) {
    int attempts = 0;
    for (const VerifiedPair *pair : byInliers) {
      if (pair->medianAngleDeg < minAngle) continue;
      if (++attempts > 20) break;
      if (mapper.initialize(*pair)) {
        std::println("Starting from {} and {}: {} matches, {:.1f} degree baseline, {} points", manifest.images[static_cast<std::size_t>(pair->a)].file,
                     manifest.images[static_cast<std::size_t>(pair->b)].file, pair->inliers.size(), pair->medianAngleDeg, mapper.point_count());
        started = true;
        break;
      }
    }
    if (started) break;
  }
  if (!started) {
    throw std::runtime_error("no pair of images has enough baseline to start from; move the camera sideways between shots, not just around");
  }

  std::println("Registering images...");
  mapper.grow();
  mapper.finish();

  // Store the result in COLMAP's format.
  ColmapModel model;
  for (std::size_t i = 0; i < manifest.cameras.size(); ++i) {
    const IngestCamera &source = manifest.cameras[i];
    const Lens &lens = mapper.lenses()[i];
    ColmapCamera camera;
    camera.id = static_cast<std::uint32_t>(i + 1);
    camera.model = 2; // SIMPLE_RADIAL
    camera.width = static_cast<std::uint64_t>(source.width);
    camera.height = static_cast<std::uint64_t>(source.height);
    camera.params = {lens.f, lens.cx, lens.cy, lens.k};
    model.cameras[camera.id] = std::move(camera);
  }
  for (std::size_t i = 0; i < count; ++i) {
    const auto &pose = mapper.poses()[i];
    if (!pose) continue;
    ColmapImage image;
    image.id = static_cast<std::uint32_t>(i + 1);
    const Eigen::Quaterniond q(pose->R);
    image.rotation = {q.w(), q.x(), q.y(), q.z()};
    image.translation = {pose->t.x(), pose->t.y(), pose->t.z()};
    image.cameraId = static_cast<std::uint32_t>(lensOfImage[i] + 1);
    image.name = manifest.images[i].file;
    model.images[image.id] = std::move(image);
  }
  for (const Track &track : mapper.tracks()) {
    if (!track.hasPoint) continue;
    ColmapPoint point;
    point.id = model.points.size() + 1;
    point.position = {track.point.x(), track.point.y(), track.point.z()};
    double errorSum = 0.0;
    for (const Observation &o : track.observations) {
      if (!o.active) continue;
      const Keypoint &k = features[static_cast<std::size_t>(o.image)].keypoints[static_cast<std::size_t>(o.feature)];
      if (point.track.empty()) point.color = k.color;
      ColmapImage &image = model.images.at(static_cast<std::uint32_t>(o.image + 1));
      point.track.push_back({image.id, static_cast<std::uint32_t>(image.observations.size())});
      image.observations.push_back({{k.x, k.y}, point.id});
      const Pose &pose = *mapper.poses()[static_cast<std::size_t>(o.image)];
      Eigen::Vector2d pixel;
      if (project_point(mapper.lenses()[static_cast<std::size_t>(lensOfImage[static_cast<std::size_t>(o.image)])], pose.R * track.point + pose.t, pixel)) {
        errorSum += (pixel - Eigen::Vector2d(k.x, k.y)).norm();
      }
    }
    point.error = errorSum / static_cast<double>(point.track.size());
    model.points.push_back(std::move(point));
  }

  const fs::path output = options.project / "sparse" / "0";
  save_colmap_model(model, output);
  save_ply(preview_cloud(model), output / "preview.ply");

  SfmReport report;
  report.stats = compute_stats(model);
  report.ingested = count;
  report.unregistered = count - model.images.size();
  report.pairsTried = pairs.size();
  report.pairsVerified = goodPairs.size();
  return report;
}

} // namespace splat
