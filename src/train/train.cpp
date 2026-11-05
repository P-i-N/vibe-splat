#include "train/train.hpp"

#include "core/parallel.hpp"
#include "core/progress.hpp"
#include "ingest/image.hpp"
#include "render/camera.hpp"
#include "sfm/colmap_model.hpp"
#include "splat/align.hpp"
#include "splat/cgc.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/dense_points.hpp"
#include "train/densify.hpp"
#include "train/evaluate.hpp"
#include "train/mip.hpp"
#include "train/schedule.hpp"
#include "train/trainer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <tuple>

namespace splat {

namespace fs = std::filesystem;

namespace {

constexpr float kInitialOpacity = 0.1f;
constexpr float kResetOpacity = 0.01f;
// Larger than either of these and a Gaussian is removed. They only catch true
// outliers. Tight limits take big Gaussians out of walls and floors faster than
// new ones can fill the holes, and quality collapses: 20 pixels on screen cost
// 10 dB in a run of 7000 iterations at 800x600.
constexpr double kMaxWorldFraction = 1.0;    // Of the scene's extent
constexpr double kMaxScreenFraction = 0.25;  // Of the image's longer side
// A pixel is high error above this share of its view's worst, as in FastGS.
constexpr float kScoreErrorShare = 0.1f;
constexpr int kOpacityResetInterval = 3000;
// A cloud just after an opacity reset renders almost black, so checkpoints wait this long.
constexpr int kCheckpointRecovery = 1000;
// Recovery-aware pruning looks this long after a reset.
constexpr int kRecoveryDelay = 300;
constexpr std::uint64_t kBudgetReserve = 256ull * 1024 * 1024; // Sort scratch, staging, the driver's share
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kExposureRate = 5e-3f; // Adam's step for the per-view exposure transforms
// Second order, as in 3DGS^2-TR: the trust region shrinks over the rest of training.
constexpr float kTrustStart = 1e-6f, kTrustEnd = 1e-8f;
constexpr int kCurvatureInterval = 10;
constexpr double kPoseStart = 0.1; // Of training: until then the Gaussians are too rough to say much about the cameras
constexpr int kSpectrumViews = 8; // Photos looked at to choose the image sizes

glm::dvec2 focal_of(const ColmapCamera &c) {
  // PINHOLE, OPENCV and FULL_OPENCV list fx and fy; the others share one f.
  if (c.model == 1 || c.model == 4 || c.model == 6) return {c.params[0], c.params[1]};
  return {c.params[0], c.params[0]};
}

glm::dvec2 principal_of(const ColmapCamera &c) {
  if (c.model == 1 || c.model == 4 || c.model == 6) return {c.params[2], c.params[3]};
  return {c.params[1], c.params[2]};
}

// Resamples an image as a pinhole camera would have seen it. The renderer has
// no lens distortion, so the training images must not have any either.
RgbImage undistort(const RgbImage &image, const ColmapCamera &camera) {
  const glm::dvec2 f = focal_of(camera), c = principal_of(camera);
  RgbImage out;
  out.width = image.width;
  out.height = image.height;
  out.pixels.assign(image.pixels.size(), 0);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      const glm::dvec3 ray((x + 0.5 - c.x) / f.x, (y + 0.5 - c.y) / f.y, 1.0);
      glm::dvec2 source;
      if (!colmap_project(camera, ray, source)) {
        throw std::runtime_error(std::format("training does not support camera model {}", colmap_model_name(camera.model)));
      }
      const double sx = source.x - 0.5, sy = source.y - 0.5;
      const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= image.width || y0 + 1 >= image.height) continue;
      const double u = sx - x0, v = sy - y0;
      const std::uint8_t *row0 = image.pixels.data() + (static_cast<std::size_t>(y0) * image.width + x0) * 3;
      const std::uint8_t *row1 = row0 + static_cast<std::size_t>(image.width) * 3;
      for (int k = 0; k < 3; ++k) {
        const double top = row0[k] * (1.0 - u) + row0[k + 3] * u;
        const double bottom = row1[k] * (1.0 - u) + row1[k + 3] * u;
        out.pixels[(static_cast<std::size_t>(y) * out.width + x) * 3 + k] = static_cast<std::uint8_t>(std::lround(top * (1.0 - v) + bottom * v));
      }
    }
  }
  return out;
}

bool has_distortion(const ColmapCamera &camera) {
  const std::size_t first = camera.model == 0 ? 3 : camera.model == 1 ? 4 : camera.model == 2 || camera.model == 3 ? 3 : 4;
  return std::any_of(camera.params.begin() + static_cast<std::ptrdiff_t>(std::min(first, camera.params.size())), camera.params.end(),
                     [](double p) { return p != 0.0; });
}

std::vector<std::uint8_t> read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

TrainView load_view(const fs::path &project, const ColmapImage &image, const ColmapCamera &camera, int maxSize) {
  const fs::path file = project / image.name;
  RgbImage rgb = decode_image(read_file(file));
  if (static_cast<std::uint64_t>(rgb.width) != camera.width || static_cast<std::uint64_t>(rgb.height) != camera.height) {
    throw std::runtime_error(std::format("{} is {}x{} but its camera says {}x{}", file.string(), rgb.width, rgb.height, camera.width, camera.height));
  }
  if (has_distortion(camera)) rgb = undistort(rgb, camera);
  const int original = std::max(rgb.width, rgb.height);
  rgb = downscale_to_fit(std::move(rgb), maxSize);
  const double scale = static_cast<double>(std::max(rgb.width, rgb.height)) / original;

  TrainView view;
  view.name = image.name;
  view.camera = posed_camera(image, camera, scale);
  view.camera.width = rgb.width;
  view.camera.height = rgb.height;
  view.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
  for (std::size_t i = 0; i < static_cast<std::size_t>(rgb.width) * rgb.height; ++i) {
    std::copy_n(rgb.pixels.data() + i * 3, 3, view.rgba.data() + i * 4);
    view.rgba[i * 4 + 3] = 255;
  }
  return view;
}

// The distance to each point's 3 nearest neighbours, as a starting size. A
// uniform grid keeps this close to linear in the number of points.
std::vector<float> neighbour_distances(const std::vector<ColmapPoint> &points) {
  glm::dvec3 low(std::numeric_limits<double>::max()), high(std::numeric_limits<double>::lowest());
  for (const ColmapPoint &p : points) {
    low = glm::min(low, p.position);
    high = glm::max(high, p.position);
  }
  const glm::dvec3 size = high - low;
  const double cell = std::max(std::cbrt(std::max(size.x * size.y * size.z, 1e-30) / static_cast<double>(points.size()) * 8.0),
                               std::max({size.x, size.y, size.z}) / 512.0);
  using Key = std::tuple<int, int, int>;
  const auto key_of = [&](const glm::dvec3 &p) {
    return Key{static_cast<int>((p.x - low.x) / cell), static_cast<int>((p.y - low.y) / cell), static_cast<int>((p.z - low.z) / cell)};
  };
  std::map<Key, std::vector<std::size_t>> grid;
  for (std::size_t i = 0; i < points.size(); ++i) grid[key_of(points[i].position)].push_back(i);

  std::vector<float> out(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto [cx, cy, cz] = key_of(points[i].position);
    double best[3] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for (int r = 1; r <= 4 && best[2] == std::numeric_limits<double>::max(); ++r) {
      best[0] = best[1] = best[2] = std::numeric_limits<double>::max();
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            const auto it = grid.find({cx + dx, cy + dy, cz + dz});
            if (it == grid.end()) continue;
            for (const std::size_t j : it->second) {
              if (j == i) continue;
              const glm::dvec3 d = points[j].position - points[i].position;
              const double dist = glm::dot(d, d);
              if (dist < best[2]) {
                best[2] = dist;
                if (best[2] < best[1]) std::swap(best[1], best[2]);
                if (best[1] < best[0]) std::swap(best[0], best[1]);
              }
            }
          }
        }
      }
    }
    const double mean = best[2] == std::numeric_limits<double>::max()
                            ? cell
                            : (std::sqrt(best[0]) + std::sqrt(best[1]) + std::sqrt(best[2])) / 3.0;
    out[i] = static_cast<float>(std::max(mean, 1e-7));
  }
  return out;
}

GaussianCloud initial_cloud(const std::vector<ColmapPoint> &points, int shDegree) {
  GaussianCloud cloud;
  cloud.resize(points.size(), shDegree);
  const std::vector<float> distances = neighbour_distances(points);
  const auto coefficients = static_cast<std::size_t>(cloud.shCoeffs());
  const float logitOpacity = std::log(kInitialOpacity / (1.f - kInitialOpacity));
  for (std::size_t i = 0; i < points.size(); ++i) {
    const ColmapPoint &p = points[i];
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.positions[i * 3 + a] = static_cast<float>(p.position[static_cast<int>(a)]);
      cloud.scales[i * 3 + a] = std::log(distances[i]);
      cloud.sh[i * coefficients * 3 + a] = (static_cast<float>(p.color[a]) / 255.f - 0.5f) / kShC0;
    }
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = logitOpacity;
  }
  return cloud;
}

void write_preview(const std::vector<float> &render, const TrainView &view, const fs::path &path) {
  // Rendered on the left, the real image on the right.
  RgbImage out;
  out.width = view.camera.width * 2;
  out.height = view.camera.height;
  out.pixels.resize(static_cast<std::size_t>(out.width) * out.height * 3);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < view.camera.width; ++x) {
      const std::size_t at = static_cast<std::size_t>(y) * view.camera.width + x;
      for (std::size_t c = 0; c < 3; ++c) {
        const std::size_t base = static_cast<std::size_t>(y) * out.width;
        out.pixels[(base + x) * 3 + c] = static_cast<std::uint8_t>(std::clamp(render[at * 4 + c], 0.f, 1.f) * 255.f + 0.5f);
        out.pixels[(base + view.camera.width + x) * 3 + c] = view.rgba[at * 4 + c];
      }
    }
  }
  write_image(out, path, 95);
}

// Half the size, each pixel the mean of four. Pixel x of the result covers
// pixels 2x and 2x + 1, so the focal length and principal point just halve.
TrainView half_size(const TrainView &view) {
  TrainView out;
  out.name = view.name;
  out.camera = view.camera;
  out.camera.width = view.camera.width / 2;
  out.camera.height = view.camera.height / 2;
  out.camera.fx *= 0.5f;
  out.camera.fy *= 0.5f;
  out.camera.cx *= 0.5f;
  out.camera.cy *= 0.5f;
  const auto w = static_cast<std::size_t>(out.camera.width), h = static_cast<std::size_t>(out.camera.height);
  const auto stride = static_cast<std::size_t>(view.camera.width) * 4;
  out.rgba.resize(w * h * 4);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const std::uint8_t *p = view.rgba.data() + 2 * y * stride + 2 * x * 4;
      for (std::size_t k = 0; k < 4; ++k) {
        out.rgba[(y * w + x) * 4 + k] = static_cast<std::uint8_t>((p[k] + p[k + 4] + p[stride + k] + p[stride + k + 4] + 2) / 4);
      }
    }
  }
  return out;
}

} // namespace

// How a training view's camera exposed its photo, learned with its own Adam:
// a view comes round only once per pass over all of them, too seldom to share
// the Gaussians' optimiser.
struct TrainSession::ViewExposure {
  Exposure value = kNeutralExposure;
  std::array<float, 12> first{}, second{};
  int steps = 0;

  void step(const std::array<float, 12> &gradient) {
    ++steps;
    const float correction1 = 1.f - std::pow(0.9f, static_cast<float>(steps));
    const float correction2 = 1.f - std::pow(0.999f, static_cast<float>(steps));
    for (std::size_t k = 0; k < 12; ++k) {
      first[k] = 0.9f * first[k] + 0.1f * gradient[k];
      second[k] = 0.999f * second[k] + 0.001f * gradient[k] * gradient[k];
      value[k] -= kExposureRate * (first[k] / correction1) / (std::sqrt(second[k] / correction2) + 1e-15f);
    }
  }
};

// The same for a training view's camera. Its rotation and its translation
// have their own steps, as they are in different units.
struct TrainSession::ViewPose {
  std::array<float, 6> first{}, second{};
  int steps = 0;

  std::array<float, 6> step(const std::array<float, 6> &gradient, float rotationRate, float translationRate) {
    ++steps;
    const float correction1 = 1.f - std::pow(0.9f, static_cast<float>(steps));
    const float correction2 = 1.f - std::pow(0.999f, static_cast<float>(steps));
    std::array<float, 6> move{};
    for (std::size_t k = 0; k < 6; ++k) {
      first[k] = 0.9f * first[k] + 0.1f * gradient[k];
      second[k] = 0.999f * second[k] + 0.001f * gradient[k] * gradient[k];
      move[k] = -(k < 3 ? rotationRate : translationRate) * (first[k] / correction1) / (std::sqrt(second[k] / correction2) + 1e-15f);
    }
    return move;
  }
};

namespace {

// Through a temporary file, so that a crash never leaves half a file.
void save_through_temporary(const GaussianCloud &cloud, const fs::path &path, CloudFormat format, const CgcOptions &options) {
  const fs::path temporary = fs::path(path).concat(".tmp");
  save_cloud(cloud, temporary, format, options);
  fs::rename(temporary, path);
}

} // namespace

Camera posed_camera(const ColmapImage &image, const ColmapCamera &camera, double scale) {
  Camera out;
  const glm::dvec2 f = focal_of(camera) * scale, c = principal_of(camera) * scale;
  out.fx = static_cast<float>(f.x);
  out.fy = static_cast<float>(f.y);
  out.cx = static_cast<float>(c.x);
  out.cy = static_cast<float>(c.y);
  out.width = static_cast<int>(std::lround(static_cast<double>(camera.width) * scale));
  out.height = static_cast<int>(std::lround(static_cast<double>(camera.height) * scale));

  // World to camera, from the quaternion (w, x, y, z).
  const double w = image.rotation[0], x = image.rotation[1], y = image.rotation[2], z = image.rotation[3];
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  const double qw = w / n, qx = x / n, qy = y / n, qz = z / n;
  const glm::dmat3 R(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy),
                     2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx),
                     2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
  glm::dmat4 viewMatrix(R);
  viewMatrix[3] = glm::dvec4(image.translation, 1.0);
  out.view = glm::mat4(viewMatrix);
  out.position = glm::vec3(image.center());
  return out;
}

std::array<fs::path, 2> output_paths(const TrainOptions &options) {
  return {fs::path(options.out).replace_extension(".ply"), fs::path(options.out).replace_extension(".cgc")};
}

SceneAlignment scene_alignment(const ColmapModel &model) {
  std::vector<Camera> cameras;
  cameras.reserve(model.images.size());
  for (const auto &[id, image] : model.images) cameras.push_back(posed_camera(image, model.cameras.at(image.cameraId)));
  // The sparse points as opaque Gaussians, which is all the floor search looks at.
  GaussianCloud points;
  points.resize(model.points.size(), 0);
  for (std::size_t i = 0; i < model.points.size(); ++i) {
    for (int a = 0; a < 3; ++a) points.positions[i * 3 + static_cast<std::size_t>(a)] = static_cast<float>(model.points[i].position[a]);
    points.opacities[i] = 8.f;
  }
  SceneAlignment alignment;
  alignment.floor = estimate_floor(points, cameras);
  alignment.transform = floor_alignment(alignment.floor, points);
  alignment.tiltDeg = glm::degrees(std::acos(std::clamp(glm::dot(alignment.floor.normal, estimate_up(cameras)), -1.0, 1.0)));
  return alignment;
}

std::string SceneAlignment::describe() const {
  return floor.fitted ? std::format("floor plane through {} of {} sparse points, {:.1f} deg off the cameras' up", floor.inliers, floor.candidates, tiltDeg)
                      : std::string("no floor plane in the sparse points, levelled by the cameras' up");
}

double growth_factor(const TrainOptions &options) noexcept {
  constexpr double kPerRun = 4.0, kRunLength = 7000.0;
  if (options.maxGrowth >= 0.0) return options.maxGrowth;
  return kPerRun * std::max(1.0, static_cast<double>(options.iterations) / kRunLength);
}

TrainData load_train_data(const TrainOptions &options, const std::function<void(std::size_t, std::size_t)> &progress) {
  TrainData data;
  data.model = load_colmap_model(options.project / "sparse" / "0");
  const ColmapModel &model = data.model;
  if (model.points.empty() || model.images.empty()) throw std::runtime_error("the sparse model has no points or no images");

  // In file name order, as 3DGS and gsplat sort them, not by image id: COLMAP
  // numbers images in the order it read them, so an imported model would
  // otherwise hold out different photos than one from `splat sfm`.
  std::vector<const ColmapImage *> chosen;
  for (const auto &[id, image] : model.images) chosen.push_back(&image);
  std::stable_sort(chosen.begin(), chosen.end(), [](const ColmapImage *a, const ColmapImage *b) { return a->name < b->name; });
  if (options.maxViews > 0 && chosen.size() > static_cast<std::size_t>(options.maxViews)) chosen.resize(static_cast<std::size_t>(options.maxViews));

  // Decoding and undistorting is most of the time: one image per thread.
  std::vector<TrainView> views(chosen.size());
  std::atomic<std::size_t> done{0};
  std::mutex mutex; // Guards `error` and the progress callback
  std::exception_ptr error;
  parallel_for(chosen.size(), 0, [&](std::size_t i) {
    {
      const std::lock_guard lock(mutex);
      if (error) return;
    }
    try {
      views[i] = load_view(options.project, *chosen[i], model.cameras.at(chosen[i]->cameraId), options.maxSize);
    } catch (...) {
      const std::lock_guard lock(mutex);
      if (!error) error = std::current_exception();
      return;
    }
    const std::size_t finished = ++done;
    const std::lock_guard lock(mutex);
    if (progress) progress(finished, chosen.size());
  });
  if (error) std::rethrow_exception(error);

  // The first view and every nth after it, as 3DGS and gsplat split, so that
  // held-out PSNRs compare with published ones. On the Mip-NeRF 360 bicycle
  // scene the photos alternate between tilted down and tilted up, and holding
  // out the nth instead of the first tested only the upward ones: 4 dB less.
  for (std::size_t i = 0; i < views.size(); ++i) {
    if (options.holdout > 0 && i % static_cast<std::size_t>(options.holdout) == 0) {
      data.holdoutViews.push_back(std::move(views[i]));
    } else {
      data.trainViews.push_back(std::move(views[i]));
    }
  }
  if (data.trainViews.empty()) throw std::runtime_error("no training views");

  // Coarse to fine, as in DashGaussian: the first iterations only need the
  // broad shapes, which smaller images show as well for far fewer pixels.
  // smallerViews[l][i] is view i at 1 / 2^(l + 1) of its size.
  if (!options.fullResolution) {
    data.smallerViews.resize(kCoarseLevels);
    for (std::size_t level = 0; level < kCoarseLevels; ++level) {
      for (const TrainView &view : level == 0 ? data.trainViews : data.smallerViews[level - 1]) {
        data.smallerViews[level].push_back(half_size(view));
      }
    }
    if (options.adaptiveSchedule) {
      std::array<double, 3> energy{};
      const std::size_t step = std::max<std::size_t>(data.trainViews.size() / kSpectrumViews, 1);
      for (std::size_t i = 0; i < data.trainViews.size(); i += step) {
        const TrainView &view = data.trainViews[i];
        const std::vector<double> x = frequency_energy(view.rgba, view.camera.width, view.camera.height, {1, 2, 4});
        for (std::size_t k = 0; k < energy.size(); ++k) energy[k] += x[k];
      }
      data.coarseUntil = resolution_switches(energy[0], energy[1], energy[2]);
    }
  }

  // The scene's size sets the position learning rate and the near plane.
  glm::dvec3 middle(0.0);
  for (const auto &[id, image] : model.images) middle += image.center();
  middle /= static_cast<double>(model.images.size());
  double extent = 0.0;
  for (const auto &[id, image] : model.images) extent = std::max(extent, glm::length(image.center() - middle));
  extent = std::max(extent * 1.1, 1e-6);
  std::vector<double> depths;
  for (const ColmapPoint &p : model.points) depths.push_back(glm::length(p.position - middle));
  // Cameras close together say little about the size of what they look at.
  if (options.pointExtent) extent = std::max(std::accumulate(depths.begin(), depths.end(), 0.0) / static_cast<double>(depths.size()), 1e-6);
  data.extent = extent;
  std::nth_element(depths.begin(), depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2), depths.end());
  data.typicalDepth = std::max(depths[depths.size() / 2] + extent, 1e-6);

  // Besides the sparse points, more from matching patches between the photos.
  data.startingPoints = model.points;
  if (options.sparsePoints > 0 && data.startingPoints.size() > options.sparsePoints) {
    std::mt19937 rng(options.seed);
    std::shuffle(data.startingPoints.begin(), data.startingPoints.end(), rng);
    data.startingPoints.resize(options.sparsePoints);
  }
  if (options.densePoints > 0) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<DensePointsView> densePointsViews;
    for (const TrainView &view : data.trainViews) densePointsViews.push_back({view.camera, view.rgba});
    const std::vector<ColmapPoint> dense = dense_points(densePointsViews, model.points, options.densePoints);
    data.startingPoints.insert(data.startingPoints.end(), dense.begin(), dense.end());
    std::println("Patch matching found {} more starting points in {:.1f} s", dense.size(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  }
  return data;
}

TrainSession::TrainSession(vk::Context &ctx, const TrainOptions &options, TrainData data, std::function<void(const std::string &)> log)
    : ctx_(ctx), options_(options), data_(std::move(data)), log_(std::move(log)), rng_(options.seed), densifyRng_(options.seed + 1),
      growth_(data_.startingPoints.size(), growth_factor(options)), lastOpacityReset_(-kCheckpointRecovery) {
  if (!log_) log_ = [](const std::string &line) { std::println("{}", line); };
  if (options_.adaptiveSchedule && !options_.fullResolution) {
    log_(std::format("Image sizes: quarter until {:.0f}% of training, half until {:.0f}%, then full", 100.0 * data_.coarseUntil[1],
                     100.0 * data_.coarseUntil[0]));
  }
  trainerOptions_.lambdaSsim = options_.lambdaSsim;
  trainerOptions_.selectiveAdam = !options_.fullAdam;
  trainerOptions_.poseGradients = options_.refinePoses;
  trainerOptions_.opacityReg = options_.opacityReg;
  trainerOptions_.scaleReg = options_.scaleReg;
  trainerOptions_.positionNoise = options_.positionNoise;
  trainerOptions_.mipFilter2d = options_.mipFilter2d;
  trainerOptions_.nearPlane = static_cast<float>(data_.typicalDepth * 0.01);
  // Asked for before anything is allocated, so that it is all ours to spend.
  budgetGiB_ = options_.vramBudgetGiB;
  if (budgetGiB_ <= 0.0) budgetGiB_ = options_.vramShare * static_cast<double>(ctx_.free_device_memory()) / (1024.0 * 1024.0 * 1024.0);
  const auto budgetBytes = static_cast<std::uint64_t>(budgetGiB_ * 1024.0 * 1024.0 * 1024.0);
  const std::uint64_t reserve = std::max(kBudgetReserve, budgetBytes / 20);
  trainerOptions_.memoryBudget = budgetBytes > reserve ? budgetBytes - reserve : 0;
  std::uint64_t largestView = 0;
  for (const TrainView &view : data_.trainViews) {
    largestView = std::max(largestView, static_cast<std::uint64_t>(view.camera.width) * view.camera.height);
  }
  if (largestView * Trainer::kBytesPerPixel >= trainerOptions_.memoryBudget) {
    throw std::runtime_error(std::format("{:.2f} GiB of VRAM is not enough for images of this size; try --max-size", budgetGiB_));
  }
  trainer_ = std::make_unique<Trainer>(ctx_, initial_cloud(data_.startingPoints, options_.shDegree), trainerOptions_);

  // The schedule of the paper for 30000 iterations, proportional when shorter.
  densifyFrom_ = std::min(500, options_.iterations / 20 + 1);
  densifyUntil_ = options_.iterations / 2;
  pruneAgainAt_ = (densifyUntil_ + options_.iterations) / 2;
  shInterval_ = std::max(1, std::min(1000, options_.iterations / 10));
  trainer_->set_active_sh_degree(0);
  exposures_.resize(data_.trainViews.size());
  poses_.resize(data_.trainViews.size());
  positionStart_ = 1.6e-4 * data_.extent;
  positionEnd_ = 1.6e-6 * data_.extent;
  log_(std::format("Training from {} Gaussians on {} views ({} held out) at {}x{} for {} iterations, SH degree {}, VRAM budget {:.1f} GiB{}",
                   data_.startingPoints.size(), data_.trainViews.size(), data_.holdoutViews.size(), data_.trainViews[0].camera.width,
                   data_.trainViews[0].camera.height, options_.iterations, options_.shDegree, budgetGiB_,
                   trainer_->uses_fixed_point() ? " (fixed-point gradients: this GPU has no float atomics)" : ""));
  if (options_.alignFloor) {
    alignment_ = scene_alignment(data_.model);
    log_(std::format("Saved with the floor level: {}", alignment_->describe()));
  }
  refresh_filter(1);
}

TrainSession::~TrainSession() = default;

// A view that does not fit gives up its largest Gaussians, rather than the run
// ending. They grow back where they are needed, within the budget.
template <class F> auto TrainSession::fitting(F &&work) {
  for (;;) {
    try {
      return work();
    } catch (const BudgetExceeded &e) {
      const std::size_t removed = trainer_->remove_largest_gaussians(0.05);
      filterLevel_ = ~std::size_t{0};
      log_(std::format("  {}: removed the {} largest Gaussians", e.what(), removed));
      if (removed == 0) throw;
    }
  }
}

std::size_t TrainSession::coarse_level(int iteration) const noexcept {
  if (data_.smallerViews.empty()) return kCoarseLevels;
  const double progress = static_cast<double>(iteration - 1) / std::max(options_.iterations - 1, 1);
  for (std::size_t level = kCoarseLevels; level-- > 0;) {
    if (progress < data_.coarseUntil[level]) return level;
  }
  return kCoarseLevels;
}

const TrainView &TrainSession::view_at(std::size_t index, int iteration) const {
  const std::size_t level = coarse_level(iteration);
  return level == kCoarseLevels ? data_.trainViews[index] : data_.smallerViews[level][index];
}

void TrainSession::refill_order() {
  if (!order_.empty()) return;
  order_.resize(data_.trainViews.size());
  std::iota(order_.begin(), order_.end(), std::size_t{0});
  std::shuffle(order_.begin(), order_.end(), rng_);
}

// Mip-Splatting's 3D smoothing filter, from the training cameras of the
// current image size; recomputed after every change of the cloud or the size.
void TrainSession::refresh_filter(int iteration) {
  if (!options_.mipFilter3d) return;
  std::vector<Camera> cameras;
  cameras.reserve(data_.trainViews.size());
  for (std::size_t i = 0; i < data_.trainViews.size(); ++i) cameras.push_back(view_at(i, iteration).camera);
  filter3d_ = smoothing_filter(trainer_->cloud(), cameras);
  trainer_->set_filter3d(filter3d_);
  filterLevel_ = coarse_level(iteration);
}

StepStats TrainSession::step() {
  if (finished()) throw std::logic_error("training has already finished");
  const auto start = std::chrono::steady_clock::now();
  const int iteration = ++iteration_;
  Trainer &trainer = *trainer_;
  const TrainOptions &options = options_;
  refill_order();
  lastView_ = order_.back();
  const std::size_t viewIndex = lastView_;
  const TrainView &view = view_at(viewIndex, iteration);
  order_.pop_back();
  refill_order();
  const TrainView &nextView = view_at(order_.back(), iteration + 1);
  if (options.mipFilter3d && coarse_level(iteration) != filterLevel_) refresh_filter(iteration);

  const double progress = static_cast<double>(iteration - 1) / std::max(options.iterations - 1, 1);
  // With the adaptive schedule, held while the image is still growing.
  const double held = options.adaptiveSchedule && !options.fullResolution ? data_.coarseUntil[0] : 0.0;
  const double decay = std::clamp((progress - held) / std::max(1.0 - held, 1e-9), 0.0, 1.0);
  rates_.position = static_cast<float>(positionStart_ * std::pow(positionEnd_ / positionStart_, decay));
  trainer.set_active_sh_degree((iteration - 1) / shInterval_);
  trainer.set_exposure(exposures_[viewIndex].value);
  if (options.secondOrder && iteration > densifyUntil_) {
    if (rates_.trustRegion == 0.f) trainer.enable_second_order();
    const double rest = static_cast<double>(iteration - densifyUntil_) / std::max(options.iterations - densifyUntil_, 1);
    rates_.trustRegion = static_cast<float>(kTrustStart * std::pow(kTrustEnd / kTrustStart, rest));
  }
  if (options.shDropout > 0.f) {
    trainer.set_sh_dropout(options.shDropout, progress < 0.2 ? 0 : progress < 0.4 ? 1 : 2, static_cast<std::uint32_t>(iteration));
  }
  const StepStats stats = fitting([&] { return trainer.step(view.camera, view.rgba, rates_, nextView.rgba); });
  if (rates_.trustRegion > 0.f && (iteration - densifyUntil_) % kCurvatureInterval == 1) trainer.update_curvature();
  if (options.exposure) exposures_[viewIndex].step(stats.exposureGradient);
  if (options.refinePoses && progress >= kPoseStart) {
    const std::array<float, 6> move =
        poses_[viewIndex].step(stats.poseGradient, rates_.position / static_cast<float>(data_.extent), rates_.position);
    // The smaller images hold the same camera at other intrinsics, which this keeps.
    Camera &full = data_.trainViews[viewIndex].camera;
    const Camera moved = refined_pose(full, move);
    full.view = moved.view;
    full.position = moved.position;
    for (auto &level : data_.smallerViews) {
      level[viewIndex].camera.view = moved.view;
      level[viewIndex].camera.position = moved.position;
    }
  }
  if (!std::isfinite(stats.loss)) throw std::runtime_error(std::format("the loss became {} at iteration {}", stats.loss, iteration));
  if (iteration == 1) report_.firstLoss = stats.loss;
  report_.lastLoss = stats.loss;
  lossSum_ += stats.loss;
  ++lossCount_;
  if (stats.visibleSplats > 0) {
    entriesPerVisible_ = 0.98 * entriesPerVisible_ + 0.02 * stats.tileEntries / stats.visibleSplats;
  }
  report_.peakGpuBytes = std::max(report_.peakGpuBytes, trainer.gpu_bytes());

  // Measured now, before this iteration's pruning and opacity reset: a cloud
  // just after a reset is nearly transparent and its PSNR says nothing about
  // what training reached. Reported below, once the Gaussian count is final.
  const bool reportDue = iteration % options.reportEvery == 0 || iteration == options.iterations;
  double probePsnr = 0.0;
  if (reportDue) {
    const TrainView &probe = data_.holdoutViews.empty() ? data_.trainViews[0] : data_.holdoutViews[0];
    probePsnr = psnr_of(fitting([&] { return trainer.render(probe.camera); }), probe.rgba);
  }

  // Before densification, which may cap the opacities the ranking depends on.
  if (options.pruneShare > 0.0 && (iteration == densifyUntil_ || iteration == pruneAgainAt_)) {
    const auto pruneStart = std::chrono::steady_clock::now();
    const std::size_t before = trainer.gaussian_count();
    // The first time, the statistics only cover the last densification
    // interval, so every view is rendered once to score them. The second
    // time, they have run since the first and seen every view many times.
    if (iteration == densifyUntil_) {
      trainer.reset_densify_stats();
      for (std::size_t i = 0; i < data_.trainViews.size(); ++i) {
        trainer.set_exposure(exposures_[i].value);
        fitting([&] { return trainer.collect_statistics(data_.trainViews[i].camera, data_.trainViews[i].rgba); });
      }
    }
    const std::size_t removed = trainer.remove_least_contributing_gaussians(options.pruneShare);
    filterLevel_ = ~std::size_t{0};
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - pruneStart).count();
    pruneSeconds_ += seconds;
    log_(std::format("  iteration {}: removed {} of {} Gaussians, those adding least to the views ({:.1f} s)", iteration, removed, before,
                     seconds));
  }

  if (options.densify && iteration >= densifyFrom_ && iteration <= densifyUntil_ && iteration % options.densifyInterval == 0) {
    // What a Gaussian costs, with the tile entries it brings and room for them to grow.
    const double perGaussian = static_cast<double>(trainer.bytes_per_gaussian()) + 1.5 * Trainer::kBytesPerTileEntry * entriesPerVisible_;
    const std::uint64_t available = trainerOptions_.memoryBudget;
    const std::uint64_t pixelBytes = static_cast<std::uint64_t>(view.camera.width) * view.camera.height * Trainer::kBytesPerPixel;
    DensifyOptions densifyOptions;
    const double densifyProgress = static_cast<double>(iteration) / std::max(densifyUntil_, 1);
    const auto rise = static_cast<float>(std::pow(options.thresholdRise, densifyProgress - 0.5));
    densifyOptions.gradientThreshold = options.gradientThreshold * rise;
    densifyOptions.splitThreshold = options.splitThreshold * rise;
    densifyOptions.denseScale = static_cast<float>(options.denseFraction * data_.extent);
    densifyOptions.maxWorldScale = static_cast<float>(kMaxWorldFraction * data_.extent);
    densifyOptions.maxScreenRadius = static_cast<float>(kMaxScreenFraction * std::max(view.camera.width, view.camera.height));
    densifyOptions.pruneLarge = iteration > kOpacityResetInterval && !options.significancePrune;
    if (options.significancePrune) densifyOptions.transparentShare = 0.5;
    densifyOptions.exploreSplits = densifyOptions.exploreSeeds = static_cast<std::size_t>(options.explore);
    densifyOptions.edgeSplit = options.edgeSplit;
    densifyOptions.maxGaussians = available > pixelBytes ? static_cast<std::size_t>(static_cast<double>(available - pixelBytes) / perGaussian) : 0;
    if (options.maxGaussians > 0) densifyOptions.maxGaussians = std::min(densifyOptions.maxGaussians, options.maxGaussians);
    const auto densifyStart = std::chrono::steady_clock::now();
    if (options.scoreViews > 0 && options.densifyMode == TrainOptions::DensifyMode::Adc) { // Scores play no part in relocation
      densifyOptions.scoredViews = options.scoreViews;
      densifyOptions.minMarkedPixels = options.scoreMinPixels;
      std::uniform_int_distribution<std::size_t> pick(0, data_.trainViews.size() - 1);
      for (int k = 0; k < options.scoreViews; ++k) {
        const std::size_t i = pick(densifyRng_);
        const TrainView &scored = view_at(i, iteration);
        // A pixel of a smaller image stands for several of the full one.
        const float area = static_cast<float>(data_.trainViews[i].camera.width) / static_cast<float>(scored.camera.width);
        trainer.set_exposure(exposures_[i].value);
        fitting([&] { return trainer.score_view(scored.camera, scored.rgba, kScoreErrorShare, area * area); });
      }
    }
    if (options.adaptiveSchedule) {
      const std::size_t level = coarse_level(iteration);
      const double factor = level == kCoarseLevels ? 1.0 : std::pow(2.0, static_cast<double>(level + 1));
      densifyOptions.maxGaussians = std::min(densifyOptions.maxGaussians, growth_.limit(factor, progress));
    } else {
      // A fixed cap (see growth_factor()): at 7000 iterations on the bicycle
      // scene quality saturates from about 5 times the starting points, and 4
      // keeps it above 100 it/s.
      densifyOptions.maxGaussians = std::min(densifyOptions.maxGaussians, growth_.limit(1.0, 1.0));
    }
    if (options.densifyMode == TrainOptions::DensifyMode::Mcmc) {
      RelocateOptions relocateOptions;
      relocateOptions.maxGaussians = densifyOptions.maxGaussians;
      trainer.relocate(relocateOptions, densifyRng_);
    } else {
      const DensifyResult result = trainer.densify(densifyOptions, densifyRng_);
      if (options.adaptiveSchedule) growth_.observe(result.cloned + result.split + result.heldBack);
      report_.heldBack += result.heldBack;
      // Not on the last round: as in the paper, and the cloud would have
      // no densification left to recover with.
      if (iteration % kOpacityResetInterval == 0 && iteration < densifyUntil_) {
        trainer.reset_opacity(kResetOpacity);
        lastOpacityReset_ = iteration;
      }
    }
    refresh_filter(iteration);
    densifySeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - densifyStart).count();
    ++densifyRounds_;
  }

  if (options.recoveryPrune > 0.0 && lastOpacityReset_ > 0 && iteration == lastOpacityReset_ + kRecoveryDelay) {
    const std::size_t before = trainer.gaussian_count();
    const std::size_t removed = trainer.remove_least_opaque_gaussians(options.recoveryPrune);
    log_(std::format("  iteration {}: removed {} of {} Gaussians, the least opaque since the reset", iteration, removed, before));
    refresh_filter(iteration);
  }
  trainingSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  if (reportDue) {
    lastPsnr_ = probePsnr;
    report_.progressPsnr.emplace_back(iteration, probePsnr);
    log_(std::format("  iteration {:6}: loss {:.4f}, {} PSNR {:.2f} dB, {} Gaussians, {} visible, {} tile entries, {:.2f} GiB, {:.1f} it/s", iteration,
                     lossSum_ / lossCount_, data_.holdoutViews.empty() ? "training view" : "held-out view", probePsnr, trainer.gaussian_count(),
                     stats.visibleSplats, stats.tileEntries, static_cast<double>(trainer.gpu_bytes()) / (1024.0 * 1024.0 * 1024.0),
                     iteration / std::max(trainingSeconds_, 1e-9)));
    lossSum_ = 0.0;
    lossCount_ = 0;
  }
  if (options.saveEvery > 0 && iteration % options.saveEvery == 0) checkpointDue_ = true;
  if (checkpointDue_ && iteration - lastOpacityReset_ >= kCheckpointRecovery && iteration < options.iterations) {
    checkpointDue_ = false;
    save();
  }
  return stats;
}

std::function<void()> TrainSession::save_outputs(bool quick) {
  if (options_.mipFilter3d && (filterLevel_ == ~std::size_t{0} || filter3d_.size() != trainer_->gaussian_count())) {
    refresh_filter(std::max(iteration_, 1));
  }
  auto cloud = std::make_shared<GaussianCloud>(trainer_->cloud());
  if (options_.mipFilter3d) bake_filter(*cloud, filter3d_);
  if (alignment_) transform_cloud(*cloud, alignment_->transform);
  const auto [ply, cgc] = output_paths(options_);
  save_through_temporary(*cloud, ply, CloudFormat::Ply, {});
  // CGC refuses NaN and infinity, which a few Gaussians can pick up after
  // densification has stopped pruning them, so those are left out of it.
  if (const std::size_t removed = remove_non_finite(*cloud); removed > 0) {
    log_(std::format("  left out {} Gaussians with NaN or infinite values from {}", removed, cgc.filename().string()));
  }
  // Quick settings for checkpoints, so that they don't hold up training. At
  // full quality a million Gaussians take about a minute.
  return [cloud, cgc, quick, log = log_] {
    const auto start = std::chrono::steady_clock::now();
    if (!quick) log(std::format("Compressing {} Gaussians into {}", cloud->size(), cgc.filename().string()));
    save_through_temporary(*cloud, cgc, CloudFormat::Cgc, CgcOptions{.fast = quick});
    if (!quick) {
      log(std::format("Wrote {} in {:.1f} s ({:.1f} MB)", cgc.string(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), static_cast<double>(fs::file_size(cgc)) / 1e6));
    }
  };
}

void TrainSession::save() {
  save_outputs(true)();
  const auto [ply, cgc] = output_paths(options_);
  log_(std::format("  iteration {:6}: checkpoint saved to {} and {}", iteration_, ply.string(), cgc.filename().string()));
}

TrainReport TrainSession::finish() {
  std::function<void()> writeCgc;
  TrainReport report = finish(writeCgc);
  writeCgc();
  return report;
}

TrainReport TrainSession::finish(std::function<void()> &writeCgc) {
  Trainer &trainer = *trainer_;
  report_.iterationsPerSecond = iteration_ / std::max(trainingSeconds_, 1e-9);

  trainer.set_active_sh_degree(options_.shDegree);
  trainer.set_sh_dropout(0.f, 0, 0);
  // Training views through the exposures they learned, or they would be
  // compared with photos the Gaussians were never fitted to.
  report_.trainPsnr = 0.0;
  for (std::size_t i = 0; i < data_.trainViews.size(); ++i) {
    std::vector<float> rendered = fitting([&] { return trainer.render(data_.trainViews[i].camera); });
    apply_exposure(rendered, exposures_[i].value);
    report_.trainPsnr += psnr_of(rendered, data_.trainViews[i].rgba) / static_cast<double>(data_.trainViews.size());
  }
  // A held-out view has no learned exposure, so it is scored both as it is and
  // with one fitted to its own photo. Only the second is fair to --exposure.
  report_.holdoutPsnr = report_.holdoutFittedPsnr = 0.0;
  for (const TrainView &view : data_.holdoutViews) {
    const std::vector<float> rendered = fitting([&] { return trainer.render(view.camera); });
    report_.holdoutPsnr += psnr_of(rendered, view.rgba) / static_cast<double>(data_.holdoutViews.size());
    report_.holdoutFittedPsnr +=
        cross_fitted_psnr(rendered, view.rgba, view.camera.width, view.camera.height) / static_cast<double>(data_.holdoutViews.size());
  }

  report_.gaussians = trainer.gaussian_count();
  writeCgc = save_outputs(false);
  write_preview(trainer.render(data_.trainViews[0].camera), data_.trainViews[0], options_.project / "preview_train.png");
  if (!data_.holdoutViews.empty()) {
    write_preview(trainer.render(data_.holdoutViews[0].camera), data_.holdoutViews[0], options_.project / "preview_holdout.png");
    log_(std::format("PSNR {:.2f} dB on training views, {:.2f} dB on held-out views ({:.2f} dB with exposure fitted to each)", report_.trainPsnr,
                     report_.holdoutPsnr, report_.holdoutFittedPsnr));
  } else {
    log_(std::format("PSNR {:.2f} dB on training views", report_.trainPsnr));
  }
  log_(std::format("{} Gaussians, peak {:.2f} GiB of VRAM{}", report_.gaussians, static_cast<double>(report_.peakGpuBytes) / (1024.0 * 1024.0 * 1024.0),
                   report_.heldBack > 0 ? std::format(" ({} densifications held back by the budget)", report_.heldBack) : ""));
  std::string profile;
  double profileTotal = 0.0;
  for (const auto &[stage, ms] : trainer.timings()) {
    profile += std::format("{} {:.2f}, ", stage, ms);
    profileTotal += ms;
  }
  log_(std::format("GPU time per iteration: {:.1f} ms ({})", profileTotal, profile.substr(0, profile.size() > 2 ? profile.size() - 2 : 0)));
  if (densifyRounds_ > 0) {
    log_(std::format("Densification: {} rounds, {:.1f} s in total ({:.0f} ms each)", densifyRounds_, densifySeconds_,
                     1000.0 * densifySeconds_ / densifyRounds_));
  }
  if (pruneSeconds_ > 0.0) log_(std::format("Pruning: {:.1f} s in total", pruneSeconds_));
  log_(std::format("Finished in {:.1f} s ({:.1f} it/s) -> {}", trainingSeconds_, report_.iterationsPerSecond, output_paths(options_)[0].string()));
  return report_;
}

TrainReport run_train(vk::Context &ctx, const TrainOptions &options) {
  // Drawn from the first iteration on. Output in the meantime goes through it.
  Progress bar("Train", static_cast<std::size_t>(options.iterations));
  bool training = false;
  TrainSession session(ctx, options, load_train_data(options), [&](const std::string &line) {
    if (training) bar.message(line);
    else std::println("{}", line);
  });
  training = true;
  bar.start();
  while (!session.finished()) {
    const StepStats stats = session.step();
    const int iteration = session.iteration();
    if (iteration % 10 == 0 || session.finished()) {
      bar.set(static_cast<std::size_t>(iteration), std::format("loss {:.4f}  {} Gaussians  {:.1f} it/s", stats.loss,
                                                               session.trainer().gaussian_count(), iteration / std::max(session.training_seconds(), 1e-9)));
    }
  }
  bar.finish();
  training = false;
  return session.finish();
}

} // namespace splat
