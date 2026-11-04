#include "ingest/image.hpp"
#include "ingest/ingest.hpp"
#include "sfm/bundle.hpp"
#include "sfm/features.hpp"
#include "sfm/gpu_matcher.hpp"
#include "sfm/sfm.hpp"
#include "sfm/test_views.hpp"
#include "splat/gaussian_cloud.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <print>
#include <random>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace splat;
using Catch::Matchers::WithinAbs;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const char *name) {
  const fs::path path = fs::temp_directory_path() / name;
  fs::remove_all(path);
  fs::create_directories(path);
  return path;
}

// Two PINHOLE cameras looking at points near the origin, with perfect observations.
ColmapModel synthetic_model() {
  ColmapModel model;
  ColmapCamera camera;
  camera.id = 7;
  camera.model = 1;
  camera.width = 800;
  camera.height = 600;
  camera.params = {500.0, 510.0, 400.0, 300.0};
  model.cameras[camera.id] = camera;

  const double s = std::sqrt(0.5);
  const auto add_image = [&](std::uint32_t id, std::array<double, 4> q, glm::dvec3 t, const char *name) {
    ColmapImage image;
    image.id = id;
    image.rotation = q;
    image.translation = t;
    image.cameraId = camera.id;
    image.name = name;
    model.images[id] = image;
  };
  add_image(1, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 5.0}, "photo_000001.jpg");
  add_image(2, {s, 0.0, s, 0.0}, {0.0, 0.0, 5.0}, "sub/photo_000002.jpg"); // 90 degrees around y

  const std::vector<glm::dvec3> positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.5, -0.5, 0.5}};
  for (std::size_t i = 0; i < positions.size(); ++i) {
    ColmapPoint point;
    point.id = 100 + i;
    point.position = positions[i];
    point.color = {10, 20, 30};
    point.error = 0.25;
    for (auto &[id, image] : model.images) {
      glm::dvec2 pixel;
      REQUIRE(colmap_project(camera, image.to_camera(point.position), pixel));
      point.track.push_back({id, static_cast<std::uint32_t>(image.observations.size())});
      image.observations.push_back({pixel, point.id});
    }
    model.points.push_back(point);
  }
  model.images[1].observations.push_back({{5.0, 6.0}, kNoPoint});
  return model;
}

} // namespace

TEST_CASE("COLMAP model round-trips through the binary format", "[sfm]") {
  const fs::path dir = temp_dir("splat_colmap_roundtrip");
  const ColmapModel model = synthetic_model();
  save_colmap_model(model, dir);
  const ColmapModel loaded = load_colmap_model(dir);

  REQUIRE(loaded.cameras.size() == 1);
  CHECK(loaded.cameras.at(7).params == model.cameras.at(7).params);
  CHECK(loaded.cameras.at(7).width == 800);
  REQUIRE(loaded.images.size() == 2);
  CHECK(loaded.images.at(2).name == "sub/photo_000002.jpg");
  CHECK(loaded.images.at(2).rotation == model.images.at(2).rotation);
  CHECK(loaded.images.at(1).observations.back().pointId == kNoPoint);
  REQUIRE(loaded.points.size() == 4);
  CHECK(loaded.points[3].position == model.points[3].position);
  CHECK(loaded.points[3].color == model.points[3].color);
  CHECK(loaded.points[3].track.size() == 2);
  fs::remove_all(dir);
}

TEST_CASE("COLMAP files have the documented byte layout", "[sfm]") {
  // A hand-built cameras.bin, as COLMAP writes it: one SIMPLE_PINHOLE camera.
  const fs::path dir = temp_dir("splat_colmap_layout");
  {
    ColmapModel empty;
    save_colmap_model(empty, dir);
    std::ofstream out(dir / "cameras.bin", std::ios::binary);
    const auto put = [&](const auto &v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
    put(std::uint64_t{1});
    put(std::uint32_t{3});
    put(std::int32_t{0});
    put(std::uint64_t{640});
    put(std::uint64_t{480});
    put(600.0);
    put(320.0);
    put(240.0);
  }
  const ColmapModel loaded = load_colmap_model(dir);
  REQUIRE(loaded.cameras.size() == 1);
  CHECK(colmap_model_name(loaded.cameras.at(3).model) == std::string("SIMPLE_PINHOLE"));
  CHECK(loaded.cameras.at(3).params == std::vector<double>{600.0, 320.0, 240.0});
  fs::remove_all(dir);
}

TEST_CASE("COLMAP loader rejects corrupt files cleanly", "[sfm]") {
  const fs::path dir = temp_dir("splat_colmap_corrupt");
  save_colmap_model(synthetic_model(), dir);
  for (const char *file : {"cameras.bin", "images.bin", "points3D.bin"}) {
    const fs::path path = dir / file;
    std::ifstream in(path, std::ios::binary);
    const std::string original{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    in.close();

    // Every truncation must throw rather than crash or over-allocate.
    for (std::size_t size = 0; size < original.size(); size += 3) {
      std::ofstream(path, std::ios::binary).write(original.data(), static_cast<std::streamsize>(size));
      CHECK_THROWS(load_colmap_model(dir));
    }
    // An absurd element count too.
    std::string huge = original;
    huge.replace(0, 8, std::string(8, '\xFF'));
    std::ofstream(path, std::ios::binary).write(huge.data(), static_cast<std::streamsize>(huge.size()));
    CHECK_THROWS(load_colmap_model(dir));

    std::ofstream(path, std::ios::binary).write(original.data(), static_cast<std::streamsize>(original.size()));
  }
  CHECK_NOTHROW(load_colmap_model(dir));
  fs::remove(dir / "images.bin");
  CHECK_THROWS(load_colmap_model(dir));
  fs::remove_all(dir);
}

TEST_CASE("Poses follow COLMAP's world-to-camera convention", "[sfm]") {
  const ColmapModel model = synthetic_model();
  const ColmapImage &front = model.images.at(1);
  CHECK(front.center() == glm::dvec3(0.0, 0.0, -5.0));
  CHECK(front.to_camera({0.0, 0.0, 0.0}) == glm::dvec3(0.0, 0.0, 5.0));

  // R maps world +x to camera -z for 90 degrees around y, so this camera sits at +x.
  const ColmapImage &side = model.images.at(2);
  CHECK_THAT(side.center().x, WithinAbs(5.0, 1e-12));
  CHECK_THAT(side.center().z, WithinAbs(0.0, 1e-12));
  CHECK_THAT(side.to_camera(side.center()).z, WithinAbs(0.0, 1e-12));
}

TEST_CASE("Projection applies each lens model", "[sfm]") {
  glm::dvec2 pixel;
  ColmapCamera camera;
  const glm::dvec3 point(1.0, 2.0, 2.0); // Normalised (0.5, 1), r^2 = 1.25

  camera.model = 0;
  camera.params = {100.0, 10.0, 20.0};
  REQUIRE(colmap_project(camera, point, pixel));
  CHECK(pixel == glm::dvec2(60.0, 120.0));

  camera.model = 2;
  camera.params = {100.0, 10.0, 20.0, 0.1};
  REQUIRE(colmap_project(camera, point, pixel));
  CHECK_THAT(pixel.x, WithinAbs(100.0 * 1.125 * 0.5 + 10.0, 1e-9));
  CHECK_THAT(pixel.y, WithinAbs(100.0 * 1.125 * 1.0 + 20.0, 1e-9));

  camera.model = 4; // With zero distortion OPENCV equals PINHOLE
  camera.params = {100.0, 200.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0};
  REQUIRE(colmap_project(camera, point, pixel));
  CHECK(pixel == glm::dvec2(60.0, 220.0));

  CHECK_FALSE(colmap_project(camera, {0.0, 0.0, -1.0}, pixel));
  camera.model = 5; // Fisheye: not supported, reported rather than wrong
  CHECK_FALSE(colmap_project(camera, point, pixel));
}

TEST_CASE("Statistics recompute the reprojection error", "[sfm]") {
  ColmapModel model = synthetic_model();
  auto stats = compute_stats(model);
  CHECK(stats.images == 2);
  CHECK(stats.points == 4);
  CHECK_THAT(stats.meanTrackLength, WithinAbs(2.0, 1e-12));
  CHECK_THAT(stats.meanObservationsPerImage, WithinAbs(4.0, 1e-12));
  CHECK_THAT(stats.meanReprojectionError, WithinAbs(0.0, 1e-9));

  // Move every observed keypoint 3 px right. The stored error field must not be trusted.
  for (auto &[id, image] : model.images) {
    for (ColmapObservation &o : image.observations) o.xy.x += 3.0;
  }
  CHECK_THAT(compute_stats(model).meanReprojectionError, WithinAbs(3.0, 1e-9));
}

TEST_CASE("Importing a model matches images to the project", "[sfm]") {
  const fs::path dir = temp_dir("splat_sfm_import");
  const fs::path project = dir / "project";
  const fs::path modelDir = dir / "model";

  fs::create_directories(project / "images");
  IngestManifest manifest;
  manifest.cameras.push_back({});
  for (const char *file : {"images/photo_000001.jpg", "images/photo_000002.jpg", "images/photo_000003.jpg"}) {
    manifest.images.push_back({file, 1, "x.jpg", std::nullopt, ""});
  }
  save_manifest(manifest, project / "ingest.json");

  ColmapModel model = synthetic_model();
  save_colmap_model(model, modelDir);
  const SfmImportReport report = import_colmap_model(project, modelDir);
  CHECK(report.ingested == 3);
  CHECK(report.unregistered == 1);
  CHECK(report.stats.images == 2);

  const ColmapModel stored = load_colmap_model(project / "sparse" / "0");
  CHECK(stored.images.at(1).name == "images/photo_000001.jpg");
  CHECK(stored.images.at(2).name == "images/photo_000002.jpg"); // Subfolder from the model is dropped

  // An image from outside the project goes, with tracks left under two views.
  model.images.at(2).name = "someone_else.jpg";
  save_colmap_model(model, modelDir);
  const SfmImportReport partial = import_colmap_model(project, modelDir);
  CHECK(partial.stats.images == 1);
  CHECK(partial.stats.points == 0);
  const ColmapModel trimmed = load_colmap_model(project / "sparse" / "0");
  for (const ColmapObservation &o : trimmed.images.at(1).observations) CHECK(o.pointId == kNoPoint);

  model.images.at(1).name = "also_not_ours.jpg";
  save_colmap_model(model, modelDir);
  CHECK_THROWS(import_colmap_model(project, modelDir));
  fs::remove_all(dir);
}

namespace {

Pose random_pose(std::mt19937 &rng) {
  std::normal_distribution<double> n(0.0, 1.0);
  Pose pose;
  pose.R = rotation_from_vector(Eigen::Vector3d(n(rng), n(rng), n(rng)) * 0.3);
  pose.t = Eigen::Vector3d(n(rng), n(rng), n(rng) * 0.2).normalized();
  return pose;
}

std::vector<Eigen::Vector3d> random_points(std::mt19937 &rng, int count, bool planar = false) {
  std::uniform_real_distribution<double> u(-2.0, 2.0), depth(4.0, 9.0);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < count; ++i) {
    const double x = u(rng), y = u(rng);
    points.emplace_back(x, y, planar ? 6.0 + 0.3 * x - 0.2 * y : depth(rng));
  }
  return points;
}

Eigen::Vector3d bearing_of(const Pose &pose, const Eigen::Vector3d &point) {
  const Eigen::Vector3d c = pose.R * point + pose.t;
  return c / c.z();
}

double rotation_error_deg(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
  const double cosine = std::clamp(((a.transpose() * b).trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine) * 180.0 / 3.14159265358979323846;
}

Eigen::Matrix3d essential_of(const Pose &pose) {
  Eigen::Matrix3d skew;
  skew << 0, -pose.t.z(), pose.t.y(), pose.t.z(), 0, -pose.t.x(), -pose.t.y(), pose.t.x(), 0;
  return (skew * pose.R).normalized();
}

} // namespace

TEST_CASE("Five-point solver finds the true essential matrix", "[sfm][geometry]") {
  std::mt19937 rng(42);
  for (const bool planar : {false, true}) {
    int found = 0;
    constexpr int kRounds = 200;
    for (int round = 0; round < kRounds; ++round) {
      const Pose pose = random_pose(rng);
      const auto points = random_points(rng, 5, planar);
      std::array<Eigen::Vector3d, 5> x1, x2;
      for (std::size_t i = 0; i < 5; ++i) {
        x1[i] = points[i] / points[i].z();
        x2[i] = bearing_of(pose, points[i]);
      }
      const Eigen::Matrix3d truth = essential_of(pose);
      for (const Eigen::Matrix3d &E : essential_five_point(x1, x2)) {
        if ((E - truth).norm() < 1e-5 || (E + truth).norm() < 1e-5) {
          ++found;
          break;
        }
      }
    }
    INFO("planar: " << planar);
    CHECK(found >= kRounds * 98 / 100);
  }
}

TEST_CASE("Relative pose survives outliers and planar scenes", "[sfm][geometry]") {
  std::mt19937 rng(7);
  for (const bool planar : {false, true}) {
    const Pose pose = random_pose(rng);
    const auto points = random_points(rng, 200, planar);
    std::vector<Eigen::Vector3d> x1, x2;
    std::uniform_real_distribution<double> u(-0.6, 0.6);
    std::normal_distribution<double> noise(0.0, 0.5 / 500.0);
    for (const Eigen::Vector3d &point : points) {
      Eigen::Vector3d a = point / point.z(), b = bearing_of(pose, point);
      a.head<2>() += Eigen::Vector2d(noise(rng), noise(rng));
      b.head<2>() += Eigen::Vector2d(noise(rng), noise(rng));
      x1.push_back(a);
      x2.push_back(b);
    }
    for (int i = 0; i < 80; ++i) { // 30% of the matches are simply wrong
      x1.emplace_back(u(rng), u(rng), 1.0);
      x2.emplace_back(u(rng), u(rng), 1.0);
    }
    const auto result = estimate_relative_pose(x1, x2, 2.0 / 500.0, rng);
    INFO("planar: " << planar);
    REQUIRE(result);
    CHECK(rotation_error_deg(result->pose.R, pose.R) < 1.0);
    CHECK(result->pose.t.normalized().dot(pose.t.normalized()) > 0.99);
    CHECK(result->inliers.size() >= 180);
    CHECK(result->medianAngleDeg > 0.0);
  }
}

TEST_CASE("Triangulation recovers a point from many views", "[sfm][geometry]") {
  std::mt19937 rng(3);
  const Eigen::Vector3d point(0.5, -1.0, 6.0);
  std::vector<Pose> poses;
  std::vector<Eigen::Vector3d> bearings;
  for (int i = 0; i < 4; ++i) {
    poses.push_back(random_pose(rng));
    bearings.push_back(bearing_of(poses.back(), point));
  }
  CHECK((triangulate(poses, bearings) - point).norm() < 1e-9);
}

TEST_CASE("P3P finds the true pose among its solutions", "[sfm][geometry]") {
  std::mt19937 rng(11);
  int found = 0;
  constexpr int kRounds = 500;
  for (int round = 0; round < kRounds; ++round) {
    const Pose pose = random_pose(rng);
    const auto points = random_points(rng, 3);
    const std::array<Eigen::Vector3d, 3> world = {points[0], points[1], points[2]};
    const std::array<Eigen::Vector3d, 3> rays = {bearing_of(pose, points[0]), bearing_of(pose, points[1]), bearing_of(pose, points[2])};
    for (const Pose &solution : solve_p3p(rays, world)) {
      if (rotation_error_deg(solution.R, pose.R) < 1e-3 && (solution.t - pose.t).norm() < 1e-5) {
        ++found;
        break;
      }
    }
  }
  CHECK(found >= kRounds * 98 / 100);
}

TEST_CASE("Absolute pose survives outliers", "[sfm][geometry]") {
  std::mt19937 rng(5);
  const Pose pose = random_pose(rng);
  const auto points = random_points(rng, 150);
  std::vector<Eigen::Vector3d> bearings;
  std::uniform_real_distribution<double> u(-0.6, 0.6);
  for (std::size_t i = 0; i < points.size(); ++i) {
    bearings.push_back(i % 3 == 0 ? Eigen::Vector3d(u(rng), u(rng), 1.0) : bearing_of(pose, points[i]));
  }
  const auto result = estimate_absolute_pose(bearings, points, 2.0 / 500.0, rng);
  REQUIRE(result);
  CHECK(rotation_error_deg(result->pose.R, pose.R) < 0.5);
  CHECK(result->inliers.size() >= 95);
  CHECK(result->inliers.size() <= 105);
}

TEST_CASE("Distortion round-trips between pixels and bearings", "[sfm][geometry]") {
  Lens lens;
  lens.f = 600.0;
  lens.cx = 320.0;
  lens.cy = 240.0;
  lens.k = -0.12;
  Eigen::Vector2d pixel;
  REQUIRE(project_point(lens, {0.7, -0.4, 2.0}, pixel));
  const Eigen::Vector3d bearing = unproject_pixel(lens, pixel);
  CHECK_THAT(bearing.x(), WithinAbs(0.35, 1e-6));
  CHECK_THAT(bearing.y(), WithinAbs(-0.2, 1e-6));
}

TEST_CASE("Bundle adjustment undoes perturbed poses, points and focal length", "[sfm][bundle]") {
  std::mt19937 rng(19);
  std::vector<Lens> truthLenses(1);
  truthLenses[0].f = 550.0;
  truthLenses[0].cx = 320.0;
  truthLenses[0].cy = 240.0;
  truthLenses[0].k = -0.05;
  truthLenses[0].fPrior = 550.0;
  truthLenses[0].fPriorSigma = 500.0;

  std::vector<BundleView> truthViews;
  for (int i = 0; i < 8; ++i) {
    BundleView view;
    view.pose.R = rotation_from_vector(Eigen::Vector3d(0.02 * i, 0.08 * (i - 4), 0.01 * i));
    view.pose.t = Eigen::Vector3d(0.5 * (i - 4), 0.1 * (i % 3), 0.2 * (i % 2));
    truthViews.push_back(view);
  }
  truthViews[0].fixed = true;
  const auto truthPoints = random_points(rng, 150);

  std::vector<BundleObservation> observations;
  for (std::size_t v = 0; v < truthViews.size(); ++v) {
    for (std::size_t p = 0; p < truthPoints.size(); ++p) {
      Eigen::Vector2d pixel;
      REQUIRE(project_point(truthLenses[0], truthViews[v].pose.R * truthPoints[p] + truthViews[v].pose.t, pixel));
      observations.push_back({static_cast<int>(v), static_cast<int>(p), pixel});
    }
  }

  auto lenses = truthLenses;
  auto views = truthViews;
  auto points = truthPoints;
  std::normal_distribution<double> n(0.0, 1.0);
  lenses[0].f = 600.0;
  lenses[0].k = 0.0;
  for (std::size_t v = 1; v < views.size(); ++v) {
    views[v].pose.R = rotation_from_vector(Eigen::Vector3d(n(rng), n(rng), n(rng)) * 0.005) * views[v].pose.R;
    views[v].pose.t += Eigen::Vector3d(n(rng), n(rng), n(rng)) * 0.01;
  }
  for (auto &point : points) point += Eigen::Vector3d(n(rng), n(rng), n(rng)) * 0.02;

  BundleOptions options;
  options.iterations = 60;
  const BundleResult result = bundle_adjust(lenses, views, points, observations, options);
  CHECK(result.initialRms > 3.0);
  CHECK(result.finalRms < 0.05);
  CHECK(std::abs(lenses[0].f - truthLenses[0].f) < 25.0);
}

TEST_CASE("Features are found again after rotation and scaling", "[sfm][features]") {
  const auto views = make_test_views(1, 480, 480, 400.0);
  const RgbImage &image = views[0].image;
  const ImageFeatures original = extract_features(image, 2000);
  REQUIRE(original.keypoints.size() > 200);
  for (const Keypoint &k : original.keypoints) {
    REQUIRE(k.x >= 0.f);
    REQUIRE(k.x <= 480.f);
    REQUIRE(k.y <= 480.f);
    REQUIRE(k.scale > 0.f);
  }

  // Quarter turn clockwise: (x, y) in the source lands on (height - y, x).
  const ImageFeatures rotated = extract_features(apply_orientation(image, 6), 2000);
  auto matches = match_features(original, rotated);
  REQUIRE(matches.size() > 100);
  int correct = 0;
  for (const auto &[a, b] : matches) {
    const Keypoint &ka = original.keypoints[static_cast<std::size_t>(a)];
    const Keypoint &kb = rotated.keypoints[static_cast<std::size_t>(b)];
    if (std::hypot(480.f - ka.y - kb.x, ka.x - kb.y) < 2.f) ++correct;
  }
  CHECK(correct >= static_cast<int>(matches.size()) * 9 / 10);

  const ImageFeatures smaller = extract_features(downscale_to_fit(image, 320), 2000);
  matches = match_features(original, smaller);
  REQUIRE(matches.size() > 50);
  correct = 0;
  const float scale = 320.f / 480.f;
  for (const auto &[a, b] : matches) {
    const Keypoint &ka = original.keypoints[static_cast<std::size_t>(a)];
    const Keypoint &kb = smaller.keypoints[static_cast<std::size_t>(b)];
    if (std::hypot(ka.x * scale - kb.x, ka.y * scale - kb.y) < 2.f) ++correct;
  }
  CHECK(correct >= static_cast<int>(matches.size()) * 8 / 10);
}

namespace {

// Writes rendered views as an ingested project whose focal length is off by
// `focalError`, as it is when ingest has to guess.
fs::path make_project(const char *name, const std::vector<TestView> &views, double focal, double focalError, FocalSource source) {
  const fs::path project = temp_dir(name);
  fs::create_directories(project / "images");
  IngestManifest manifest;
  IngestCamera camera;
  camera.id = 1;
  camera.width = views[0].image.width;
  camera.height = views[0].image.height;
  camera.fx = camera.fy = focal * focalError;
  camera.cx = camera.width * 0.5;
  camera.cy = camera.height * 0.5;
  camera.focalSource = source;
  manifest.cameras.push_back(camera);
  for (std::size_t i = 0; i < views.size(); ++i) {
    const std::string file = "images/view_" + std::to_string(i) + ".png";
    write_image(views[i].image, project / file, 95);
    manifest.images.push_back({file, 1, "test", std::nullopt, ""});
  }
  save_manifest(manifest, project / "ingest.json");
  return project;
}

// Aligns the result to the ground truth, since poses are only defined up to a
// similarity transform, and checks what is left.
void check_against_truth(const ColmapModel &model, const std::vector<TestView> &views) {
  REQUIRE(model.images.size() == views.size());
  Eigen::Matrix3Xd estimated(3, views.size()), truth(3, views.size());
  std::size_t column = 0;
  std::vector<Eigen::Matrix3d> rotations;
  for (const auto &[id, image] : model.images) {
    const glm::dvec3 c = image.center();
    estimated.col(static_cast<Eigen::Index>(column)) = Eigen::Vector3d(c.x, c.y, c.z);
    truth.col(static_cast<Eigen::Index>(column)) = views[id - 1].pose.center();
    Eigen::Quaterniond q(image.rotation[0], image.rotation[1], image.rotation[2], image.rotation[3]);
    rotations.push_back(q.normalized().toRotationMatrix());
    ++column;
  }
  const Eigen::Matrix4d similarity = Eigen::umeyama(estimated, truth, true);
  const Eigen::Matrix3d scaled = similarity.topLeftCorner<3, 3>();

  double worstCenter = 0.0, worstRotation = 0.0;
  for (std::size_t i = 0; i < views.size(); ++i) {
    const Eigen::Index c = static_cast<Eigen::Index>(i);
    const Eigen::Vector3d mapped = scaled * estimated.col(c) + similarity.topRightCorner<3, 1>();
    worstCenter = std::max(worstCenter, (mapped - truth.col(c)).norm());
    // Rotation between views, which any global rotation leaves alone. The
    // centres alone pin the alignment down poorly: they lie on a nearly flat arc.
    worstRotation = std::max(worstRotation, rotation_error_deg(rotations[i] * rotations[0].transpose(),
                                                               views[i].pose.R * views[0].pose.R.transpose()));
  }
  std::println("  worst camera position error {:.4f} units (scene is about 9 wide), worst rotation error {:.3f} degrees", worstCenter, worstRotation);
  CHECK(worstCenter < 0.05);
  CHECK(worstRotation < 0.3);
}

} // namespace

TEST_CASE("Poses are recovered from rendered images", "[sfm][pipeline]") {
  constexpr double kFocal = 400.0;
  const auto views = make_test_views(10, 512, 384, kFocal);

  SECTION("known focal length, exhaustive matching") {
    const fs::path project = make_project("splat_sfm_exact", views, kFocal, 1.0, FocalSource::Manual);
    SfmOptions options;
    options.project = project;
    const SfmReport report = run_sfm(options);
    CHECK(report.unregistered == 0);
    CHECK(report.stats.points > 300);
    CHECK(report.stats.meanReprojectionError < 1.0);
    const ColmapModel model = load_colmap_model(project / "sparse" / "0");
    check_against_truth(model, views);
    CHECK(model.cameras.at(1).params[0] == kFocal);
    REQUIRE(report.cameras.size() == 1);
    CHECK(report.cameras[0].id == 1);
    CHECK(report.cameras[0].focalSource == FocalSource::Manual);
    CHECK(report.cameras[0].initialFocal == kFocal);
    CHECK(report.cameras[0].focal == kFocal);
    fs::remove_all(project);
  }

  SECTION("known focal length, matching on the GPU") {
    vk::Context *ctx = splat::test::context();
    if (ctx == nullptr) SKIP("no Vulkan device");
    const fs::path project = make_project("splat_sfm_gpu", views, kFocal, 1.0, FocalSource::Manual);
    SfmOptions options;
    options.project = project;
    options.gpu = ctx;
    const SfmReport report = run_sfm(options);
    CHECK(report.unregistered == 0);
    CHECK(report.pairsVerified > 30);
    CHECK(report.stats.points > 300);
    CHECK(report.stats.meanReprojectionError < 1.0);
    check_against_truth(load_colmap_model(project / "sparse" / "0"), views);
    fs::remove_all(project);
  }

  SECTION("a focal length 15% off gets corrected, sequential matching") {
    const fs::path project = make_project("splat_sfm_guess", views, kFocal, 1.15, FocalSource::Default);
    SfmOptions options;
    options.project = project;
    options.matcher = Matcher::Sequential;
    options.window = 4;
    const SfmReport report = run_sfm(options);
    CHECK(report.unregistered == 0);
    const ColmapModel model = load_colmap_model(project / "sparse" / "0");
    check_against_truth(model, views);
    const double focal = model.cameras.at(1).params[0];
    std::println("  focal length refined from {:.1f} to {:.1f}, true value {:.1f}", kFocal * 1.15, focal, kFocal);
    CHECK(std::abs(focal - kFocal) < kFocal * 0.02);
    REQUIRE(report.cameras.size() == 1);
    CHECK(report.cameras[0].focalSource == FocalSource::Default);
    CHECK_THAT(report.cameras[0].initialFocal, Catch::Matchers::WithinRel(kFocal * 1.15, 1e-9));
    CHECK(report.cameras[0].focal == focal);
    CHECK(report.cameras[0].width == model.cameras.at(1).width);
    CHECK(load_ply(project / "sparse" / "0" / "preview.ply").size() == model.points.size());
    fs::remove_all(project);
  }
}

TEST_CASE("GPU matching agrees with the CPU matcher", "[sfm][features][gpu]") {
  vk::Context *ctx = splat::test::context();
  if (ctx == nullptr) SKIP("no Vulkan device");

  const auto views = make_test_views(8, 640, 480, 500.0);
  std::vector<ImageFeatures> features;
  for (const auto &view : views) features.push_back(extract_features(view.image, 4000));
  features.push_back({}); // An image without features must not break anything

  std::vector<std::pair<int, int>> pairs;
  for (int a = 0; a < static_cast<int>(features.size()); ++a) {
    for (int b = a + 1; b < static_cast<int>(features.size()); ++b) pairs.emplace_back(a, b);
  }
  GpuMatcher matcher(*ctx, features);
  const auto gpu = matcher.match(pairs);
  REQUIRE(gpu.size() == pairs.size());

  std::size_t cpuTotal = 0, agreed = 0;
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const auto cpu = match_features(features[static_cast<std::size_t>(pairs[p].first)], features[static_cast<std::size_t>(pairs[p].second)]);
    cpuTotal += cpu.size();
    for (const auto &m : gpu[p]) {
      if (std::binary_search(cpu.begin(), cpu.end(), m)) ++agreed;
    }
    CHECK(std::is_sorted(gpu[p].begin(), gpu[p].end()));
    if (pairs[p].second == static_cast<int>(features.size()) - 1) CHECK(gpu[p].empty());
    // Quantisation shifts a few borderline ratio tests either way, never more.
    if (cpu.size() >= 50) {
      CHECK(gpu[p].size() * 10 >= cpu.size() * 9);
      CHECK(gpu[p].size() * 9 <= cpu.size() * 10);
    }
  }
  REQUIRE(cpuTotal > 300);
  std::println("  GPU matching: {} of {} CPU matches found again", agreed, cpuTotal);
  CHECK(agreed * 100 >= cpuTotal * 95);

  // A second call must give the same lists: the compaction order is sorted away.
  CHECK(matcher.match(pairs) == gpu);
}

TEST_CASE("Mutual nearest neighbours and the ratio test on a tiny set", "[sfm][features]") {
  ImageFeatures a, b;
  auto add = [](ImageFeatures &f, int hot, float spread) {
    f.keypoints.push_back({});
    std::vector<float> d(static_cast<std::size_t>(kDescriptorSize), spread);
    d[static_cast<std::size_t>(hot)] = 1.f;
    float norm = 0.f;
    for (const float v : d) norm += v * v;
    for (float &v : d) v /= std::sqrt(norm);
    f.descriptors.insert(f.descriptors.end(), d.begin(), d.end());
  };
  add(a, 0, 0.f);  // Matches b0 cleanly
  add(a, 5, 0.f);  // b1 and b2 are equally close: fails the ratio test
  add(a, 9, 0.f);  // Prefers b3, but b3 prefers a3
  add(a, 9, 0.01f);
  add(b, 0, 0.f);
  add(b, 5, 0.05f);
  add(b, 5, 0.05f);
  add(b, 9, 0.01f);
  const auto matches = match_features(a, b);
  REQUIRE(matches.size() == 2);
  CHECK(matches[0] == std::pair{0, 0});
  CHECK(matches[1] == std::pair{3, 3});
}

namespace {

void add_descriptor(ImageFeatures &f, const std::vector<float> &d) {
  f.keypoints.push_back({});
  float norm = 0.f;
  for (const float v : d) norm += v * v;
  for (const float v : d) f.descriptors.push_back(v / std::sqrt(norm));
}

// x_b^T E x_a = 0 for b = R a + t.
Eigen::Matrix3d essential_of(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
  Eigen::Matrix3d tx;
  tx << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
  return tx * R;
}

} // namespace

TEST_CASE("Guided matching settles repeated patterns along epipolar lines", "[sfm][features]") {
  // A sideways step: epipolar lines are horizontal, y_b = y_a.
  const Eigen::Matrix3d E = essential_of(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));
  // b's descriptors differ a little from a's, as they would in a real photo.
  auto hot = [](int index, float spread = 0.02f) {
    std::vector<float> d(static_cast<std::size_t>(kDescriptorSize), spread);
    d[static_cast<std::size_t>(index)] = 1.f;
    return d;
  };
  ImageFeatures a, b;
  std::vector<Eigen::Vector2d> pa, pb;
  add_descriptor(a, hot(5)), pa.emplace_back(0.1, 0.2);  // Two copies in b, one on its line
  add_descriptor(a, hot(0)), pa.emplace_back(0.0, -0.3); // A unique match on its line
  add_descriptor(a, hot(9)), pa.emplace_back(0.3, 0.4);  // Its twin in b is off the line
  add_descriptor(b, hot(5, 0.04f)), pb.emplace_back(-0.2, 0.2);
  add_descriptor(b, hot(5, 0.04f)), pb.emplace_back(0.0, -0.1);
  add_descriptor(b, hot(0, 0.04f)), pb.emplace_back(0.2, -0.3);
  add_descriptor(b, hot(9, 0.04f)), pb.emplace_back(0.0, 0.0);

  // Against the whole image, the repeated pattern fails the ratio test.
  const auto plain = match_features(a, b);
  CHECK(std::find(plain.begin(), plain.end(), std::pair{0, 0}) == plain.end());
  CHECK(std::find(plain.begin(), plain.end(), std::pair{2, 3}) != plain.end());

  const auto guided = match_features_guided(a, b, pa, pb, E, 0.01, 0.01);
  REQUIRE(guided.size() == 2);
  CHECK(guided[0] == std::pair{0, 0});
  CHECK(guided[1] == std::pair{1, 2});
}

TEST_CASE("Guided matching finds what a search of every pair finds", "[sfm][features]") {
  // Descriptors drawn around a few shared directions, so features have many
  // look-alikes and the ratio test has work to do; positions spread over a
  // wide image. The grid search must give exactly what checking every pair
  // gives.
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.f, 1.f);
  std::uniform_real_distribution<double> position(-0.8, 0.8);
  std::vector<std::vector<float>> palette(12);
  for (auto &p : palette) {
    p.resize(static_cast<std::size_t>(kDescriptorSize));
    for (float &v : p) v = std::abs(noise(rng));
  }
  auto draw = [&] {
    std::vector<float> d = palette[static_cast<std::size_t>(rng() % palette.size())];
    for (float &v : d) v = std::max(0.f, v + 0.15f * noise(rng));
    return d;
  };
  ImageFeatures a, b;
  std::vector<Eigen::Vector2d> pa, pb;
  for (int i = 0; i < 600; ++i) add_descriptor(a, draw()), pa.emplace_back(position(rng), position(rng) * 1.3);
  for (int i = 0; i < 1500; ++i) add_descriptor(b, draw()), pb.emplace_back(position(rng), position(rng) * 1.3);

  for (const Eigen::Vector3d &t : {Eigen::Vector3d(1.0, 0.1, 0.05), Eigen::Vector3d(0.05, 1.0, 0.2), Eigen::Vector3d(0.1, 0.2, 1.0)}) {
    const Eigen::Matrix3d R = rotation_from_vector(Eigen::Vector3d(0.1, -0.2, 0.15));
    const Eigen::Matrix3d E = essential_of(R, t.normalized());
    constexpr double kThreshold = 0.01;
    const auto guided = match_features_guided(a, b, pa, pb, E, kThreshold, kThreshold);

    // The same rules, checking every pair of features.
    const int na = static_cast<int>(pa.size()), nb = static_cast<int>(pb.size());
    std::vector<int> bestOfA(static_cast<std::size_t>(na), -1), bestOfB(static_cast<std::size_t>(nb), -1);
    std::vector<float> bestDotOfB(static_cast<std::size_t>(nb), -2.f);
    for (int i = 0; i < na; ++i) {
      const Eigen::Vector3d xa(pa[static_cast<std::size_t>(i)].x(), pa[static_cast<std::size_t>(i)].y(), 1.0);
      float first = -2.f, second = -2.f;
      int firstIndex = -1;
      for (int j = 0; j < nb; ++j) {
        const Eigen::Vector3d xb(pb[static_cast<std::size_t>(j)].x(), pb[static_cast<std::size_t>(j)].y(), 1.0);
        const Eigen::Vector3d inB = E * xa, inA = E.transpose() * xb;
        const double residual = std::abs(xb.dot(inB));
        if (residual >= kThreshold * inB.head<2>().norm() || residual >= kThreshold * inA.head<2>().norm()) continue;
        float dot = 0.f;
        for (int d = 0; d < kDescriptorSize; ++d) dot += a.descriptor(static_cast<std::size_t>(i))[d] * b.descriptor(static_cast<std::size_t>(j))[d];
        if (dot > first) {
          second = first;
          first = dot;
          firstIndex = j;
        } else if (dot > second) {
          second = dot;
        }
        if (dot > bestDotOfB[static_cast<std::size_t>(j)]) {
          bestDotOfB[static_cast<std::size_t>(j)] = dot;
          bestOfB[static_cast<std::size_t>(j)] = i;
        }
      }
      if (firstIndex >= 0 && first >= 0.75f && (second <= -2.f || 2.f - 2.f * first < 0.64f * (2.f - 2.f * second))) {
        bestOfA[static_cast<std::size_t>(i)] = firstIndex;
      }
    }
    std::vector<std::pair<int, int>> expected;
    for (int i = 0; i < na; ++i) {
      const int j = bestOfA[static_cast<std::size_t>(i)];
      if (j >= 0 && bestOfB[static_cast<std::size_t>(j)] == i) expected.emplace_back(i, j);
    }
    REQUIRE(expected.size() > 20);
    // The reference works in double where the matcher uses float, so a
    // candidate right at the threshold may come out either way.
    std::size_t same = 0;
    for (const auto &m : guided) same += std::find(expected.begin(), expected.end(), m) != expected.end() ? 1 : 0;
    CHECK(same + 2 >= expected.size());
    CHECK(guided.size() <= expected.size() + 2);
  }
}
