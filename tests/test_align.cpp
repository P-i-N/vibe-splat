#include "render/camera.hpp"
#include "splat/align.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/reference.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace splat;
using Catch::Matchers::WithinAbs;

namespace {

RigidTransform tilt_transform() {
  RigidTransform t;
  t.rotation = glm::dmat3(glm::rotate(glm::dmat4(1.0), glm::radians(23.0), glm::normalize(glm::dvec3(1.0, 0.3, -0.6))));
  t.translation = {0.7, -1.9, 3.2};
  return t;
}

// The same camera in a world moved by `t`.
Camera moved(const Camera &camera, const RigidTransform &t) {
  glm::dmat4 m(t.rotation);
  m[3] = glm::dvec4(t.translation, 1.0);
  Camera out = camera;
  out.view = glm::mat4(glm::dmat4(camera.view) * glm::inverse(m));
  out.position = glm::vec3(t.rotation * glm::dvec3(camera.position) + t.translation);
  return out;
}

// A ring of cameras above the test scene's floor (-y up), looking down at it.
std::vector<Camera> ring_cameras() {
  std::vector<Camera> cameras;
  for (int i = 0; i < 12; ++i) {
    const double a = 2.0 * std::numbers::pi * i / 12.0;
    const glm::vec3 eye(static_cast<float>(5.0 * std::cos(a)), -2.5f, static_cast<float>(5.0 * std::sin(a)));
    cameras.push_back(Camera::look_at(eye, glm::vec3(0.f, 0.3f, 0.f), glm::vec3(0.f, -1.f, 0.f), 60.f, 64, 48));
  }
  return cameras;
}

} // namespace

TEST_CASE("A transformed cloud looks the same from a camera moved with it", "[align]") {
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0.f, 1.f);
  GaussianCloud cloud;
  cloud.resize(40, 3);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      cloud.positions[i * 3 + k] = 0.6f * normal(rng);
      cloud.scales[i * 3 + k] = std::log(0.15f) + 0.4f * normal(rng);
    }
    for (int k = 0; k < 4; ++k) cloud.rotations[i * 4 + k] = normal(rng);
    cloud.opacities[i] = normal(rng);
  }
  for (float &c : cloud.sh) c = 0.3f * normal(rng); // Every band, so all get turned

  const Camera camera = Camera::look_at({0.4f, -0.8f, -4.f}, {0.f, 0.f, 0.f}, {0.f, -1.f, 0.f}, 50.f, 48, 36);
  const RigidTransform t = tilt_transform();
  GaussianCloud turned = cloud;
  transform_cloud(turned, t);

  const ReferenceImage before = reference_render(cloud, camera, {});
  const ReferenceImage after = reference_render(turned, moved(camera, t), {});
  REQUIRE(before.rgb.size() == after.rgb.size());
  double worst = 0.0, total = 0.0;
  for (std::size_t p = 0; p < before.rgb.size(); ++p) {
    const glm::dvec3 d = glm::abs(before.rgb[p] - after.rgb[p]);
    worst = std::max({worst, d.x, d.y, d.z});
    total += before.rgb[p].x + before.rgb[p].y + before.rgb[p].z;
  }
  CHECK(total > 1.0); // Something is in view
  CHECK(worst < 1e-3);
}

TEST_CASE("Camera up comes from their level x axes, not where they look", "[align]") {
  // Every camera looks steeply down, so their mean up leans inwards.
  std::vector<Camera> cameras;
  for (int i = 0; i < 8; ++i) {
    const double a = 2.0 * std::numbers::pi * i / 8.0;
    const glm::vec3 eye(static_cast<float>(2.0 * std::cos(a)), -4.f, static_cast<float>(2.0 * std::sin(a)));
    cameras.push_back(Camera::look_at(eye, glm::vec3(0.f), glm::vec3(0.f, -1.f, 0.f), 60.f, 64, 48));
  }
  const glm::dvec3 up = estimate_up(cameras);
  CHECK_THAT(up.y, WithinAbs(-1.0, 1e-6));
  CHECK(estimate_up({}) == glm::dvec3(0.0, -1.0, 0.0));
}

TEST_CASE("The floor of a tilted scene is found and levelled", "[align]") {
  const GaussianCloud original = make_test_scene(30000);
  const RigidTransform tilt = tilt_transform();
  GaussianCloud cloud = original;
  transform_cloud(cloud, tilt);
  std::vector<Camera> cameras;
  for (const Camera &camera : ring_cameras()) cameras.push_back(moved(camera, tilt));

  const FloorEstimate floor = estimate_floor(cloud, cameras);
  REQUIRE(floor.fitted);
  const glm::dvec3 trueUp = tilt.rotation * glm::dvec3(0.0, -1.0, 0.0);
  CHECK(glm::degrees(std::acos(std::min(1.0, glm::dot(floor.normal, trueUp)))) < 0.5);
  // 40% of the test scene is floor
  CHECK(floor.inliers > floor.candidates / 3);

  transform_cloud(cloud, floor_alignment(floor, cloud));
  // The floor Gaussians, at y = 0.6 in the original, are now at y = 0.
  double worst = 0.0;
  for (std::size_t i = 0; i < original.size(); ++i) {
    if (original.positions[i * 3 + 1] == 0.6f) worst = std::max(worst, std::abs(static_cast<double>(cloud.positions[i * 3 + 1])));
  }
  CHECK(worst < 0.01);
  // The spheres stand on it, above: towards -y.
  CHECK(cloud.positions[1] < 0.f);
}

TEST_CASE("Without a plane the floor falls back to the cameras' up", "[align]") {
  // A ball of Gaussians has no floor.
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.f, 1.f);
  GaussianCloud cloud;
  cloud.resize(5000, 0);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    for (int k = 0; k < 3; ++k) cloud.positions[i * 3 + k] = normal(rng);
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = 4.f;
  }
  const FloorEstimate floor = estimate_floor(cloud, ring_cameras());
  CHECK_FALSE(floor.fitted);
  CHECK_THAT(floor.normal.y, WithinAbs(-1.0, 1e-6));
  CHECK(floor.offset < -1.5); // Under the lowest 2%: n.p = -y, so y > 1.5
}
