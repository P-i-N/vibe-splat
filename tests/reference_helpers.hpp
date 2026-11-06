#pragma once

#include "render/camera.hpp"
#include "sfm/test_views.hpp"
#include "splat/gaussian_cloud.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace splat::test {

inline constexpr float kShC0 = 0.28209479177387814f;

// Random Gaussians in front of train_camera().
inline GaussianCloud random_cloud(std::size_t count, int degree, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f), unit(0.f, 1.f);
  GaussianCloud cloud;
  cloud.resize(count, degree);
  const auto coefficients = static_cast<std::size_t>(cloud.shCoeffs());
  for (std::size_t i = 0; i < count; ++i) {
    cloud.positions[i * 3] = u(rng) * 1.4f;
    cloud.positions[i * 3 + 1] = u(rng) * 1.1f;
    cloud.positions[i * 3 + 2] = 3.f + unit(rng) * 3.f;
    for (std::size_t a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = std::log(0.06f + 0.2f * unit(rng));
    for (std::size_t a = 0; a < 4; ++a) cloud.rotations[i * 4 + a] = u(rng);
    cloud.opacities[i] = u(rng) * 1.5f;
    for (std::size_t c = 0; c < 3; ++c) cloud.sh[i * coefficients * 3 + c] = (unit(rng) - 0.5f) / kShC0;
    for (std::size_t k = 3; k < coefficients * 3; ++k) cloud.sh[i * coefficients * 3 + k] = u(rng) * 0.2f;
  }
  return cloud;
}

// At the origin, looking along +z.
inline Camera train_camera(int width = 64, int height = 48) {
  Camera camera;
  camera.fx = camera.fy = 60.f;
  camera.cx = static_cast<float>(width) * 0.5f;
  camera.cy = static_cast<float>(height) * 0.5f;
  camera.width = width;
  camera.height = height;
  return camera;
}

// RGBA8. Smooth rather than white noise, so that SSIM has structure to compare.
inline std::vector<std::uint8_t> smooth_target(int width, int height, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint8_t> target(static_cast<std::size_t>(width) * height * 4);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t at = (static_cast<std::size_t>(y) * width + x) * 4;
      target[at] = static_cast<std::uint8_t>(127.f + 100.f * std::sin(0.2f * static_cast<float>(x)) + static_cast<float>(rng() % 16));
      target[at + 1] = static_cast<std::uint8_t>(127.f + 100.f * std::cos(0.15f * static_cast<float>(y)) + static_cast<float>(rng() % 16));
      target[at + 2] = static_cast<std::uint8_t>(60 + (x + y) % 120);
      target[at + 3] = 255;
    }
  }
  return target;
}

inline Camera camera_of(const TestView &view, int width, int height, float focal) {
  Camera camera = train_camera(width, height);
  camera.fx = camera.fy = focal;
  glm::dmat4 matrix(1.0);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) matrix[c][r] = view.pose.R(r, c);
    matrix[3][r] = view.pose.t(r);
  }
  camera.view = glm::mat4(matrix);
  const Eigen::Vector3d center = view.pose.center();
  camera.position = glm::vec3(center.x(), center.y(), center.z());
  return camera;
}

// A side x side grid of grey Gaussians facing the camera, 2 units ahead.
inline GaussianCloud grid_in_front(const Camera &camera, std::size_t side) {
  GaussianCloud cloud;
  cloud.resize(side * side, 0);
  const glm::mat3 toWorld = glm::transpose(glm::mat3(camera.view));
  const float spacing = 2.6f / static_cast<float>(side);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const float x = (static_cast<float>(i % side) + 0.5f - static_cast<float>(side) * 0.5f) * spacing;
    const float y = (static_cast<float>(i / side) + 0.5f - static_cast<float>(side) * 0.5f) * spacing;
    const glm::vec3 world = camera.position + toWorld * glm::vec3(x, y, 2.f);
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.positions[i * 3 + a] = world[static_cast<int>(a)];
      cloud.scales[i * 3 + a] = std::log(spacing * 0.75f);
    }
    cloud.rotations[i * 4] = 1.f;
  }
  return cloud;
}

} // namespace splat::test
