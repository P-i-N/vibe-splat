#include "core/vk_buffer.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

using namespace splat;

namespace {

constexpr float kShC0 = 0.28209479177387814f;

// Adds one Gaussian with an RGB colour, isotropic std-dev `sigma` and opacity.
void add(GaussianCloud &c, float x, float y, float z, float sigma, float r, float g, float b, float opacity) {
  const std::size_t i = c.size();
  c.positions.insert(c.positions.end(), {x, y, z});
  c.scales.insert(c.scales.end(), {std::log(sigma), std::log(sigma), std::log(sigma)});
  c.rotations.insert(c.rotations.end(), {1.f, 0.f, 0.f, 0.f});
  c.opacities.push_back(std::log(opacity / (1.f - opacity)));
  c.sh.insert(c.sh.end(), {(r - 0.5f) / kShC0, (g - 0.5f) / kShC0, (b - 0.5f) / kShC0});
  (void)i;
}

struct Pixel {
  int r, g, b, a;
};

Pixel at(const RenderedImage &img, int x, int y) {
  const std::size_t o = (static_cast<std::size_t>(y) * img.width + x) * 4;
  return {img.rgba[o], img.rgba[o + 1], img.rgba[o + 2], img.rgba[o + 3]};
}

// Camera at the origin looking down +z (identity view), 64x64, 90 deg FOV.
Camera front_camera() {
  return Camera::look_at({0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, -1.f, 0.f}, 90.f, 64, 64);
}

} // namespace

TEST_CASE("A single Gaussian renders at the projected centre", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  GaussianCloud cloud;
  // Right of centre by 1 unit at depth 4: x pixel = 32 + 32 * 1/4 = 40.
  add(cloud, 1.f, 0.f, 4.f, 0.15f, 1.f, 0.f, 0.f, 0.99f);

  SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const RenderedImage img = render_offscreen(*ctx, renderer, front_camera());

  CHECK(img.visibleSplats == 1);
  const Pixel c = at(img, 40, 32);
  CHECK(c.r > 200);
  CHECK(c.g < 10);
  CHECK(c.b < 10);
  // Far from the splat stays background.
  const Pixel bg = at(img, 5, 5);
  CHECK(bg.r == 0);
  CHECK(bg.g == 0);
  CHECK(bg.b == 0);
}

TEST_CASE("Nearer Gaussians occlude farther ones regardless of order", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  for (bool nearFirst : {true, false}) {
    GaussianCloud cloud;
    if (nearFirst) {
      add(cloud, 0.f, 0.f, 2.f, 0.3f, 1.f, 0.f, 0.f, 0.99f); // near: red
      add(cloud, 0.f, 0.f, 5.f, 0.6f, 0.f, 0.f, 1.f, 0.99f); // far: blue
    } else {
      add(cloud, 0.f, 0.f, 5.f, 0.6f, 0.f, 0.f, 1.f, 0.99f);
      add(cloud, 0.f, 0.f, 2.f, 0.3f, 1.f, 0.f, 0.f, 0.99f);
    }
    SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
    renderer.set_scene(cloud);
    const RenderedImage img = render_offscreen(*ctx, renderer, front_camera());
    const Pixel c = at(img, 32, 32);
    INFO("near splat first in file: " << nearFirst);
    CHECK(c.r > 200);
    CHECK(c.b < 20);
  }
}

TEST_CASE("Gaussians behind the camera are culled", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  GaussianCloud cloud;
  add(cloud, 0.f, 0.f, -3.f, 0.5f, 1.f, 1.f, 1.f, 0.99f);
  add(cloud, 0.f, 0.f, 3.f, 0.2f, 0.f, 1.f, 0.f, 0.99f);
  SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const RenderedImage img = render_offscreen(*ctx, renderer, front_camera());
  CHECK(img.visibleSplats == 1);
}

TEST_CASE("Test scene renders without errors", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  const GaussianCloud cloud = make_test_scene(20'000);
  SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const Camera cam = default_orbit(cloud).camera(160, 90);
  const RenderedImage img = render_offscreen(*ctx, renderer, cam);
  CHECK(img.visibleSplats > 10'000);
  std::size_t lit = 0;
  for (std::size_t i = 0; i < img.rgba.size(); i += 4) {
    lit += (img.rgba[i] + img.rgba[i + 1] + img.rgba[i + 2]) > 30 ? 1 : 0;
  }
  CHECK(lit > 160 * 90 / 5); // the scene fills a good part of the frame
}

TEST_CASE("Gaussians packed on the GPU render like those uploaded from the CPU", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  GaussianCloud cloud = make_test_scene(20'000);
  // Unnormalised rotations: the shader must normalise them as set_scene() does.
  for (float &q : cloud.rotations) q *= 1.7f;
  // Degree 3 with view-dependent colour of about the size 3DGS trains, which
  // the CPU path packs to bytes and the GPU path reads as the trainer keeps it.
  std::mt19937 rng(5);
  std::normal_distribution<float> rest(0.f, 0.1f);
  std::vector<float> degree3;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    degree3.insert(degree3.end(), cloud.sh.begin() + static_cast<std::ptrdiff_t>(i * 3 * cloud.shCoeffs()),
              cloud.sh.begin() + static_cast<std::ptrdiff_t>(i * 3 * cloud.shCoeffs() + 3));
    for (int k = 0; k < 45; ++k) degree3.push_back(rest(rng));
  }
  cloud.shDegree = 3;
  cloud.sh = std::move(degree3);
  const Camera cam = default_orbit(cloud).camera(160, 90);

  SplatRenderer fromCpu(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  fromCpu.set_scene(cloud);
  const RenderedImage expected = render_offscreen(*ctx, fromCpu, cam);

  const auto buffer = [&](const std::vector<float> &values) {
    vk::Buffer b(*ctx, values.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::MemoryUsage::GpuOnly);
    b.upload(*ctx, std::span<const float>(values));
    return b;
  };
  const vk::Buffer positions = buffer(cloud.positions), scales = buffer(cloud.scales), rotations = buffer(cloud.rotations),
                   opacities = buffer(cloud.opacities);
  const std::vector<std::uint32_t> trainerSh = pack_half_sh(cloud.sh, cloud.shCoeffs());
  vk::Buffer sh(*ctx, trainerSh.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::MemoryUsage::GpuOnly);
  sh.upload(*ctx, std::span<const std::uint32_t>(trainerSh));
  SplatRenderer fromGpu(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  ctx->submit_and_wait([&](VkCommandBuffer cmd) {
    fromGpu.set_scene(cmd, {&positions, &scales, &rotations, &opacities, &sh, cloud.size(), cloud.shDegree, cloud.shDegree});
  });
  const RenderedImage actual = render_offscreen(*ctx, fromGpu, cam);

  CHECK(actual.visibleSplats == expected.visibleSplats);
  REQUIRE(actual.rgba.size() == expected.rgba.size());
  int worst = 0;
  for (std::size_t i = 0; i < actual.rgba.size(); ++i) worst = std::max(worst, std::abs(actual.rgba[i] - expected.rgba[i]));
  const double value = test::psnr(actual, expected);
  INFO("PSNR " << value << " dB, worst " << worst);
  CHECK(value > 50.0);
  CHECK(worst <= 8);
}

TEST_CASE("Point mode draws every Gaussian as an opaque dot", "[vulkan][render]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  GaussianCloud cloud;
  // Faint and wide: as a splat it barely shows, as a point it is solid.
  add(cloud, 0.f, 0.f, 4.f, 1.f, 0.f, 1.f, 0.f, 0.1f);
  SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const RenderedImage splat = render_offscreen(*ctx, renderer, front_camera());
  renderer.settings.points = true;
  renderer.settings.pointRadius = 3.f;
  const RenderedImage point = render_offscreen(*ctx, renderer, front_camera());

  CHECK(at(splat, 32, 32).g < 40);
  CHECK(at(point, 32, 32).g > 200);
  CHECK(at(point, 32 + 8, 32).g < 10); // Beyond the radius
}

TEST_CASE("An orbit camera placed along a direction looks that way from there", "[render]") {
  OrbitCamera orbit;
  orbit.up = {0.f, -1.f, 0.f};
  const glm::vec3 eye(1.f, -2.f, 3.f);
  const glm::vec3 forward = glm::normalize(glm::vec3(-0.3f, 0.4f, -1.f));
  orbit.look_along(eye, forward, 5.f);
  CHECK(glm::length(orbit.eye() - eye) < 1e-4f);
  CHECK(glm::dot(glm::normalize(orbit.target - orbit.eye()), forward) > 0.99999f);
  CHECK(std::abs(orbit.distance - 5.f) < 1e-5f);
}
