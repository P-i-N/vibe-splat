#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

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
