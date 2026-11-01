#include "render/offscreen.hpp"

#include "core/vk_image.hpp"
#include "render/splat_renderer.hpp"

#include <chrono>
#include <cmath>
#include <format>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace splat {

RenderedImage render_offscreen(vk::Context &ctx, SplatRenderer &renderer, const Camera &camera,
                               std::array<float, 4> background) {
  const VkExtent2D extent{static_cast<std::uint32_t>(camera.width), static_cast<std::uint32_t>(camera.height)};
  vk::Image target(ctx, extent, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  const auto start = std::chrono::steady_clock::now();
  ctx.submit_and_wait([&](VkCommandBuffer cmd) {
    vk::transition_image(cmd, target.handle(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    renderer.record(cmd, camera, target.view(), extent, background);
    vk::transition_image(cmd, target.handle(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  });
  const auto end = std::chrono::steady_clock::now();

  RenderedImage image;
  image.width = camera.width;
  image.height = camera.height;
  image.rgba = target.read_rgba8(ctx);
  image.visibleSplats = renderer.last_visible_count();
  image.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  return image;
}

void write_png(const RenderedImage &image, const std::filesystem::path &path) {
  const std::string file = path.string();
  if (stbi_write_png(file.c_str(), image.width, image.height, 4, image.rgba.data(), image.width * 4) == 0) {
    throw std::runtime_error(std::format("Failed to write '{}'", file));
  }
}

OrbitCamera default_orbit(const GaussianCloud &cloud, glm::vec3 up) {
  const Bounds b = compute_bounds(cloud);
  OrbitCamera orbit;
  orbit.up = up;
  orbit.target = {b.center[0], b.center[1], b.center[2]};
  orbit.fovDeg = 60.f;
  orbit.pitchDeg = 25.f;
  orbit.yawDeg = 0.f;
  // Fit a sphere of `radius` into the vertical field of view, with margin.
  orbit.distance = b.radius / std::sin(glm::radians(orbit.fovDeg * 0.5f)) * 1.1f;
  return orbit;
}

} // namespace splat
