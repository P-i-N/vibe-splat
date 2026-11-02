#pragma once

#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace splat {
namespace vk {
class Context;
}
class SplatRenderer;

struct RenderedImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgba; // 8-bit RGBA, row-major, top row first
  std::uint32_t visibleSplats = 0;
  double milliseconds = 0.0;      // GPU submit-to-finish wall time
};

// Renders one frame offscreen (RGBA8) with an existing renderer whose scene
// is already set. The renderer must have been created for VK_FORMAT_R8G8B8A8_UNORM.
[[nodiscard]] RenderedImage render_offscreen(vk::Context &ctx, SplatRenderer &renderer, const Camera &camera,
                                             std::array<float, 4> background = {0.f, 0.f, 0.f, 1.f});

// Writes an RGBA image as PNG. Throws std::runtime_error.
void write_png(const RenderedImage &image, const std::filesystem::path &path);

// A camera that frames the scene: orbiting its robust centre from a distance
// that fits the bulk of the Gaussians into view.
[[nodiscard]] OrbitCamera default_orbit(const GaussianCloud &cloud, glm::vec3 up = {0.f, -1.f, 0.f});

} // namespace splat
