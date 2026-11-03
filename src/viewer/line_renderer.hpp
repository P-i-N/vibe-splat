#pragma once

#include "core/vk_buffer.hpp"
#include "render/camera.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace splat {

// Collects world-space lines each frame and draws them over the splats:
// camera frustums, the ground grid, axes. No depth test, so they stay visible
// inside the scene.
class LineRenderer {
public:
  LineRenderer(vk::Context &ctx, VkFormat colorFormat);
  ~LineRenderer();
  LineRenderer(const LineRenderer &) = delete;
  LineRenderer &operator=(const LineRenderer &) = delete;

  void clear() noexcept { vertices_.clear(); }
  void line(glm::vec3 a, glm::vec3 b, glm::vec4 color);

  // Records into an active rendering pass over the whole target.
  void record(VkCommandBuffer cmd, const Camera &camera, VkExtent2D extent, float nearPlane);

private:
  struct Vertex {
    glm::vec3 position;
    std::uint32_t color;
  };

  vk::Context &ctx_;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  std::vector<Vertex> vertices_;
  std::unique_ptr<vk::Buffer> buffer_; // Host-visible, rewritten every frame
};

} // namespace splat
