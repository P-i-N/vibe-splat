#pragma once

#include "core/radix_sort.hpp"
#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"
#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace splat {

// Renders Gaussian splats with Vulkan:
//   1. preprocess (compute): cull, project, SH colour, depth keys
//   2. GPU radix sort by depth (back to front)
//   3. one instanced quad per visible splat, alpha-blended ("over")
// The renderer records into a caller-provided command buffer and renders
// into any colour attachment (offscreen image or swapchain image).
class SplatRenderer {
public:
  SplatRenderer(vk::Context &ctx, VkFormat colorFormat);
  ~SplatRenderer();
  SplatRenderer(const SplatRenderer &) = delete;
  SplatRenderer &operator=(const SplatRenderer &) = delete;

  // Uploads a scene (activating opacity, scale and rotation). Waits for the
  // GPU. Must not be called while a frame using this renderer is in flight.
  void set_scene(const GaussianCloud &cloud);
  [[nodiscard]] std::size_t splat_count() const noexcept { return count_; }

  // Records a full frame. `target` must be in COLOR_ATTACHMENT_OPTIMAL layout
  // and stays in it. Only one frame may be in flight at a time.
  void record(VkCommandBuffer cmd, const Camera &camera, VkImageView target, VkExtent2D extent,
              std::array<float, 4> clearColor = {0.f, 0.f, 0.f, 1.f});

  // Number of splats drawn in the last completed frame (reads GPU memory;
  // call after the frame's fence has signalled).
  [[nodiscard]] std::uint32_t last_visible_count() const;

private:
  void create_graphics_pipeline(VkFormat colorFormat);

  vk::Context &ctx_;
  vk::ComputePipeline preprocess_;
  vk::RadixSorter sorter_;

  VkDescriptorSetLayout drawSetLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout drawLayout_ = VK_NULL_HANDLE;
  VkPipeline drawPipeline_ = VK_NULL_HANDLE;

  std::size_t count_ = 0;
  int shDegree_ = 0;
  std::uint32_t shFloats_ = 3;
  std::unique_ptr<vk::Buffer> gaussians_;
  std::unique_ptr<vk::Buffer> sh_;
  std::unique_ptr<vk::Buffer> splats2d_;
  std::unique_ptr<vk::Buffer> keys_, values_, tmpKeys_, tmpValues_;
  std::unique_ptr<vk::Buffer> camera_;   // host-visible, rewritten each frame
  std::unique_ptr<vk::Buffer> drawArgs_; // indirect draw arguments
};

} // namespace splat
