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

// Gaussians already on the GPU, in GaussianCloud's raw layout but for `sh`,
// which is in the trainer's (see half_sh_words()): the trainer's parameter
// buffers. The renderer reads `sh` in place, so it must outlive every frame
// recorded with it.
struct GpuCloud {
  const vk::Buffer *positions = nullptr;
  const vk::Buffer *scales = nullptr;
  const vk::Buffer *rotations = nullptr;
  const vk::Buffer *opacities = nullptr;
  const vk::Buffer *sh = nullptr;
  std::size_t count = 0;
  int shDegree = 0;     // Of the data
  int activeDegree = 0; // Degrees that hold anything yet
};

// How to draw, for the viewer. The defaults render a scene as trained.
struct RenderSettings {
  float splatScale = 1.f;
  int maxShDegree = 3;
  bool points = false;     // Every Gaussian as an opaque dot at its centre
  float pointRadius = 2.f; // Pixels
};

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

  // Uploads a scene, activating opacity, scale and rotation and packing them
  // and the SH coefficients (see shaders/splat_types.glsl). Waits for the GPU.
  // Must not be called while a frame using this renderer is in flight.
  void set_scene(const GaussianCloud &cloud);
  // Records the conversion of Gaussians already on the GPU. Waits for nothing:
  // no frame using this renderer may be in flight.
  void set_scene(VkCommandBuffer cmd, const GpuCloud &cloud);
  [[nodiscard]] std::size_t splat_count() const noexcept { return count_; }
  [[nodiscard]] int sh_degree() const noexcept { return shDegree_; }

  RenderSettings settings;

  // Records a full frame. `target` must be in COLOR_ATTACHMENT_OPTIMAL layout
  // and stays in it. Only one frame may be in flight at a time. With `clear`
  // false, the splats are drawn over what the target holds.
  void record(VkCommandBuffer cmd, const Camera &camera, VkImageView target, VkExtent2D extent,
              std::array<float, 4> clearColor = {0.f, 0.f, 0.f, 1.f}, bool clear = true);

  // Number of splats drawn in the last completed frame (reads GPU memory;
  // call after the frame's fence has signalled).
  [[nodiscard]] std::uint32_t last_visible_count() const;

private:
  void create_graphics_pipeline(VkFormat colorFormat);
  void allocate(std::size_t count);

  vk::Context &ctx_;
  vk::ComputePipeline preprocess_;
  vk::ComputePipeline pack_;
  vk::ComputePipeline packSh_;
  vk::RadixSorter sorter_;

  VkDescriptorSetLayout drawSetLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout drawLayout_ = VK_NULL_HANDLE;
  VkPipeline drawPipeline_ = VK_NULL_HANDLE;

  std::size_t count_ = 0;
  std::size_t capacity_ = 0;
  const vk::Buffer *shExternal_ = nullptr; // From set_scene(cmd, GpuCloud), in the trainer's layout, instead of the packed sh_
  int shDegree_ = 0;
  std::uint32_t shStride_ = 3; // Words of sh_ or shExternal_ per Gaussian
  std::unique_ptr<vk::Buffer> gaussians_;
  std::unique_ptr<vk::Buffer> sh_;
  std::unique_ptr<vk::Buffer> splats2d_;
  std::unique_ptr<vk::Buffer> keys_, values_, tmpKeys_, tmpValues_;
  std::unique_ptr<vk::Buffer> camera_;   // host-visible, rewritten each frame
  std::unique_ptr<vk::Buffer> drawArgs_; // indirect draw arguments
};

} // namespace splat
