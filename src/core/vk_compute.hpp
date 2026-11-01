#pragma once

#include "core/vk_buffer.hpp"

#include <cstdint>
#include <initializer_list>
#include <span>

namespace splat::vk {

// A compute pipeline whose descriptor set is N storage buffers at bindings
// 0..N-1, plus an optional push-constant block. That covers every kernel the
// trainer and viewer need; anything fancier gets its own class.
class ComputePipeline {
public:
  ComputePipeline(const Context &ctx, std::span<const std::uint32_t> spirv,
                  std::uint32_t storageBufferCount, std::uint32_t pushConstantBytes = 0);
  ~ComputePipeline();
  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;

  // Binds the pipeline and buffers, pushes constants and dispatches.
  void dispatch(VkCommandBuffer cmd, std::initializer_list<const Buffer *> buffers,
                std::span<const std::byte> pushConstants, std::uint32_t groupsX,
                std::uint32_t groupsY = 1, std::uint32_t groupsZ = 1);

private:
  VkDevice device_ = VK_NULL_HANDLE;
  std::uint32_t bufferCount_ = 0;
  std::uint32_t pushBytes_ = 0;
  VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
};

// Full compute-to-compute and compute-to-host memory barrier.
void compute_barrier(VkCommandBuffer cmd);

} // namespace splat::vk
