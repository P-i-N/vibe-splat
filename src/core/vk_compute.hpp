#pragma once

#include "core/vk_buffer.hpp"

#include <cstdint>
#include <initializer_list>
#include <span>

namespace splat::vk {

// A compute pipeline whose inputs are N storage buffers at bindings 0..N-1
// plus an optional push-constant block. Buffers are bound with push
// descriptors, so one pipeline can be dispatched any number of times in a
// command buffer with different buffers (needed for ping-pong passes).
class ComputePipeline {
public:
  ComputePipeline(const Context &ctx, std::span<const std::uint32_t> spirv,
                  std::uint32_t storageBufferCount, std::uint32_t pushConstantBytes = 0);
  ~ComputePipeline();
  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;

  void dispatch(VkCommandBuffer cmd, std::initializer_list<const Buffer *> buffers,
                std::span<const std::byte> pushConstants, std::uint32_t groupsX,
                std::uint32_t groupsY = 1, std::uint32_t groupsZ = 1) const;

  template <class T>
  void dispatch(VkCommandBuffer cmd, std::initializer_list<const Buffer *> buffers, const T &push,
                std::uint32_t groupsX, std::uint32_t groupsY = 1, std::uint32_t groupsZ = 1) const {
    dispatch(cmd, buffers, std::as_bytes(std::span{&push, 1}), groupsX, groupsY, groupsZ);
  }

private:
  VkDevice device_ = VK_NULL_HANDLE;
  std::uint32_t bufferCount_ = 0;
  std::uint32_t pushBytes_ = 0;
  VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// Pushes storage-buffer descriptors for bindings 0..N-1 of set 0.
void push_storage_buffers(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint, VkPipelineLayout layout,
                          std::initializer_list<const Buffer *> buffers);

// Creates a push-descriptor set layout of `count` storage buffers.
[[nodiscard]] VkDescriptorSetLayout create_storage_set_layout(VkDevice device, std::uint32_t count,
                                                              VkShaderStageFlags stages);

[[nodiscard]] VkShaderModule create_shader_module(VkDevice device, std::span<const std::uint32_t> spirv);

// Compute/transfer writes -> compute reads/writes, indirect draw arguments,
// vertex-shader reads and host reads.
void compute_barrier(VkCommandBuffer cmd);

// Ceil-division helper for dispatch sizes.
[[nodiscard]] constexpr std::uint32_t div_up(std::uint32_t n, std::uint32_t d) noexcept {
  return (n + d - 1) / d;
}

} // namespace splat::vk
