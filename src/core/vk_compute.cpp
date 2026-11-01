#include "core/vk_compute.hpp"

#include "core/vk_check.hpp"

#include <array>
#include <vector>

namespace splat::vk {

VkDescriptorSetLayout create_storage_set_layout(VkDevice device, std::uint32_t count, VkShaderStageFlags stages) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  dlci.bindingCount = count;
  dlci.pBindings = bindings.data();
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  check(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &layout), "vkCreateDescriptorSetLayout");
  return layout;
}

VkShaderModule create_shader_module(VkDevice device, std::span<const std::uint32_t> spirv) {
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv.size_bytes();
  smci.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device, &smci, nullptr, &module), "vkCreateShaderModule");
  return module;
}

void push_storage_buffers(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint, VkPipelineLayout layout,
                          std::initializer_list<const Buffer *> buffers) {
  constexpr std::size_t kMax = 16;
  if (buffers.size() > kMax) {
    throw Error("push_storage_buffers: too many buffers");
  }
  std::array<VkDescriptorBufferInfo, kMax> infos{};
  std::array<VkWriteDescriptorSet, kMax> writes{};
  std::uint32_t i = 0;
  for (const Buffer *b : buffers) {
    infos[i] = {b->handle(), 0, VK_WHOLE_SIZE};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
    ++i;
  }
  vkCmdPushDescriptorSetKHR(cmd, bindPoint, layout, 0, i, writes.data());
}

ComputePipeline::ComputePipeline(const Context &ctx, std::span<const std::uint32_t> spirv,
                                 std::uint32_t storageBufferCount, std::uint32_t pushConstantBytes)
    : device_(ctx.device()), bufferCount_(storageBufferCount), pushBytes_(pushConstantBytes) {
  setLayout_ = create_storage_set_layout(device_, storageBufferCount, VK_SHADER_STAGE_COMPUTE_BIT);

  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantBytes};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &setLayout_;
  plci.pushConstantRangeCount = pushConstantBytes > 0 ? 1 : 0;
  plci.pPushConstantRanges = &range;
  check(vkCreatePipelineLayout(device_, &plci, nullptr, &layout_), "vkCreatePipelineLayout");

  VkShaderModule module = create_shader_module(device_, spirv);
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = module;
  cpci.stage.pName = "main";
  cpci.layout = layout_;
  const VkResult created = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_);
  vkDestroyShaderModule(device_, module, nullptr);
  check(created, "vkCreateComputePipelines");
}

ComputePipeline::~ComputePipeline() {
  vkDestroyPipeline(device_, pipeline_, nullptr);
  vkDestroyPipelineLayout(device_, layout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
}

void ComputePipeline::dispatch(VkCommandBuffer cmd, std::initializer_list<const Buffer *> buffers,
                               std::span<const std::byte> pushConstants, std::uint32_t groupsX,
                               std::uint32_t groupsY, std::uint32_t groupsZ) const {
  if (buffers.size() != bufferCount_) {
    throw Error("ComputePipeline::dispatch: wrong number of buffers");
  }
  if (pushConstants.size() != pushBytes_) {
    throw Error("ComputePipeline::dispatch: wrong push-constant size");
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  push_storage_buffers(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, buffers);
  if (pushBytes_ > 0) {
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes_, pushConstants.data());
  }
  if (groupsX > 0 && groupsY > 0 && groupsZ > 0) {
    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
  }
}

void compute_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_HOST_BIT;
  mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                     VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
                     VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.memoryBarrierCount = 1;
  dep.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace splat::vk
