#include "core/vk_compute.hpp"

#include "core/vk_check.hpp"

#include <vector>

namespace splat::vk {

ComputePipeline::ComputePipeline(const Context &ctx, std::span<const std::uint32_t> spirv,
                                 std::uint32_t storageBufferCount, std::uint32_t pushConstantBytes)
    : device_(ctx.device()), bufferCount_(storageBufferCount), pushBytes_(pushConstantBytes) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(storageBufferCount);
  for (std::uint32_t i = 0; i < storageBufferCount; ++i) {
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = storageBufferCount;
  dlci.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &setLayout_), "vkCreateDescriptorSetLayout");

  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantBytes};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &setLayout_;
  plci.pushConstantRangeCount = pushConstantBytes > 0 ? 1 : 0;
  plci.pPushConstantRanges = &range;
  check(vkCreatePipelineLayout(device_, &plci, nullptr, &layout_), "vkCreatePipelineLayout");

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv.size_bytes();
  smci.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(device_, &smci, nullptr, &module), "vkCreateShaderModule");

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = module;
  cpci.stage.pName = "main";
  cpci.layout = layout_;
  const VkResult created = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_);
  vkDestroyShaderModule(device_, module, nullptr);
  check(created, "vkCreateComputePipelines");

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBufferCount};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &poolSize;
  check(vkCreateDescriptorPool(device_, &dpci, nullptr, &pool_), "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = pool_;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &setLayout_;
  check(vkAllocateDescriptorSets(device_, &dsai, &set_), "vkAllocateDescriptorSets");
}

ComputePipeline::~ComputePipeline() {
  vkDestroyDescriptorPool(device_, pool_, nullptr);
  vkDestroyPipeline(device_, pipeline_, nullptr);
  vkDestroyPipelineLayout(device_, layout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
}

void ComputePipeline::dispatch(VkCommandBuffer cmd, std::initializer_list<const Buffer *> buffers,
                               std::span<const std::byte> pushConstants, std::uint32_t groupsX,
                               std::uint32_t groupsY, std::uint32_t groupsZ) {
  if (buffers.size() != bufferCount_) {
    throw Error("ComputePipeline::dispatch: wrong number of buffers");
  }
  if (pushConstants.size() != pushBytes_) {
    throw Error("ComputePipeline::dispatch: wrong push-constant size");
  }

  // The descriptor set is updated at record time, so one pipeline object can
  // only be recorded once per submission. Good enough for the skeleton.
  std::vector<VkDescriptorBufferInfo> infos;
  std::vector<VkWriteDescriptorSet> writes;
  infos.reserve(buffers.size());
  writes.reserve(buffers.size());
  std::uint32_t binding = 0;
  for (const Buffer *b : buffers) {
    infos.push_back({b->handle(), 0, VK_WHOLE_SIZE});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set_;
    w.dstBinding = binding++;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &infos.back();
    writes.push_back(w);
  }
  vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
  if (pushBytes_ > 0) {
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes_, pushConstants.data());
  }
  vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
}

void compute_barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
  mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.memoryBarrierCount = 1;
  dep.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace splat::vk
