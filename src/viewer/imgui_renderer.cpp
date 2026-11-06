#include "viewer/imgui_renderer.hpp"

#include "core/vk_check.hpp"
#include "core/vk_compute.hpp"
#include "splat_shaders/shaders.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace splat {
namespace {

struct Push {
  float scale[2];
  float translate[2];
};

} // namespace

ImGuiRenderer::ImGuiRenderer(vk::Context &ctx, VkFormat colorFormat) : ctx_(ctx) {
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = VK_FILTER_LINEAR;
  sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.maxLod = 0.25f;
  vk::check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler");
  create_pipeline(colorFormat);
}

ImGuiRenderer::~ImGuiRenderer() {
  VkDevice dev = ctx_.device();
  vkDeviceWaitIdle(dev);
  // Textures ImGui still holds (the font atlas).
  if (ImGui::GetCurrentContext() != nullptr) {
    for (ImTextureData *tex : ImGui::GetPlatformIO().Textures) {
      if (tex->BackendUserData != nullptr) {
        delete static_cast<Texture *>(tex->BackendUserData);
        tex->BackendUserData = nullptr;
        tex->SetTexID(ImTextureID_Invalid);
        tex->SetStatus(ImTextureStatus_Destroyed);
      }
    }
  }
  vkDestroyPipeline(dev, pipeline_, nullptr);
  vkDestroyPipelineLayout(dev, layout_, nullptr);
  vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
  vkDestroySampler(dev, sampler_, nullptr);
}

void ImGuiRenderer::create_pipeline(VkFormat colorFormat) {
  VkDevice dev = ctx_.device();
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  dlci.bindingCount = 1;
  dlci.pBindings = &binding;
  vk::check(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &setLayout_), "vkCreateDescriptorSetLayout");

  VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &setLayout_;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  vk::check(vkCreatePipelineLayout(dev, &plci, nullptr, &layout_), "vkCreatePipelineLayout");

  VkShaderModule vert = vk::create_shader_module(dev, shaders::imgui_vert());
  VkShaderModule frag = vk::create_shader_module(dev, shaders::imgui_frag());
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  const VkVertexInputBindingDescription vertexBinding{0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX};
  const VkVertexInputAttributeDescription attributes[3] = {
      {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, pos)},
      {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, uv)},
      {2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ImDrawVert, col)},
  };
  VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &vertexBinding;
  vertexInput.vertexAttributeDescriptionCount = 3;
  vertexInput.pVertexAttributeDescriptions = attributes;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.lineWidth = 1.f;
  VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState blend{};
  blend.blendEnable = VK_TRUE;
  blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.colorBlendOp = VK_BLEND_OP_ADD;
  blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.alphaBlendOp = VK_BLEND_OP_ADD;
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blend;
  const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamicStates;
  VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = 1;
  rendering.pColorAttachmentFormats = &colorFormat;

  VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gpci.pNext = &rendering;
  gpci.stageCount = 2;
  gpci.pStages = stages;
  gpci.pVertexInputState = &vertexInput;
  gpci.pInputAssemblyState = &inputAssembly;
  gpci.pViewportState = &viewport;
  gpci.pRasterizationState = &raster;
  gpci.pMultisampleState = &multisample;
  gpci.pColorBlendState = &colorBlend;
  gpci.pDynamicState = &dynamic;
  gpci.layout = layout_;
  const VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
  vkDestroyShaderModule(dev, vert, nullptr);
  vkDestroyShaderModule(dev, frag, nullptr);
  vk::check(r, "vkCreateGraphicsPipelines");
}

ImGuiRenderer::Texture *ImGuiRenderer::upload(Texture *texture, int width, int height, const std::uint8_t *rgba) {
  const VkExtent2D extent{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
  const bool fresh = texture == nullptr || texture->image->extent().width != extent.width ||
                     texture->image->extent().height != extent.height;
  if (texture == nullptr) texture = new Texture;
  if (fresh) {
    texture->image = std::make_unique<vk::Image>(ctx_, extent, VK_FORMAT_R8G8B8A8_UNORM,
                                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }
  const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
  vk::Buffer staging(ctx_, bytes, 0, vk::MemoryUsage::Upload);
  std::memcpy(staging.mapped().data(), rgba, static_cast<std::size_t>(bytes));
  staging.flush();
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    vk::transition_image(cmd, texture->image->handle(),
                         fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyBufferToImage(cmd, staging.handle(), texture->image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vk::transition_image(cmd, texture->image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  });
  return texture;
}

void ImGuiRenderer::update_textures(ImDrawData *data) {
  if (data == nullptr || data->Textures == nullptr) return;
  for (ImTextureData *tex : *data->Textures) {
    if (tex->Status == ImTextureStatus_OK) continue;
    if (tex->Status == ImTextureStatus_WantDestroy) {
      // ImGui only asks once the texture went unused, and no frame is in flight.
      delete static_cast<Texture *>(tex->BackendUserData);
      tex->BackendUserData = nullptr;
      tex->SetTexID(ImTextureID_Invalid);
      tex->SetStatus(ImTextureStatus_Destroyed);
      continue;
    }
    // WantCreate or WantUpdates: the atlas is small, so upload all of it.
    std::vector<std::uint8_t> rgba;
    const auto *pixels = static_cast<const std::uint8_t *>(tex->GetPixels());
    if (tex->Format == ImTextureFormat_Alpha8) {
      rgba.resize(static_cast<std::size_t>(tex->Width) * tex->Height * 4, 255);
      for (std::size_t i = 0; i < static_cast<std::size_t>(tex->Width) * tex->Height; ++i) rgba[i * 4 + 3] = pixels[i];
      pixels = rgba.data();
    }
    auto *texture = upload(static_cast<Texture *>(tex->BackendUserData), tex->Width, tex->Height, pixels);
    tex->BackendUserData = texture;
    tex->SetTexID(reinterpret_cast<ImTextureID>(texture));
    tex->SetStatus(ImTextureStatus_OK);
  }
}

ImTextureID ImGuiRenderer::create_texture(int width, int height, std::span<const std::uint8_t> pixels, int channels) {
  if (channels == 4) return reinterpret_cast<ImTextureID>(upload(nullptr, width, height, pixels.data()));
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) * height * 4, 255);
  for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
    std::copy_n(pixels.data() + i * channels, std::min(channels, 3), rgba.data() + i * 4);
  }
  return reinterpret_cast<ImTextureID>(upload(nullptr, width, height, rgba.data()));
}

void ImGuiRenderer::destroy_texture(ImTextureID id) {
  if (id == ImTextureID_Invalid) return;
  delete reinterpret_cast<Texture *>(id);
}

void ImGuiRenderer::record(VkCommandBuffer cmd, ImDrawData *data, VkExtent2D extent) {
  if (data == nullptr || data->TotalVtxCount == 0 || extent.width == 0 || extent.height == 0) return;

  const VkDeviceSize vertexBytes = VkDeviceSize(data->TotalVtxCount) * sizeof(ImDrawVert);
  const VkDeviceSize indexBytes = VkDeviceSize(data->TotalIdxCount) * sizeof(ImDrawIdx);
  if (!vertices_ || vertices_->size() < vertexBytes) {
    vertices_ = std::make_unique<vk::Buffer>(ctx_, vertexBytes * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk::MemoryUsage::Upload);
  }
  if (!indices_ || indices_->size() < indexBytes) {
    indices_ = std::make_unique<vk::Buffer>(ctx_, indexBytes * 2, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk::MemoryUsage::Upload);
  }
  auto *vtx = reinterpret_cast<ImDrawVert *>(vertices_->mapped().data());
  auto *idx = reinterpret_cast<ImDrawIdx *>(indices_->mapped().data());
  for (const ImDrawList *list : data->CmdLists) {
    std::memcpy(vtx, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
    std::memcpy(idx, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx += list->VtxBuffer.Size;
    idx += list->IdxBuffer.Size;
  }
  vertices_->flush();
  indices_->flush();

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  const VkBuffer vb = vertices_->handle();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
  vkCmdBindIndexBuffer(cmd, indices_->handle(), 0, sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
  const VkViewport vp{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  Push push{};
  push.scale[0] = 2.f / data->DisplaySize.x;
  push.scale[1] = 2.f / data->DisplaySize.y;
  push.translate[0] = -1.f - data->DisplayPos.x * push.scale[0];
  push.translate[1] = -1.f - data->DisplayPos.y * push.scale[1];
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

  const ImVec2 clipOffset = data->DisplayPos;
  const ImVec2 clipScale = data->FramebufferScale;
  std::uint32_t vertexBase = 0, indexBase = 0;
  VkImageView bound = VK_NULL_HANDLE;
  for (const ImDrawList *list : data->CmdLists) {
    for (const ImDrawCmd &pcmd : list->CmdBuffer) {
      if (pcmd.UserCallback != nullptr) {
        if (pcmd.UserCallback != ImDrawCallback_ResetRenderState) pcmd.UserCallback(list, &pcmd);
        continue;
      }
      const float x0 = std::max((pcmd.ClipRect.x - clipOffset.x) * clipScale.x, 0.f);
      const float y0 = std::max((pcmd.ClipRect.y - clipOffset.y) * clipScale.y, 0.f);
      const float x1 = std::min((pcmd.ClipRect.z - clipOffset.x) * clipScale.x, static_cast<float>(extent.width));
      const float y1 = std::min((pcmd.ClipRect.w - clipOffset.y) * clipScale.y, static_cast<float>(extent.height));
      if (x1 <= x0 || y1 <= y0) continue;
      const VkRect2D scissor{{static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0)},
                             {static_cast<std::uint32_t>(x1 - x0), static_cast<std::uint32_t>(y1 - y0)}};
      vkCmdSetScissor(cmd, 0, 1, &scissor);

      const auto *texture = reinterpret_cast<const Texture *>(pcmd.GetTexID());
      if (texture == nullptr) continue;
      if (texture->image->view() != bound) {
        bound = texture->image->view();
        VkDescriptorImageInfo info{sampler_, bound, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &write);
      }
      vkCmdDrawIndexed(cmd, pcmd.ElemCount, 1, indexBase + pcmd.IdxOffset, static_cast<std::int32_t>(vertexBase + pcmd.VtxOffset), 0);
    }
    vertexBase += static_cast<std::uint32_t>(list->VtxBuffer.Size);
    indexBase += static_cast<std::uint32_t>(list->IdxBuffer.Size);
  }
}

} // namespace splat
