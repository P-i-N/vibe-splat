#include "viewer/line_renderer.hpp"

#include "core/vk_check.hpp"
#include "core/vk_compute.hpp"
#include "splat_shaders/shaders.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>

namespace splat {
namespace {

// Must match shaders/debug_line.vert.
struct Push {
  float view[16];
  float intrinsics[4];
  float viewport[4];
};

std::uint32_t pack_color(glm::vec4 c) {
  const auto channel = [](float v) { return static_cast<std::uint32_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); };
  return channel(c.r) | channel(c.g) << 8 | channel(c.b) << 16 | channel(c.a) << 24;
}

} // namespace

LineRenderer::LineRenderer(vk::Context &ctx, VkFormat colorFormat) : ctx_(ctx) {
  VkDevice dev = ctx_.device();
  VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  vk::check(vkCreatePipelineLayout(dev, &plci, nullptr, &layout_), "vkCreatePipelineLayout");

  VkShaderModule vert = vk::create_shader_module(dev, shaders::debug_line_vert);
  VkShaderModule frag = vk::create_shader_module(dev, shaders::debug_line_frag);
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  const VkVertexInputAttributeDescription attributes[2] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
      {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(Vertex, color)},
  };
  VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount = 2;
  vertexInput.pVertexAttributeDescriptions = attributes;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
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

LineRenderer::~LineRenderer() {
  VkDevice dev = ctx_.device();
  vkDeviceWaitIdle(dev);
  vkDestroyPipeline(dev, pipeline_, nullptr);
  vkDestroyPipelineLayout(dev, layout_, nullptr);
}

void LineRenderer::line(glm::vec3 a, glm::vec3 b, glm::vec4 color) {
  const std::uint32_t packed = pack_color(color);
  vertices_.push_back({a, packed});
  vertices_.push_back({b, packed});
}

void LineRenderer::record(VkCommandBuffer cmd, const Camera &camera, VkExtent2D extent, float nearPlane) {
  if (vertices_.empty()) return;
  const VkDeviceSize bytes = vertices_.size() * sizeof(Vertex);
  if (!buffer_ || buffer_->size() < bytes) {
    buffer_ = std::make_unique<vk::Buffer>(ctx_, bytes * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk::MemoryUsage::Upload);
  }
  std::memcpy(buffer_->mapped().data(), vertices_.data(), static_cast<std::size_t>(bytes));
  buffer_->flush();

  Push push{};
  std::memcpy(push.view, glm::value_ptr(camera.view), sizeof(push.view));
  push.intrinsics[0] = camera.fx;
  push.intrinsics[1] = camera.fy;
  push.intrinsics[2] = camera.cx;
  push.intrinsics[3] = camera.cy;
  push.viewport[0] = static_cast<float>(extent.width);
  push.viewport[1] = static_cast<float>(extent.height);
  push.viewport[2] = nearPlane;

  const VkViewport vp{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  const VkRect2D scissor{{0, 0}, extent};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
  const VkBuffer vb = buffer_->handle();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);
  vkCmdDraw(cmd, static_cast<std::uint32_t>(vertices_.size()), 1, 0, 0);
}

} // namespace splat
