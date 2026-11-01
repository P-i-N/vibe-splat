#include "render/splat_renderer.hpp"

#include "core/vk_check.hpp"
#include "splat_shaders/shaders.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace splat {
namespace {

using vk::Buffer;
using vk::MemoryUsage;

// Must match shaders/splat_types.glsl (std430).
struct GpuGaussian {
  float posOpacity[4];
  float rotation[4];
  float scale[4];
};
static_assert(sizeof(GpuGaussian) == 48);

struct GpuSplat2D {
  float data[12];
};

struct GpuCamera {
  float view[16];
  float position[4];
  float intrinsics[4];
  float viewport[4];
  std::uint32_t counts[4];
};
static_assert(sizeof(GpuCamera) == 128);

struct DrawArgs {
  std::uint32_t vertexCount, instanceCount, firstVertex, firstInstance;
};

constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

} // namespace

SplatRenderer::SplatRenderer(vk::Context &ctx, VkFormat colorFormat)
    : ctx_(ctx), preprocess_(ctx, shaders::preprocess_comp, 7), sorter_(ctx) {
  camera_ = std::make_unique<Buffer>(ctx_, sizeof(GpuCamera), kStorage, MemoryUsage::Upload);
  drawArgs_ = std::make_unique<Buffer>(ctx_, sizeof(DrawArgs),
                                       kStorage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryUsage::GpuOnly);
  create_graphics_pipeline(colorFormat);
}

SplatRenderer::~SplatRenderer() {
  VkDevice dev = ctx_.device();
  vkDeviceWaitIdle(dev);
  vkDestroyPipeline(dev, drawPipeline_, nullptr);
  vkDestroyPipelineLayout(dev, drawLayout_, nullptr);
  vkDestroyDescriptorSetLayout(dev, drawSetLayout_, nullptr);
}

void SplatRenderer::create_graphics_pipeline(VkFormat colorFormat) {
  VkDevice dev = ctx_.device();
  drawSetLayout_ = vk::create_storage_set_layout(dev, 2, VK_SHADER_STAGE_VERTEX_BIT);
  VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &drawSetLayout_;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  vk::check(vkCreatePipelineLayout(dev, &plci, nullptr, &drawLayout_), "vkCreatePipelineLayout");

  VkShaderModule vert = vk::create_shader_module(dev, shaders::splat_vert);
  VkShaderModule frag = vk::create_shader_module(dev, shaders::splat_frag);
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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
  // Premultiplied "over", back to front.
  blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
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
  gpci.layout = drawLayout_;
  const VkResult r = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &drawPipeline_);
  vkDestroyShaderModule(dev, vert, nullptr);
  vkDestroyShaderModule(dev, frag, nullptr);
  vk::check(r, "vkCreateGraphicsPipelines");
}

void SplatRenderer::set_scene(const GaussianCloud &cloud) {
  cloud.validate();
  vkDeviceWaitIdle(ctx_.device());
  count_ = cloud.size();
  shDegree_ = cloud.shDegree;
  shFloats_ = static_cast<std::uint32_t>(3 * cloud.shCoeffs());
  if (count_ == 0) {
    return;
  }

  std::vector<GpuGaussian> gpu(count_);
  for (std::size_t i = 0; i < count_; ++i) {
    GpuGaussian &g = gpu[i];
    for (int a = 0; a < 3; ++a) {
      g.posOpacity[a] = cloud.positions[i * 3 + a];
      g.scale[a] = std::exp(cloud.scales[i * 3 + a]);
    }
    g.posOpacity[3] = sigmoid(cloud.opacities[i]);
    g.scale[3] = 0.f;
    float q[4];
    float len = 0.f;
    for (int a = 0; a < 4; ++a) {
      q[a] = cloud.rotations[i * 4 + a];
      len += q[a] * q[a];
    }
    len = std::sqrt(len);
    for (int a = 0; a < 4; ++a) {
      g.rotation[a] = len > 0.f ? q[a] / len : (a == 0 ? 1.f : 0.f);
    }
  }

  const VkDeviceSize n = count_;
  gaussians_ = std::make_unique<Buffer>(ctx_, n * sizeof(GpuGaussian), kStorage, MemoryUsage::GpuOnly);
  gaussians_->upload(ctx_, std::span<const GpuGaussian>(gpu));
  sh_ = std::make_unique<Buffer>(ctx_, cloud.sh.size() * sizeof(float), kStorage, MemoryUsage::GpuOnly);
  sh_->upload(ctx_, std::span<const float>(cloud.sh));
  splats2d_ = std::make_unique<Buffer>(ctx_, n * sizeof(GpuSplat2D), kStorage, MemoryUsage::GpuOnly);
  keys_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  values_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  tmpKeys_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  tmpValues_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
}

void SplatRenderer::record(VkCommandBuffer cmd, const Camera &camera, VkImageView target, VkExtent2D extent,
                           std::array<float, 4> clearColor) {
  const auto count = static_cast<std::uint32_t>(count_);

  if (count > 0) {
    GpuCamera gc{};
    std::memcpy(gc.view, glm::value_ptr(camera.view), sizeof(gc.view));
    gc.position[0] = camera.position.x;
    gc.position[1] = camera.position.y;
    gc.position[2] = camera.position.z;
    gc.intrinsics[0] = camera.fx;
    gc.intrinsics[1] = camera.fy;
    gc.intrinsics[2] = camera.cx;
    gc.intrinsics[3] = camera.cy;
    gc.viewport[0] = static_cast<float>(extent.width);
    gc.viewport[1] = static_cast<float>(extent.height);
    gc.viewport[2] = 0.5f * static_cast<float>(extent.width) / camera.fx;
    gc.viewport[3] = 0.5f * static_cast<float>(extent.height) / camera.fy;
    gc.counts[0] = count;
    gc.counts[1] = static_cast<std::uint32_t>(shDegree_);
    gc.counts[2] = shFloats_;
    std::memcpy(camera_->mapped().data(), &gc, sizeof(gc));
    camera_->flush();

    const DrawArgs args{4, 0, 0, 0};
    vkCmdUpdateBuffer(cmd, drawArgs_->handle(), 0, sizeof(args), &args);
    vk::compute_barrier(cmd);

    preprocess_.dispatch(cmd, {gaussians_.get(), sh_.get(), camera_.get(), splats2d_.get(), keys_.get(),
                               values_.get(), drawArgs_.get()},
                         std::span<const std::byte>{}, vk::div_up(count, 256));
    vk::compute_barrier(cmd);
    sorter_.record(cmd, *keys_, *values_, *tmpKeys_, *tmpValues_, count);
  }

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = target;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  std::memcpy(color.clearValue.color.float32, clearColor.data(), sizeof(float) * 4);
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, extent};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  vkCmdBeginRendering(cmd, &ri);

  if (count > 0) {
    const VkViewport vp{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline_);
    vk::push_storage_buffers(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawLayout_, {splats2d_.get(), values_.get()});
    const float viewportSize[2] = {static_cast<float>(extent.width), static_cast<float>(extent.height)};
    vkCmdPushConstants(cmd, drawLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(viewportSize), viewportSize);
    vkCmdDrawIndirect(cmd, drawArgs_->handle(), 0, 1, sizeof(DrawArgs));
  }
  vkCmdEndRendering(cmd);
}

std::uint32_t SplatRenderer::last_visible_count() const {
  if (count_ == 0) {
    return 0;
  }
  const auto args = drawArgs_->download_as<DrawArgs>(ctx_);
  return args.empty() ? 0 : args[0].instanceCount;
}

} // namespace splat
