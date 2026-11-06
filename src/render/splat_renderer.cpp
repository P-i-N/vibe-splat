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

// Must match shaders/splat_types.glsl (std430). Only their sizes matter here:
// the shaders write both.
struct GpuGaussian {
  std::uint32_t data[6];
};
static_assert(sizeof(GpuGaussian) == 24);

struct GpuSplat2D {
  std::uint32_t data[5];
};
static_assert(sizeof(GpuSplat2D) == 20);

// Must match the push constants of shaders/splat.vert.
struct DrawPush {
  float viewport[2];
  float pointRadius;
};

struct GpuCamera {
  float view[16];
  float position[4];
  float intrinsics[4];
  float viewport[4];
  std::uint32_t counts[4];
};
static_assert(sizeof(GpuCamera) == 128);

// Must match the push constants of shaders/preprocess.comp.
struct PreprocessPush {
  float splatScale;
  std::uint32_t maxShDegree;
  std::uint32_t pointMode;
  float pointRadius;
  std::uint32_t packedSh;
};

struct PackPush {
  std::uint32_t count;
};

// Must match the push constants of shaders/pack_sh.comp.
struct PackShPush {
  std::uint32_t count, coefficients;
};

// As packed_sh_words() in shaders/splat_types.glsl.
std::uint32_t packed_sh_words(std::uint32_t coefficients) {
  const std::uint32_t bits = coefficients > 1 ? 72 + 12 * (coefficients - 4) : 0;
  return (coefficients >= 9 ? 3 : 2) + (bits + 31) / 32;
}

struct DrawArgs {
  std::uint32_t vertexCount, instanceCount, firstVertex, firstInstance;
};

constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

} // namespace

SplatRenderer::SplatRenderer(vk::Context &ctx, VkFormat colorFormat)
    : ctx_(ctx), preprocess_(ctx, shaders::preprocess_comp(), 7, sizeof(PreprocessPush)),
      pack_(ctx, shaders::pack_gaussians_comp(), 5, sizeof(PackPush)), packSh_(ctx, shaders::pack_sh_comp(), 2, sizeof(PackShPush)),
      sorter_(ctx) {
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
  VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawPush)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &drawSetLayout_;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  vk::check(vkCreatePipelineLayout(dev, &plci, nullptr, &drawLayout_), "vkCreatePipelineLayout");

  VkShaderModule vert = vk::create_shader_module(dev, shaders::splat_vert());
  VkShaderModule frag = vk::create_shader_module(dev, shaders::splat_frag());
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
  const auto coefficients = static_cast<std::uint32_t>(cloud.shCoeffs());
  shStride_ = packed_sh_words(coefficients);
  shExternal_ = nullptr;
  sh_.reset();
  if (count_ == 0) {
    return;
  }

  // Packed on the GPU, by the shaders that pack a training scene, from raw
  // values held only until then.
  const auto upload = [&](const std::vector<float> &values) {
    auto b = std::make_unique<Buffer>(ctx_, values.size() * sizeof(float), kStorage, MemoryUsage::GpuOnly);
    b->upload(ctx_, std::span<const float>(values));
    return b;
  };
  const auto positions = upload(cloud.positions), scales = upload(cloud.scales), rotations = upload(cloud.rotations),
             opacities = upload(cloud.opacities), sh = upload(cloud.sh);
  capacity_ = 0;
  allocate(count_);
  sh_ = std::make_unique<Buffer>(ctx_, VkDeviceSize{count_} * shStride_ * sizeof(std::uint32_t), kStorage, MemoryUsage::GpuOnly);
  const auto count = static_cast<std::uint32_t>(count_);
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    pack_.dispatch(cmd, {positions.get(), scales.get(), rotations.get(), opacities.get(), gaussians_.get()}, PackPush{count},
                   vk::div_up(count, 256));
    packSh_.dispatch(cmd, {sh.get(), sh_.get()}, PackShPush{count, coefficients}, vk::div_up(count, 256));
    vk::compute_barrier(cmd);
  });
}

void SplatRenderer::allocate(std::size_t count) {
  if (count <= capacity_) return;
  // Room to grow, so that a trainer adding Gaussians does not reallocate every time.
  const VkDeviceSize n = capacity_ == 0 ? count : count + count / 4;
  gaussians_.reset();
  splats2d_.reset();
  gaussians_ = std::make_unique<Buffer>(ctx_, n * sizeof(GpuGaussian), kStorage, MemoryUsage::GpuOnly);
  splats2d_ = std::make_unique<Buffer>(ctx_, n * sizeof(GpuSplat2D), kStorage, MemoryUsage::GpuOnly);
  keys_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  values_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  tmpKeys_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  tmpValues_ = std::make_unique<Buffer>(ctx_, n * 4, kStorage, MemoryUsage::GpuOnly);
  capacity_ = n;
}

void SplatRenderer::set_scene(VkCommandBuffer cmd, const GpuCloud &cloud) {
  count_ = cloud.count;
  shDegree_ = std::min(cloud.activeDegree, cloud.shDegree);
  shStride_ = half_sh_words((cloud.shDegree + 1) * (cloud.shDegree + 1));
  sh_.reset();
  shExternal_ = cloud.sh;
  if (count_ == 0) return;
  allocate(count_);
  vk::compute_barrier(cmd);
  const auto count = static_cast<std::uint32_t>(count_);
  pack_.dispatch(cmd, {cloud.positions, cloud.scales, cloud.rotations, cloud.opacities, gaussians_.get()}, PackPush{count},
                 vk::div_up(count, 256));
  vk::compute_barrier(cmd);
}

void SplatRenderer::record(VkCommandBuffer cmd, const Camera &camera, VkImageView target, VkExtent2D extent,
                           std::array<float, 4> clearColor, bool clear) {
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
    gc.counts[1] = static_cast<std::uint32_t>(std::max(std::min(shDegree_, settings.maxShDegree), 0));
    gc.counts[2] = shStride_;
    gc.counts[3] = static_cast<std::uint32_t>((shDegree_ + 1) * (shDegree_ + 1));
    std::memcpy(camera_->mapped().data(), &gc, sizeof(gc));
    camera_->flush();

    const DrawArgs args{4, 0, 0, 0};
    vkCmdUpdateBuffer(cmd, drawArgs_->handle(), 0, sizeof(args), &args);
    vk::compute_barrier(cmd);

    const PreprocessPush push{settings.splatScale, static_cast<std::uint32_t>(std::max(settings.maxShDegree, 0)),
                              settings.points ? 1u : 0u, settings.pointRadius, sh_ ? 1u : 0u};
    const Buffer *sh = sh_ ? sh_.get() : shExternal_;
    preprocess_.dispatch(cmd, {gaussians_.get(), sh, camera_.get(), splats2d_.get(), keys_.get(), values_.get(), drawArgs_.get()},
                         push, vk::div_up(count, 256));
    vk::compute_barrier(cmd);
    sorter_.record(cmd, *keys_, *values_, *tmpKeys_, *tmpValues_, count);
  }

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = target;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
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
    const DrawPush push{{static_cast<float>(extent.width), static_cast<float>(extent.height)}, settings.points ? settings.pointRadius : 0.f};
    vkCmdPushConstants(cmd, drawLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
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
