#include "train/trainer.hpp"

#include "core/vk_check.hpp"
#include "splat_shaders/shaders.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace splat {

namespace {

using vk::Buffer;
using vk::MemoryUsage;

constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
constexpr std::uint32_t kTileSize = 16;
constexpr float kBeta1 = 0.9f, kBeta2 = 0.999f, kEpsilon = 1e-15f;
constexpr std::size_t kRawGradientsPerSplat = 9;

// Must match shaders/splat_types.glsl and train_types.glsl (std430).
struct GpuCamera {
  float view[16];
  float position[4];
  float intrinsics[4];
  float viewport[4];
  std::uint32_t counts[4];
};
static_assert(sizeof(GpuCamera) == 128);

struct GpuSplat {
  float data[12];
  std::uint32_t tiles[4];
};
static_assert(sizeof(GpuSplat) == 64);

struct CountPush {
  std::uint32_t count;
};
struct EmitPush {
  std::uint32_t count, gridW;
};
struct TilePush {
  std::uint32_t width, height, gridW, tileCount;
  float background[4];
};
struct SsimPush {
  std::uint32_t width, height;
};
struct LossPush {
  std::uint32_t width, height;
  float lambda;
};
struct FixedPush {
  std::uint32_t count;
  float scale;
};
struct AdamPush {
  std::uint32_t count;
  float learningRate, beta1, beta2, epsilon, correction1, correction2;
  std::uint32_t dcStride;
  float restScale;
};

std::unique_ptr<Buffer> gpu_buffer(vk::Context &ctx, VkDeviceSize bytes) {
  return std::make_unique<Buffer>(ctx, std::max<VkDeviceSize>(bytes, 4), kStorage, MemoryUsage::GpuOnly);
}

void clear(VkCommandBuffer cmd, const Buffer &buffer) { vkCmdFillBuffer(cmd, buffer.handle(), 0, VK_WHOLE_SIZE, 0); }

} // namespace

struct Trainer::Group {
  std::unique_ptr<Buffer> values, gradients, first, second;

  Group(vk::Context &ctx, std::span<const float> initial) {
    const VkDeviceSize bytes = initial.size_bytes();
    values = gpu_buffer(ctx, bytes);
    values->upload(ctx, initial);
    gradients = gpu_buffer(ctx, bytes);
    first = gpu_buffer(ctx, bytes);
    second = gpu_buffer(ctx, bytes);
    ctx.submit_and_wait([&](VkCommandBuffer cmd) {
      clear(cmd, *first);
      clear(cmd, *second);
      vk::compute_barrier(cmd);
    });
  }
};

Trainer::Trainer(vk::Context &ctx, const GaussianCloud &initial, const TrainerOptions &options)
    : ctx_(ctx), options_(options), preprocess_(ctx, shaders::train_preprocess_comp, 9),
      binCount_(ctx, shaders::train_bin_count_comp, 5, sizeof(CountPush)),
      binEmit_(ctx, shaders::train_bin_emit_comp, 6, sizeof(EmitPush)),
      tileRanges_(ctx, shaders::train_tile_ranges_comp, 2, sizeof(CountPush)),
      forward_(ctx, shaders::train_forward_comp, 4, sizeof(TilePush)),
      ssim_(ctx, shaders::train_ssim_comp, 4, sizeof(SsimPush)),
      loss_(ctx, shaders::train_loss_comp, 5, sizeof(LossPush)),
      gradient_(ctx, shaders::train_gradient_comp, 12),
      adam_(ctx, shaders::train_adam_comp, 4, sizeof(AdamPush)), sorter_(ctx) {
  initial.validate();
  if (initial.empty()) throw std::runtime_error("nothing to train: the cloud is empty");
  count_ = static_cast<std::uint32_t>(initial.size());
  shDegree_ = initial.shDegree;
  shFloats_ = static_cast<std::uint32_t>(3 * initial.shCoeffs());

  fixedPoint_ = options.forceFixedPoint || !ctx.info().floatAtomicAdd;
  if (fixedPoint_) {
    backward_.emplace(ctx, shaders::train_backward_fixed_comp, 7, sizeof(TilePush));
    fixedToFloat_.emplace(ctx, shaders::train_fixed_to_float_comp, 1, sizeof(FixedPush));
    fixedScale_ = std::make_unique<Buffer>(ctx, sizeof(float), kStorage, MemoryUsage::Upload);
  } else {
    backward_.emplace(ctx, shaders::train_backward_float_comp, 6, sizeof(TilePush));
  }

  groups_.push_back(std::make_unique<Group>(ctx, initial.positions));
  groups_.push_back(std::make_unique<Group>(ctx, initial.scales));
  groups_.push_back(std::make_unique<Group>(ctx, initial.rotations));
  groups_.push_back(std::make_unique<Group>(ctx, initial.opacities));
  groups_.push_back(std::make_unique<Group>(ctx, initial.sh));

  const VkDeviceSize n = count_;
  camera_ = std::make_unique<Buffer>(ctx, sizeof(GpuCamera), kStorage, MemoryUsage::Upload);
  counters_ = std::make_unique<Buffer>(ctx, sizeof(std::uint32_t) * 2, kStorage, MemoryUsage::Readback);
  splats_ = gpu_buffer(ctx, n * sizeof(GpuSplat));
  keys_ = gpu_buffer(ctx, n * 4);
  values_ = gpu_buffer(ctx, n * 4);
  tmpKeys_ = gpu_buffer(ctx, n * 4);
  tmpValues_ = gpu_buffer(ctx, n * 4);
  coverage_ = gpu_buffer(ctx, n * 4);
  rawGradients_ = gpu_buffer(ctx, n * kRawGradientsPerSplat * sizeof(float));
}

Trainer::~Trainer() { vkDeviceWaitIdle(ctx_.device()); }

void Trainer::ensure_size(int width, int height) {
  if (width == width_ && height == height_) return;
  if (width <= 0 || height <= 0) throw std::runtime_error("bad image size");
  width_ = width;
  height_ = height;
  gridW_ = vk::div_up(static_cast<std::uint32_t>(width), kTileSize);
  tileCount_ = gridW_ * vk::div_up(static_cast<std::uint32_t>(height), kTileSize);
  if (tileCount_ > 65536) throw std::runtime_error("the image is too large for the 16-bit tile index");
  const VkDeviceSize pixels = VkDeviceSize{static_cast<std::uint32_t>(width)} * static_cast<std::uint32_t>(height);
  if (!image_ || image_->size() < pixels * 16) {
    image_ = gpu_buffer(ctx_, pixels * 16);
    adjoint_ = gpu_buffer(ctx_, pixels * 16);
    fields_ = gpu_buffer(ctx_, pixels * 48);
    target_ = std::make_unique<Buffer>(ctx_, pixels * 4, kStorage, MemoryUsage::Upload);
  }
  if (!ranges_ || ranges_->size() < VkDeviceSize{tileCount_} * 8) ranges_ = gpu_buffer(ctx_, VkDeviceSize{tileCount_} * 8);
  const VkDeviceSize rows = static_cast<std::uint32_t>(height);
  if (!rowTotals_ || rowTotals_->size() < rows * 8) {
    rowTotals_ = std::make_unique<Buffer>(ctx_, rows * 8, kStorage, MemoryUsage::Readback);
  }
}

void Trainer::upload_camera(const Camera &camera) {
  ensure_size(camera.width, camera.height);
  GpuCamera gc{};
  std::memcpy(gc.view, glm::value_ptr(camera.view), sizeof(gc.view));
  gc.position[0] = camera.position.x;
  gc.position[1] = camera.position.y;
  gc.position[2] = camera.position.z;
  gc.position[3] = options_.nearPlane;
  gc.intrinsics[0] = camera.fx;
  gc.intrinsics[1] = camera.fy;
  gc.intrinsics[2] = camera.cx;
  gc.intrinsics[3] = camera.cy;
  gc.viewport[0] = static_cast<float>(camera.width);
  gc.viewport[1] = static_cast<float>(camera.height);
  gc.viewport[2] = 0.5f * static_cast<float>(camera.width) / camera.fx;
  gc.viewport[3] = 0.5f * static_cast<float>(camera.height) / camera.fy;
  gc.counts[0] = count_;
  gc.counts[1] = static_cast<std::uint32_t>(shDegree_);
  gc.counts[2] = shFloats_;
  std::memcpy(camera_->mapped().data(), &gc, sizeof(gc));
  camera_->flush();
}

// First submission: everything needed to learn how many tile entries this view
// has, so that the buffers and the sort can be sized for it.
std::uint32_t Trainer::record_prepare() {
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    clear(cmd, *counters_);
    vk::compute_barrier(cmd);
    preprocess_.dispatch(cmd,
                         {groups_[0]->values.get(), groups_[1]->values.get(), groups_[2]->values.get(),
                          groups_[3]->values.get(), groups_[4]->values.get(), camera_.get(), splats_.get(), keys_.get(),
                          values_.get()},
                         std::span<const std::byte>{}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    sorter_.record(cmd, *keys_, *values_, *tmpKeys_, *tmpValues_, count_);
    vk::compute_barrier(cmd);
    binCount_.dispatch(cmd, {keys_.get(), values_.get(), splats_.get(), coverage_.get(), counters_.get()},
                       CountPush{count_}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    sorter_.record_scan(cmd, *coverage_, count_);
    vk::compute_barrier(cmd);
  });
  counters_->invalidate();
  const std::uint32_t entries = counters_->as<std::uint32_t>()[1];

  if (!entryKeys_ || entryKeys_->size() < VkDeviceSize{entries} * 4) {
    const VkDeviceSize bytes = (VkDeviceSize{entries} + entries / 2 + 1024) * 4; // Room to grow as splats change
    entryKeys_ = gpu_buffer(ctx_, bytes);
    entryValues_ = gpu_buffer(ctx_, bytes);
    tmpEntryKeys_ = gpu_buffer(ctx_, bytes);
    tmpEntryValues_ = gpu_buffer(ctx_, bytes);
  }
  return entries;
}

void Trainer::record_forward(VkCommandBuffer cmd, std::uint32_t entries) {
  clear(cmd, *ranges_);
  vk::compute_barrier(cmd);
  if (entries > 0) {
    binEmit_.dispatch(cmd, {keys_.get(), values_.get(), splats_.get(), coverage_.get(), entryKeys_.get(), entryValues_.get()},
                      EmitPush{count_, gridW_}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    sorter_.record(cmd, *entryKeys_, *entryValues_, *tmpEntryKeys_, *tmpEntryValues_, entries, 16);
    vk::compute_barrier(cmd);
    tileRanges_.dispatch(cmd, {entryKeys_.get(), ranges_.get()}, CountPush{entries}, vk::div_up(entries, 256));
    vk::compute_barrier(cmd);
  }
  const TilePush push{static_cast<std::uint32_t>(width_), static_cast<std::uint32_t>(height_), gridW_, tileCount_,
                      {options_.background[0], options_.background[1], options_.background[2], 0.f}};
  forward_.dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get()}, push, vk::div_up(tileCount_, 16));
  vk::compute_barrier(cmd);
}

void Trainer::record_backward(VkCommandBuffer cmd) {
  const auto w = static_cast<std::uint32_t>(width_), h = static_cast<std::uint32_t>(height_);
  const std::uint32_t rowGroups = vk::div_up(h, 16);
  ssim_.dispatch(cmd, {image_.get(), target_.get(), fields_.get(), rowTotals_.get()}, SsimPush{w, h}, rowGroups);
  vk::compute_barrier(cmd);
  loss_.dispatch(cmd, {image_.get(), target_.get(), fields_.get(), adjoint_.get(), rowTotals_.get()},
                 LossPush{w, h, options_.lambdaSsim}, rowGroups);
  vk::compute_barrier(cmd);

  clear(cmd, *rawGradients_);
  for (const auto &group : groups_) clear(cmd, *group->gradients);
  vk::compute_barrier(cmd);
  const TilePush push{w, h, gridW_, tileCount_, {options_.background[0], options_.background[1], options_.background[2], 0.f}};
  if (fixedPoint_) {
    backward_->dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get(), adjoint_.get(), rawGradients_.get(), fixedScale_.get()},
                        push, vk::div_up(tileCount_, 16));
    vk::compute_barrier(cmd);
    const auto rawCount = static_cast<std::uint32_t>(count_ * kRawGradientsPerSplat);
    fixedToFloat_->dispatch(cmd, {rawGradients_.get()}, FixedPush{rawCount, fixedScale_->as<float>()[0]}, vk::div_up(rawCount, 256));
  } else {
    backward_->dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get(), adjoint_.get(), rawGradients_.get()},
                        push, vk::div_up(tileCount_, 16));
  }
  vk::compute_barrier(cmd);
  gradient_.dispatch(cmd,
                     {groups_[0]->values.get(), groups_[1]->values.get(), groups_[2]->values.get(), groups_[3]->values.get(),
                      groups_[4]->values.get(), camera_.get(), rawGradients_.get(), groups_[0]->gradients.get(),
                      groups_[1]->gradients.get(), groups_[2]->gradients.get(), groups_[3]->gradients.get(), groups_[4]->gradients.get()},
                     std::span<const std::byte>{}, vk::div_up(count_, 256));
  vk::compute_barrier(cmd);
}

std::vector<float> Trainer::render(const Camera &camera) {
  upload_camera(camera);
  const std::uint32_t entries = record_prepare();
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) { record_forward(cmd, entries); });
  std::vector<float> pixels = image_->download_as<float>(ctx_);
  pixels.resize(static_cast<std::size_t>(width_) * height_ * 4);
  return pixels;
}

StepStats Trainer::compute_gradients(const Camera &camera, std::span<const std::uint8_t> target) {
  upload_camera(camera);
  const std::size_t bytes = static_cast<std::size_t>(width_) * height_ * 4;
  if (target.size() != bytes) throw std::runtime_error("the target image does not match the camera");
  std::memcpy(target_->mapped().data(), target.data(), bytes);
  target_->flush();
  if (fixedPoint_) {
    // Per-pixel derivatives carry a factor 1 / (3 pixels), which this undoes.
    // The rest leaves a resolution of 1/64 and room for the sums.
    fixedScale_->as<float>()[0] = 3.f * static_cast<float>(width_) * static_cast<float>(height_) * 64.f;
    fixedScale_->flush();
  }

  const std::uint32_t entries = record_prepare();
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    record_forward(cmd, entries);
    record_backward(cmd);
  });

  rowTotals_->invalidate();
  counters_->invalidate();
  const auto rows = rowTotals_->as<float>();
  double ssimSum = 0.0, l1Sum = 0.0;
  for (int row = 0; row < height_; ++row) {
    ssimSum += rows[static_cast<std::size_t>(row) * 2];
    l1Sum += rows[static_cast<std::size_t>(row) * 2 + 1];
  }
  const double perValue = 1.0 / (3.0 * static_cast<double>(width_) * height_);
  StepStats stats;
  stats.l1 = static_cast<float>(l1Sum * perValue);
  stats.ssim = static_cast<float>(ssimSum * perValue);
  stats.loss = (1.f - options_.lambdaSsim) * stats.l1 + options_.lambdaSsim * (1.f - stats.ssim);
  stats.visibleSplats = counters_->as<std::uint32_t>()[0];
  stats.tileEntries = entries;
  return stats;
}

void Trainer::apply_gradients(const LearningRates &rates) {
  ++adamStep_;
  const float correction1 = 1.f - std::pow(kBeta1, static_cast<float>(adamStep_));
  const float correction2 = 1.f - std::pow(kBeta2, static_cast<float>(adamStep_));
  const float learning[5] = {rates.position, rates.scale, rates.rotation, rates.opacity, rates.sh};
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    for (std::size_t g = 0; g < groups_.size(); ++g) {
      const Group &group = *groups_[g];
      const auto floats = static_cast<std::uint32_t>(group.values->size() / sizeof(float));
      const AdamPush push{floats, learning[g], kBeta1, kBeta2, kEpsilon, correction1, correction2,
                          g == 4 ? shFloats_ : 0u, rates.shRestScale};
      adam_.dispatch(cmd, {group.values.get(), group.gradients.get(), group.first.get(), group.second.get()}, push, vk::div_up(floats, 256));
    }
    vk::compute_barrier(cmd);
  });
}

StepStats Trainer::step(const Camera &camera, std::span<const std::uint8_t> target, const LearningRates &rates) {
  const StepStats stats = compute_gradients(camera, target);
  apply_gradients(rates);
  return stats;
}

GaussianCloud Trainer::cloud() const {
  GaussianCloud out;
  out.resize(count_, shDegree_);
  out.positions = groups_[0]->values->download_as<float>(ctx_);
  out.scales = groups_[1]->values->download_as<float>(ctx_);
  out.rotations = groups_[2]->values->download_as<float>(ctx_);
  out.opacities = groups_[3]->values->download_as<float>(ctx_);
  out.sh = groups_[4]->values->download_as<float>(ctx_);
  return out;
}

Trainer::Gradients Trainer::gradients() const {
  Gradients out;
  out.positions = groups_[0]->gradients->download_as<float>(ctx_);
  out.scales = groups_[1]->gradients->download_as<float>(ctx_);
  out.rotations = groups_[2]->gradients->download_as<float>(ctx_);
  out.opacities = groups_[3]->gradients->download_as<float>(ctx_);
  out.sh = groups_[4]->gradients->download_as<float>(ctx_);
  return out;
}

} // namespace splat
