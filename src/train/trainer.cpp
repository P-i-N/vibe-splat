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
constexpr std::size_t kRawGradientsPerSplat = 14; // RAW_PER_SPLAT in train_types.glsl
constexpr float kCurvatureBeta = 0.999f;
constexpr std::uint32_t kMaxMarks = 16;

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
};
static_assert(sizeof(GpuSplat) == 48);

struct CountPush {
  std::uint32_t count;
};
struct BinPush {
  std::uint32_t count, gridW, gridH;
};
struct TilePush {
  std::uint32_t width, height, gridW, tileCount;
  float background[4];
};
struct SsimPush {
  std::uint32_t width, height, gridW;
};
struct ExposurePush {
  float transform[12];
  std::uint32_t pixels;
};
struct LossPush {
  std::uint32_t width, height, gridW;
  float lambda;
  std::uint32_t scoring;
};
struct ErrorMaskPush {
  std::uint32_t pixels;
  float share;
};
constexpr std::uint32_t kStatisticsFlag = 1, kPoseFlag = 2, kSelectiveFlag = 4, kSecondOrderFlag = 8;

struct GradientPush {
  float opacityReg, scaleReg; // Per Gaussian: the option divided by the count
  std::uint32_t mipFilter;
  float shDropout;
  std::uint32_t shRetained, shSeed, mode, flags;
  float rates[5];
  float restScale, beta1, beta2, epsilon, correction1, correction2, trustRegion, curvatureScale, curvatureBeta, perValue;
  std::uint32_t seed, shFloats;
};
static_assert(sizeof(GradientPush) <= 128);

struct PreprocessPush {
  std::uint32_t mipFilter;
  float shDropout;
  std::uint32_t shRetained, shSeed;
};

struct NoisePush {
  std::uint32_t count;
  float scaler;
  std::uint32_t iteration;
};

struct ScorePush {
  std::uint32_t count;
  float toPixels, loss;
};
struct FixedPush {
  std::uint32_t count;
  float scale;
};
struct HutchinsonPush {
  std::uint32_t pixels, seed;
};
struct GatherPush {
  std::uint32_t count, stride;
};

std::unique_ptr<Buffer> gpu_buffer(vk::Context &ctx, VkDeviceSize bytes) {
  return std::make_unique<Buffer>(ctx, std::max<VkDeviceSize>(bytes, 4), kStorage, MemoryUsage::GpuOnly);
}

void clear(VkCommandBuffer cmd, const Buffer &buffer) { vkCmdFillBuffer(cmd, buffer.handle(), 0, VK_WHOLE_SIZE, 0); }

} // namespace

// What the chain-rule pass does with the gradients, as MODE_* in train_gradient.comp.
enum class Trainer::Chain : std::uint32_t { Statistics, Gradients, Step, Curvature };

Camera refined_pose(const Camera &camera, const std::array<float, 6> &move) {
  const glm::dvec3 w(move[0], move[1], move[2]), v(move[3], move[4], move[5]);
  const double angle = glm::length(w);
  const glm::dmat3 K(0.0, w.z, -w.y, -w.z, 0.0, w.x, w.y, -w.x, 0.0);
  // Rodrigues' formula, with the limits of its factors at small angles.
  const double a = angle > 1e-8 ? std::sin(angle) / angle : 1.0, b = angle > 1e-8 ? (1.0 - std::cos(angle)) / (angle * angle) : 0.5;
  const glm::dmat3 R = glm::dmat3(1.0) + a * K + b * K * K;
  const glm::dmat4 before(camera.view);
  const glm::dmat3 rotation = R * glm::dmat3(before);
  const glm::dvec3 translation = R * glm::dvec3(before[3]) + v;
  Camera out = camera;
  glm::dmat4 after(rotation);
  after[3] = glm::dvec4(translation, 1.0);
  out.view = glm::mat4(after);
  out.position = glm::vec3(-glm::transpose(rotation) * translation);
  return out;
}

// The optimiser's state is a word per parameter, as train_gradient.comp
// describes. The values are a float each, but for SH (see half_sh_words()).
// Gradients are only stored for compute_gradients(), so they are made then,
// a float per parameter.
struct Trainer::Group {
  std::unique_ptr<Buffer> values, state, gradients;

  Group() = default;
  template <class T> Group(vk::Context &ctx, std::span<const T> initial, std::size_t parameters) {
    values = gpu_buffer(ctx, initial.size_bytes());
    values->upload(ctx, initial);
    state = gpu_buffer(ctx, VkDeviceSize{parameters} * sizeof(std::uint32_t));
    ctx.submit_and_wait([&](VkCommandBuffer cmd) {
      clear(cmd, *state);
      vk::compute_barrier(cmd);
    });
  }
};

Trainer::Trainer(vk::Context &ctx, const GaussianCloud &initial, const TrainerOptions &options)
    : ctx_(ctx), options_(options), preprocess_(ctx, shaders::train_preprocess_comp, 10, sizeof(PreprocessPush)),
      binCount_(ctx, shaders::train_bin_count_comp, 5, sizeof(BinPush)),
      binEmit_(ctx, shaders::train_bin_emit_comp, 6, sizeof(BinPush)),
      tileRanges_(ctx, shaders::train_tile_ranges_comp, 2, sizeof(CountPush)),
      forward_(ctx, shaders::train_forward_comp, 5, sizeof(TilePush)),
      exposure_(ctx, shaders::train_exposure_comp, 2, sizeof(ExposurePush)),
      ssim_(ctx, shaders::train_ssim_comp, 4, sizeof(SsimPush)),
      loss_(ctx, shaders::train_loss_comp, 7, sizeof(LossPush)),
      exposureBackward_(ctx, shaders::train_exposure_backward_comp, 3, sizeof(ExposurePush)),
      errorMask_(ctx, shaders::train_error_mask_comp, 2, sizeof(ErrorMaskPush)),
      score_(ctx, shaders::train_score_comp, 2, sizeof(ScorePush)),
      hutchinson_(ctx, shaders::train_hutchinson_comp, 1, sizeof(HutchinsonPush)),
      gradient_(ctx, shaders::train_gradient_comp, 21, sizeof(GradientPush)), shStep_(ctx, shaders::train_sh_step_comp, 4, sizeof(GradientPush)),
      noise_(ctx, shaders::train_noise_comp, 4, sizeof(NoisePush)), gather_(ctx, shaders::train_gather_comp, 3, sizeof(GatherPush)),
      sorter_(ctx) {
  initial.validate();
  if (initial.empty()) throw std::runtime_error("nothing to train: the cloud is empty");
  count_ = static_cast<std::uint32_t>(initial.size());
  shDegree_ = initial.shDegree;
  activeShDegree_ = initial.shDegree;
  shFloats_ = static_cast<std::uint32_t>(3 * initial.shCoeffs());
  shWords_ = half_sh_words(initial.shCoeffs());

  fixedPoint_ = options.forceFixedPoint || !ctx.info().floatAtomicAdd;
  if (fixedPoint_) {
    backward_.emplace(ctx, shaders::train_backward_fixed_comp, 9, sizeof(TilePush));
    fixedToFloat_.emplace(ctx, shaders::train_fixed_to_float_comp, 1, sizeof(FixedPush));
    fixedScale_ = std::make_unique<Buffer>(ctx, sizeof(float), kStorage, MemoryUsage::Upload);
  } else {
    backward_.emplace(ctx, shaders::train_backward_float_comp, 8, sizeof(TilePush));
  }

  for (const auto *values : {&initial.positions, &initial.scales, &initial.rotations, &initial.opacities}) {
    groups_.push_back(std::make_unique<Group>(ctx, std::span<const float>(*values), values->size()));
  }
  const std::vector<std::uint32_t> sh = pack_half_sh(initial.sh, initial.shCoeffs());
  groups_.push_back(std::make_unique<Group>(ctx, std::span<const std::uint32_t>(sh), initial.sh.size()));

  camera_ = std::make_unique<Buffer>(ctx, sizeof(GpuCamera), kStorage, MemoryUsage::Upload);
  counters_ = std::make_unique<Buffer>(ctx, sizeof(std::uint32_t) * 2, kStorage, MemoryUsage::Readback);
  errorMax_ = gpu_buffer(ctx, sizeof(std::uint32_t));
  allocate_per_gaussian();

  VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qci.queryCount = kMaxMarks;
  vk::check(vkCreateQueryPool(ctx.device(), &qci, nullptr, &timestamps_), "vkCreateQueryPool");
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
  nanosecondsPerTick_ = properties.limits.timestampPeriod;
}

void Trainer::begin_timing(VkCommandBuffer cmd) {
  marks_.clear();
  vkCmdResetQueryPool(cmd, timestamps_, 0, kMaxMarks);
  vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps_, 0);
}

void Trainer::mark(VkCommandBuffer cmd, const char *stage) {
  if (marks_.size() + 1 >= kMaxMarks) return;
  marks_.push_back(stage);
  vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, timestamps_, static_cast<std::uint32_t>(marks_.size()));
}

void Trainer::finish_timing() {
  std::uint64_t ticks[kMaxMarks]{};
  const auto count = static_cast<std::uint32_t>(marks_.size() + 1);
  vk::check(vkGetQueryPoolResults(ctx_.device(), timestamps_, 0, count, sizeof(ticks), ticks, sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults");
  for (std::size_t m = 0; m < marks_.size(); ++m) {
    const double ms = static_cast<double>(ticks[m + 1] - ticks[m]) * nanosecondsPerTick_ * 1e-6;
    const auto it = std::find_if(stageTotals_.begin(), stageTotals_.end(), [&](const auto &s) { return s.first == marks_[m]; });
    if (it == stageTotals_.end()) stageTotals_.emplace_back(marks_[m], ms);
    else it->second += ms;
  }
}

std::vector<std::pair<std::string, double>> Trainer::timings() const {
  auto out = stageTotals_;
  for (auto &stage : out) stage.second /= static_cast<double>(std::max<std::uint64_t>(timedIterations_, 1));
  return out;
}

void Trainer::allocate_per_gaussian() {
  const VkDeviceSize n = count_;
  splats_ = gpu_buffer(ctx_, n * sizeof(GpuSplat));
  keys_ = gpu_buffer(ctx_, n * 4);
  values_ = gpu_buffer(ctx_, n * 4);
  tmpKeys_ = gpu_buffer(ctx_, n * 4);
  tmpValues_ = gpu_buffer(ctx_, n * 4);
  coverage_ = gpu_buffer(ctx_, n * 4);
  rawGradients_ = gpu_buffer(ctx_, n * kRawGradientsPerSplat * sizeof(float));
  stats_ = gpu_buffer(ctx_, n * 32);
  filter3d_ = gpu_buffer(ctx_, n * sizeof(float));
  if (options_.poseGradients) {
    poseTotals_ = std::make_unique<Buffer>(ctx_, VkDeviceSize{vk::div_up(count_, 256u)} * 6 * sizeof(float), kStorage, MemoryUsage::Readback);
  }
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    clear(cmd, *stats_);
    clear(cmd, *filter3d_);
    clear(cmd, *rawGradients_);
    vk::compute_barrier(cmd);
  });
}

Trainer::~Trainer() {
  if (prefetchWorker_.joinable()) {
    prefetchWorker_.request_stop();
    prefetchChanged_.notify_all();
    prefetchWorker_.join();
  }
  vkDeviceWaitIdle(ctx_.device());
  vkDestroyQueryPool(ctx_.device(), timestamps_, nullptr);
}

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
    stops_ = gpu_buffer(ctx_, pixels * 4);
    exposed_ = gpu_buffer(ctx_, pixels * 16);
    exposureTotals_ = std::make_unique<Buffer>(ctx_, (pixels + 255) / 256 * 12 * sizeof(float), kStorage, MemoryUsage::Readback);
    adjoint_ = gpu_buffer(ctx_, pixels * 16);
    edge_ = gpu_buffer(ctx_, pixels * 4);
    fields_ = gpu_buffer(ctx_, pixels * 48);
    // The target is written by the CPU into staging memory and copied to the
    // device once per iteration, as the loss passes read it several times.
    targetStaging_ = std::make_unique<Buffer>(ctx_, pixels * 4, kStorage, MemoryUsage::Upload);
    target_ = gpu_buffer(ctx_, pixels * 4);
  }
  if (!ranges_ || ranges_->size() < VkDeviceSize{tileCount_} * 8) ranges_ = gpu_buffer(ctx_, VkDeviceSize{tileCount_} * 8);
  if (!groupTotals_ || groupTotals_->size() < VkDeviceSize{tileCount_} * 8) {
    groupTotals_ = std::make_unique<Buffer>(ctx_, VkDeviceSize{tileCount_} * 8, kStorage, MemoryUsage::Readback);
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
  gc.counts[1] = static_cast<std::uint32_t>(activeShDegree_);
  gc.counts[2] = shWords_;
  gc.counts[3] = static_cast<std::uint32_t>((activeShDegree_ + 1) * (activeShDegree_ + 1));
  std::memcpy(camera_->mapped().data(), &gc, sizeof(gc));
  camera_->flush();
}

// First submission: everything needed to learn how many tile entries this view
// has, so that the buffers and the sort can be sized for it.
std::uint32_t Trainer::record_prepare() {
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    begin_timing(cmd);
    clear(cmd, *counters_);
    vk::compute_barrier(cmd);
    preprocess_.dispatch(cmd,
                         {groups_[0]->values.get(), groups_[1]->values.get(), groups_[2]->values.get(),
                          groups_[3]->values.get(), groups_[4]->values.get(), camera_.get(), splats_.get(), keys_.get(),
                          values_.get(), filter3d_.get()},
                         PreprocessPush{options_.mipFilter2d ? 1u : 0u, dropoutNow_, shRetained_, shSeed_}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    mark(cmd, "project");
    sorter_.record(cmd, *keys_, *values_, *tmpKeys_, *tmpValues_, count_);
    vk::compute_barrier(cmd);
    mark(cmd, "depth sort");
    binCount_.dispatch(cmd, {keys_.get(), values_.get(), splats_.get(), coverage_.get(), counters_.get()},
                       BinPush{count_, gridW_, tileCount_ / gridW_}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    sorter_.record_scan(cmd, *coverage_, count_);
    vk::compute_barrier(cmd);
    mark(cmd, "count tiles");
  });
  finish_timing();
  counters_->invalidate();
  const std::uint32_t entries = counters_->as<std::uint32_t>()[1];

  if (!entryKeys_ || entryKeys_->size() < VkDeviceSize{entries} * 4) {
    // Check before allocating: running out of memory is not recoverable, this is.
    const auto entryBytes = [](std::uint64_t n) { return n * Trainer::kBytesPerTileEntry; };
    const std::uint64_t others = gpu_bytes() - (entryKeys_ ? entryKeys_->size() * 4 : 0);
    if (others + entryBytes(entries) > options_.memoryBudget) {
      throw BudgetExceeded(std::format("{} tile entries do not fit the memory budget", entries));
    }
    // Room to grow as splats change, as far as the budget lets it.
    const std::uint64_t wanted = std::uint64_t{entries} + entries / 2 + 1024;
    const std::uint64_t affordable = options_.memoryBudget == ~std::uint64_t{0} ? wanted : (options_.memoryBudget - others) / kBytesPerTileEntry;
    const VkDeviceSize bytes = std::max<std::uint64_t>(std::min(wanted, affordable), entries) * 4;
    entryKeys_.reset();
    entryValues_.reset();
    tmpEntryKeys_.reset();
    tmpEntryValues_.reset();
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
                      BinPush{count_, gridW_, tileCount_ / gridW_}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
    sorter_.record(cmd, *entryKeys_, *entryValues_, *tmpEntryKeys_, *tmpEntryValues_, entries, 16);
    vk::compute_barrier(cmd);
    tileRanges_.dispatch(cmd, {entryKeys_.get(), ranges_.get()}, CountPush{entries}, vk::div_up(entries, 256));
    vk::compute_barrier(cmd);
  }
  const TilePush push{static_cast<std::uint32_t>(width_), static_cast<std::uint32_t>(height_), gridW_, tileCount_,
                      {options_.background[0], options_.background[1], options_.background[2], 0.f}};
  mark(cmd, "tile lists");
  forward_.dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get(), stops_.get()}, push, tileCount_);
  vk::compute_barrier(cmd);
  mark(cmd, "composite");
}

void Trainer::record_backward(VkCommandBuffer cmd) {
  const auto w = static_cast<std::uint32_t>(width_), h = static_cast<std::uint32_t>(height_);
  // A neutral exposure changes nothing, so its passes are left out.
  exposing_ = exposureValues_ != kNeutralExposure;
  ExposurePush exposure{};
  std::copy(exposureValues_.begin(), exposureValues_.end(), exposure.transform);
  exposure.pixels = w * h;
  if (exposing_) {
    exposure_.dispatch(cmd, {image_.get(), exposed_.get()}, exposure, vk::div_up(w * h, 256));
    vk::compute_barrier(cmd);
  }
  const vk::Buffer *seen = exposing_ ? exposed_.get() : image_.get();
  if (scoreErrorShare_ > 0.f) clear(cmd, *errorMax_);
  ssim_.dispatch(cmd, {seen, target_.get(), fields_.get(), groupTotals_.get()}, SsimPush{w, h, gridW_}, tileCount_);
  vk::compute_barrier(cmd);
  loss_.dispatch(cmd, {seen, target_.get(), fields_.get(), adjoint_.get(), groupTotals_.get(), errorMax_.get(), edge_.get()},
                 LossPush{w, h, gridW_, options_.lambdaSsim, scoreErrorShare_ > 0.f ? 1u : 0u}, tileCount_);
  vk::compute_barrier(cmd);
  if (exposing_) {
    exposureBackward_.dispatch(cmd, {image_.get(), adjoint_.get(), exposureTotals_.get()}, exposure, vk::div_up(w * h, 256));
    vk::compute_barrier(cmd);
  }
  if (scoreErrorShare_ > 0.f) {
    errorMask_.dispatch(cmd, {adjoint_.get(), errorMax_.get()}, ErrorMaskPush{w * h, scoreErrorShare_}, vk::div_up(w * h, 256));
    vk::compute_barrier(cmd);
  }
  mark(cmd, "loss");
  record_backward_from_adjoint(cmd);
}

void Trainer::record_backward_from_adjoint(VkCommandBuffer cmd) {
  const auto w = static_cast<std::uint32_t>(width_), h = static_cast<std::uint32_t>(height_);
  clear(cmd, *rawGradients_);
  vk::compute_barrier(cmd);
  const TilePush push{w, h, gridW_, tileCount_, {options_.background[0], options_.background[1], options_.background[2], 0.f}};
  if (fixedPoint_) {
    backward_->dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get(), adjoint_.get(), stops_.get(), rawGradients_.get(), fixedScale_.get(), edge_.get()},
                        push, tileCount_);
    vk::compute_barrier(cmd);
    const auto rawCount = static_cast<std::uint32_t>(count_ * kRawGradientsPerSplat);
    fixedToFloat_->dispatch(cmd, {rawGradients_.get()}, FixedPush{rawCount, fixedScale_->as<float>()[0]}, vk::div_up(rawCount, 256));
  } else {
    backward_->dispatch(cmd, {splats_.get(), entryValues_.get(), ranges_.get(), image_.get(), adjoint_.get(), stops_.get(), rawGradients_.get(), edge_.get()},
                        push, tileCount_);
  }
  vk::compute_barrier(cmd);
  mark(cmd, "backward");
}

void Trainer::record_chain(VkCommandBuffer cmd, Chain mode, bool statistics, const LearningRates *rates) {
  GradientPush push{};
  push.opacityReg = options_.opacityReg / static_cast<float>(count_);
  push.scaleReg = options_.scaleReg / static_cast<float>(count_);
  push.mipFilter = options_.mipFilter2d ? 1u : 0u;
  push.shDropout = dropoutNow_;
  push.shRetained = shRetained_;
  push.shSeed = shSeed_;
  push.mode = static_cast<std::uint32_t>(mode);
  push.flags = (statistics ? kStatisticsFlag : 0u) | (options_.poseGradients && mode != Chain::Curvature ? kPoseFlag : 0u) |
               (options_.selectiveAdam ? kSelectiveFlag : 0u) | (secondOrder_ ? kSecondOrderFlag : 0u);
  push.beta1 = kBeta1;
  push.beta2 = kBeta2;
  push.epsilon = kEpsilon;
  push.curvatureBeta = kCurvatureBeta;
  push.curvatureScale = 1.f / (1.f - std::pow(kCurvatureBeta, static_cast<float>(std::max(curvatureUpdates_, 1u))));
  if (mode == Chain::Step) {
    ++adamStep_;
    const float learning[5] = {rates->position, rates->scale, rates->rotation, rates->opacity, rates->sh};
    std::copy(std::begin(learning), std::end(learning), push.rates);
    push.restScale = rates->shRestScale;
    push.correction1 = 1.f - std::pow(kBeta1, static_cast<float>(adamStep_));
    push.correction2 = 1.f - std::pow(kBeta2, static_cast<float>(adamStep_));
    push.trustRegion = rates->trustRegion;
    push.seed = adamStep_;
  } else if (mode == Chain::Curvature) {
    push.perValue = 1.f / (3.f * static_cast<float>(width_) * static_cast<float>(height_));
    push.seed = curvatureUpdates_;
  } else if (mode == Chain::Gradients) {
    for (const auto &group : groups_) {
      if (!group->gradients) group->gradients = gpu_buffer(ctx_, group->state->size());
    }
  }
  push.shFloats = shFloats_;
  // Bindings a pass does not use get any buffer: they are never touched. A step
  // or curvature update replaces the backward pass's sums with what the SH pass needs.
  const auto gradients = [&](std::size_t g) { return mode == Chain::Gradients ? groups_[g]->gradients.get() : rawGradients_.get(); };
  gradient_.dispatch(cmd,
                     {groups_[0]->values.get(), groups_[1]->values.get(), groups_[2]->values.get(), groups_[3]->values.get(),
                      groups_[4]->values.get(), camera_.get(), rawGradients_.get(), splats_.get(), stats_.get(), filter3d_.get(),
                      groups_[0]->state.get(), groups_[1]->state.get(), groups_[2]->state.get(), groups_[3]->state.get(), groups_[4]->state.get(),
                      gradients(0), gradients(1), gradients(2), gradients(3), gradients(4),
                      poseTotals_ ? poseTotals_.get() : rawGradients_.get()},
                     push, vk::div_up(count_, 256));
  vk::compute_barrier(cmd);
  if (mode == Chain::Step || mode == Chain::Curvature) {
    shStep_.dispatch(cmd, {groups_[4]->values.get(), groups_[4]->state.get(), rawGradients_.get(), camera_.get()}, push,
                     vk::div_up(count_ * shWords_, 256));
    vk::compute_barrier(cmd);
  }
  mark(cmd, mode == Chain::Step ? "chain rule and step" : "chain rule");
}

std::vector<float> Trainer::render(const Camera &camera) {
  upload_camera(camera);
  dropoutNow_ = 0.f; // Reports and evaluation see the whole model
  const std::uint32_t entries = record_prepare();
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    begin_timing(cmd);
    record_forward(cmd, entries);
  });
  marks_.clear(); // Only training iterations count towards the timings
  std::vector<float> pixels = image_->download_as<float>(ctx_);
  pixels.resize(static_cast<std::size_t>(width_) * height_ * 4);
  return pixels;
}

StepStats Trainer::compute_gradients(const Camera &camera, std::span<const std::uint8_t> target) {
  return run(camera, target, Chain::Gradients, nullptr);
}

StepStats Trainer::collect_statistics(const Camera &camera, std::span<const std::uint8_t> target) {
  return run(camera, target, Chain::Statistics, nullptr);
}

StepStats Trainer::score_view(const Camera &camera, std::span<const std::uint8_t> target, float errorShare, float pixelWeight) {
  scoreErrorShare_ = errorShare;
  const StepStats stats = run(camera, target, Chain::Statistics, nullptr);
  scoreErrorShare_ = 0.f;
  // The raw sums are over 3 pixels of the image, like the derivatives.
  const float toPixels = 3.f * static_cast<float>(width_) * static_cast<float>(height_) * pixelWeight;
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    score_.dispatch(cmd, {rawGradients_.get(), stats_.get()}, ScorePush{count_, toPixels, stats.loss}, vk::div_up(count_, 256));
    vk::compute_barrier(cmd);
  });
  return stats;
}

void Trainer::wait_for_prefetch() {
  std::unique_lock lock(prefetchMutex_);
  prefetchChanged_.wait(lock, [&] { return !prefetchPending_; });
}

void Trainer::prefetch_target(std::span<const std::uint8_t> target) {
  wait_for_prefetch();
  if (!targetNext_ || targetNext_->size() < target.size()) {
    targetNext_ = std::make_unique<Buffer>(ctx_, std::max<VkDeviceSize>(target.size(), 4), kStorage, MemoryUsage::Upload);
  }
  if (!prefetchWorker_.joinable()) {
    prefetchWorker_ = std::jthread([this](std::stop_token stop) {
      std::unique_lock lock(prefetchMutex_);
      for (;;) {
        prefetchChanged_.wait(lock, [&] { return prefetchPending_ || stop.stop_requested(); });
        if (stop.stop_requested()) return;
        std::memcpy(prefetchDestination_, prefetched_, prefetchedBytes_);
        prefetchPending_ = false;
        prefetchChanged_.notify_all();
      }
    });
  }
  {
    const std::lock_guard lock(prefetchMutex_);
    prefetched_ = target.data();
    prefetchedBytes_ = target.size();
    prefetchDestination_ = reinterpret_cast<std::uint8_t *>(targetNext_->mapped().data());
    prefetchPending_ = true;
  }
  prefetchChanged_.notify_all();
}

StepStats Trainer::run(const Camera &camera, std::span<const std::uint8_t> target, Chain chain, const LearningRates *rates,
                       std::span<const std::uint8_t> nextTarget) {
  upload_camera(camera);
  dropoutNow_ = rates != nullptr ? shDropout_ : 0.f; // Dropout trains; scoring and statistics see the whole model
  const std::size_t bytes = static_cast<std::size_t>(width_) * height_ * 4;
  if (target.size() != bytes) throw std::runtime_error("the target image does not match the camera");
  if (prefetched_ == target.data() && prefetchedBytes_ == bytes) {
    // Started during the previous iteration, so normally long finished.
    wait_for_prefetch();
    std::swap(targetStaging_, targetNext_);
    prefetched_ = nullptr;
  } else {
    // A prefetch of some other image carries on in the background. The staging
    // buffer may be one that prefetch sized for a smaller image, and the image
    // buffers, which decide on reallocating, may be larger already.
    if (targetStaging_->size() < bytes) targetStaging_ = std::make_unique<Buffer>(ctx_, bytes, kStorage, MemoryUsage::Upload);
    std::memcpy(targetStaging_->mapped().data(), target.data(), bytes);
  }
  targetStaging_->flush();
  if (!nextTarget.empty()) prefetch_target(nextTarget);
  if (fixedPoint_) {
    // Per-pixel derivatives carry a factor 1 / (3 pixels), which this undoes.
    // The rest leaves a resolution of 1/64 and room for the sums.
    fixedScale_->as<float>()[0] = 3.f * static_cast<float>(width_) * static_cast<float>(height_) * 64.f;
    fixedScale_->flush();
  }

  const std::uint32_t entries = record_prepare();
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    begin_timing(cmd);
    const VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, targetStaging_->handle(), target_->handle(), 1, &region);
    record_forward(cmd, entries);
    record_backward(cmd);
    record_chain(cmd, chain, true, rates);
    if (rates != nullptr && options_.positionNoise > 0.f) record_noise(cmd, options_.positionNoise * rates->position, adamStep_);
  });
  finish_timing();
  ++timedIterations_;

  groupTotals_->invalidate();
  counters_->invalidate();
  const auto totals = groupTotals_->as<float>();
  double ssimSum = 0.0, l1Sum = 0.0;
  for (std::uint32_t g = 0; g < tileCount_; ++g) {
    ssimSum += totals[static_cast<std::size_t>(g) * 2];
    l1Sum += totals[static_cast<std::size_t>(g) * 2 + 1];
  }
  const double perValue = 1.0 / (3.0 * static_cast<double>(width_) * height_);
  StepStats stats;
  stats.l1 = static_cast<float>(l1Sum * perValue);
  stats.ssim = static_cast<float>(ssimSum * perValue);
  stats.loss = (1.f - options_.lambdaSsim) * stats.l1 + options_.lambdaSsim * (1.f - stats.ssim);
  stats.visibleSplats = counters_->as<std::uint32_t>()[0];
  stats.tileEntries = entries;
  if (exposing_) {
    exposureTotals_->invalidate();
    const auto exposureTotals = exposureTotals_->as<float>();
    std::array<double, 12> exposureSums{};
    for (std::uint32_t g = 0; g < vk::div_up(static_cast<std::uint32_t>(width_ * height_), 256u); ++g) {
      for (std::size_t k = 0; k < 12; ++k) exposureSums[k] += exposureTotals[g * 12 + k];
    }
    for (std::size_t k = 0; k < 12; ++k) stats.exposureGradient[k] = static_cast<float>(exposureSums[k]);
  }
  if (options_.poseGradients) {
    poseTotals_->invalidate();
    const auto poseTotals = poseTotals_->as<float>();
    std::array<double, 6> poseSums{};
    for (std::uint32_t g = 0; g < vk::div_up(count_, 256u); ++g) {
      for (std::size_t k = 0; k < 6; ++k) poseSums[k] += poseTotals[g * 6 + k];
    }
    for (std::size_t k = 0; k < 6; ++k) stats.poseGradient[k] = static_cast<float>(poseSums[k]);
  }
  return stats;
}

void Trainer::enable_second_order() {
  if (fixedPoint_) throw std::runtime_error("second-order training needs float atomics");
  if (secondOrder_) return;
  // The curvature takes the place of the second moment, from zero.
  for (std::size_t g = 0; g < groups_.size(); ++g) {
    if (g == 2) continue; // Rotation stays with Adam: the paper's bound for it has a constant it never gives
    std::vector<std::uint32_t> state = groups_[g]->state->download_as<std::uint32_t>(ctx_);
    for (std::uint32_t &word : state) word &= 0xFFFFu;
    groups_[g]->state->upload(ctx_, std::span<const std::uint32_t>(state));
  }
  secondOrder_ = true;
  curvatureUpdates_ = 0;
}

void Trainer::update_curvature() {
  const auto pixels = static_cast<std::uint32_t>(width_) * static_cast<std::uint32_t>(height_);
  ++curvatureUpdates_;
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    begin_timing(cmd);
    hutchinson_.dispatch(cmd, {adjoint_.get()}, HutchinsonPush{pixels, curvatureUpdates_}, vk::div_up(pixels, 256));
    vk::compute_barrier(cmd);
    record_backward_from_adjoint(cmd);
    record_chain(cmd, Chain::Curvature, false, nullptr);
  });
  marks_.clear(); // Only training iterations count towards the timings
}

void Trainer::apply_gradients(const LearningRates &rates) {
  // From the backward pass's sums, which compute_gradients() left: the
  // gradients follow from them again. The step uses them up (see record_chain()).
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    begin_timing(cmd);
    record_chain(cmd, Chain::Step, false, &rates);
  });
  finish_timing();
}

StepStats Trainer::step(const Camera &camera, std::span<const std::uint8_t> target, const LearningRates &rates, std::span<const std::uint8_t> nextTarget) {
  // One submission for the whole iteration, and the step taken as the gradients are found.
  return run(camera, target, Chain::Step, &rates, nextTarget);
}

void Trainer::set_active_sh_degree(int degree) { activeShDegree_ = std::clamp(degree, 0, shDegree_); }

DensifyStats Trainer::densify_stats() const {
  const std::vector<float> raw = stats_->download_as<float>(ctx_);
  DensifyStats stats;
  stats.gradientSum.resize(count_);
  stats.visits.resize(count_);
  stats.maxRadius.resize(count_);
  stats.contribution.resize(count_);
  stats.absoluteGradientSum.resize(count_);
  stats.markedPixels.resize(count_);
  stats.markedLoss.resize(count_);
  stats.edgeScore.resize(count_);
  for (std::size_t i = 0; i < count_; ++i) {
    stats.gradientSum[i] = raw[i * 8];
    stats.visits[i] = raw[i * 8 + 1];
    stats.maxRadius[i] = raw[i * 8 + 2];
    stats.contribution[i] = raw[i * 8 + 3];
    stats.absoluteGradientSum[i] = raw[i * 8 + 4];
    stats.markedPixels[i] = raw[i * 8 + 5];
    stats.markedLoss[i] = raw[i * 8 + 6];
    stats.edgeScore[i] = raw[i * 8 + 7];
  }
  return stats;
}

DensifyResult Trainer::densify(DensifyOptions options, std::mt19937 &rng) {
  DensifyResult result = densify_cloud(cloud(), densify_stats(), options, rng);
  if (result.cloud.empty()) throw std::runtime_error("densification removed every Gaussian");
  apply_changes(result);
  return result;
}

DensifyResult Trainer::relocate(const RelocateOptions &options, std::mt19937 &rng) {
  DensifyResult result = relocate_cloud(cloud(), options, rng);
  if (result.cloud.empty()) throw std::runtime_error("relocation removed every Gaussian");
  apply_changes(result);
  return result;
}

void Trainer::reset_densify_stats() {
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    clear(cmd, *stats_);
    vk::compute_barrier(cmd);
  });
}

std::size_t Trainer::remove_least_contributing_gaussians(double share) {
  const DensifyResult result = remove_least_contributing(cloud(), densify_stats().contribution, share);
  apply_changes(result);
  return result.pruned;
}

std::size_t Trainer::remove_least_opaque_gaussians(double share) {
  const DensifyResult result = remove_least_opaque(cloud(), share);
  apply_changes(result);
  return result.pruned;
}

std::size_t Trainer::remove_largest_gaussians(double share) {
  const DensifyResult result = remove_largest(cloud(), share);
  apply_changes(result);
  return result.pruned;
}

void Trainer::apply_changes(const DensifyResult &result) {
  const std::size_t newCount = result.cloud.size();
  const std::vector<float> *newValues[5] = {&result.cloud.positions, &result.cloud.scales, &result.cloud.rotations,
                                            &result.cloud.opacities, &result.cloud.sh};
  const std::vector<std::uint32_t> newSh = pack_half_sh(result.cloud.sh, result.cloud.shCoeffs());

  // The survivors keep their optimiser state: gathered on the GPU through
  // this table, so that the moments never cross to the CPU. One old buffer is
  // freed before the next new one is made, for the sake of the budget.
  Buffer sourceTable(ctx_, std::max<VkDeviceSize>(newCount * sizeof(std::int32_t), 4), kStorage, MemoryUsage::Upload);
  {
    auto table = sourceTable.as<std::int32_t>();
    for (std::size_t i = 0; i < newCount; ++i) {
      table[i] = result.source[i] < 0 ? -1 : static_cast<std::int32_t>(result.source[i]);
    }
    sourceTable.flush();
  }
  for (std::size_t g = 0; g < groups_.size(); ++g) {
    const auto stride = static_cast<std::uint32_t>(newValues[g]->size() / newCount);
    const auto floats = static_cast<std::uint32_t>(newValues[g]->size());
    auto moved = std::make_unique<Group>();
    moved->state = gpu_buffer(ctx_, VkDeviceSize{floats} * sizeof(std::uint32_t));
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
      gather_.dispatch(cmd, {&sourceTable, groups_[g]->state.get(), moved->state.get()}, GatherPush{floats, stride}, vk::div_up(floats, 256u));
      vk::compute_barrier(cmd);
    });
    groups_[g].reset();
    if (g == 4) {
      moved->values = gpu_buffer(ctx_, newSh.size() * sizeof(std::uint32_t));
      moved->values->upload(ctx_, std::span<const std::uint32_t>(newSh));
    } else {
      moved->values = gpu_buffer(ctx_, VkDeviceSize{floats} * sizeof(float));
      moved->values->upload(ctx_, std::span<const float>(*newValues[g]));
    }
    groups_[g] = std::move(moved);
  }
  count_ = static_cast<std::uint32_t>(newCount);
  allocate_per_gaussian();
}

void Trainer::record_noise(VkCommandBuffer cmd, float scaler, std::uint32_t iteration) {
  vk::compute_barrier(cmd); // After Adam's writes to the positions
  noise_.dispatch(cmd, {groups_[0]->values.get(), groups_[1]->values.get(), groups_[2]->values.get(), groups_[3]->values.get()},
                  NoisePush{count_, scaler, iteration}, vk::div_up(count_, 256));
}

void Trainer::set_sh_dropout(float probability, int retainedDegree, std::uint32_t seed) {
  shDropout_ = std::clamp(probability, 0.f, 1.f);
  shRetained_ = static_cast<std::uint32_t>(std::clamp(retainedDegree, 0, 3));
  shSeed_ = seed;
}

void Trainer::set_filter3d(std::span<const float> filter) {
  if (filter.size() != count_) throw std::invalid_argument("the filter must have one value per Gaussian");
  filter3d_->upload(ctx_, filter);
}

void Trainer::add_position_noise(float scaler, std::uint32_t iteration) {
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) { record_noise(cmd, scaler, iteration); });
}

void Trainer::reset_opacity(float maximum) {
  const float limit = std::log(maximum / (1.f - maximum));
  std::vector<float> values = groups_[3]->values->download_as<float>(ctx_);
  for (float &v : values) v = std::min(v, limit);
  groups_[3]->values->upload(ctx_, std::span<const float>(values));
  ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
    clear(cmd, *groups_[3]->state);
    vk::compute_barrier(cmd);
  });
}

std::uint64_t Trainer::bytes_per_gaussian() const noexcept {
  const std::uint64_t floats = 3 + 3 + 4 + 1 + shFloats_;
  // The values, floats but for SH, and the optimiser's state, a word each. The
  // gradients stored by compute_gradients() are left out, as training never makes them.
  const std::uint64_t parameters = (floats - shFloats_ + shWords_) * sizeof(float) + floats * sizeof(std::uint32_t);
  // Sort keys and values with their copies, coverage and the 3D filter, then the backward sums and the densification stats.
  return parameters + sizeof(GpuSplat) + 4 * 6 + kRawGradientsPerSplat * sizeof(float) + 32;
}

std::uint64_t Trainer::gpu_bytes() const {
  std::uint64_t total = bytes_per_gaussian() * count_;
  for (const auto *buffer : {image_.get(), stops_.get(), exposed_.get(), adjoint_.get(), fields_.get(), targetStaging_.get(), target_.get(), ranges_.get(), entryKeys_.get(),
                             entryValues_.get(), tmpEntryKeys_.get(), tmpEntryValues_.get()}) {
    if (buffer != nullptr) total += buffer->size();
  }
  return total;
}

GaussianCloud Trainer::cloud() const {
  GaussianCloud out;
  out.shDegree = shDegree_;
  out.positions = groups_[0]->values->download_as<float>(ctx_);
  out.scales = groups_[1]->values->download_as<float>(ctx_);
  out.rotations = groups_[2]->values->download_as<float>(ctx_);
  out.opacities = groups_[3]->values->download_as<float>(ctx_);
  out.sh = unpack_half_sh(groups_[4]->values->download_as<std::uint32_t>(ctx_), out.shCoeffs());
  return out;
}

const vk::Buffer &Trainer::parameters(std::size_t group) const { return *groups_.at(group)->values; }

Trainer::Gradients Trainer::gradients() const {
  if (!groups_[0]->gradients) throw std::logic_error("no gradients: compute_gradients() stores them");
  Gradients out;
  out.positions = groups_[0]->gradients->download_as<float>(ctx_);
  out.scales = groups_[1]->gradients->download_as<float>(ctx_);
  out.rotations = groups_[2]->gradients->download_as<float>(ctx_);
  out.opacities = groups_[3]->gradients->download_as<float>(ctx_);
  out.sh = groups_[4]->gradients->download_as<float>(ctx_);
  return out;
}

} // namespace splat
