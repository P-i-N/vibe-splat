#pragma once

#include "core/radix_sort.hpp"
#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"
#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/densify.hpp"

#include <array>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splat {

struct TrainerOptions {
  float lambdaSsim = 0.2f;
  std::array<float, 3> background{0.f, 0.f, 0.f};
  float nearPlane = 0.01f;
  bool forceFixedPoint = false; // Use the path for GPUs without float atomics
  // Adam leaves Gaussians out of view alone, which is about a tenth faster.
  bool selectiveAdam = true;
  // 3DGS-MCMC's penalties on the loss: opacityReg * mean(sigmoid(opacity)) and
  // scaleReg * mean(exp(scale)) over the Gaussians. 0 is off.
  float opacityReg = 0.f;
  float scaleReg = 0.f;
  // 3DGS-MCMC's exploration: after each step, positions move by this times the
  // position rate, scaled by the covariance and gated to the nearly transparent.
  float positionNoise = 0.f;
  // Mip-Splatting's 2D filter, see ReferenceOptions::mipFilter2d.
  bool mipFilter2d = false;
  bool poseGradients = false; // Fill StepStats::poseGradient, for refining cameras
  std::uint64_t memoryBudget = ~std::uint64_t{0}; // Bytes; never allocates beyond this
};

// Thrown when a view would need more memory than the budget allows. The trainer
// is still usable: remove_largest() and try again.
class BudgetExceeded : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct LearningRates {
  float position = 1.6e-4f;
  float scale = 0.005f;
  float rotation = 0.001f;
  float opacity = 0.05f;
  float sh = 0.025f;
  float shRestScale = 0.05f; // SH above degree 0, relative to `sh`
  // Above 0, once enable_second_order() was called: position, scale, opacity
  // and SH step by curvature within this trust region, rotation still by Adam.
  float trustRegion = 0.f;
};

struct StepStats {
  float loss = 0.f;
  float l1 = 0.f;
  float ssim = 0.f;
  std::uint32_t visibleSplats = 0;
  std::uint32_t tileEntries = 0;
  // dL/d(exposure transform), as set_exposure() takes it.
  std::array<float, 12> exposureGradient{};
  // dL/d(rotation, translation) of the camera, in camera space. See refined_pose().
  std::array<float, 6> poseGradient{};
};

// rgb' = A rgb + b for a view's camera, as rows (A row, b): identity by default.
using Exposure = std::array<float, 12>;
inline constexpr Exposure kNeutralExposure{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f};

// The camera moved by a rotation (axis times angle) and then a translation, both in camera space.
[[nodiscard]] Camera refined_pose(const Camera &camera, const std::array<float, 6> &move);

// Trains Gaussians with Vulkan compute only:
//   forward   parameters -> splats -> depth sort -> per-tile lists -> composite
//   loss      L1 + D-SSIM, and its derivative at every pixel
//   backward  the derivative through compositing, down to every parameter
//   optimise  Adam
// See the README for how it fits together, and src/train/reference.cpp for
// the same maths on the CPU.
class Trainer {
public:
  Trainer(vk::Context &ctx, const GaussianCloud &initial, const TrainerOptions &options = {});
  ~Trainer();
  Trainer(const Trainer &) = delete;
  Trainer &operator=(const Trainer &) = delete;

  // Renders one view and returns width * height * 4 floats: rgb, and the log
  // of the transmittance.
  [[nodiscard]] std::vector<float> render(const Camera &camera);

  // Forward, loss and backward, leaving the gradients ready. `target` is
  // RGBA8, width * height * 4 bytes.
  StepStats compute_gradients(const Camera &camera, std::span<const std::uint8_t> target);

  // The same, and also counts for each Gaussian the pixels it reaches whose
  // error is above errorShare of this view's worst, each counting as
  // pixelWeight, for densification to use (FastGS).
  StepStats score_view(const Camera &camera, std::span<const std::uint8_t> target, float errorShare, float pixelWeight = 1.f);

  // Second-order training: makes room for the curvature, and from then on
  // update_curvature() adds an estimate to it from the view rendered last, on
  // top of what compute_gradients() or step() left. Needs float atomics.
  void enable_second_order();
  void update_curvature();

  // One Adam step from the current gradients.
  void apply_gradients(const LearningRates &rates);

  // Both of the above. A nextTarget, the image of the step after this one, is
  // copied into staging memory on a worker thread while the GPU works on this one.
  StepStats step(const Camera &camera, std::span<const std::uint8_t> target, const LearningRates &rates,
                 std::span<const std::uint8_t> nextTarget = {});

  // Starts that copy on its own. A later step() or compute_gradients() with the same span uses it.
  void prefetch_target(std::span<const std::uint8_t> target);

  [[nodiscard]] GaussianCloud cloud() const;

  struct Gradients {
    std::vector<float> positions, scales, rotations, opacities, sh;
  };
  [[nodiscard]] Gradients gradients() const;

  [[nodiscard]] bool uses_fixed_point() const noexcept { return fixedPoint_; }

  // How many SH degrees are used; the rest stay at their values. Training
  // starts low and raises this, so that view dependence does not soak up what
  // geometry should explain.
  void set_active_sh_degree(int degree);
  // How the camera of the next compute_gradients() or step() exposed its
  // photo: the loss compares the photo with the render through it.
  void set_exposure(const Exposure &exposure) { exposureValues_ = exposure; }

  // Replaces Gaussians by densify_cloud() from what has been collected, keeping
  // the optimiser state of survivors, and starts collecting anew.
  DensifyResult densify(DensifyOptions options, std::mt19937 &rng);
  // 3DGS-MCMC relocation and growth in place of densify(); see relocate_cloud().
  DensifyResult relocate(const RelocateOptions &options, std::mt19937 &rng);

  // Caps opacity, so that Gaussians no longer needed fade below the prune limit.
  void reset_opacity(float maximum);
  // One round of the position noise of TrainerOptions::positionNoise, with the given scaler (noise_lr times the position rate).
  void add_position_noise(float scaler, std::uint32_t iteration);
  // Mip-Splatting's 3D smoothing filter per Gaussian (see smoothing_filter()); zero until set, and
  // zero again after any change of the count.
  void set_filter3d(std::span<const float> filter);
  // SH dropout for the coming iterations: with `probability`, a Gaussian uses and
  // trains only the coefficients up to `retainedDegree`; drawn afresh per `seed`.
  void set_sh_dropout(float probability, int retainedDegree, std::uint32_t seed);

  // Frees memory when a view does not fit. See remove_largest().
  std::size_t remove_largest_gaussians(double share);

  [[nodiscard]] DensifyStats densify_stats() const;
  // Starts the statistics again, which otherwise run from the last change to the cloud.
  void reset_densify_stats();
  // Removes the given share of Gaussians that contributed least since the
  // statistics started. See remove_least_contributing().
  std::size_t remove_least_contributing_gaussians(double share);
  // Removes the least opaque share, see remove_least_opaque().
  std::size_t remove_least_opaque_gaussians(double share);

  // Average GPU time per iteration in milliseconds, by stage, in order.
  [[nodiscard]] std::vector<std::pair<std::string, double>> timings() const;
  [[nodiscard]] std::size_t gaussian_count() const noexcept { return count_; }
  // What the buffers take now, and what one more Gaussian would take. The tile
  // entries it brings are not included, as they depend on the view.
  [[nodiscard]] std::uint64_t gpu_bytes() const;
  // How large an image the staging memory can hold at the moment.
  [[nodiscard]] std::uint64_t staging_bytes() const noexcept { return targetStaging_ ? targetStaging_->size() : 0; }
  [[nodiscard]] std::uint64_t bytes_per_gaussian() const noexcept;
  static constexpr std::uint64_t kBytesPerTileEntry = 16;
  // Image, exposed image and adjoint 16 each, SSIM fields 48, stops 4, and
  // the target three times (device, staging, prefetch) at 4.
  static constexpr std::uint64_t kBytesPerPixel = 112;

private:
  struct Group; // A flat parameter array with its gradient and Adam state

  void allocate_per_gaussian();
  void apply_changes(const DensifyResult &result);
  void mark(VkCommandBuffer cmd, const char *stage);
  void begin_timing(VkCommandBuffer cmd);
  void finish_timing();
  void ensure_size(int width, int height);
  void upload_camera(const Camera &camera);
  std::uint32_t record_prepare();
  void record_forward(VkCommandBuffer cmd, std::uint32_t entries);
  void record_backward(VkCommandBuffer cmd);
  void record_backward_from_adjoint(VkCommandBuffer cmd);
  void record_adam(VkCommandBuffer cmd, const LearningRates &rates);
  void record_noise(VkCommandBuffer cmd, float scaler, std::uint32_t iteration);
  // Forward, loss, backward and, when rates are given, the Adam step, in one submission.
  StepStats run(const Camera &camera, std::span<const std::uint8_t> target, const LearningRates *rates,
                std::span<const std::uint8_t> nextTarget = {});

  vk::Context &ctx_;
  TrainerOptions options_;
  std::uint32_t count_ = 0;
  int shDegree_ = 0;
  std::uint32_t shFloats_ = 3;
  int activeShDegree_ = 0;
  std::uint32_t adamStep_ = 0;
  float shDropout_ = 0.f;
  std::uint32_t shRetained_ = 0, shSeed_ = 0;
  bool fixedPoint_ = false;
  int width_ = 0, height_ = 0;
  std::uint32_t gridW_ = 0, tileCount_ = 0;

  vk::ComputePipeline preprocess_, binCount_, binEmit_, tileRanges_, forward_, exposure_, ssim_, loss_, exposureBackward_, errorMask_, score_, pose_, hutchinson_, curvature_, secondOrder_, gradient_,
      adam_, noise_, gather_;
  std::uint32_t curvatureUpdates_ = 0;
  Exposure exposureValues_ = kNeutralExposure;
  bool exposing_ = false; // The last iteration ran the exposure passes
  float scoreErrorShare_ = 0.f; // Above 0: this iteration marks high-error pixels
  std::optional<vk::ComputePipeline> backward_, fixedToFloat_;
  vk::RadixSorter sorter_;

  std::vector<std::unique_ptr<Group>> groups_; // positions, scales, rotations, opacities, sh

  std::unique_ptr<vk::Buffer> camera_, counters_, fixedScale_;
  std::unique_ptr<vk::Buffer> splats_, keys_, values_, tmpKeys_, tmpValues_, coverage_, rawGradients_, stats_, filter3d_, edge_;
  std::unique_ptr<vk::Buffer> entryKeys_, entryValues_, tmpEntryKeys_, tmpEntryValues_;
  std::unique_ptr<vk::Buffer> ranges_, image_, stops_, exposed_, exposureTotals_, adjoint_, fields_, targetStaging_, target_, groupTotals_, errorMax_, poseTotals_;
  // prefetch_target() hands one copy at a time to a worker thread that lives
  // as long as the trainer, so that no thread is started per iteration.
  std::unique_ptr<vk::Buffer> targetNext_;
  const std::uint8_t *prefetched_ = nullptr;
  std::size_t prefetchedBytes_ = 0;
  std::mutex prefetchMutex_;
  std::condition_variable prefetchChanged_;
  bool prefetchPending_ = false; // A copy is queued or running
  std::uint8_t *prefetchDestination_ = nullptr;
  std::jthread prefetchWorker_;
  void wait_for_prefetch();

  VkQueryPool timestamps_ = VK_NULL_HANDLE;
  double nanosecondsPerTick_ = 1.0;
  std::vector<const char *> marks_;
  std::vector<std::pair<std::string, double>> stageTotals_;
  std::uint64_t timedIterations_ = 0;
};

} // namespace splat
