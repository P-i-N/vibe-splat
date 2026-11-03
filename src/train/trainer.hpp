#pragma once

#include "core/radix_sort.hpp"
#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"
#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/densify.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
};

struct StepStats {
  float loss = 0.f;
  float l1 = 0.f;
  float ssim = 0.f;
  std::uint32_t visibleSplats = 0;
  std::uint32_t tileEntries = 0;
};

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

  // One Adam step from the current gradients.
  void apply_gradients(const LearningRates &rates);

  // Both of the above.
  StepStats step(const Camera &camera, std::span<const std::uint8_t> target, const LearningRates &rates);

  [[nodiscard]] GaussianCloud cloud() const;

  // The raw parameters on the GPU, laid out as in GaussianCloud: 0 positions,
  // 1 scales, 2 rotations, 3 opacities, 4 SH. Valid until the Gaussians change
  // in number (densify, remove_largest_gaussians).
  [[nodiscard]] const vk::Buffer &parameters(std::size_t group) const;
  [[nodiscard]] int sh_degree() const noexcept { return shDegree_; }
  [[nodiscard]] int active_sh_degree() const noexcept { return activeShDegree_; }

  struct Gradients {
    std::vector<float> positions, scales, rotations, opacities, sh;
  };
  [[nodiscard]] Gradients gradients() const;

  [[nodiscard]] bool uses_fixed_point() const noexcept { return fixedPoint_; }

  // How many SH degrees are used; the rest stay at their values. Training
  // starts low and raises this, so that view dependence does not soak up what
  // geometry should explain.
  void set_active_sh_degree(int degree);

  // Replaces Gaussians by densify_cloud() from what has been collected, keeping
  // the optimiser state of survivors, and starts collecting anew.
  DensifyResult densify(DensifyOptions options, std::mt19937 &rng);

  // Caps opacity, so that Gaussians no longer needed fade below the prune limit.
  void reset_opacity(float maximum);

  // Frees memory when a view does not fit. See remove_largest().
  std::size_t remove_largest_gaussians(double share);

  [[nodiscard]] DensifyStats densify_stats() const;

  // Average GPU time per iteration in milliseconds, by stage, in order.
  [[nodiscard]] std::vector<std::pair<std::string, double>> timings() const;
  [[nodiscard]] std::size_t gaussian_count() const noexcept { return count_; }
  // What the buffers take now, and what one more Gaussian would take. The tile
  // entries it brings are not included, as they depend on the view.
  [[nodiscard]] std::uint64_t gpu_bytes() const;
  [[nodiscard]] std::uint64_t bytes_per_gaussian() const noexcept;
  static constexpr std::uint64_t kBytesPerTileEntry = 16;
  static constexpr std::uint64_t kBytesPerPixel = 180;

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

  vk::Context &ctx_;
  TrainerOptions options_;
  std::uint32_t count_ = 0;
  int shDegree_ = 0;
  std::uint32_t shFloats_ = 3;
  int activeShDegree_ = 0;
  std::uint32_t adamStep_ = 0;
  bool fixedPoint_ = false;
  int width_ = 0, height_ = 0;
  std::uint32_t gridW_ = 0, tileCount_ = 0;

  vk::ComputePipeline preprocess_, binCount_, binEmit_, tileRanges_, forward_, ssimH_, ssim_, lossH_, loss_, gradient_, adam_;
  std::optional<vk::ComputePipeline> backward_, fixedToFloat_;
  vk::RadixSorter sorter_;

  std::vector<std::unique_ptr<Group>> groups_; // positions, scales, rotations, opacities, sh

  std::unique_ptr<vk::Buffer> camera_, counters_, fixedScale_;
  std::unique_ptr<vk::Buffer> splats_, keys_, values_, tmpKeys_, tmpValues_, coverage_, rawGradients_, stats_;
  std::unique_ptr<vk::Buffer> entryKeys_, entryValues_, tmpEntryKeys_, tmpEntryValues_;
  std::unique_ptr<vk::Buffer> ranges_, image_, adjoint_, fields_, moments_, behind_, target_, rowTotals_;

  VkQueryPool timestamps_ = VK_NULL_HANDLE;
  double nanosecondsPerTick_ = 1.0;
  std::vector<const char *> marks_;
  std::vector<std::pair<std::string, double>> stageTotals_;
  std::uint64_t timedIterations_ = 0;
};

} // namespace splat
