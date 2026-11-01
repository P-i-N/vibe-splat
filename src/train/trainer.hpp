#pragma once

#include "core/radix_sort.hpp"
#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"
#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace splat {

struct TrainerOptions {
  float lambdaSsim = 0.2f;
  std::array<float, 3> background{0.f, 0.f, 0.f};
  float nearPlane = 0.01f;
  bool forceFixedPoint = false; // Use the path for GPUs without float atomics
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

  struct Gradients {
    std::vector<float> positions, scales, rotations, opacities, sh;
  };
  [[nodiscard]] Gradients gradients() const;

  [[nodiscard]] bool uses_fixed_point() const noexcept { return fixedPoint_; }

private:
  struct Group; // A flat parameter array with its gradient and Adam state

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
  std::uint32_t adamStep_ = 0;
  bool fixedPoint_ = false;
  int width_ = 0, height_ = 0;
  std::uint32_t gridW_ = 0, tileCount_ = 0;

  vk::ComputePipeline preprocess_, binCount_, binEmit_, tileRanges_, forward_, ssim_, loss_, gradient_, adam_;
  std::optional<vk::ComputePipeline> backward_, fixedToFloat_;
  vk::RadixSorter sorter_;

  std::vector<std::unique_ptr<Group>> groups_; // positions, scales, rotations, opacities, sh

  std::unique_ptr<vk::Buffer> camera_, counters_, fixedScale_;
  std::unique_ptr<vk::Buffer> splats_, keys_, values_, tmpKeys_, tmpValues_, coverage_, rawGradients_;
  std::unique_ptr<vk::Buffer> entryKeys_, entryValues_, tmpEntryKeys_, tmpEntryValues_;
  std::unique_ptr<vk::Buffer> ranges_, image_, adjoint_, fields_, target_, rowTotals_;
};

} // namespace splat
