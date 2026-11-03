#pragma once

#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"
#include "sfm/features.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace splat {

// Brute-force feature matching on the GPU: the same mutual nearest neighbours
// under Lowe's ratio test as match_features, for many image pairs at once.
// Descriptors are uploaded once, quantised to 8 bits (RootSIFT values lie in
// [0, 1]), so the scores differ slightly from the float version.
class GpuMatcher {
public:
  // Throws std::runtime_error if an image has more than 65535 keypoints or
  // the descriptors do not fit in device memory.
  GpuMatcher(vk::Context &ctx, const std::vector<ImageFeatures> &features, float ratio = 0.8f);

  // Matches each (a, b) pair of image indices. Each list is sorted by the
  // feature index in a, and so is the same from run to run.
  [[nodiscard]] std::vector<std::vector<std::pair<int, int>>> match(std::span<const std::pair<int, int>> pairs);

  // Pairs handled in one submission. Longer lists go in several.
  static constexpr std::size_t kBatch = 256;

  // Bytes of device memory the descriptors of these features take.
  [[nodiscard]] static std::uint64_t descriptor_bytes(const std::vector<ImageFeatures> &features);

private:
  vk::Context &ctx_;
  float ratio_;
  std::uint32_t stride_ = 0; // The most keypoints any image has
  std::vector<std::uint32_t> offsets_;
  std::vector<std::uint32_t> counts_;
  vk::Buffer descriptors_;
  vk::Buffer pairs_;
  vk::Buffer bestA_;
  vk::Buffer bestB_;
  vk::Buffer matchCounts_;
  vk::Buffer matches_;
  vk::Buffer readback_;
  vk::ComputePipeline score_;
  vk::ComputePipeline compact_;
};

} // namespace splat
