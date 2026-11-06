#include "sfm/gpu_matcher.hpp"

#include "splat_shaders/shaders.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace splat {

namespace {

constexpr std::uint32_t kMaxKeypoints = 0xFFFF; // The index shares a word with the score
constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

struct PairJob {
  std::uint32_t offsetA;
  std::uint32_t offsetB;
  std::uint32_t countA;
  std::uint32_t countB;
};

struct ScorePush {
  std::uint32_t stride;
};

struct CompactPush {
  std::uint32_t stride;
  float ratioSquared;
};

} // namespace

std::uint64_t GpuMatcher::descriptor_bytes(const std::vector<ImageFeatures> &features) {
  std::uint64_t total = 0;
  for (const ImageFeatures &f : features) total += f.keypoints.size();
  return total * kDescriptorSize;
}

GpuMatcher::GpuMatcher(vk::Context &ctx, const std::vector<ImageFeatures> &features, float ratio)
    : ctx_(ctx), ratio_(ratio), score_(ctx, shaders::match_score_comp(), 4, sizeof(ScorePush)),
      compact_(ctx, shaders::match_compact_comp(), 5, sizeof(CompactPush)) {
  std::uint32_t total = 0;
  for (const ImageFeatures &f : features) {
    if (f.keypoints.size() > kMaxKeypoints) {
      throw std::runtime_error(std::format("GPU matching handles at most {} features per image", kMaxKeypoints));
    }
    offsets_.push_back(total);
    counts_.push_back(static_cast<std::uint32_t>(f.keypoints.size()));
    total += counts_.back();
    stride_ = std::max(stride_, counts_.back());
  }
  stride_ = std::max(stride_, 1u);

  const std::uint64_t bytes = std::max<std::uint64_t>(descriptor_bytes(features), 4);
  const std::uint64_t scratch = std::uint64_t{kBatch} * stride_ * (16 + 4 + 8) * 2;
  const std::uint64_t free = ctx.free_device_memory();
  if (bytes + scratch > free) {
    throw std::runtime_error(std::format("GPU matching needs {} MB of device memory, {} MB free", (bytes + scratch) >> 20, free >> 20));
  }

  // Quantise to bytes, round(255 d), straight into upload memory.
  {
    vk::Buffer staging(ctx, bytes, 0, vk::MemoryUsage::Upload);
    auto out = staging.as<std::uint8_t>();
    std::size_t at = 0;
    for (const ImageFeatures &f : features) {
      for (const float d : f.descriptors) {
        out[at++] = static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(d * 255.f)), 0, 255));
      }
    }
    staging.flush();
    descriptors_ = vk::Buffer(ctx, bytes, kStorage, vk::MemoryUsage::GpuOnly);
    ctx.submit_and_wait([&](VkCommandBuffer cmd) {
      const VkBufferCopy region{0, 0, bytes};
      vkCmdCopyBuffer(cmd, staging.handle(), descriptors_.handle(), 1, &region);
      vk::compute_barrier(cmd);
    });
  }

  const VkDeviceSize rows = VkDeviceSize{kBatch} * stride_;
  pairs_ = vk::Buffer(ctx, kBatch * sizeof(PairJob), kStorage, vk::MemoryUsage::Upload);
  bestA_ = vk::Buffer(ctx, rows * 16, kStorage, vk::MemoryUsage::GpuOnly);
  bestB_ = vk::Buffer(ctx, rows * 4, kStorage, vk::MemoryUsage::GpuOnly);
  matchCounts_ = vk::Buffer(ctx, kBatch * 4, kStorage, vk::MemoryUsage::GpuOnly);
  matches_ = vk::Buffer(ctx, rows * 8, kStorage, vk::MemoryUsage::GpuOnly);
  readback_ = vk::Buffer(ctx, rows * 8 + kBatch * 4, 0, vk::MemoryUsage::Readback);
}

std::vector<std::vector<std::pair<int, int>>> GpuMatcher::match(std::span<const std::pair<int, int>> pairs) {
  std::vector<std::vector<std::pair<int, int>>> result(pairs.size());
  for (std::size_t start = 0; start < pairs.size(); start += kBatch) {
    const auto batch = static_cast<std::uint32_t>(std::min(kBatch, pairs.size() - start));
    auto jobs = pairs_.as<PairJob>();
    std::uint32_t widest = 1;
    for (std::uint32_t p = 0; p < batch; ++p) {
      const auto [a, b] = pairs[start + p];
      jobs[p] = {offsets_[static_cast<std::size_t>(a)], offsets_[static_cast<std::size_t>(b)], counts_[static_cast<std::size_t>(a)],
                 counts_[static_cast<std::size_t>(b)]};
      widest = std::max(widest, jobs[p].countA);
    }
    pairs_.flush();

    const VkDeviceSize matchBytes = VkDeviceSize{batch} * stride_ * 8;
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
      vkCmdFillBuffer(cmd, bestB_.handle(), 0, VkDeviceSize{batch} * stride_ * 4, 0);
      vkCmdFillBuffer(cmd, matchCounts_.handle(), 0, VkDeviceSize{batch} * 4, 0);
      vk::compute_barrier(cmd);
      score_.dispatch(cmd, {&descriptors_, &pairs_, &bestA_, &bestB_}, ScorePush{stride_}, vk::div_up(widest, 64), batch);
      vk::compute_barrier(cmd);
      compact_.dispatch(cmd, {&pairs_, &bestA_, &bestB_, &matchCounts_, &matches_}, CompactPush{stride_, ratio_ * ratio_},
                        vk::div_up(widest, 256), batch);
      vk::compute_barrier(cmd);
      const VkBufferCopy matchRegion{0, 0, matchBytes};
      const VkBufferCopy countRegion{0, matchBytes, VkDeviceSize{batch} * 4};
      vkCmdCopyBuffer(cmd, matches_.handle(), readback_.handle(), 1, &matchRegion);
      vkCmdCopyBuffer(cmd, matchCounts_.handle(), readback_.handle(), 1, &countRegion);
      vk::compute_barrier(cmd);
    });
    readback_.invalidate();

    const auto bytes = readback_.mapped();
    const auto *entries = reinterpret_cast<const std::uint32_t *>(bytes.data());
    const auto *counts = reinterpret_cast<const std::uint32_t *>(bytes.data() + matchBytes);
    for (std::uint32_t p = 0; p < batch; ++p) {
      std::vector<std::pair<int, int>> &list = result[start + p];
      const std::uint32_t n = std::min(counts[p], jobs[p].countA);
      list.reserve(n);
      const std::uint32_t *row = entries + std::size_t{p} * stride_ * 2;
      for (std::uint32_t k = 0; k < n; ++k) list.emplace_back(static_cast<int>(row[k * 2]), static_cast<int>(row[k * 2 + 1]));
      std::sort(list.begin(), list.end());
    }
  }
  return result;
}

} // namespace splat
