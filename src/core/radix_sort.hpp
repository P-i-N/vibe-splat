#pragma once

#include "core/vk_buffer.hpp"
#include "core/vk_compute.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace splat::vk {

// Stable GPU LSD radix sort of 32-bit keys with 32-bit values, 8 bits per
// pass. Each pass: per-tile histogram, multi-level exclusive scan of the
// histograms, then a stable scatter.
class RadixSorter {
public:
  explicit RadixSorter(Context &ctx);

  // Records the sort. On return `keys`/`values` hold the sorted data;
  // `tmpKeys`/`tmpValues` are scratch buffers at least as large. Only the low
  // `keyBits` bits of each key are compared (rounded up to a multiple of 8).
  // Scratch memory grows here, so do not record while a previous submission
  // using this sorter is still in flight. Several sorts and scans may go into
  // one command buffer: scratch that grows is kept, not destroyed.
  void record(VkCommandBuffer cmd, const Buffer &keys, const Buffer &values, const Buffer &tmpKeys,
              const Buffer &tmpValues, std::uint32_t count, std::uint32_t keyBits = 32);

  // Records an in-place exclusive scan (prefix sum) of `data[0..count)`.
  void record_scan(VkCommandBuffer cmd, const Buffer &data, std::uint32_t count) { record_scan(cmd, data, count, 0); }

  static constexpr std::uint32_t kTile = 4096;
  static constexpr std::uint32_t kScanBlock = 1024;

private:
  // Records an exclusive scan of `data[0..count)` using scratch level `level`.
  void record_scan(VkCommandBuffer cmd, const Buffer &data, std::uint32_t count, std::size_t level);
  const Buffer &scan_level(std::size_t level, std::uint32_t blocks);
  // `buffer`, grown to at least `bytes` if smaller.
  const Buffer &scratch(std::unique_ptr<Buffer> &buffer, VkDeviceSize bytes);

  Context &ctx_;
  ComputePipeline histogram_;
  ComputePipeline scatter_;
  ComputePipeline scanBlocks_;
  ComputePipeline scanAdd_;
  std::unique_ptr<Buffer> hist_;
  std::vector<std::unique_ptr<Buffer>> scanSums_;
  std::vector<std::unique_ptr<Buffer>> retired_; // Outgrown scratch, see scratch()
};

} // namespace splat::vk
