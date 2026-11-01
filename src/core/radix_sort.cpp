#include "core/radix_sort.hpp"

#include "splat_shaders/shaders.hpp"

#include <utility>

namespace splat::vk {
namespace {

struct RadixPush {
  std::uint32_t count;
  std::uint32_t shift;
  std::uint32_t numTiles;
};

struct ScanPush {
  std::uint32_t count;
};

constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

void copy_buffer(VkCommandBuffer cmd, const Buffer &src, const Buffer &dst, VkDeviceSize bytes) {
  const VkBufferCopy region{0, 0, bytes};
  vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
}

} // namespace

RadixSorter::RadixSorter(Context &ctx)
    : ctx_(ctx), histogram_(ctx, shaders::radix_histogram_comp, 2, sizeof(RadixPush)),
      scatter_(ctx, shaders::radix_scatter_comp, 5, sizeof(RadixPush)),
      scanBlocks_(ctx, shaders::scan_blocks_comp, 2, sizeof(ScanPush)),
      scanAdd_(ctx, shaders::scan_add_comp, 2, sizeof(ScanPush)) {}

const Buffer &RadixSorter::scan_level(std::size_t level, std::uint32_t blocks) {
  if (scanSums_.size() <= level) {
    scanSums_.resize(level + 1);
  }
  const VkDeviceSize bytes = VkDeviceSize{blocks} * sizeof(std::uint32_t);
  if (!scanSums_[level] || scanSums_[level]->size() < bytes) {
    scanSums_[level] = std::make_unique<Buffer>(ctx_, bytes, kStorage, MemoryUsage::GpuOnly);
  }
  return *scanSums_[level];
}

void RadixSorter::record_scan(VkCommandBuffer cmd, const Buffer &data, std::uint32_t count, std::size_t level) {
  const std::uint32_t blocks = div_up(count, kScanBlock);
  const Buffer &sums = scan_level(level, blocks);
  scanBlocks_.dispatch(cmd, {&data, &sums}, ScanPush{count}, blocks);
  compute_barrier(cmd);
  if (blocks > 1) {
    record_scan(cmd, sums, blocks, level + 1);
    scanAdd_.dispatch(cmd, {&data, &sums}, ScanPush{count}, blocks);
    compute_barrier(cmd);
  }
}

void RadixSorter::record(VkCommandBuffer cmd, const Buffer &keys, const Buffer &values, const Buffer &tmpKeys,
                         const Buffer &tmpValues, std::uint32_t count, std::uint32_t keyBits) {
  if (count <= 1) {
    return;
  }
  const std::uint32_t numTiles = div_up(count, kTile);
  const VkDeviceSize histBytes = VkDeviceSize{numTiles} * 256 * sizeof(std::uint32_t);
  if (!hist_ || hist_->size() < histBytes) {
    hist_ = std::make_unique<Buffer>(ctx_, histBytes, kStorage, MemoryUsage::GpuOnly);
  }

  const Buffer *srcK = &keys, *srcV = &values, *dstK = &tmpKeys, *dstV = &tmpValues;
  const std::uint32_t passes = div_up(std::min(keyBits, 32u), 8u);
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    const RadixPush push{count, pass * 8, numTiles};
    histogram_.dispatch(cmd, {srcK, hist_.get()}, push, numTiles);
    compute_barrier(cmd);
    record_scan(cmd, *hist_, numTiles * 256, 0);
    scatter_.dispatch(cmd, {srcK, srcV, dstK, dstV, hist_.get()}, push, numTiles);
    compute_barrier(cmd);
    std::swap(srcK, dstK);
    std::swap(srcV, dstV);
  }
  if (srcK != &keys) {
    const VkDeviceSize bytes = VkDeviceSize{count} * sizeof(std::uint32_t);
    copy_buffer(cmd, *srcK, keys, bytes);
    copy_buffer(cmd, *srcV, values, bytes);
    compute_barrier(cmd);
  }
}

} // namespace splat::vk
