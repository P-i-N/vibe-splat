#include "core/vk_buffer.hpp"

#include "core/vk_check.hpp"

#include <utility>

namespace splat::vk {

Buffer::Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, MemoryUsage memory)
    : allocator_(ctx.allocator()), size_(size) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;
  switch (memory) {
  case MemoryUsage::GpuOnly:
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    break;
  case MemoryUsage::Upload:
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    break;
  case MemoryUsage::Readback:
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    break;
  }

  VmaAllocationInfo info{};
  check(vmaCreateBuffer(allocator_, &bci, &aci, &buffer_, &allocation_, &info), "vmaCreateBuffer");
  mapped_ = info.pMappedData;
}

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer &&other) noexcept
    : allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      mapped_(std::exchange(other.mapped_, nullptr)), size_(std::exchange(other.size_, 0)) {}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    mapped_ = std::exchange(other.mapped_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

std::span<std::byte> Buffer::mapped() const noexcept {
  if (mapped_ == nullptr) {
    return {};
  }
  return {static_cast<std::byte *>(mapped_), static_cast<std::size_t>(size_)};
}

void Buffer::flush() const {
  check(vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
}

void Buffer::invalidate() const {
  check(vmaInvalidateAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation");
}

void Buffer::upload(Context &ctx, std::span<const std::byte> data, VkDeviceSize offset) const {
  if (offset + data.size() > size_) {
    throw Error("Buffer::upload: data does not fit");
  }
  if (data.empty()) {
    return;
  }
  if (mapped_ != nullptr) {
    std::memcpy(static_cast<std::byte *>(mapped_) + offset, data.data(), data.size());
    flush();
    return;
  }
  Buffer staging(ctx, data.size(), 0, MemoryUsage::Upload);
  std::memcpy(staging.mapped().data(), data.data(), data.size());
  staging.flush();
  ctx.submit_and_wait([&](VkCommandBuffer cmd) {
    const VkBufferCopy region{0, offset, data.size()};
    vkCmdCopyBuffer(cmd, staging.handle(), buffer_, 1, &region);
  });
}

std::vector<std::byte> Buffer::download(Context &ctx) const {
  std::vector<std::byte> out(static_cast<std::size_t>(size_));
  if (mapped_ != nullptr) {
    invalidate();
    std::memcpy(out.data(), mapped_, out.size());
    return out;
  }
  Buffer staging(ctx, size_, 0, MemoryUsage::Readback);
  ctx.submit_and_wait([&](VkCommandBuffer cmd) {
    const VkBufferCopy region{0, 0, size_};
    vkCmdCopyBuffer(cmd, buffer_, staging.handle(), 1, &region);
  });
  staging.invalidate();
  std::memcpy(out.data(), staging.mapped().data(), out.size());
  return out;
}

void Buffer::reset() noexcept {
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
  size_ = 0;
}

} // namespace splat::vk
