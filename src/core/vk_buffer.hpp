#pragma once

#include "core/vk_context.hpp"

#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

namespace splat::vk {

enum class MemoryUsage {
  GpuOnly,  // Device-local, not mappable. Training state lives here.
  Upload,   // Host-visible, written sequentially by the CPU.
  Readback, // Host-visible and cached, read by the CPU.
};

// A VMA-backed VkBuffer. Move-only; destroyed with its owning Context alive.
// Every buffer also gets TRANSFER_SRC/DST usage so it can be filled, cleared
// and read back.
class Buffer {
public:
  Buffer() = default;
  Buffer(const Context &ctx, VkDeviceSize size, VkBufferUsageFlags usage, MemoryUsage memory);
  ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
  [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }

  // Only valid for Upload / Readback buffers.
  [[nodiscard]] std::span<std::byte> mapped() const noexcept;

  template <class T> [[nodiscard]] std::span<T> as() const noexcept {
    auto bytes = mapped();
    return {reinterpret_cast<T *>(bytes.data()), bytes.size() / sizeof(T)};
  }

  // Makes CPU writes visible to the GPU, and GPU writes visible to the CPU,
  // on memory that is not host-coherent. No-op otherwise.
  void flush() const;
  void invalidate() const;

  // Copies bytes into a buffer of any kind (uses a staging buffer when the
  // destination is not mappable) and waits for completion.
  void upload(Context &ctx, std::span<const std::byte> data, VkDeviceSize offset = 0) const;
  // Reads the whole buffer back to the CPU and waits for completion.
  [[nodiscard]] std::vector<std::byte> download(Context &ctx) const;

  template <class T> void upload(Context &ctx, std::span<const T> data) const {
    upload(ctx, std::as_bytes(data));
  }
  template <class T> [[nodiscard]] std::vector<T> download_as(Context &ctx) const {
    const auto bytes = download(ctx);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
  }

private:
  void reset() noexcept;

  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  void *mapped_ = nullptr;
  VkDeviceSize size_ = 0;
};

} // namespace splat::vk
