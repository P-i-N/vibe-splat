#pragma once

#include "core/vk_context.hpp"

#include <cstdint>
#include <vector>

namespace splat::vk {

// A 2D colour image with one mip level and a view, backed by VMA. Used as an
// offscreen render target for headless rendering and tests.
class Image {
public:
  Image(const Context &ctx, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage);
  ~Image();
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;

  [[nodiscard]] VkImage handle() const noexcept { return image_; }
  [[nodiscard]] VkImageView view() const noexcept { return view_; }
  [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
  [[nodiscard]] VkFormat format() const noexcept { return format_; }

  // Reads back an RGBA8 image (must be in TRANSFER_SRC_OPTIMAL layout).
  [[nodiscard]] std::vector<std::uint8_t> read_rgba8(Context &ctx) const;

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkImage image_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkExtent2D extent_{};
  VkFormat format_ = VK_FORMAT_UNDEFINED;
};

// Records a full-image layout transition with conservative stage/access masks.
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to);

} // namespace splat::vk
