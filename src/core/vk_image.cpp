#include "core/vk_image.hpp"

#include "core/vk_buffer.hpp"
#include "core/vk_check.hpp"

#include <cstring>

namespace splat::vk {

Image::Image(const Context &ctx, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage)
    : device_(ctx.device()), allocator_(ctx.allocator()), extent_(extent), format_(format) {
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = format;
  ici.extent = {extent.width, extent.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = usage;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  check(vmaCreateImage(allocator_, &ici, &aci, &image_, &allocation_, nullptr), "vmaCreateImage");

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = image_;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = format;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  const VkResult r = vkCreateImageView(device_, &vci, nullptr, &view_);
  if (r != VK_SUCCESS) {
    vmaDestroyImage(allocator_, image_, allocation_);
    check(r, "vkCreateImageView");
  }
}

Image::~Image() {
  vkDestroyImageView(device_, view_, nullptr);
  vmaDestroyImage(allocator_, image_, allocation_);
}

std::vector<std::uint8_t> Image::read_rgba8(Context &ctx) const {
  const VkDeviceSize bytes = VkDeviceSize{extent_.width} * extent_.height * 4;
  Buffer staging(ctx, bytes, 0, MemoryUsage::Readback);
  ctx.submit_and_wait([&](VkCommandBuffer cmd) {
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {extent_.width, extent_.height, 1};
    vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.handle(), 1, &region);
  });
  staging.invalidate();
  std::vector<std::uint8_t> out(static_cast<std::size_t>(bytes));
  std::memcpy(out.data(), staging.mapped().data(), out.size());
  return out;
}

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace splat::vk
