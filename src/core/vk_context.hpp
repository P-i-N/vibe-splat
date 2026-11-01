#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace splat::vk {

// What we know about one Vulkan physical device, used by `splat info` and for
// picking a device.
struct DeviceInfo {
  std::uint32_t index = 0;
  std::string name;
  std::string driver;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  std::uint32_t vendorId = 0;
  std::uint32_t apiVersion = 0;
  std::uint64_t deviceLocalBytes = 0;
  std::uint32_t subgroupSize = 0;
  bool hasComputeQueue = false;
  bool meetsRequirements = false; // Vulkan 1.3 + compute + subgroup arithmetic
  bool floatAtomicAdd = false;    // VK_EXT_shader_atomic_float, buffer add
};

[[nodiscard]] const char *to_string(VkPhysicalDeviceType type) noexcept;
[[nodiscard]] const char *vendor_name(std::uint32_t vendorId) noexcept;

struct ContextOptions {
  bool validation = false;                   // Enable Khronos validation layer if installed
  std::optional<std::uint32_t> deviceIndex;  // Force a device; otherwise best is picked
};

// Owns the Vulkan instance, the chosen device, its compute queue and the VMA
// allocator. Move-only. Everything GPU-side in splat hangs off one Context.
class Context {
public:
  explicit Context(const ContextOptions &options = {});
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other) noexcept;
  Context &operator=(Context &&other) noexcept;

  // Lists devices without creating a full context.
  [[nodiscard]] static std::vector<DeviceInfo> enumerate_devices();

  [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
  [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_; }
  [[nodiscard]] VkDevice device() const noexcept { return device_; }
  [[nodiscard]] VkQueue compute_queue() const noexcept { return computeQueue_; }
  [[nodiscard]] std::uint32_t compute_family() const noexcept { return computeFamily_; }
  [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }
  [[nodiscard]] const DeviceInfo &info() const noexcept { return info_; }

  // Records commands via `record`, submits them on the compute queue and
  // waits for completion. Fine for setup work and tests; the trainer will use
  // its own frame-based submission.
  template <class F> void submit_and_wait(F &&record) {
    VkCommandBuffer cmd = begin_one_shot();
    record(cmd);
    end_one_shot(cmd);
  }

private:
  VkCommandBuffer begin_one_shot();
  void end_one_shot(VkCommandBuffer cmd);
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue computeQueue_ = VK_NULL_HANDLE;
  std::uint32_t computeFamily_ = 0;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  DeviceInfo info_;
};

} // namespace splat::vk
