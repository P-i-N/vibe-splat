#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>
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
  bool meetsRequirements = false; // Vulkan 1.3, graphics+compute queue, subgroup arithmetic, push descriptors
  bool floatAtomicAdd = false;    // VK_EXT_shader_atomic_float, buffer add
};

[[nodiscard]] const char *to_string(VkPhysicalDeviceType type) noexcept;
[[nodiscard]] const char *vendor_name(std::uint32_t vendorId) noexcept;

struct ContextOptions {
  bool validation = false;                   // Enable Khronos validation layer if installed
  std::optional<std::uint32_t> deviceIndex;  // Force a device; otherwise best is picked

  // For on-screen rendering: extra instance extensions (from the windowing
  // library) and a callback that creates the window surface once the instance
  // exists. The chosen device must then be able to present to that surface,
  // and VK_KHR_swapchain is enabled. The Context owns the surface.
  std::vector<const char *> instanceExtensions;
  std::function<VkSurfaceKHR(VkInstance)> createSurface;
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
  [[nodiscard]] VkSurfaceKHR surface() const noexcept { return surface_; }
  // The one queue does graphics, compute and (with a surface) present.
  [[nodiscard]] VkQueue queue() const noexcept { return computeQueue_; }
  [[nodiscard]] VkCommandPool command_pool() const noexcept { return commandPool_; }
  [[nodiscard]] const DeviceInfo &info() const noexcept { return info_; }

  // Device-local memory this process may still allocate. Accurate if the driver
  // has VK_EXT_memory_budget, which accounts for other programs; otherwise 80%
  // of the largest heap, minus nothing, as there is no way to know more.
  [[nodiscard]] std::uint64_t free_device_memory() const;

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
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue computeQueue_ = VK_NULL_HANDLE;
  std::uint32_t computeFamily_ = 0;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  DeviceInfo info_;
  bool hasMemoryBudget_ = false;
};

} // namespace splat::vk
