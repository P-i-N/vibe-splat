#include "core/vk_context.hpp"

#include "core/vk_check.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <format>
#include <mutex>
#include <utility>

namespace splat::vk {
namespace {

constexpr std::uint32_t kRequiredApi = VK_API_VERSION_1_3;
constexpr const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";

void init_loader() {
  static std::once_flag once;
  static VkResult result = VK_SUCCESS;
  std::call_once(once, [] { result = volkInitialize(); });
  if (result != VK_SUCCESS) {
    throw Error("No Vulkan loader found. Install or update your GPU driver.", result);
  }
}

bool has_extension(VkPhysicalDevice pd, const char *name) {
  std::uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> exts(count);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data());
  return std::ranges::any_of(exts, [&](const VkExtensionProperties &e) {
    return std::strcmp(e.extensionName, name) == 0;
  });
}

bool has_layer(const char *name) {
  std::uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());
  return std::ranges::any_of(layers, [&](const VkLayerProperties &l) {
    return std::strcmp(l.layerName, name) == 0;
  });
}

std::optional<std::uint32_t> find_compute_family(VkPhysicalDevice pd, VkSurfaceKHR surface = VK_NULL_HANDLE) {
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
  // One queue family that does graphics + compute (and present, when a
  // surface is given): the viewer, trainer and live preview all share it.
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto flags = families[i].queueFlags;
    if ((flags & VK_QUEUE_COMPUTE_BIT) == 0 || (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
      continue;
    }
    if (surface != VK_NULL_HANDLE) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
      if (present == VK_FALSE) {
        continue;
      }
    }
    return i;
  }
  return std::nullopt;
}

DeviceInfo describe(VkPhysicalDevice pd, std::uint32_t index, VkSurfaceKHR surface = VK_NULL_HANDLE) {
  VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  v11.pNext = &driver;
  VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props.pNext = &v11;
  vkGetPhysicalDeviceProperties2(pd, &props);

  DeviceInfo info;
  info.index = index;
  info.name = props.properties.deviceName;
  info.driver = std::format("{} {}", driver.driverName, driver.driverInfo);
  info.type = props.properties.deviceType;
  info.vendorId = props.properties.vendorID;
  info.apiVersion = props.properties.apiVersion;
  info.subgroupSize = v11.subgroupSize;

  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(pd, &mem);
  for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      info.deviceLocalBytes = std::max<std::uint64_t>(info.deviceLocalBytes, mem.memoryHeaps[i].size);
    }
  }

  info.hasComputeQueue = find_compute_family(pd, surface).has_value();

  if (has_extension(pd, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &af;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    info.floatAtomicAdd = af.shaderBufferFloat32AtomicAdd == VK_TRUE;
  }

  const bool subgroupArith =
      (v11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      (v11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  const bool pushDescriptor = has_extension(pd, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  const bool swapchain = surface == VK_NULL_HANDLE || has_extension(pd, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  info.meetsRequirements =
      info.apiVersion >= kRequiredApi && info.hasComputeQueue && subgroupArith && pushDescriptor && swapchain;
  return info;
}

int type_rank(VkPhysicalDeviceType type) {
  switch (type) {
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
  case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
  default: return 0;
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT,
                                              const VkDebugUtilsMessengerCallbackDataEXT *data,
                                              void *) {
  const char *level = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning";
  std::fprintf(stderr, "[vulkan %s] %s\n", level, data->pMessage);
  return VK_FALSE;
}

VkInstance create_instance(bool validation, VkDebugUtilsMessengerEXT *messenger,
                           const std::vector<const char *> &extraExtensions = {}) {
  init_loader();

  std::uint32_t loaderVersion = VK_API_VERSION_1_0;
  vkEnumerateInstanceVersion(&loaderVersion);
  if (loaderVersion < kRequiredApi) {
    throw Error("The installed Vulkan loader is older than 1.3. Update your GPU driver.");
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "splat";
  app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app.pEngineName = "splat";
  app.apiVersion = kRequiredApi;

  std::vector<const char *> layers;
  std::vector<const char *> extensions = extraExtensions;
  const bool useValidation = validation && has_layer(kValidationLayer);
  if (validation && !useValidation) {
    std::fprintf(stderr, "warning: %s not installed, continuing without validation\n", kValidationLayer);
  }
  if (useValidation) {
    layers.push_back(kValidationLayer);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app;
  ci.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
  ci.ppEnabledLayerNames = layers.data();
  ci.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
  ci.ppEnabledExtensionNames = extensions.data();

  VkInstance instance = VK_NULL_HANDLE;
  check(vkCreateInstance(&ci, nullptr, &instance), "vkCreateInstance");
  volkLoadInstanceOnly(instance);

  if (useValidation && messenger != nullptr) {
    VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mci.pfnUserCallback = debug_callback;
    check(vkCreateDebugUtilsMessengerEXT(instance, &mci, nullptr, messenger),
          "vkCreateDebugUtilsMessengerEXT");
  }
  return instance;
}

std::vector<VkPhysicalDevice> physical_devices(VkInstance instance) {
  std::uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
  std::vector<VkPhysicalDevice> devices(count);
  check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
  return devices;
}

} // namespace

const char *to_string(VkPhysicalDeviceType type) noexcept {
  switch (type) {
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
  case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU (software)";
  default: return "other";
  }
}

const char *vendor_name(std::uint32_t vendorId) noexcept {
  switch (vendorId) {
  case 0x1002: return "AMD";
  case 0x10DE: return "NVIDIA";
  case 0x8086: return "Intel";
  case 0x13B5: return "ARM";
  case 0x5143: return "Qualcomm";
  case 0x10005: return "Mesa";
  default: return "unknown";
  }
}

std::vector<DeviceInfo> Context::enumerate_devices() {
  VkInstance instance = create_instance(false, nullptr);
  std::vector<DeviceInfo> result;
  try {
    const auto devices = physical_devices(instance);
    for (std::uint32_t i = 0; i < devices.size(); ++i) {
      result.push_back(describe(devices[i], i));
    }
  } catch (...) {
    vkDestroyInstance(instance, nullptr);
    throw;
  }
  vkDestroyInstance(instance, nullptr);
  return result;
}

Context::Context(const ContextOptions &options) {
  try {
    instance_ = create_instance(options.validation, &messenger_, options.instanceExtensions);
    if (options.createSurface) {
      surface_ = options.createSurface(instance_);
      if (surface_ == VK_NULL_HANDLE) {
        throw Error("Could not create a window surface.");
      }
    }

    const auto devices = physical_devices(instance_);
    if (devices.empty()) {
      throw Error("No Vulkan devices found.");
    }

    if (options.deviceIndex) {
      if (*options.deviceIndex >= devices.size()) {
        throw Error(std::format("Device {} does not exist ({} found).", *options.deviceIndex, devices.size()));
      }
      info_ = describe(devices[*options.deviceIndex], *options.deviceIndex, surface_);
      if (!info_.meetsRequirements) {
        throw Error(std::format("Device {} ({}) lacks Vulkan 1.3, a graphics+compute queue, subgroup arithmetic, "
                                "push descriptors or presentation support.",
                                info_.index, info_.name));
      }
    } else {
      std::optional<DeviceInfo> best;
      for (std::uint32_t i = 0; i < devices.size(); ++i) {
        DeviceInfo d = describe(devices[i], i, surface_);
        if (!d.meetsRequirements) {
          continue;
        }
        if (!best || type_rank(d.type) > type_rank(best->type) ||
            (type_rank(d.type) == type_rank(best->type) && d.deviceLocalBytes > best->deviceLocalBytes)) {
          best = d;
        }
      }
      if (!best) {
        throw Error("No usable Vulkan device found (needs Vulkan 1.3, a graphics+compute queue, subgroup "
                    "arithmetic and push descriptors). Run 'splat info' for details.");
      }
      info_ = *best;
    }
    physical_ = devices[info_.index];
    computeFamily_ = *find_compute_family(physical_, surface_);

    // Features: Vulkan 1.3 core bits we rely on, plus float atomics when present.
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    atomicFloat.shaderBufferFloat32AtomicAdd = info_.floatAtomicAdd ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.synchronization2 = VK_TRUE;
    v13.maintenance4 = VK_TRUE;
    v13.dynamicRendering = VK_TRUE;
    v13.pNext = info_.floatAtomicAdd ? &atomicFloat : nullptr;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.timelineSemaphore = VK_TRUE;
    v12.pNext = &v13;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &v12;

    std::vector<const char *> extensions{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    if (surface_ != VK_NULL_HANDLE) {
      extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    if (info_.floatAtomicAdd) {
      extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    hasMemoryBudget_ = has_extension(physical_, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (hasMemoryBudget_) {
      extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = computeFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, computeFamily_, 0, &computeQueue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = computeFamily_;
    check(vkCreateCommandPool(device_, &pci, nullptr, &commandPool_), "vkCreateCommandPool");

    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.vulkanApiVersion = kRequiredApi;
    aci.instance = instance_;
    aci.physicalDevice = physical_;
    aci.device = device_;
    aci.pVulkanFunctions = &fns;
    check(vmaCreateAllocator(&aci, &allocator_), "vmaCreateAllocator");
  } catch (...) {
    destroy();
    throw;
  }
}

Context::~Context() { destroy(); }

Context::Context(Context &&other) noexcept
    : instance_(std::exchange(other.instance_, VK_NULL_HANDLE)),
      messenger_(std::exchange(other.messenger_, VK_NULL_HANDLE)),
      surface_(std::exchange(other.surface_, VK_NULL_HANDLE)),
      physical_(std::exchange(other.physical_, VK_NULL_HANDLE)),
      device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      computeQueue_(std::exchange(other.computeQueue_, VK_NULL_HANDLE)),
      computeFamily_(other.computeFamily_),
      commandPool_(std::exchange(other.commandPool_, VK_NULL_HANDLE)),
      allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
      info_(std::move(other.info_)) {}

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    destroy();
    instance_ = std::exchange(other.instance_, VK_NULL_HANDLE);
    messenger_ = std::exchange(other.messenger_, VK_NULL_HANDLE);
    surface_ = std::exchange(other.surface_, VK_NULL_HANDLE);
    physical_ = std::exchange(other.physical_, VK_NULL_HANDLE);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    computeQueue_ = std::exchange(other.computeQueue_, VK_NULL_HANDLE);
    computeFamily_ = other.computeFamily_;
    commandPool_ = std::exchange(other.commandPool_, VK_NULL_HANDLE);
    allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
    info_ = std::move(other.info_);
  }
  return *this;
}

void Context::destroy() noexcept {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }
  if (allocator_ != VK_NULL_HANDLE) {
    vmaDestroyAllocator(allocator_);
    allocator_ = VK_NULL_HANDLE;
  }
  if (commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    commandPool_ = VK_NULL_HANDLE;
  }
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }
  if (messenger_ != VK_NULL_HANDLE) {
    vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    messenger_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
}

VkCommandBuffer Context::begin_one_shot() {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = commandPool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device_, &ai, &cmd), "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
  return cmd;
}

void Context::end_one_shot(VkCommandBuffer cmd) {
  check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  check(vkCreateFence(device_, &fci, nullptr, &fence), "vkCreateFence");

  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cbi.commandBuffer = cmd;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.commandBufferInfoCount = 1;
  si.pCommandBufferInfos = &cbi;
  const VkResult submitted = vkQueueSubmit2(computeQueue_, 1, &si, fence);
  const VkResult waited =
      submitted == VK_SUCCESS ? vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX) : submitted;
  vkDestroyFence(device_, fence, nullptr);
  vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
  check(submitted, "vkQueueSubmit2");
  check(waited, "vkWaitForFences");
}

std::uint64_t Context::free_device_memory() const {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  if (hasMemoryBudget_) properties.pNext = &budget;
  vkGetPhysicalDeviceMemoryProperties2(physical_, &properties);
  std::uint64_t best = 0;
  for (std::uint32_t h = 0; h < properties.memoryProperties.memoryHeapCount; ++h) {
    const VkMemoryHeap &heap = properties.memoryProperties.memoryHeaps[h];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
    const std::uint64_t available = !hasMemoryBudget_ ? heap.size / 5 * 4
                                    : budget.heapBudget[h] > budget.heapUsage[h] ? budget.heapBudget[h] - budget.heapUsage[h]
                                                                                 : 0;
    best = std::max(best, available);
  }
  return best;
}

} // namespace splat::vk
