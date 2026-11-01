#include "viewer/viewer.hpp"

#include "core/vk_check.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"

#include <volk.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <print>
#include <vector>

namespace splat {
namespace {

using vk::check;

// Owns the swapchain and its image views, recreated on resize.
class Swapchain {
public:
  explicit Swapchain(vk::Context &ctx) : ctx_(ctx) { choose_format(); }
  ~Swapchain() { destroy(); }
  Swapchain(const Swapchain &) = delete;
  Swapchain &operator=(const Swapchain &) = delete;

  [[nodiscard]] VkFormat format() const noexcept { return format_.format; }
  [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
  [[nodiscard]] VkSwapchainKHR handle() const noexcept { return swapchain_; }
  [[nodiscard]] VkImage image(std::uint32_t i) const { return images_.at(i); }
  [[nodiscard]] VkImageView view(std::uint32_t i) const { return views_.at(i); }
  [[nodiscard]] VkSemaphore render_finished(std::uint32_t i) const { return renderFinished_.at(i); }

  void create(VkExtent2D wanted, bool vsync) {
    VkDevice dev = ctx_.device();
    vkDeviceWaitIdle(dev);

    VkSurfaceCapabilitiesKHR caps{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physical_device(), ctx_.surface(), &caps),
          "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    extent_ = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : wanted;
    extent_.width = std::clamp(extent_.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent_.height = std::clamp(extent_.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    if (extent_.width == 0 || extent_.height == 0) {
      return; // minimised
    }

    std::uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
      imageCount = std::min(imageCount, caps.maxImageCount);
    }

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // always supported
    if (!vsync) {
      std::uint32_t n = 0;
      vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical_device(), ctx_.surface(), &n, nullptr);
      std::vector<VkPresentModeKHR> modes(n);
      vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical_device(), ctx_.surface(), &n, modes.data());
      for (VkPresentModeKHR m : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
        if (std::ranges::find(modes, m) != modes.end()) {
          presentMode = m;
          break;
        }
      }
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = ctx_.surface();
    sci.minImageCount = imageCount;
    sci.imageFormat = format_.format;
    sci.imageColorSpace = format_.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) == 0) {
      for (std::uint32_t bit = 1; bit != 0; bit <<= 1) {
        if (caps.supportedCompositeAlpha & bit) {
          sci.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(bit);
          break;
        }
      }
    }
    sci.presentMode = presentMode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;
    VkSwapchainKHR next = VK_NULL_HANDLE;
    check(vkCreateSwapchainKHR(dev, &sci, nullptr, &next), "vkCreateSwapchainKHR");
    destroy();
    swapchain_ = next;

    std::uint32_t n = 0;
    vkGetSwapchainImagesKHR(dev, swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(dev, swapchain_, &n, images_.data());
    for (VkImage img : images_) {
      VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vci.image = img;
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = format_.format;
      vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      check(vkCreateImageView(dev, &vci, nullptr, &view), "vkCreateImageView");
      views_.push_back(view);
      VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VkSemaphore sem = VK_NULL_HANDLE;
      check(vkCreateSemaphore(dev, &semInfo, nullptr, &sem), "vkCreateSemaphore");
      renderFinished_.push_back(sem);
    }
  }

private:
  void choose_format() {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical_device(), ctx_.surface(), &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical_device(), ctx_.surface(), &n, formats.data());
    if (formats.empty()) {
      throw vk::Error("The window surface reports no formats.");
    }
    // Splat colours are already sRGB-encoded (as in 3DGS training), so write
    // them unchanged to a UNORM swapchain rather than an _SRGB one.
    format_ = formats.front();
    for (const auto &f : formats) {
      if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
          f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        format_ = f;
        break;
      }
    }
  }

  void destroy() {
    VkDevice dev = ctx_.device();
    for (VkImageView v : views_) vkDestroyImageView(dev, v, nullptr);
    for (VkSemaphore s : renderFinished_) vkDestroySemaphore(dev, s, nullptr);
    views_.clear();
    renderFinished_.clear();
    images_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(dev, swapchain_, nullptr);
      swapchain_ = VK_NULL_HANDLE;
    }
  }

  vk::Context &ctx_;
  VkSurfaceFormatKHR format_{};
  VkExtent2D extent_{};
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  std::vector<VkImageView> views_;
  std::vector<VkSemaphore> renderFinished_; // one per image: safe reuse across presents
};

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                   VkAccessFlags2 dstAccess) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
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

struct SdlSession {
  SdlSession() {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
      throw std::runtime_error(std::format("SDL_Init failed: {}", SDL_GetError()));
    }
  }
  ~SdlSession() { SDL_Quit(); }
};

struct WindowDeleter {
  void operator()(SDL_Window *w) const { SDL_DestroyWindow(w); }
};

float stick(SDL_Gamepad *pad, SDL_GamepadAxis axis) {
  const float v = static_cast<float>(SDL_GetGamepadAxis(pad, axis)) / 32767.f;
  constexpr float kDeadZone = 0.15f;
  if (std::abs(v) < kDeadZone) {
    return 0.f;
  }
  return std::copysign((std::abs(v) - kDeadZone) / (1.f - kDeadZone), v);
}

} // namespace

int run_viewer(const GaussianCloud &cloud, const ViewerOptions &options) {
  SdlSession sdl;
  std::unique_ptr<SDL_Window, WindowDeleter> window(
      SDL_CreateWindow(options.title.c_str(), 1280, 720,
                       SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY));
  if (!window) {
    throw std::runtime_error(std::format("Could not create a window: {}", SDL_GetError()));
  }

  vk::ContextOptions co = options.context;
  Uint32 extCount = 0;
  const char *const *exts = SDL_Vulkan_GetInstanceExtensions(&extCount);
  if (exts == nullptr) {
    throw std::runtime_error(std::format("SDL has no Vulkan support: {}", SDL_GetError()));
  }
  co.instanceExtensions.assign(exts, exts + extCount);
  co.createSurface = [&](VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window.get(), instance, nullptr, &surface)) {
      throw std::runtime_error(std::format("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError()));
    }
    return surface;
  };

  // Scope: everything Vulkan is destroyed before the window.
  {
    vk::Context ctx(co);
    std::println("Device: {} ({})", ctx.info().name, vk::vendor_name(ctx.info().vendorId));
    VkDevice dev = ctx.device();

    Swapchain swapchain(ctx);
    auto pixel_size = [&] {
      int w = 0, h = 0;
      SDL_GetWindowSizeInPixels(window.get(), &w, &h);
      return VkExtent2D{static_cast<std::uint32_t>(std::max(w, 0)), static_cast<std::uint32_t>(std::max(h, 0))};
    };
    swapchain.create(pixel_size(), options.vsync);

    SplatRenderer renderer(ctx, swapchain.format());
    renderer.set_scene(cloud);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = ctx.command_pool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(dev, &cbai, &cmd), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence frameFence = VK_NULL_HANDLE;
    check(vkCreateFence(dev, &fci, nullptr, &frameFence), "vkCreateFence");
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    check(vkCreateSemaphore(dev, &sci, nullptr, &imageAvailable), "vkCreateSemaphore");

    const OrbitCamera home = default_orbit(cloud, options.up);
    OrbitCamera orbit = home;
    SDL_Gamepad *pad = nullptr;
    bool running = true;
    bool needResize = false;
    Uint64 last = SDL_GetTicksNS();
    double titleTimer = 0.0;
    int framesSinceTitle = 0;

    while (running) {
      SDL_Event e;
      while (SDL_PollEvent(&e)) {
        switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
          running = false;
          break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
          needResize = true;
          break;
        case SDL_EVENT_KEY_DOWN:
          if (e.key.key == SDLK_ESCAPE) running = false;
          if (e.key.key == SDLK_R) orbit = home;
          break;
        case SDL_EVENT_MOUSE_MOTION: {
          const float h = static_cast<float>(std::max(1u, swapchain.extent().height));
          const float panScale = 2.f * std::tan(glm::radians(orbit.fovDeg * 0.5f)) / h;
          if (e.motion.state & SDL_BUTTON_LMASK) {
            orbit.orbit(e.motion.xrel * 0.3f, e.motion.yrel * 0.3f);
          } else if (e.motion.state & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) {
            orbit.pan(-e.motion.xrel * panScale, -e.motion.yrel * panScale);
          }
          break;
        }
        case SDL_EVENT_MOUSE_WHEEL:
          orbit.zoom(std::pow(0.88f, e.wheel.y));
          break;
        case SDL_EVENT_GAMEPAD_ADDED:
          if (pad == nullptr) pad = SDL_OpenGamepad(e.gdevice.which);
          break;
        case SDL_EVENT_GAMEPAD_REMOVED:
          if (pad != nullptr && SDL_GetGamepadID(pad) == e.gdevice.which) {
            SDL_CloseGamepad(pad);
            pad = nullptr;
          }
          break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
          if (e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) orbit = home;
          break;
        default:
          break;
        }
      }
      if (!running) break;

      if (SDL_GetWindowFlags(window.get()) & SDL_WINDOW_MINIMIZED) {
        SDL_WaitEventTimeout(nullptr, 100);
        last = SDL_GetTicksNS();
        continue;
      }

      const Uint64 now = SDL_GetTicksNS();
      const float dt = std::min(0.1f, static_cast<float>(now - last) * 1e-9f);
      last = now;

      // Continuous input: keyboard and gamepad.
      const bool *keys = SDL_GetKeyboardState(nullptr);
      const float speed = 0.6f * dt;
      float fwd = static_cast<float>(keys[SDL_SCANCODE_W]) - static_cast<float>(keys[SDL_SCANCODE_S]);
      float right = static_cast<float>(keys[SDL_SCANCODE_D]) - static_cast<float>(keys[SDL_SCANCODE_A]);
      float upward = static_cast<float>(keys[SDL_SCANCODE_E]) - static_cast<float>(keys[SDL_SCANCODE_Q]);
      if (pad != nullptr) {
        fwd -= stick(pad, SDL_GAMEPAD_AXIS_LEFTY);
        right += stick(pad, SDL_GAMEPAD_AXIS_LEFTX);
        orbit.orbit(stick(pad, SDL_GAMEPAD_AXIS_RIGHTX) * 120.f * dt, stick(pad, SDL_GAMEPAD_AXIS_RIGHTY) * 90.f * dt);
        const float zoomIn = std::max(0.f, stick(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
        const float zoomOut = std::max(0.f, stick(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
        orbit.zoom(std::pow(0.3f, (zoomIn - zoomOut) * dt));
      }
      if (fwd != 0.f || right != 0.f || upward != 0.f) {
        orbit.move(fwd * speed, right * speed, upward * speed);
      }

      check(vkWaitForFences(dev, 1, &frameFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

      if (needResize || swapchain.handle() == VK_NULL_HANDLE) {
        swapchain.create(pixel_size(), options.vsync);
        needResize = false;
        if (swapchain.handle() == VK_NULL_HANDLE) continue;
      }

      std::uint32_t imageIndex = 0;
      VkResult acquired =
          vkAcquireNextImageKHR(dev, swapchain.handle(), UINT64_MAX, imageAvailable, VK_NULL_HANDLE, &imageIndex);
      if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        needResize = true;
        continue;
      }
      if (acquired != VK_SUBOPTIMAL_KHR) {
        check(acquired, "vkAcquireNextImageKHR");
      }
      check(vkResetFences(dev, 1, &frameFence), "vkResetFences");

      const VkExtent2D extent = swapchain.extent();
      const Camera camera = orbit.camera(static_cast<int>(extent.width), static_cast<int>(extent.height));

      check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
      const VkImage image = swapchain.image(imageIndex);
      image_barrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      renderer.record(cmd, camera, swapchain.view(imageIndex), extent, {0.f, 0.f, 0.f, 1.f});
      image_barrier(cmd, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_NONE, 0);
      check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

      VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      wait.semaphore = imageAvailable;
      wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      signal.semaphore = swapchain.render_finished(imageIndex);
      signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      cbsi.commandBuffer = cmd;
      VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submit.waitSemaphoreInfoCount = 1;
      submit.pWaitSemaphoreInfos = &wait;
      submit.commandBufferInfoCount = 1;
      submit.pCommandBufferInfos = &cbsi;
      submit.signalSemaphoreInfoCount = 1;
      submit.pSignalSemaphoreInfos = &signal;
      check(vkQueueSubmit2(ctx.queue(), 1, &submit, frameFence), "vkQueueSubmit2");

      const VkSemaphore renderFinished = swapchain.render_finished(imageIndex);
      const VkSwapchainKHR sc = swapchain.handle();
      VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
      present.waitSemaphoreCount = 1;
      present.pWaitSemaphores = &renderFinished;
      present.swapchainCount = 1;
      present.pSwapchains = &sc;
      present.pImageIndices = &imageIndex;
      const VkResult presented = vkQueuePresentKHR(ctx.queue(), &present);
      if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
        needResize = true;
      } else {
        check(presented, "vkQueuePresentKHR");
      }

      ++framesSinceTitle;
      titleTimer += dt;
      if (titleTimer >= 0.5) {
        const double fps = framesSinceTitle / titleTimer;
        const std::string title = std::format("{} - {} splats - {:.0f} fps ({:.1f} ms)", options.title,
                                              renderer.splat_count(), fps, 1000.0 / fps);
        SDL_SetWindowTitle(window.get(), title.c_str());
        titleTimer = 0.0;
        framesSinceTitle = 0;
      }
    }

    vkDeviceWaitIdle(dev);
    if (pad != nullptr) SDL_CloseGamepad(pad);
    vkDestroySemaphore(dev, imageAvailable, nullptr);
    vkDestroyFence(dev, frameFence, nullptr);
    vkFreeCommandBuffers(dev, ctx.command_pool(), 1, &cmd);
  }
  return 0;
}

} // namespace splat
