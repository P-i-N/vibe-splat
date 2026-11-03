#include "viewer/viewer.hpp"

#include "core/vk_check.hpp"
#include "ingest/image.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "viewer/imgui_renderer.hpp"
#include "viewer/line_renderer.hpp"

#include <volk.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace splat {
namespace {

namespace fs = std::filesystem;
using vk::check;

constexpr float kShC0 = 0.28209479177387814f;

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
  // Whether images can be copied out, for screenshots.
  [[nodiscard]] bool readable() const noexcept { return readable_; }
  [[nodiscard]] bool vsync() const noexcept { return vsync_; }

  void create(VkExtent2D wanted, bool vsync) {
    VkDevice dev = ctx_.device();
    vkDeviceWaitIdle(dev);
    vsync_ = vsync;

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

    readable_ = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = ctx_.surface();
    sci.minImageCount = imageCount;
    sci.imageFormat = format_.format;
    sci.imageColorSpace = format_.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (readable_ ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
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
  bool readable_ = false;
  bool vsync_ = true;
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

struct ImGuiSession {
  ImGuiSession() { ImGui::CreateContext(); }
  ~ImGuiSession() { ImGui::DestroyContext(); }
};

float stick(SDL_Gamepad *pad, SDL_GamepadAxis axis) {
  const float v = static_cast<float>(SDL_GetGamepadAxis(pad, axis)) / 32767.f;
  constexpr float kDeadZone = 0.15f;
  if (std::abs(v) < kDeadZone) {
    return 0.f;
  }
  return std::copysign((std::abs(v) - kDeadZone) / (1.f - kDeadZone), v);
}

// 1234 -> "1234", 51234 -> "51.2k", 2345678 -> "2.35M".
std::string human(double n) {
  if (n < 10'000.0) return std::format("{:.0f}", n);
  if (n < 1'000'000.0) return std::format("{:.1f}k", n / 1e3);
  return std::format("{:.2f}M", n / 1e6);
}

std::string clock_text(double seconds) {
  const auto s = static_cast<long long>(std::max(seconds, 0.0));
  if (s >= 3600) return std::format("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60);
  return std::format("{}:{:02}", s / 60, s % 60);
}

// A camera whose image is fitted into a window of another size: same view,
// scaled so the whole photo is visible.
Camera fit_to(const Camera &c, int width, int height) {
  const float s = std::min(static_cast<float>(width) / static_cast<float>(c.width),
                           static_cast<float>(height) / static_cast<float>(c.height));
  Camera out = c;
  out.fx *= s;
  out.fy *= s;
  out.cx = 0.5f * static_cast<float>(width) + (c.cx - 0.5f * static_cast<float>(c.width)) * s;
  out.cy = 0.5f * static_cast<float>(height) + (c.cy - 0.5f * static_cast<float>(c.height)) * s;
  out.width = width;
  out.height = height;
  return out;
}

// Projects a world point to pixels, or nothing if it is behind the camera.
std::optional<glm::vec2> project(const Camera &c, glm::vec3 world) {
  const glm::vec3 p = glm::vec3(c.view * glm::vec4(world, 1.f));
  if (p.z <= 1e-6f) return std::nullopt;
  return glm::vec2(c.fx * p.x / p.z + c.cx, c.fy * p.y / p.z + c.cy);
}

glm::vec3 forward_of(const Camera &c) { return {c.view[0][2], c.view[1][2], c.view[2][2]}; }

// One photo with a known pose.
struct PosedView {
  std::string name;
  Camera camera; // At the photo's own size
  glm::mat4 toWorld{1.f};
  bool holdout = false;
};

std::vector<PosedView> posed_views(const ColmapModel &model, const TrainOptions &train) {
  std::vector<PosedView> out;
  int number = 0;
  for (const auto &[id, image] : model.images) {
    PosedView view;
    view.name = image.name;
    view.camera = posed_camera(image, model.cameras.at(image.cameraId));
    view.toWorld = glm::inverse(view.camera.view);
    // The same rule as load_train_data().
    view.holdout = train.holdout > 0 && number % train.holdout == train.holdout - 1;
    out.push_back(std::move(view));
    ++number;
  }
  return out;
}

// Half the typical distance between neighbouring cameras: frustums that size
// do not overlap much, however the photos were taken.
float camera_spacing(const std::vector<PosedView> &views) {
  if (views.size() < 2) return 0.f;
  std::vector<float> nearest;
  for (const PosedView &a : views) {
    float best = std::numeric_limits<float>::max();
    for (const PosedView &b : views) {
      if (&a != &b) best = std::min(best, glm::length(a.camera.position - b.camera.position));
    }
    nearest.push_back(best);
  }
  std::nth_element(nearest.begin(), nearest.begin() + static_cast<std::ptrdiff_t>(nearest.size() / 2), nearest.end());
  return 0.5f * nearest[nearest.size() / 2];
}

// The sparse points as tiny opaque Gaussians, for the point renderer.
GaussianCloud sparse_cloud(const ColmapModel &model) {
  GaussianCloud cloud;
  cloud.resize(model.points.size(), 0);
  for (std::size_t i = 0; i < model.points.size(); ++i) {
    const ColmapPoint &p = model.points[i];
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.positions[i * 3 + a] = static_cast<float>(p.position[static_cast<int>(a)]);
      cloud.scales[i * 3 + a] = -7.f;
      cloud.sh[i * 3 + a] = (static_cast<float>(p.color[a]) / 255.f - 0.5f) / kShC0;
    }
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = 8.f;
  }
  return cloud;
}

GpuCloud gpu_cloud(const Trainer &trainer) {
  GpuCloud out;
  out.positions = &trainer.parameters(0);
  out.scales = &trainer.parameters(1);
  out.rotations = &trainer.parameters(2);
  out.opacities = &trainer.parameters(3);
  out.sh = &trainer.parameters(4);
  out.count = trainer.gaussian_count();
  out.shDegree = trainer.sh_degree();
  out.activeDegree = trainer.active_sh_degree();
  return out;
}

struct Photo {
  std::size_t view = 0;
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgb;
};

Photo load_photo(const fs::path &file, std::size_t view) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", file.string()));
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  RgbImage image = downscale_to_fit(decode_image(bytes), 2048);
  return {view, image.width, image.height, std::move(image.pixels)};
}

enum class TrainState { Idle, Loading, Running, Paused, Finished, Failed };

const ImVec4 kDim{0.62f, 0.66f, 0.72f, 1.f};
const ImVec4 kAccent{0.45f, 0.75f, 1.f, 1.f};
const ImVec4 kWarn{1.f, 0.62f, 0.3f, 1.f};

// Everything the window shows and does. One instance per run_viewer().
class Viewer {
public:
  Viewer(const ViewerOptions &options, SDL_Window *window, vk::Context &ctx);
  ~Viewer();
  void run();

private:
  // Loading
  void open_project(const fs::path &project);
  void open_scene(const fs::path &path);
  void open_dropped(const fs::path &path);
  void reset_home();
  void start_training();
  void stop_training(bool save);

  // Per frame
  void handle_event(const SDL_Event &e);
  void handle_key(const SDL_KeyboardEvent &key);
  void update_input(float dt);
  void after_fence();
  void train_tick();
  void build_ui();
  void build_lines(const Camera &camera);
  void record(VkCommandBuffer cmd, std::uint32_t imageIndex);

  // UI panels
  bool begin_corner(const char *id, int corner);
  void panel_scene();
  void panel_help();
  void panel_training();
  void panel_view();
  void overlay_photo();
  void overlay_toast();

  // Helpers
  [[nodiscard]] Camera render_camera(VkExtent2D extent) const;
  [[nodiscard]] float pixels_per_point() const;
  [[nodiscard]] std::optional<std::size_t> view_at(glm::vec2 pixel, float radius) const;
  void snap_to(std::size_t view);
  void unsnap();
  void step_view(int delta);
  void request_photo(std::size_t view);
  void toast(std::string text, bool warning = false);
  void log(const std::string &line);
  [[nodiscard]] bool training_active() const noexcept {
    return state_ == TrainState::Running || state_ == TrainState::Paused;
  }

  ViewerOptions options_;
  SDL_Window *window_;
  vk::Context &ctx_;
  Swapchain swapchain_;
  std::unique_ptr<SplatRenderer> renderer_, points_;
  std::unique_ptr<LineRenderer> lines_;
  std::unique_ptr<ImGuiRenderer> imgui_;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence frameFence_ = VK_NULL_HANDLE;
  VkSemaphore imageAvailable_ = VK_NULL_HANDLE;
  float uiScale_ = 1.f;

  // Scene
  std::string sceneName_ = "nothing loaded";
  fs::path scenePath_;
  fs::path project_;
  std::optional<ColmapModel> model_;
  std::vector<PosedView> views_;
  float cameraSpacing_ = 0.f;
  Bounds bounds_;
  OrbitCamera home_, orbit_;
  std::optional<std::pair<Bounds, OrbitCamera>> sceneHome_; // Of the loaded .ply
  std::uint32_t visible_ = 0;

  // View settings
  bool running_ = true;
  bool showUi_ = true, showHelp_ = true, showSettings_ = false;
  bool showCameras_ = true, showPoints_ = false, showGrid_ = false, showAxes_ = false;
  bool pointMode_ = false, autoRotate_ = false, vsync_ = true;
  float splatScale_ = 1.f, pointRadius_ = 2.f, frustumScale_ = 1.f;
  int shDegree_ = 3;
  float background_[3] = {0.f, 0.f, 0.f};

  // Input
  bool dragging_ = false;       // A drag that started over the scene, not the UI
  bool showPhoto_ = false;      // X held
  std::optional<std::size_t> snapped_, hovered_;
  SDL_Gamepad *pad_ = nullptr;
  bool needResize_ = false;

  // Photo of the snapped view
  std::optional<std::size_t> photoWanted_;
  std::future<Photo> photoLoading_;
  ImTextureID photoTexture_ = ImTextureID_Invalid;
  std::size_t photoView_ = ~std::size_t{0};
  int photoWidth_ = 0, photoHeight_ = 0;

  // Training
  TrainState state_ = TrainState::Idle;
  TrainOptions trainOptions_;
  std::future<TrainData> trainLoading_;
  std::shared_ptr<std::atomic<std::size_t>> loadDone_ = std::make_shared<std::atomic<std::size_t>>(0);
  std::shared_ptr<std::atomic<std::size_t>> loadTotal_ = std::make_shared<std::atomic<std::size_t>>(0);
  std::unique_ptr<TrainSession> session_;
  StepStats lastStep_;
  float trainBudgetMs_ = 30.f;
  std::vector<float> lossHistory_;
  double lossBucket_ = 0.0;
  int lossBucketCount_ = 0;
  std::deque<std::string> log_;
  std::string failure_;
  TrainReport finalReport_;

  // Frame timing, toasts, screenshots, work for after the fence
  double frameMs_ = 16.0;
  double visibleTimer_ = 0.0;
  std::string toast_;
  bool toastWarning_ = false;
  double toastTime_ = 0.0;
  std::vector<std::function<void()>> pending_;
  bool screenshotWanted_ = false;
  std::unique_ptr<vk::Buffer> screenshotBuffer_;
  VkExtent2D screenshotExtent_{};
  bool screenshotInFlight_ = false;
  std::vector<std::future<void>> tasks_;
};

Viewer::Viewer(const ViewerOptions &options, SDL_Window *window, vk::Context &ctx)
    : options_(options), window_(window), ctx_(ctx), swapchain_(ctx), vsync_(options.vsync) {
  VkDevice dev = ctx_.device();
  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(window_, &w, &h);
  swapchain_.create({static_cast<std::uint32_t>(std::max(w, 0)), static_cast<std::uint32_t>(std::max(h, 0))}, vsync_);

  renderer_ = std::make_unique<SplatRenderer>(ctx_, swapchain_.format());
  points_ = std::make_unique<SplatRenderer>(ctx_, swapchain_.format());
  points_->settings.points = true;
  lines_ = std::make_unique<LineRenderer>(ctx_, swapchain_.format());

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = ctx_.command_pool();
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(dev, &cbai, &cmd_), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  check(vkCreateFence(dev, &fci, nullptr, &frameFence_), "vkCreateFence");
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  check(vkCreateSemaphore(dev, &sci, nullptr, &imageAvailable_), "vkCreateSemaphore");

  // Dear ImGui: compact, translucent, scaled for the display.
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  io.BackendRendererName = "splat";
  ImGui_ImplSDL3_InitForVulkan(window_);
  ImGui_ImplSDL3_SetGamepadMode(ImGui_ImplSDL3_GamepadMode_Manual, nullptr, 0);
  uiScale_ = std::max(1.f, SDL_GetWindowDisplayScale(window_) / std::max(SDL_GetWindowPixelDensity(window_), 1.f));
  ImGui::StyleColorsDark();
  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 6.f;
  style.ChildRounding = 4.f;
  style.FrameRounding = 4.f;
  style.GrabRounding = 4.f;
  style.PopupRounding = 4.f;
  style.WindowBorderSize = 0.f;
  style.WindowPadding = ImVec2(10.f, 8.f);
  style.FramePadding = ImVec2(6.f, 3.f);
  style.ItemSpacing = ImVec2(6.f, 4.f);
  style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.07f, 0.09f, 0.78f);
  style.Colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.09f, 0.11f, 0.95f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.2f, 0.24f, 0.7f);
  style.Colors[ImGuiCol_PlotLines] = kAccent;
  style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.3f, 0.55f, 0.85f, 1.f);
  style.ScaleAllSizes(uiScale_);
  style.FontScaleDpi = uiScale_;
  style.FontSizeBase = 14.f;
  io.Fonts->AddFontDefaultVector();
  imgui_ = std::make_unique<ImGuiRenderer>(ctx_, swapchain_.format());

  trainOptions_ = options_.train.value_or(TrainOptions{});
  // The viewer keeps a copy of the Gaussians to draw, so training gets less.
  if (trainOptions_.vramBudgetGiB <= 0.0) trainOptions_.vramShare = 0.75;
  if (!options_.project.empty()) open_project(options_.project);
  if (!options_.scene.empty()) open_scene(options_.scene);
  if (renderer_->splat_count() == 0 && points_->splat_count() > 0) showPoints_ = true;
  reset_home();
  orbit_ = home_;
  if (options_.train) start_training();
}

Viewer::~Viewer() {
  VkDevice dev = ctx_.device();
  vkDeviceWaitIdle(dev);
  for (auto &task : tasks_) task.wait();
  if (photoLoading_.valid()) photoLoading_.wait();
  if (trainLoading_.valid()) trainLoading_.wait();
  imgui_->destroy_texture(photoTexture_);
  session_.reset();
  imgui_.reset(); // Before the ImGui context goes, as it releases ImGui's textures
  ImGui_ImplSDL3_Shutdown();
  if (pad_ != nullptr) SDL_CloseGamepad(pad_);
  vkDestroySemaphore(dev, imageAvailable_, nullptr);
  vkDestroyFence(dev, frameFence_, nullptr);
  vkFreeCommandBuffers(dev, ctx_.command_pool(), 1, &cmd_);
}

// --- Loading ----------------------------------------------------------------

void Viewer::open_project(const fs::path &project) {
  project_ = project;
  model_.reset();
  views_.clear();
  snapped_.reset();
  photoWanted_.reset();
  photoView_ = ~std::size_t{0};
  const fs::path sparse = project / "sparse" / "0";
  if (fs::exists(sparse / "images.bin")) {
    model_ = load_colmap_model(sparse);
    views_ = posed_views(*model_, trainOptions_);
    cameraSpacing_ = camera_spacing(views_);
    points_->set_scene(sparse_cloud(*model_));
    std::println("Project {}: {} posed images, {} points", project.string(), views_.size(), model_->points.size());
  } else {
    points_->set_scene(GaussianCloud{});
  }
  trainOptions_.project = project;
  if (trainOptions_.out.empty() || !options_.train) {
    // Trained again in the format it was trained in before.
    const fs::path previous = project_scene(project);
    trainOptions_.out = project / (previous.empty() ? fs::path("splat.ply") : previous.filename());
  }
  sceneName_ = project.filename().string();
}

void Viewer::open_scene(const fs::path &path) {
  const auto start = std::chrono::steady_clock::now();
  const GaussianCloud cloud = load_cloud(path);
  const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::println("Loaded {} Gaussians (SH degree {}) from {} in {:.0f} ms", cloud.size(), cloud.shDegree, path.string(), ms);
  renderer_->set_scene(cloud);
  sceneHome_.emplace(compute_bounds(cloud), default_orbit(cloud, options_.up));
  shDegree_ = cloud.shDegree;
  scenePath_ = path;
  sceneName_ = path.filename().string();
  SDL_SetWindowTitle(window_, std::format("splat - {}", path.string()).c_str());
}

void Viewer::open_dropped(const fs::path &path) {
  if (training_active() || state_ == TrainState::Loading) {
    toast("Stop training before opening something else", true);
    return;
  }
  try {
    session_.reset();
    state_ = TrainState::Idle;
    sceneHome_.reset();
    renderer_->set_scene(GaussianCloud{});
    if (fs::is_directory(path)) {
      open_project(path);
      if (const fs::path scene = project_scene(path); !scene.empty()) open_scene(scene);
    } else {
      // A scene inside a project brings its cameras along.
      const fs::path parent = path.parent_path();
      if (fs::exists(parent / "sparse" / "0" / "images.bin")) {
        open_project(parent);
      } else {
        project_.clear();
        model_.reset();
        views_.clear();
        snapped_.reset();
        photoWanted_.reset();
        photoView_ = ~std::size_t{0};
        points_->set_scene(GaussianCloud{});
      }
      open_scene(path);
    }
    showPoints_ = renderer_->splat_count() == 0 && points_->splat_count() > 0;
    reset_home();
    orbit_ = home_;
    toast(std::format("Opened {}", path.filename().string()));
  } catch (const std::exception &e) {
    toast(std::format("Could not open {}: {}", path.filename().string(), e.what()), true);
  }
}

void Viewer::reset_home() {
  GaussianCloud probe;
  if (session_) {
    probe = sparse_cloud(session_->data().model);
  } else if (sceneHome_) {
    bounds_ = sceneHome_->first;
    home_ = sceneHome_->second;
    return;
  } else if (model_) {
    probe = sparse_cloud(*model_);
  }
  if (probe.empty()) {
    bounds_ = Bounds{};
    home_ = OrbitCamera{};
    home_.up = options_.up;
    return;
  }
  bounds_ = compute_bounds(probe);
  home_ = default_orbit(probe, options_.up);
}

void Viewer::start_training() {
  if (!model_) {
    toast("Training needs a project with camera poses (run 'splat sfm')", true);
    return;
  }
  if (state_ == TrainState::Loading || training_active()) return;
  session_.reset();
  lossHistory_.clear();
  lossBucket_ = 0.0;
  lossBucketCount_ = 0;
  failure_.clear();
  log_.clear();
  loadDone_->store(0);
  loadTotal_->store(views_.size());
  // Views are decoded on another thread; the trainer starts once they are in.
  const TrainOptions options = trainOptions_;
  auto done = loadDone_, total = loadTotal_;
  trainLoading_ = std::async(std::launch::async, [options, done, total] {
    return load_train_data(options, [done, total](std::size_t d, std::size_t t) {
      done->store(d);
      total->store(t);
    });
  });
  state_ = TrainState::Loading;
  views_ = posed_views(*model_, trainOptions_); // Holdout may have changed
  log(std::format("Loading {} images from {}", views_.size(), project_.string()));
}

void Viewer::stop_training(bool save) {
  if (!session_) return;
  try {
    if (save) {
      finalReport_ = session_->finish();
      toast(std::format("Saved {} Gaussians to {}", human(static_cast<double>(finalReport_.gaussians)), session_->options().out.filename().string()));
      scenePath_ = session_->options().out;
    }
    state_ = TrainState::Finished;
  } catch (const std::exception &e) {
    failure_ = e.what();
    state_ = TrainState::Failed;
    toast(std::format("Training failed: {}", e.what()), true);
  }
}

// --- Frame loop -------------------------------------------------------------

void Viewer::run() {
  Uint64 last = SDL_GetTicksNS();
  while (running_) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) handle_event(e);
    if (!running_) break;

    if (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) {
      // Keep training while minimised, but do not spin.
      check(vkWaitForFences(ctx_.device(), 1, &frameFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
      if (state_ == TrainState::Running) {
        train_tick();
      } else {
        SDL_WaitEventTimeout(nullptr, 100);
      }
      last = SDL_GetTicksNS();
      continue;
    }

    const Uint64 now = SDL_GetTicksNS();
    const float dt = std::min(0.1f, static_cast<float>(now - last) * 1e-9f);
    last = now;
    frameMs_ = 0.9 * frameMs_ + 0.1 * dt * 1000.0;
    if (toastTime_ > 0.0) toastTime_ -= dt;

    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    update_input(dt);
    build_ui();
    ImGui::Render();

    check(vkWaitForFences(ctx_.device(), 1, &frameFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    visibleTimer_ += dt;
    after_fence();
    train_tick();

    // Present without waiting for the display while training, so that the time
    // goes to training rather than to vsync.
    const bool wantVsync = vsync_ && state_ != TrainState::Running;
    if (needResize_ || swapchain_.handle() == VK_NULL_HANDLE || wantVsync != swapchain_.vsync()) {
      int w = 0, h = 0;
      SDL_GetWindowSizeInPixels(window_, &w, &h);
      swapchain_.create({static_cast<std::uint32_t>(std::max(w, 0)), static_cast<std::uint32_t>(std::max(h, 0))}, wantVsync);
      needResize_ = false;
      if (swapchain_.handle() == VK_NULL_HANDLE) continue;
    }

    std::uint32_t imageIndex = 0;
    const VkResult acquired =
        vkAcquireNextImageKHR(ctx_.device(), swapchain_.handle(), UINT64_MAX, imageAvailable_, VK_NULL_HANDLE, &imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
      needResize_ = true;
      continue;
    }
    if (acquired != VK_SUBOPTIMAL_KHR) check(acquired, "vkAcquireNextImageKHR");
    check(vkResetFences(ctx_.device(), 1, &frameFence_), "vkResetFences");

    imgui_->update_textures(ImGui::GetDrawData());
    check(vkResetCommandBuffer(cmd_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd_, &bi), "vkBeginCommandBuffer");
    record(cmd_, imageIndex);
    check(vkEndCommandBuffer(cmd_), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = imageAvailable_;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = swapchain_.render_finished(imageIndex);
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbsi.commandBuffer = cmd_;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cbsi;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    check(vkQueueSubmit2(ctx_.queue(), 1, &submit, frameFence_), "vkQueueSubmit2");

    const VkSemaphore renderFinished = swapchain_.render_finished(imageIndex);
    const VkSwapchainKHR sc = swapchain_.handle();
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished;
    present.swapchainCount = 1;
    present.pSwapchains = &sc;
    present.pImageIndices = &imageIndex;
    const VkResult presented = vkQueuePresentKHR(ctx_.queue(), &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
      needResize_ = true;
    } else {
      check(presented, "vkQueuePresentKHR");
    }
  }

  check(vkWaitForFences(ctx_.device(), 1, &frameFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  // Closing mid-training keeps what was learnt so far.
  if (session_ && training_active() && !session_->finished()) {
    session_->save(session_->options().out);
    std::println("Saved iteration {} to {}", session_->iteration(), session_->options().out.string());
  }
}

void Viewer::handle_event(const SDL_Event &e) {
  ImGui_ImplSDL3_ProcessEvent(&e);
  const ImGuiIO &io = ImGui::GetIO();
  switch (e.type) {
  case SDL_EVENT_QUIT:
  case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
    running_ = false;
    break;
  case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    needResize_ = true;
    break;
  case SDL_EVENT_KEY_DOWN:
    if (!io.WantCaptureKeyboard) handle_key(e.key);
    break;
  case SDL_EVENT_KEY_UP:
    if (e.key.key == SDLK_X) showPhoto_ = false;
    break;
  case SDL_EVENT_MOUSE_BUTTON_DOWN:
    if (!io.WantCaptureMouse) {
      dragging_ = true;
      if (e.button.button == SDL_BUTTON_LEFT && e.button.clicks == 2 && showCameras_) {
        const float ppp = pixels_per_point();
        if (const auto view = view_at({e.button.x * ppp, e.button.y * ppp}, 14.f * uiScale_ * ppp)) snap_to(*view);
      }
    }
    break;
  case SDL_EVENT_MOUSE_BUTTON_UP:
    if ((SDL_GetMouseState(nullptr, nullptr) & (SDL_BUTTON_LMASK | SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) == 0) dragging_ = false;
    break;
  case SDL_EVENT_MOUSE_MOTION: {
    if (!dragging_) break;
    const float h = static_cast<float>(std::max(1u, swapchain_.extent().height));
    const float panScale = 2.f * std::tan(glm::radians(orbit_.fovDeg * 0.5f)) / h * pixels_per_point();
    if (e.motion.state & SDL_BUTTON_LMASK) {
      unsnap();
      // Looking around in place by default; Ctrl swings around the target instead.
      if (SDL_GetModState() & SDL_KMOD_CTRL) {
        orbit_.orbit(e.motion.xrel * 0.3f, e.motion.yrel * 0.3f);
      } else {
        orbit_.look(-e.motion.xrel * 0.2f, e.motion.yrel * 0.2f);
      }
    } else if (e.motion.state & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) {
      unsnap();
      orbit_.pan(-e.motion.xrel * panScale, -e.motion.yrel * panScale);
    }
    break;
  }
  case SDL_EVENT_MOUSE_WHEEL:
    if (!io.WantCaptureMouse) {
      unsnap();
      if (SDL_GetModState() & SDL_KMOD_CTRL) {
        orbit_.fovDeg = std::clamp(orbit_.fovDeg - e.wheel.y * 3.f, 10.f, 120.f);
      } else {
        orbit_.zoom(std::pow(0.88f, e.wheel.y));
      }
    }
    break;
  case SDL_EVENT_DROP_FILE:
    if (e.drop.data != nullptr) {
      const fs::path path = reinterpret_cast<const char8_t *>(e.drop.data);
      pending_.push_back([this, path] { open_dropped(path); });
    }
    break;
  case SDL_EVENT_GAMEPAD_ADDED:
    if (pad_ == nullptr) pad_ = SDL_OpenGamepad(e.gdevice.which);
    break;
  case SDL_EVENT_GAMEPAD_REMOVED:
    if (pad_ != nullptr && SDL_GetGamepadID(pad_) == e.gdevice.which) {
      SDL_CloseGamepad(pad_);
      pad_ = nullptr;
    }
    break;
  case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    switch (e.gbutton.button) {
    case SDL_GAMEPAD_BUTTON_SOUTH:
      unsnap();
      orbit_ = home_;
      break;
    case SDL_GAMEPAD_BUTTON_EAST: unsnap(); break;
    case SDL_GAMEPAD_BUTTON_NORTH: showUi_ = !showUi_; break;
    case SDL_GAMEPAD_BUTTON_WEST: pointMode_ = !pointMode_; break;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: step_view(-1); break;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: step_view(1); break;
    case SDL_GAMEPAD_BUTTON_START:
      if (state_ == TrainState::Running) state_ = TrainState::Paused;
      else if (state_ == TrainState::Paused) state_ = TrainState::Running;
      break;
    default: break;
    }
    break;
  default:
    break;
  }
}

void Viewer::handle_key(const SDL_KeyboardEvent &key) {
  const bool ctrl = (key.mod & SDL_KMOD_CTRL) != 0;
  if (key.key == SDLK_X) {
    showPhoto_ = true;
    return;
  }
  if (key.repeat) {
    if (key.key == SDLK_LEFTBRACKET) step_view(-1);
    if (key.key == SDLK_RIGHTBRACKET) step_view(1);
    return;
  }
  switch (key.key) {
  case SDLK_ESCAPE:
    if (snapped_) unsnap();
    else running_ = false;
    break;
  case SDLK_R:
    unsnap();
    orbit_ = home_;
    break;
  case SDLK_F: {
    // Frame the scene from where we are looking.
    unsnap();
    const float yaw = orbit_.yawDeg, pitch = orbit_.pitchDeg;
    orbit_ = home_;
    orbit_.yawDeg = yaw;
    orbit_.pitchDeg = pitch;
    break;
  }
  case SDLK_T: autoRotate_ = !autoRotate_; break;
  case SDLK_1: pointMode_ = false; break;
  case SDLK_2: pointMode_ = true; break;
  case SDLK_C: showCameras_ = !showCameras_; break;
  case SDLK_P: showPoints_ = !showPoints_; break;
  case SDLK_G: showGrid_ = !showGrid_; break;
  case SDLK_TAB: showUi_ = !showUi_; break;
  case SDLK_H:
  case SDLK_F1: showHelp_ = !showHelp_; break;
  case SDLK_LEFTBRACKET: step_view(-1); break;
  case SDLK_RIGHTBRACKET: step_view(1); break;
  case SDLK_SPACE:
    if (state_ == TrainState::Running) state_ = TrainState::Paused;
    else if (state_ == TrainState::Paused) state_ = TrainState::Running;
    break;
  case SDLK_S:
    if (ctrl && session_) {
      pending_.push_back([this] {
        session_->save(session_->options().out);
        toast(std::format("Saved iteration {} to {}", session_->iteration(), session_->options().out.filename().string()));
      });
    }
    break;
  case SDLK_F12: screenshotWanted_ = true; break;
  default: break;
  }
}

void Viewer::update_input(float dt) {
  const ImGuiIO &io = ImGui::GetIO();
  // Continuous input: keyboard and gamepad.
  const bool *keys = SDL_GetKeyboardState(nullptr);
  float fwd = 0.f, right = 0.f, upward = 0.f;
  if (!io.WantCaptureKeyboard) {
    fwd = static_cast<float>(keys[SDL_SCANCODE_W]) - static_cast<float>(keys[SDL_SCANCODE_S]);
    right = static_cast<float>(keys[SDL_SCANCODE_D]) - static_cast<float>(keys[SDL_SCANCODE_A]);
    upward = static_cast<float>(keys[SDL_SCANCODE_E]) - static_cast<float>(keys[SDL_SCANCODE_Q]);
  }
  const float speed = (SDL_GetModState() & SDL_KMOD_SHIFT ? 2.4f : 0.6f) * dt;
  if (pad_ != nullptr) {
    fwd -= stick(pad_, SDL_GAMEPAD_AXIS_LEFTY);
    right += stick(pad_, SDL_GAMEPAD_AXIS_LEFTX);
    const float yaw = stick(pad_, SDL_GAMEPAD_AXIS_RIGHTX), pitch = stick(pad_, SDL_GAMEPAD_AXIS_RIGHTY);
    const float zoomIn = std::max(0.f, stick(pad_, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
    const float zoomOut = std::max(0.f, stick(pad_, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    if (yaw != 0.f || pitch != 0.f || zoomIn != zoomOut) {
      unsnap();
      orbit_.orbit(yaw * 120.f * dt, pitch * 90.f * dt);
      orbit_.zoom(std::pow(0.3f, (zoomIn - zoomOut) * dt));
    }
  }
  if (fwd != 0.f || right != 0.f || upward != 0.f) {
    unsnap();
    orbit_.move(fwd * speed, right * speed, upward * speed);
  }
  if (autoRotate_ && !snapped_ && !dragging_) orbit_.orbit(dt * 15.f, 0.f);

  hovered_.reset();
  if (showCameras_ && !io.WantCaptureMouse && !dragging_) {
    float mx = 0.f, my = 0.f;
    SDL_GetMouseState(&mx, &my);
    const float ppp = pixels_per_point();
    hovered_ = view_at({mx * ppp, my * ppp}, 14.f * uiScale_ * ppp);
  }
}

// Work that touches the GPU, done while no frame is in flight.
void Viewer::after_fence() {
  for (auto &task : std::exchange(pending_, {})) task();
  std::erase_if(tasks_, [](std::future<void> &f) { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });

  if (visibleTimer_ > 0.5) {
    visibleTimer_ = 0.0;
    visible_ = renderer_->last_visible_count();
  }

  // Screenshot copied in the last frame: write it on another thread.
  if (screenshotInFlight_) {
    screenshotInFlight_ = false;
    screenshotBuffer_->invalidate();
    RenderedImage image;
    image.width = static_cast<int>(screenshotExtent_.width);
    image.height = static_cast<int>(screenshotExtent_.height);
    const auto bytes = screenshotBuffer_->mapped();
    image.rgba.assign(reinterpret_cast<const std::uint8_t *>(bytes.data()),
                      reinterpret_cast<const std::uint8_t *>(bytes.data()) + static_cast<std::size_t>(image.width) * image.height * 4);
    const bool bgra = swapchain_.format() == VK_FORMAT_B8G8R8A8_UNORM;
    for (std::size_t i = 0; i < image.rgba.size(); i += 4) {
      if (bgra) std::swap(image.rgba[i], image.rgba[i + 2]);
      image.rgba[i + 3] = 255;
    }
    const auto stamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    const fs::path folder = project_.empty() ? fs::current_path() : project_;
    const fs::path file = folder / std::format("screenshot_{:%Y%m%d_%H%M%S}.png", stamp);
    tasks_.push_back(std::async(std::launch::async, [image = std::move(image), file] { write_png(image, file); }));
    std::println("Screenshot: {}", file.string());
    toast(std::format("Screenshot saved: {}", file.filename().string()));
  }

  // The photo of the snapped view.
  if (photoLoading_.valid() && photoLoading_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    try {
      Photo photo = photoLoading_.get();
      if (photoWanted_ == photo.view) { // Not if another was asked for meanwhile
        imgui_->destroy_texture(photoTexture_);
        photoTexture_ = imgui_->create_texture(photo.width, photo.height, photo.rgb, 3);
        photoView_ = photo.view;
        photoWidth_ = photo.width;
        photoHeight_ = photo.height;
      }
    } catch (const std::exception &e) {
      toast(std::format("Could not load the photo: {}", e.what()), true);
      photoWanted_.reset();
    }
  }
  if (photoWanted_ && *photoWanted_ != photoView_ && !photoLoading_.valid()) {
    const std::size_t view = *photoWanted_;
    // A view being trained on is already decoded, undistorted and sized.
    const TrainView *trained = nullptr;
    if (session_) {
      for (const auto *list : {&session_->data().trainViews, &session_->data().holdoutViews}) {
        for (const TrainView &v : *list) {
          if (v.name == views_[view].name) trained = &v;
        }
      }
    }
    if (trained != nullptr) {
      imgui_->destroy_texture(photoTexture_);
      photoTexture_ = imgui_->create_texture(trained->camera.width, trained->camera.height, trained->rgba, 4);
      photoView_ = view;
      photoWidth_ = trained->camera.width;
      photoHeight_ = trained->camera.height;
    } else {
      photoLoading_ = std::async(std::launch::async, load_photo, project_ / views_[view].name, view);
    }
  }

  // Training data loaded: the trainer can start.
  if (state_ == TrainState::Loading && trainLoading_.valid() &&
      trainLoading_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    try {
      session_ = std::make_unique<TrainSession>(ctx_, trainOptions_, trainLoading_.get(), [this](const std::string &line) { log(line); });
      state_ = TrainState::Running;
      scenePath_.clear();
      showPoints_ = false;
      reset_home();
    } catch (const std::exception &e) {
      failure_ = e.what();
      state_ = TrainState::Failed;
      toast(std::format("Training failed: {}", e.what()), true);
    }
  }
}

// Trains for a share of each frame, so the window stays responsive.
void Viewer::train_tick() {
  if (state_ != TrainState::Running || !session_) return;
  try {
    const auto start = std::chrono::steady_clock::now();
    const int bucket = std::max(1, session_->options().iterations / 500);
    do {
      lastStep_ = session_->step();
      lossBucket_ += lastStep_.loss;
      if (++lossBucketCount_ >= bucket) {
        lossHistory_.push_back(static_cast<float>(lossBucket_ / lossBucketCount_));
        lossBucket_ = 0.0;
        lossBucketCount_ = 0;
      }
    } while (!session_->finished() &&
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() < trainBudgetMs_);
    if (session_->finished()) stop_training(true);
  } catch (const std::exception &e) {
    failure_ = e.what();
    state_ = TrainState::Failed;
    toast(std::format("Training failed: {}", e.what()), true);
  }
}

void Viewer::record(VkCommandBuffer cmd, std::uint32_t imageIndex) {
  const VkExtent2D extent = swapchain_.extent();
  const Camera camera = render_camera(extent);
  const VkImage image = swapchain_.image(imageIndex);
  const VkImageView target = swapchain_.view(imageIndex);
  const std::array<float, 4> background{background_[0], background_[1], background_[2], 1.f};

  if (session_) renderer_->set_scene(cmd, gpu_cloud(session_->trainer()));
  renderer_->settings.splatScale = splatScale_;
  renderer_->settings.maxShDegree = shDegree_;
  renderer_->settings.points = pointMode_;
  renderer_->settings.pointRadius = pointRadius_;
  points_->settings.pointRadius = pointRadius_;

  image_barrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  renderer_->record(cmd, camera, target, extent, background);
  if (showPoints_ && points_->splat_count() > 0) points_->record(cmd, camera, target, extent, background, false);

  // Screenshots show the scene, without the overlays.
  if (screenshotWanted_) {
    screenshotWanted_ = false;
    if (!swapchain_.readable()) {
      toast("This display does not allow screenshots", true);
    } else {
      const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
      if (!screenshotBuffer_ || screenshotBuffer_->size() < bytes) {
        screenshotBuffer_ = std::make_unique<vk::Buffer>(ctx_, bytes, 0, vk::MemoryUsage::Readback);
      }
      image_barrier(cmd, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {extent.width, extent.height, 1};
      vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, screenshotBuffer_->handle(), 1, &region);
      image_barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
      screenshotExtent_ = extent;
      screenshotInFlight_ = true;
    }
  } else {
    image_barrier(cmd, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
  }

  // Overlays: lines, then the UI.
  build_lines(camera);
  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = target;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, extent};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  vkCmdBeginRendering(cmd, &ri);
  lines_->record(cmd, camera, extent, std::max(bounds_.radius * 1e-3f, 1e-4f));
  imgui_->record(cmd, ImGui::GetDrawData(), extent);
  vkCmdEndRendering(cmd);

  image_barrier(cmd, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_NONE, 0);
}

void Viewer::build_lines(const Camera &camera) {
  lines_->clear();
  const glm::vec3 u = glm::normalize(options_.up);
  const glm::vec3 ref = std::abs(u.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
  const glm::vec3 a = glm::normalize(ref - u * glm::dot(ref, u));
  const glm::vec3 b = glm::cross(u, a);
  const glm::vec3 center(bounds_.center[0], bounds_.center[1], bounds_.center[2]);
  const float radius = std::max(bounds_.radius, 1e-3f);

  if (showGrid_) {
    const float spacing = std::pow(10.f, std::floor(std::log10(radius * 0.5f)));
    const int n = std::clamp(static_cast<int>(std::ceil(radius * 1.5f / spacing)), 2, 40);
    const float half = static_cast<float>(n) * spacing;
    for (int i = -n; i <= n; ++i) {
      const float t = static_cast<float>(i) * spacing;
      const glm::vec4 color = i == 0 ? glm::vec4(0.8f, 0.8f, 0.85f, 0.5f)
                              : i % 5 == 0 ? glm::vec4(0.6f, 0.6f, 0.65f, 0.35f)
                                           : glm::vec4(0.5f, 0.5f, 0.55f, 0.18f);
      lines_->line(center + a * t - b * half, center + a * t + b * half, color);
      lines_->line(center + b * t - a * half, center + b * t + a * half, color);
    }
  }
  if (showAxes_) {
    const float len = radius * 0.4f;
    lines_->line({0.f, 0.f, 0.f}, {len, 0.f, 0.f}, {1.f, 0.3f, 0.3f, 0.9f});
    lines_->line({0.f, 0.f, 0.f}, {0.f, len, 0.f}, {0.3f, 1.f, 0.3f, 0.9f});
    lines_->line({0.f, 0.f, 0.f}, {0.f, 0.f, len}, {0.3f, 0.5f, 1.f, 0.9f});
  }
  if (showCameras_) {
    const float depth = (cameraSpacing_ > 0.f ? std::min(cameraSpacing_, radius * 0.06f) : radius * 0.06f) * frustumScale_;
    // Through a photo, the cameras around it would hide the view: fade them.
    const float fade = snapped_ ? 0.3f : 1.f;
    const bool flash = session_ && state_ == TrainState::Running;
    std::optional<std::size_t> trained;
    if (flash) {
      const std::string &name = session_->data().trainViews[session_->last_view()].name;
      for (std::size_t i = 0; i < views_.size(); ++i) {
        if (views_[i].name == name) trained = i;
      }
    }
    for (std::size_t i = 0; i < views_.size(); ++i) {
      if (snapped_ == i) continue; // We are looking through it
      const PosedView &v = views_[i];
      glm::vec4 color = v.holdout ? glm::vec4(1.f, 0.6f, 0.2f, 0.8f) : glm::vec4(0.35f, 0.65f, 1.f, 0.65f);
      color.a *= fade;
      if (trained == i) color = {0.35f, 1.f, 0.45f, 1.f};
      if (hovered_ == i) color = {1.f, 1.f, 1.f, 1.f};
      const Camera &c = v.camera;
      const auto corner = [&](float px, float py) {
        const glm::vec3 local((px - c.cx) / c.fx * depth, (py - c.cy) / c.fy * depth, depth);
        return glm::vec3(v.toWorld * glm::vec4(local, 1.f));
      };
      const auto w = static_cast<float>(c.width), h = static_cast<float>(c.height);
      const glm::vec3 p0 = corner(0.f, 0.f), p1 = corner(w, 0.f), p2 = corner(w, h), p3 = corner(0.f, h);
      for (const glm::vec3 &p : {p0, p1, p2, p3}) lines_->line(c.position, p, color);
      lines_->line(p0, p1, color);
      lines_->line(p1, p2, color);
      lines_->line(p2, p3, color);
      lines_->line(p3, p0, color);
      // A tick on the top edge says which way is up in the photo.
      const glm::vec3 tip = corner(0.5f * w, -0.18f * h);
      lines_->line(corner(0.38f * w, 0.f), tip, color);
      lines_->line(tip, corner(0.62f * w, 0.f), color);
    }
  }
  if (dragging_ && !snapped_) {
    const float s = orbit_.distance * 0.03f;
    const glm::vec4 color{1.f, 1.f, 1.f, 0.8f};
    lines_->line(orbit_.target - a * s, orbit_.target + a * s, color);
    lines_->line(orbit_.target - b * s, orbit_.target + b * s, color);
    lines_->line(orbit_.target - u * s, orbit_.target + u * s, color);
  }
  (void)camera;
}

// --- Cameras ----------------------------------------------------------------

Camera Viewer::render_camera(VkExtent2D extent) const {
  const auto w = static_cast<int>(std::max(extent.width, 1u)), h = static_cast<int>(std::max(extent.height, 1u));
  if (snapped_) return fit_to(views_[*snapped_].camera, w, h);
  return orbit_.camera(w, h);
}

float Viewer::pixels_per_point() const {
  int w = 0, pw = 0, h = 0;
  SDL_GetWindowSize(window_, &w, &h);
  SDL_GetWindowSizeInPixels(window_, &pw, &h);
  return w > 0 ? static_cast<float>(pw) / static_cast<float>(w) : 1.f;
}

std::optional<std::size_t> Viewer::view_at(glm::vec2 pixel, float radius) const {
  const Camera camera = render_camera(swapchain_.extent());
  std::optional<std::size_t> best;
  float bestDistance = radius;
  for (std::size_t i = 0; i < views_.size(); ++i) {
    if (snapped_ == i) continue;
    const auto p = project(camera, views_[i].camera.position);
    if (!p) continue;
    const float d = glm::length(*p - pixel);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return best;
}

void Viewer::snap_to(std::size_t view) {
  if (view >= views_.size()) return;
  snapped_ = view;
  autoRotate_ = false;
  request_photo(view);
}

void Viewer::unsnap() {
  if (!snapped_) return;
  // Continue orbiting from where the photo was taken, around a point at the
  // depth of the scene's centre.
  const Camera &c = views_[*snapped_].camera;
  const glm::vec3 forward = forward_of(c);
  const glm::vec3 center(bounds_.center[0], bounds_.center[1], bounds_.center[2]);
  const float depth = std::max(glm::dot(center - c.position, forward), bounds_.radius * 0.2f);
  orbit_.fovDeg = glm::degrees(2.f * std::atan(0.5f * static_cast<float>(c.height) / c.fy));
  orbit_.look_along(c.position, forward, depth);
  snapped_.reset();
  photoWanted_.reset();
}

void Viewer::step_view(int delta) {
  if (views_.empty()) return;
  const auto n = static_cast<long long>(views_.size());
  const long long from = snapped_ ? static_cast<long long>(*snapped_) : (delta > 0 ? -1 : 0);
  snap_to(static_cast<std::size_t>(((from + delta) % n + n) % n));
}

void Viewer::request_photo(std::size_t view) {
  if (project_.empty()) return;
  photoWanted_ = view;
}

void Viewer::toast(std::string text, bool warning) {
  std::println("{}", text);
  toast_ = std::move(text);
  toastWarning_ = warning;
  toastTime_ = 4.0;
}

void Viewer::log(const std::string &line) {
  std::println("{}", line);
  log_.push_back(line);
  while (log_.size() > 200) log_.pop_front();
}

// --- UI ---------------------------------------------------------------------

bool Viewer::begin_corner(const char *id, int corner) {
  const ImGuiViewport *vp = ImGui::GetMainViewport();
  const float pad = 10.f * uiScale_;
  const ImVec2 pos((corner & 1) ? vp->WorkPos.x + vp->WorkSize.x - pad : vp->WorkPos.x + pad,
                   (corner & 2) ? vp->WorkPos.y + vp->WorkSize.y - pad : vp->WorkPos.y + pad);
  ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2((corner & 1) ? 1.f : 0.f, (corner & 2) ? 1.f : 0.f));
  constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
  return ImGui::Begin(id, nullptr, flags);
}

void Viewer::build_ui() {
  overlay_photo();
  if (renderer_->splat_count() == 0 && points_->splat_count() == 0 && state_ != TrainState::Loading) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const char *hint = "Drop a .ply, a .cgc or a project folder here";
    const ImVec2 size = ImGui::CalcTextSize(hint);
    ImGui::GetBackgroundDrawList()->AddText(ImVec2(vp->Pos.x + 0.5f * (vp->Size.x - size.x), vp->Pos.y + 0.5f * (vp->Size.y - size.y)),
                                            ImGui::GetColorU32(kDim), hint);
  }
  if (showUi_) {
    panel_scene();
    panel_help();
    panel_training();
    panel_view();
    if (hovered_) ImGui::SetTooltip("%s", views_[*hovered_].name.c_str());
  }
  overlay_toast();
}

// Top left: what is shown and how fast.
void Viewer::panel_scene() {
  if (begin_corner("##scene", 0)) {
    ImGui::TextColored(kAccent, "%s", sceneName_.c_str());
    const std::size_t count = renderer_->splat_count();
    if (count > 0) {
      ImGui::Text("%s splats", human(static_cast<double>(count)).c_str());
      ImGui::SameLine();
      ImGui::TextColored(kDim, "%s visible", human(static_cast<double>(visible_)).c_str());
    } else if (points_->splat_count() > 0) {
      ImGui::Text("%s sparse points", human(static_cast<double>(points_->splat_count())).c_str());
    }
    if (!views_.empty()) ImGui::TextColored(kDim, "%zu posed photos", views_.size());
    const VkExtent2D extent = swapchain_.extent();
    ImGui::TextColored(kDim, "%.0f fps  %.1f ms  %ux%u%s", 1000.0 / std::max(frameMs_, 0.01), frameMs_, extent.width,
                       extent.height, swapchain_.vsync() ? "" : "  no vsync");
    ImGui::TextColored(kDim, "%s", ctx_.info().name.c_str());

    if (snapped_) {
      ImGui::Separator();
      const PosedView &v = views_[*snapped_];
      ImGui::TextColored(v.holdout ? kWarn : kAccent, "Photo %zu/%zu%s", *snapped_ + 1, views_.size(), v.holdout ? "  held out" : "");
      ImGui::TextUnformatted(fs::path(v.name).filename().string().c_str());
      ImGui::TextColored(kDim, "%dx%d  f %.0f px", v.camera.width, v.camera.height, v.camera.fx);
      if (photoTexture_ != ImTextureID_Invalid && photoView_ == *snapped_) {
        const float width = 220.f * uiScale_;
        ImGui::Image(ImTextureRef(photoTexture_), ImVec2(width, width * photoHeight_ / std::max(photoWidth_, 1)));
        ImGui::TextColored(kDim, "hold X to compare");
      } else if (!project_.empty()) {
        ImGui::TextColored(kDim, "loading photo...");
      }
    }
  }
  ImGui::End();
}

// Top right: keys and mouse.
void Viewer::panel_help() {
  if (begin_corner("##help", 1)) {
    if (!showHelp_) {
      ImGui::TextColored(kDim, "H  help");
    } else {
      static constexpr const char *kRows[][2] = {
          {"L / R drag", "look / pan  +Ctrl: orbit"},
          {"wheel", "zoom   +Ctrl: field of view"},
          {"WASD QE", "move   +Shift: faster"},
          {"F R T", "frame / reset / turntable"},
          {"1 2", "splats / centres"},
          {"C P G", "cameras / points / grid"},
          {"[ ]  dbl-click", "look through a photo"},
          {"X", "hold to show the photo"},
          {"Space  Ctrl+S", "pause / save training"},
          {"F12  Tab  H", "screenshot / UI / help"},
          {"Esc", "leave photo / quit"},
      };
      if (ImGui::BeginTable("##keys", 2, ImGuiTableFlags_SizingFixedFit)) {
        for (const auto &row : kRows) {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TextColored(kAccent, "%s", row[0]);
          ImGui::TableNextColumn();
          ImGui::TextColored(kDim, "%s", row[1]);
        }
        ImGui::EndTable();
      }
      if (pad_ != nullptr) {
        ImGui::Separator();
        ImGui::TextColored(kDim, "Pad: L stick move, R stick orbit, triggers zoom,\nLB/RB photos, A reset, B leave, Y UI, X points");
      }
    }
  }
  ImGui::End();
}

// Bottom left: training, when there is something to train.
void Viewer::panel_training() {
  if (!model_ && !session_) return;
  if (!begin_corner("##training", 2)) {
    ImGui::End();
    return;
  }
  const float width = 300.f * uiScale_;
  switch (state_) {
  case TrainState::Idle:
  case TrainState::Failed: {
    if (state_ == TrainState::Failed) {
      ImGui::PushTextWrapPos(width);
      ImGui::TextColored(kWarn, "Training failed: %s", failure_.c_str());
      ImGui::PopTextWrapPos();
    }
    if (!ImGui::TreeNode("Train this project")) break;
    ImGui::PushItemWidth(110.f * uiScale_);
    ImGui::InputInt("iterations", &trainOptions_.iterations, 1000, 5000);
    ImGui::InputInt("max image size", &trainOptions_.maxSize, 100, 400);
    ImGui::InputInt("hold out every", &trainOptions_.holdout, 1, 4);
    ImGui::SliderInt("SH degree", &trainOptions_.shDegree, 0, 3);
    ImGui::PopItemWidth();
    trainOptions_.iterations = std::max(trainOptions_.iterations, 1);
    trainOptions_.maxSize = std::max(trainOptions_.maxSize, 0);
    trainOptions_.holdout = std::max(trainOptions_.holdout, 0);
    // CGC is far smaller; training itself is the same either way.
    int format = cloud_format(trainOptions_.out) == CloudFormat::Cgc ? 1 : 0;
    ImGui::TextColored(kDim, "save as");
    ImGui::SameLine();
    const bool ply = ImGui::RadioButton(".ply", &format, 0);
    ImGui::SameLine();
    const bool cgc = ImGui::RadioButton(".cgc (compressed)", &format, 1);
    if (ply || cgc) trainOptions_.out.replace_extension(format == 1 ? ".cgc" : ".ply");
    ImGui::TextColored(kDim, "-> %s", trainOptions_.out.filename().string().c_str());
    if (ImGui::Button("Start training", ImVec2(width * 0.6f, 0.f))) pending_.push_back([this] { start_training(); });
    ImGui::TreePop();
    break;
  }
  case TrainState::Loading: {
    ImGui::TextColored(kAccent, "Loading photos");
    const auto done = loadDone_->load(), total = std::max<std::size_t>(loadTotal_->load(), 1);
    ImGui::ProgressBar(static_cast<float>(done) / static_cast<float>(total), ImVec2(width, 0.f),
                       std::format("{} / {}", done, total).c_str());
    break;
  }
  case TrainState::Running:
  case TrainState::Paused:
  case TrainState::Finished: {
    if (!session_) break;
    const TrainSession &s = *session_;
    const int iterations = s.options().iterations;
    const double rate = s.iteration() / std::max(s.training_seconds(), 1e-9);
    ImGui::TextColored(kAccent, state_ == TrainState::Finished ? "Training finished" : state_ == TrainState::Paused ? "Training paused" : "Training");
    if (state_ != TrainState::Finished) {
      ImGui::SameLine();
      ImGui::TextColored(kDim, "ETA %s", clock_text((iterations - s.iteration()) / std::max(rate, 1e-9)).c_str());
    }
    ImGui::ProgressBar(static_cast<float>(s.iteration()) / static_cast<float>(iterations), ImVec2(width, 0.f),
                       std::format("{} / {}", s.iteration(), iterations).c_str());
    ImGui::Text("loss %.4f", lastStep_.loss);
    ImGui::SameLine();
    if (state_ == TrainState::Finished) {
      ImGui::Text("PSNR %.2f dB", finalReport_.trainPsnr);
      if (finalReport_.holdoutPsnr > 0.0) {
        ImGui::SameLine();
        ImGui::TextColored(kWarn, "held out %.2f dB", finalReport_.holdoutPsnr);
      }
    } else if (s.last_psnr() > 0.0) {
      ImGui::Text("PSNR %.2f dB", s.last_psnr());
    }
    ImGui::TextColored(kDim, "%s Gaussians  %.2f / %.1f GiB  %.0f it/s  SH %d",
                       human(static_cast<double>(s.trainer().gaussian_count())).c_str(),
                       static_cast<double>(s.trainer().gpu_bytes()) / (1024.0 * 1024.0 * 1024.0), s.budget_gib(), rate,
                       s.trainer().active_sh_degree());
    if (lossHistory_.size() > 1) {
      ImGui::PlotLines("##loss", lossHistory_.data(), static_cast<int>(lossHistory_.size()), 0, "loss", 0.f, FLT_MAX,
                       ImVec2(width, 46.f * uiScale_));
    }
    if (state_ != TrainState::Finished) {
      if (ImGui::Button(state_ == TrainState::Paused ? "Resume" : "Pause")) {
        state_ = state_ == TrainState::Paused ? TrainState::Running : TrainState::Paused;
      }
      ImGui::SameLine();
      if (ImGui::Button("Save")) {
        pending_.push_back([this] {
          session_->save(session_->options().out);
          toast(std::format("Saved iteration {} to {}", session_->iteration(), session_->options().out.filename().string()));
        });
      }
      ImGui::SameLine();
      if (ImGui::Button("Finish now")) pending_.push_back([this] { stop_training(true); });
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::SliderFloat("##budget", &trainBudgetMs_, 5.f, 200.f, "%.0f ms/frame", ImGuiSliderFlags_Logarithmic);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Training time per frame: more trains faster, less keeps the view smooth");
    } else {
      ImGui::TextColored(kDim, "-> %s", s.options().out.filename().string().c_str());
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", s.options().out.string().c_str());
      if (ImGui::Button("Train again")) pending_.push_back([this] {
        state_ = TrainState::Idle;
        start_training();
      });
    }
    if (!log_.empty() && ImGui::TreeNode("Log")) {
      ImGui::BeginChild("##log", ImVec2(width * 1.6f, 120.f * uiScale_), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
      for (const std::string &line : log_) ImGui::TextUnformatted(line.c_str());
      if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) ImGui::SetScrollHereY(1.f);
      ImGui::EndChild();
      ImGui::TreePop();
    }
    break;
  }
  }
  ImGui::End();
}

// Bottom right: how to draw.
void Viewer::panel_view() {
  if (!begin_corner("##view", 3)) {
    ImGui::End();
    return;
  }
  if (!showSettings_) {
    if (ImGui::SmallButton("View settings")) showSettings_ = true;
    ImGui::End();
    return;
  }
  ImGui::PushItemWidth(150.f * uiScale_);
  int mode = pointMode_ ? 1 : 0;
  ImGui::RadioButton("splats", &mode, 0);
  ImGui::SameLine();
  ImGui::RadioButton("centres", &mode, 1);
  pointMode_ = mode == 1;
  if (pointMode_ || showPoints_) ImGui::SliderFloat("point size", &pointRadius_, 0.5f, 8.f, "%.1f px");
  if (!pointMode_) ImGui::SliderFloat("splat scale", &splatScale_, 0.05f, 2.f, "%.2f");
  ImGui::SliderInt("SH degree", &shDegree_, 0, 3);
  float fov = orbit_.fovDeg;
  if (ImGui::SliderFloat("field of view", &fov, 10.f, 120.f, "%.0f deg")) {
    unsnap();
    orbit_.fovDeg = fov;
  }
  ImGui::ColorEdit3("background", background_, ImGuiColorEditFlags_NoInputs);
  ImGui::Checkbox("cameras", &showCameras_);
  ImGui::SameLine();
  ImGui::Checkbox("points", &showPoints_);
  ImGui::SameLine();
  ImGui::Checkbox("grid", &showGrid_);
  ImGui::Checkbox("axes", &showAxes_);
  ImGui::SameLine();
  ImGui::Checkbox("turntable", &autoRotate_);
  ImGui::SameLine();
  ImGui::Checkbox("vsync", &vsync_);
  if (showCameras_) ImGui::SliderFloat("camera size", &frustumScale_, 0.1f, 5.f, "%.1fx", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  if (ImGui::SmallButton("Screenshot")) screenshotWanted_ = true;
  ImGui::SameLine();
  if (ImGui::SmallButton("Reset view")) {
    unsnap();
    orbit_ = home_;
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Hide")) showSettings_ = false;
  ImGui::End();
}

// The snapped view's photo over the whole render, while X is held.
void Viewer::overlay_photo() {
  if (!showPhoto_ || !snapped_ || photoTexture_ == ImTextureID_Invalid || photoView_ != *snapped_) return;
  const ImGuiViewport *vp = ImGui::GetMainViewport();
  const Camera &c = views_[*snapped_].camera;
  const float s = std::min(vp->Size.x / static_cast<float>(c.width), vp->Size.y / static_cast<float>(c.height));
  const ImVec2 size(static_cast<float>(c.width) * s, static_cast<float>(c.height) * s);
  const ImVec2 min(vp->Pos.x + 0.5f * (vp->Size.x - size.x), vp->Pos.y + 0.5f * (vp->Size.y - size.y));
  ImGui::GetBackgroundDrawList()->AddImage(ImTextureRef(photoTexture_), min, ImVec2(min.x + size.x, min.y + size.y));
}

void Viewer::overlay_toast() {
  if (toastTime_ <= 0.0 || toast_.empty()) return;
  const ImGuiViewport *vp = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y - 10.f * uiScale_),
                          ImGuiCond_Always, ImVec2(0.5f, 1.f));
  ImGui::SetNextWindowBgAlpha(0.85f * static_cast<float>(std::min(toastTime_, 1.0)));
  constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
  if (ImGui::Begin("##toast", nullptr, flags)) {
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, static_cast<float>(std::min(toastTime_, 1.0)));
    ImGui::TextColored(toastWarning_ ? kWarn : ImVec4(1.f, 1.f, 1.f, 1.f), "%s", toast_.c_str());
    ImGui::PopStyleVar();
  }
  ImGui::End();
}

} // namespace

fs::path project_scene(const fs::path &project) {
  fs::path best;
  for (const char *name : {"splat.ply", "splat.cgc"}) {
    const fs::path candidate = project / name;
    std::error_code error;
    if (!fs::is_regular_file(candidate, error)) continue;
    if (best.empty() || fs::last_write_time(candidate, error) > fs::last_write_time(best, error)) best = candidate;
  }
  return best;
}

int run_viewer(const ViewerOptions &options) {
  SdlSession sdl;
  std::unique_ptr<SDL_Window, WindowDeleter> window(
      SDL_CreateWindow("splat", 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY));
  if (!window) {
    throw std::runtime_error(std::format("Could not create a window: {}", SDL_GetError()));
  }
  const std::string title = !options.scene.empty() ? options.scene.string() : options.project.string();
  if (!title.empty()) SDL_SetWindowTitle(window.get(), std::format("splat - {}", title).c_str());

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

  // Scope: everything Vulkan and ImGui is destroyed before the window.
  {
    vk::Context ctx(co);
    std::println("Device: {} ({})", ctx.info().name, vk::vendor_name(ctx.info().vendorId));
    ImGuiSession imgui;
    Viewer viewer(options, window.get(), ctx);
    viewer.run();
  }
  return 0;
}

} // namespace splat
