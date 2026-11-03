#pragma once

#include "core/vk_buffer.hpp"
#include "core/vk_image.hpp"

#include <imgui.h>

#include <cstdint>
#include <memory>
#include <span>

namespace splat {

// Draws Dear ImGui with Vulkan: dynamic rendering, push descriptors, and the
// textures ImGui asks for (ImGuiBackendFlags_RendererHasTextures). Written here
// rather than taken from ImGui's own backend, which links the Vulkan loader,
// while splat loads it at runtime.
class ImGuiRenderer {
public:
  ImGuiRenderer(vk::Context &ctx, VkFormat colorFormat);
  ~ImGuiRenderer();
  ImGuiRenderer(const ImGuiRenderer &) = delete;
  ImGuiRenderer &operator=(const ImGuiRenderer &) = delete;

  // Creates, updates and destroys ImGui's textures (the font atlas). Call after
  // ImGui::Render() and before recording, while no frame is in flight.
  void update_textures(ImDrawData *data);

  // Records the draw lists into an active rendering pass over the whole target.
  void record(VkCommandBuffer cmd, ImDrawData *data, VkExtent2D extent);

  // A texture for ImGui::Image(), from tightly packed RGBA8 or RGB8 pixels.
  // Waits for the upload. Destroy it while no frame using it is in flight.
  [[nodiscard]] ImTextureID create_texture(int width, int height, std::span<const std::uint8_t> pixels, int channels);
  void destroy_texture(ImTextureID id);

private:
  struct Texture {
    std::unique_ptr<vk::Image> image;
  };

  Texture *upload(Texture *texture, int width, int height, const std::uint8_t *rgba);
  void create_pipeline(VkFormat colorFormat);

  vk::Context &ctx_;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  std::unique_ptr<vk::Buffer> vertices_, indices_; // Host-visible, rewritten every frame
};

} // namespace splat
