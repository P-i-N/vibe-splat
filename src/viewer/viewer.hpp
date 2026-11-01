#pragma once

#include "core/vk_context.hpp"
#include "splat/gaussian_cloud.hpp"

#include <glm/glm.hpp>

#include <string>

namespace splat {

struct ViewerOptions {
  vk::ContextOptions context;
  glm::vec3 up{0.f, -1.f, 0.f};
  std::string title = "splat";
  bool vsync = true;
};

// Opens a window and shows the scene until the user closes it. Returns the
// process exit code.
int run_viewer(const GaussianCloud &cloud, const ViewerOptions &options);

} // namespace splat
