#pragma once

#include "core/vk_context.hpp"
#include "train/train.hpp"

#include <glm/glm.hpp>

#include <filesystem>
#include <optional>

namespace splat {

struct ViewerOptions {
  vk::ContextOptions context;
  glm::vec3 up{0.f, -1.f, 0.f};
  bool vsync = true;
  // What to open: a .ply, and a project folder whose sparse/0 gives the
  // camera poses and points, and which can be trained. Either may be empty.
  std::filesystem::path scene;
  std::filesystem::path project;
  // Starts training the project right away, with these options. Without
  // them, training can still be started from the window.
  std::optional<TrainOptions> train;
};

// Opens a window and shows the scene until the user closes it. Returns the
// process exit code.
int run_viewer(const ViewerOptions &options);

} // namespace splat
