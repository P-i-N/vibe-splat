#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace splat {

// The single-page browser viewer, web/viewer.html, with a scene built in, as
// its "Save as HTML" button writes it: `scene` is the content of a .cgc (or
// .ply) file and `name` the file name the page shows. The page opens from disk
// in any browser with WebGL 2.
[[nodiscard]] std::string scene_html(std::span<const std::uint8_t> scene, std::string_view name);

// Writes scene_html() to `path`. Throws std::runtime_error if it cannot.
void save_scene_html(const std::filesystem::path &path, std::span<const std::uint8_t> scene, std::string_view name);

} // namespace splat
