#include "splat/web_viewer.hpp"

#include <format>
#include <fstream>
#include <stdexcept>

namespace splat {

namespace embedded {
extern const std::span<const std::uint8_t> viewer_html; // generated from web/viewer.html
}

namespace {

// The page's empty scene tag; viewer.html fills the same tag when it saves itself.
constexpr std::string_view kSceneTag = R"(<script id="scene" type="application/octet-stream"></script>)";

void append_base64(std::string &out, std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  out.reserve(out.size() + (bytes.size() + 2) / 3 * 4);
  std::size_t i = 0;
  for (; i + 3 <= bytes.size(); i += 3) {
    const std::uint32_t v = std::uint32_t{bytes[i]} << 16 | std::uint32_t{bytes[i + 1]} << 8 | bytes[i + 2];
    out += {digits[v >> 18], digits[(v >> 12) & 63], digits[(v >> 6) & 63], digits[v & 63]};
  }
  if (const std::size_t rest = bytes.size() - i; rest > 0) {
    const std::uint32_t v = std::uint32_t{bytes[i]} << 16 | (rest == 2 ? std::uint32_t{bytes[i + 1]} << 8 : 0u);
    out += {digits[v >> 18], digits[(v >> 12) & 63], rest == 2 ? digits[(v >> 6) & 63] : '=', '='};
  }
}

} // namespace

std::string scene_html(std::span<const std::uint8_t> scene, std::string_view name) {
  const std::string_view page(reinterpret_cast<const char *>(embedded::viewer_html.data()), embedded::viewer_html.size());
  const std::size_t at = page.find(kSceneTag);
  if (at == std::string_view::npos) throw std::runtime_error("web/viewer.html has no empty scene tag");
  std::string attribute;
  for (const char c : name) {
    if (c == '&' || c == '"' || c == '<') attribute += std::format("&#{};", static_cast<int>(c));
    else attribute += c;
  }
  std::string html(page.substr(0, at));
  html += std::format(R"(<script id="scene" type="application/octet-stream" data-name="{}">)", attribute);
  append_base64(html, scene);
  html += "</script>";
  html += page.substr(at + kSceneTag.size());
  return html;
}

void save_scene_html(const std::filesystem::path &path, std::span<const std::uint8_t> scene, std::string_view name) {
  const std::string html = scene_html(scene, name);
  std::ofstream out(path, std::ios::binary);
  out.write(html.data(), static_cast<std::streamsize>(html.size()));
  if (!out) throw std::runtime_error(std::format("Failed to write '{}'", path.string()));
}

} // namespace splat
