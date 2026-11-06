#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace splat {

// An 8-bit RGB image, tightly packed, top row first.
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels; // width * height * 3

  [[nodiscard]] bool empty() const noexcept { return pixels.empty(); }
};

// Decodes JPEG, PNG, BMP or TGA from memory. Throws std::runtime_error.
[[nodiscard]] RgbImage decode_image(std::span<const std::uint8_t> encoded);

// Writes a JPEG (quality 1..100) or PNG, chosen from the file extension.
// Throws std::runtime_error.
void write_image(const RgbImage &image, const std::filesystem::path &path, int jpegQuality);

// Applies an EXIF orientation (1..8), so the result needs no further rotation.
[[nodiscard]] RgbImage apply_orientation(RgbImage image, int orientation);

// True if an orientation swaps width and height.
[[nodiscard]] constexpr bool orientation_swaps_axes(int orientation) noexcept { return orientation >= 5; }

// Shrinks with an area filter so the longer side is at most maxSize. Returns
// the image unchanged if it already fits, or if maxSize is 0.
[[nodiscard]] RgbImage downscale_to_fit(RgbImage image, int maxSize);

// Variance of the Laplacian of an 8-bit grey image. Higher means sharper.
[[nodiscard]] double sharpness(std::span<const std::uint8_t> grey, int width, int height, std::ptrdiff_t stride);

} // namespace splat
