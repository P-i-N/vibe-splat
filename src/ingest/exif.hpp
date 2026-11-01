#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

namespace splat {

// The EXIF tags ingest cares about. Missing or malformed tags stay empty.
struct ExifData {
  std::string make;
  std::string model;
  std::string dateTimeOriginal; // "YYYY:MM:DD HH:MM:SS"
  int orientation = 1;          // 1..8, as in the EXIF specification
  std::optional<double> focalLengthMm;
  std::optional<double> focalLength35mm;
  std::optional<double> focalPlaneXResolution; // Pixels per millimetre
  std::optional<double> focalPlaneYResolution; // Pixels per millimetre
  std::optional<std::uint32_t> pixelXDimension;
  std::optional<std::uint32_t> pixelYDimension;
};

// Parses an EXIF/TIFF block starting at the "II" or "MM" byte-order mark.
// Never throws: corrupt data yields whatever was readable before the damage.
[[nodiscard]] ExifData parse_tiff_exif(std::span<const std::uint8_t> tiff);

// Finds the EXIF block in a JPEG file in memory. Returns defaults if there is none.
[[nodiscard]] ExifData parse_jpeg_exif(std::span<const std::uint8_t> jpeg);

} // namespace splat
