#pragma once

#include "ingest/exif.hpp"
#include "ingest/video.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace splat {

// Camera information given by hand. Pixel values refer to the input image
// after rotation and before --max-size, and are scaled along with the image.
struct ManualIntrinsics {
  std::optional<double> focalPx;
  std::optional<double> focalMm;       // Needs sensorWidthMm
  std::optional<double> sensorWidthMm; // Along the image's longer side
  std::optional<double> focal35mm;
  std::optional<double> fovDeg; // Horizontal
  std::optional<double> cx;
  std::optional<double> cy;

  [[nodiscard]] bool has_focal() const noexcept {
    return focalPx.has_value() || focalMm.has_value() || focal35mm.has_value() || fovDeg.has_value();
  }
};

enum class FocalSource {
  Manual,    // Given by hand
  Exif35mm,  // FocalLengthIn35mmFilm
  ExifPlane, // FocalLength and FocalPlaneResolution
  Default,   // A guess: 1.2 x the longer side, which camera-pose estimation refines
};

[[nodiscard]] const char *to_string(FocalSource source) noexcept;
[[nodiscard]] FocalSource focal_source_from_string(const std::string &text);

struct FocalEstimate {
  double pixels = 0.0;
  FocalSource source = FocalSource::Default;
};

// The focal length in pixels for an image of the given size, after rotation.
// Priority: manual, then the 35 mm equivalent, then focal plane resolution,
// then the default guess. Throws std::runtime_error on inconsistent manual values.
[[nodiscard]] FocalEstimate estimate_focal(int width, int height, const ExifData &exif, const ManualIntrinsics &manual);

// A pinhole camera, in pixels of the ingested images.
struct IngestCamera {
  int id = 0;
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  FocalSource focalSource = FocalSource::Default;
  std::string make;
  std::string model;
};

struct IngestImage {
  std::string file;  // Relative to the project folder, with forward slashes
  int cameraId = 0;
  std::string source; // The input file this came from
  std::optional<double> videoSeconds;
  std::string captureTime; // EXIF DateTimeOriginal, if any
};

// The result of ingest, stored as ingest.json in the project folder. It is
// what later pipeline stages read.
struct IngestManifest {
  std::vector<IngestCamera> cameras;
  std::vector<IngestImage> images;
};

void save_manifest(const IngestManifest &manifest, const std::filesystem::path &path);
[[nodiscard]] IngestManifest load_manifest(const std::filesystem::path &path);

struct IngestOptions {
  std::vector<std::filesystem::path> inputs; // Image files, video files or folders of them
  std::filesystem::path project;
  int maxSize = 1600;   // Longest side of stored images; 0 keeps the original size
  int jpegQuality = 95;
  bool png = false;     // Store PNG instead of JPEG
  bool force = false;   // Replace a previous ingest in this project
  int threads = 0;      // 0 uses all cores
  VideoOptions video;
  ManualIntrinsics manual;
};

struct IngestReport {
  std::size_t images = 0;
  std::size_t videoFrames = 0;
  std::size_t failed = 0; // Files skipped because they could not be read
  std::size_t cameras = 0;
};

// Turns inputs into project/images/* plus project/ingest.json. A file that
// cannot be read is reported on stderr and skipped. Throws std::runtime_error
// if nothing usable remains or the project cannot be written.
IngestReport run_ingest(const IngestOptions &options);

// Expands folders into their image and video files, sorted naturally (so
// img2 comes before img10). Exposed for testing.
[[nodiscard]] std::vector<std::filesystem::path> expand_inputs(const std::vector<std::filesystem::path> &inputs);

} // namespace splat
