#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace splat {

// A sparse reconstruction in COLMAP's binary format (cameras.bin, images.bin,
// points3D.bin). Poses map world to camera: x_cam = R(q) * x_world + t, with
// the camera looking along +z, x right and y down.
struct ColmapCamera {
  std::uint32_t id = 0;
  int model = 0; // COLMAP's model id, see colmap_model_name()
  std::uint64_t width = 0;
  std::uint64_t height = 0;
  std::vector<double> params;
};

struct ColmapObservation {
  glm::dvec2 xy{0.0};
  std::uint64_t pointId = 0; // kNoPoint if this keypoint is not triangulated
};

struct ColmapImage {
  std::uint32_t id = 0;
  std::array<double, 4> rotation{1.0, 0.0, 0.0, 0.0}; // Quaternion (w, x, y, z)
  glm::dvec3 translation{0.0};
  std::uint32_t cameraId = 0;
  std::string name;
  std::vector<ColmapObservation> observations;

  // Where the camera is, in world space.
  [[nodiscard]] glm::dvec3 center() const;
  // Transforms a world point into this camera's space.
  [[nodiscard]] glm::dvec3 to_camera(const glm::dvec3 &world) const;
};

struct ColmapTrackElement {
  std::uint32_t imageId = 0;
  std::uint32_t pointIndex = 0;
};

struct ColmapPoint {
  std::uint64_t id = 0;
  glm::dvec3 position{0.0};
  std::array<std::uint8_t, 3> color{};
  double error = 0.0;
  std::vector<ColmapTrackElement> track;
};

inline constexpr std::uint64_t kNoPoint = ~std::uint64_t{0};

struct ColmapModel {
  std::map<std::uint32_t, ColmapCamera> cameras;
  std::map<std::uint32_t, ColmapImage> images;
  std::vector<ColmapPoint> points;
};

// The model's name ("PINHOLE", ...), or nullptr if the id is unknown.
[[nodiscard]] const char *colmap_model_name(int model) noexcept;
// Number of parameters of a camera model, or -1 if the id is unknown.
[[nodiscard]] int colmap_model_param_count(int model) noexcept;

// Reads the three .bin files of a folder. Throws std::runtime_error on missing
// or corrupt files, and if the files disagree (an image using an unknown camera).
[[nodiscard]] ColmapModel load_colmap_model(const std::filesystem::path &folder);
void save_colmap_model(const ColmapModel &model, const std::filesystem::path &folder);

// Projects a camera-space point to pixels with the camera's full lens model.
// Returns false if the point is behind the camera or the model is unsupported.
[[nodiscard]] bool colmap_project(const ColmapCamera &camera, const glm::dvec3 &cameraSpace, glm::dvec2 &pixel);

struct ColmapStats {
  std::size_t cameras = 0;
  std::size_t images = 0;
  std::size_t points = 0;
  double meanTrackLength = 0.0;
  double meanObservationsPerImage = 0.0;
  double meanReprojectionError = 0.0; // Pixels, recomputed from the geometry
};

[[nodiscard]] ColmapStats compute_stats(const ColmapModel &model);

} // namespace splat
