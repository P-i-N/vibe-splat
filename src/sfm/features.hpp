#pragma once

#include "ingest/image.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace splat {

// A SIFT keypoint. Coordinates follow COLMAP: (0.5, 0.5) is the centre of the
// first pixel.
struct Keypoint {
  float x = 0.f;
  float y = 0.f;
  float scale = 0.f; // Blur sigma in image pixels
  float angle = 0.f; // Radians
  std::array<std::uint8_t, 3> color{};
};

inline constexpr int kDescriptorSize = 128;

struct ImageFeatures {
  std::vector<Keypoint> keypoints;
  std::vector<float> descriptors; // kDescriptorSize per keypoint, RootSIFT, unit length

  [[nodiscard]] const float *descriptor(std::size_t i) const { return descriptors.data() + i * kDescriptorSize; }
};

// Detects difference-of-Gaussian extrema and describes them with SIFT
// (Lowe 2004), keeping the maxFeatures with the strongest contrast.
[[nodiscard]] ImageFeatures extract_features(const RgbImage &image, int maxFeatures);

// Mutual nearest neighbours that also pass Lowe's ratio test. Brute force.
[[nodiscard]] std::vector<std::pair<int, int>> match_features(const ImageFeatures &a, const ImageFeatures &b,
                                                              float ratio = 0.8f);

} // namespace splat
