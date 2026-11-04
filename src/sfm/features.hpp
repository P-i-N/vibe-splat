#pragma once

#include "ingest/image.hpp"

#include <Eigen/Core>

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

// Matches between two images whose relative pose is known, through the
// essential matrix E (x_b^T E x_a = 0 for normalised, undistorted coordinates
// x = (x, y, 1)). A feature only competes with features of the other image that
// lie within the threshold of its epipolar line (in normalised units of that
// image, so pixels / focal length), and the ratio test, mutual check and a
// minimum descriptor similarity apply among those. Repeated patterns and weak
// texture, which lose the ratio test against the whole image, survive it here.
[[nodiscard]] std::vector<std::pair<int, int>> match_features_guided(const ImageFeatures &a, const ImageFeatures &b,
                                                                     const std::vector<Eigen::Vector2d> &pointsA,
                                                                     const std::vector<Eigen::Vector2d> &pointsB,
                                                                     const Eigen::Matrix3d &E, double thresholdA,
                                                                     double thresholdB, float ratio = 0.8f);

} // namespace splat
