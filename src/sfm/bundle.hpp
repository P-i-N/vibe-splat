#pragma once

#include "sfm/geometry.hpp"

#include <vector>

namespace splat {

// A camera lens shared by several views: COLMAP's SIMPLE_RADIAL model.
struct Lens {
  double f = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  double k = 0.0;
  double fPrior = 1.0;      // Pulled towards this, weakly, when f is refined
  double fPriorSigma = 1.0; // In the same units as f
  bool fixedF = false;
};

struct BundleView {
  Pose pose;
  int lens = 0;
  bool fixed = false;
};

struct BundleObservation {
  int view = 0;
  int point = 0;
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct BundleOptions {
  int iterations = 20;
  double huberPixels = 2.0;
  bool refineLenses = true;
};

struct BundleResult {
  double initialRms = 0.0; // Pixels
  double finalRms = 0.0;
  int iterations = 0;
};

// Projects a camera-space point. False if it is behind the camera.
[[nodiscard]] bool project_point(const Lens &lens, const Eigen::Vector3d &cameraSpace, Eigen::Vector2d &pixel);

// Normalised, undistorted coordinates (x, y, 1) of a pixel.
[[nodiscard]] Eigen::Vector3d unproject_pixel(const Lens &lens, const Eigen::Vector2d &pixel);

// Levenberg-Marquardt on the reprojection error with a Huber loss, eliminating
// the points through the Schur complement. Updates views, lenses and points.
BundleResult bundle_adjust(std::vector<Lens> &lenses, std::vector<BundleView> &views, std::vector<Eigen::Vector3d> &points,
                           const std::vector<BundleObservation> &observations, const BundleOptions &options);

// The same for one view alone, with points held fixed.
void refine_pose(const Lens &lens, Pose &pose, const std::vector<Eigen::Vector3d> &points,
                 const std::vector<Eigen::Vector2d> &pixels);

} // namespace splat
