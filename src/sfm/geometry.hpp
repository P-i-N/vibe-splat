#pragma once

#include <Eigen/Dense>

#include <array>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace splat {

// A world-to-camera pose: x_cam = R * x_world + t.
struct Pose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  [[nodiscard]] Eigen::Vector3d center() const { return -R.transpose() * t; }
};

// Bearings below are normalised image coordinates (x, y, 1), free of lens distortion.

// Essential matrices satisfying x2^T E x1 = 0 for five correspondences, using
// the action-matrix solution of Stewenius, Engels and Nister (2006). Unlike
// the eight-point method it also handles planar scenes. At most 10 results.
[[nodiscard]] std::vector<Eigen::Matrix3d> essential_five_point(const std::array<Eigen::Vector3d, 5> &x1,
                                                                const std::array<Eigen::Vector3d, 5> &x2);

// First-order geometric error of a correspondence, squared, in normalised units.
[[nodiscard]] double sampson_squared(const Eigen::Matrix3d &E, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2);

struct EssentialResult {
  Pose pose;                  // Second camera relative to the first; |t| = 1
  std::vector<int> inliers;   // Indices of correspondences in front of both cameras
  double medianAngleDeg = 0.; // Median triangulation angle of the inliers
};

// RANSAC around the five-point solver. threshold is in normalised units.
[[nodiscard]] std::optional<EssentialResult> estimate_relative_pose(const std::vector<Eigen::Vector3d> &x1,
                                                                    const std::vector<Eigen::Vector3d> &x2,
                                                                    double threshold, std::mt19937 &rng);

// Linear (DLT) triangulation from two or more views.
[[nodiscard]] Eigen::Vector3d triangulate(const std::vector<Pose> &poses, const std::vector<Eigen::Vector3d> &bearings);

// Camera poses from three 2D-3D correspondences (Grunert's method, as
// described by Haralick et al. 1994). At most 4 results.
[[nodiscard]] std::vector<Pose> solve_p3p(const std::array<Eigen::Vector3d, 3> &bearings,
                                          const std::array<Eigen::Vector3d, 3> &points);

struct AbsolutePoseResult {
  Pose pose;
  std::vector<int> inliers;
};

// RANSAC around P3P. threshold is in normalised units.
[[nodiscard]] std::optional<AbsolutePoseResult> estimate_absolute_pose(const std::vector<Eigen::Vector3d> &bearings,
                                                                       const std::vector<Eigen::Vector3d> &points,
                                                                       double threshold, std::mt19937 &rng);

// Rotation matrix for a small rotation vector (Rodrigues).
[[nodiscard]] Eigen::Matrix3d rotation_from_vector(const Eigen::Vector3d &w);

} // namespace splat
