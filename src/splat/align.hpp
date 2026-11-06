#pragma once

#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <cstddef>
#include <span>

#include <glm/glm.hpp>

namespace splat {

// The ground of a scene: the plane n.p = offset, with n pointing up, away
// from the ground and towards the cameras.
struct FloorEstimate {
  glm::dvec3 normal{0.0, -1.0, 0.0};
  double offset = 0.0;
  // False if no plane stood out and the floor is only the cameras' up
  // direction under the lowest Gaussians.
  bool fitted = false;
  std::size_t inliers = 0;
  std::size_t candidates = 0; // Gaussians the plane was fitted to
};

// Up from the cameras: people hold cameras level, so the camera x axes all lie
// in the horizontal plane, and roughly upright, which gives the sign. (0,-1,0)
// without cameras.
[[nodiscard]] glm::dvec3 estimate_up(std::span<const Camera> cameras);

// Finds the floor: the plane with the most opaque Gaussians on it (by RANSAC),
// favouring level ones, among those within 20 degrees of the cameras' up and
// with the cameras above it. On a table scan this is the table top.
// Deterministic.
[[nodiscard]] FloorEstimate estimate_floor(const GaussianCloud &cloud, std::span<const Camera> cameras);

// A rigid motion, p' = rotation * p + translation.
struct RigidTransform {
  glm::dmat3 rotation{1.0};
  glm::dvec3 translation{0.0};
};

// Puts the floor at y = 0 with -y up, the COLMAP convention the viewer and
// other 3DGS tools expect, and the scene's median position over the origin.
// Turns the scene as little as possible otherwise.
[[nodiscard]] RigidTransform floor_alignment(const FloorEstimate &floor, const GaussianCloud &cloud);

// Moves and turns every Gaussian, turning their orientations and the
// view-dependent colour with them, so the cloud looks the same from cameras
// moved the same way.
void transform_cloud(GaussianCloud &cloud, const RigidTransform &transform);

} // namespace splat
