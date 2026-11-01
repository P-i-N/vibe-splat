#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace splat {

// A set of 3D Gaussians in the parameterisation used by the original 3D
// Gaussian Splatting paper and its .ply files. Values are stored raw, exactly
// as in the file:
//   position  xyz (world space)
//   scale     log of the standard deviations along the local axes
//   rotation  quaternion (w, x, y, z), not necessarily normalised
//   opacity   logit; the real opacity is sigmoid(opacity)
//   sh        spherical-harmonic colour coefficients, (degree+1)^2 per channel.
//             Stored per Gaussian as [coefficient][rgb]: sh[(i*coeffs + k)*3 + c].
//             Coefficient 0 is the DC term; colour = 0.28209479 * sh0 + 0.5 + ...
struct GaussianCloud {
  int shDegree = 0;
  std::vector<float> positions; // 3 per Gaussian
  std::vector<float> scales;    // 3 per Gaussian
  std::vector<float> rotations; // 4 per Gaussian
  std::vector<float> opacities; // 1 per Gaussian
  std::vector<float> sh;        // 3 * shCoeffs() per Gaussian

  [[nodiscard]] std::size_t size() const noexcept { return opacities.size(); }
  [[nodiscard]] bool empty() const noexcept { return opacities.empty(); }
  [[nodiscard]] int shCoeffs() const noexcept { return (shDegree + 1) * (shDegree + 1); }

  // Resizes every array for `count` Gaussians at the given SH degree (zeroed).
  void resize(std::size_t count, int degree);

  // Throws std::runtime_error if array sizes are inconsistent.
  void validate() const;
};

struct Bounds {
  float min[3]{};
  float max[3]{};
  float center[3]{}; // Median position: robust against far-away floaters
  float radius = 1.f; // Distance from center containing 90% of positions
};

[[nodiscard]] Bounds compute_bounds(const GaussianCloud &cloud);

// Reads a 3DGS .ply (binary little endian or ASCII). Unknown properties such
// as normals are ignored. SH degree is derived from the number of f_rest_*
// properties (0, 9, 24 or 45 -> degree 0..3). Throws std::runtime_error.
[[nodiscard]] GaussianCloud load_ply(const std::filesystem::path &path);

// Writes a binary little-endian .ply in the standard 3DGS layout, readable by
// other splat viewers.
void save_ply(const GaussianCloud &cloud, const std::filesystem::path &path);

// A small synthetic scene for demos and tests: a ring of coloured spheres on a
// checkered floor. About `count` Gaussians, SH degree 1.
[[nodiscard]] GaussianCloud make_test_scene(std::size_t count = 200'000, std::uint32_t seed = 1);

} // namespace splat
