#pragma once

#include "splat/gaussian_cloud.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace splat {

// What the GPU collected per Gaussian since the last reset.
struct DensifyStats {
  std::vector<float> gradientSum; // Sum of |dLoss/dScreenPosition| over views that saw it, per half image
  std::vector<float> visits;      // How many views saw it
  std::vector<float> maxRadius;   // Largest size on screen, in pixels
};

struct DensifyOptions {
  float gradientThreshold = 2e-4f;
  float denseScale = 0.0f;     // Larger than this along any axis: split; otherwise clone
  float pruneOpacity = 0.005f;
  bool pruneLarge = false;     // Also remove the oversized, below
  float maxScreenRadius = 20.f;
  float maxWorldScale = 0.0f;
  std::size_t maxGaussians = ~std::size_t{0};
};

struct DensifyResult {
  GaussianCloud cloud;
  // For each new Gaussian, the old one whose optimiser state it keeps, or -1
  // for a new one, which starts fresh.
  std::vector<std::int64_t> source;
  std::size_t cloned = 0;
  std::size_t split = 0;
  std::size_t pruned = 0;
  std::size_t heldBack = 0; // Candidates left out to stay within maxGaussians
};

// Adaptive density control, as in the 3DGS paper. A Gaussian whose position
// keeps wanting to move is covering something it cannot represent. If it is
// small the region lacks coverage, so it is cloned. If it is large it covers too
// much, so it becomes two smaller ones sampled from itself. Nearly transparent
// Gaussians are removed. When the limit is near, the highest gradients go first.
// Removes the given share of Gaussians, largest first: those cover the most
// tiles, so they free the most memory. For when a view would not fit the budget.
[[nodiscard]] DensifyResult remove_largest(const GaussianCloud &cloud, double share);

[[nodiscard]] DensifyResult densify_cloud(const GaussianCloud &cloud, const DensifyStats &stats, const DensifyOptions &options,
                                          std::mt19937 &rng);

} // namespace splat
