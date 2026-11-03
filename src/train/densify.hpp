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
  std::vector<float> contribution; // Its blending weight T alpha summed over the pixels of those views
  std::vector<float> absoluteGradientSum; // As gradientSum, but pixels pulling opposite ways do not cancel
  std::vector<float> markedPixels; // Over the views scored, the high-error pixels it reached
  std::vector<float> markedLoss;   // The same, each view's count weighted by its loss
};

struct DensifyOptions {
  float gradientThreshold = 2e-4f;
  float denseScale = 0.0f;     // Larger than this along any axis: split; otherwise clone
  float pruneOpacity = 0.005f;
  bool pruneLarge = false;     // Also remove the oversized, below
  float maxScreenRadius = 20.f;
  float maxWorldScale = 0.0f;
  std::size_t maxGaussians = ~std::size_t{0};
  // Above 0, large Gaussians split on the gradient that does not cancel, against
  // this limit (AbsGS). Small ones still clone on the other.
  float splitThreshold = 0.0f;
  // Above 0, the views scored (FastGS): only Gaussians reaching more than
  // minMarkedPixels high-error pixels per view densify at all.
  int scoredViews = 0;
  float minMarkedPixels = 5.f;
  // The share of the too transparent actually removed: below 1, those adding
  // least to the images, or with scored views those in the worst pixels.
  double transparentShare = 1.0;
  // Each round, regardless of gradients (the blur trap paper): this many
  // random large Gaussians split, and this many new ones are placed at random
  // in the cloud's bounding box. Places gradients never point to get a chance.
  std::size_t exploreSplits = 0;
  std::size_t exploreSeeds = 0;
};

struct DensifyResult {
  GaussianCloud cloud;
  // For each new Gaussian, the old one whose optimiser state it keeps, or -1
  // for a new one, which starts fresh.
  std::vector<std::int64_t> source;
  std::size_t cloned = 0;
  std::size_t split = 0;
  std::size_t seeded = 0;
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

// Removes the given share of Gaussians that add least to the images, by their
// blending weight summed over every pixel of every training view (the global
// significance of LightGaussian and Mini-Splatting). Most of a trained cloud
// adds almost nothing: faint, hidden or tiny Gaussians the others cover.
[[nodiscard]] DensifyResult remove_least_contributing(const GaussianCloud &cloud, const std::vector<float> &contribution, double share);

[[nodiscard]] DensifyResult densify_cloud(const GaussianCloud &cloud, const DensifyStats &stats, const DensifyOptions &options,
                                          std::mt19937 &rng);

struct RelocateOptions {
  float minOpacity = 0.005f; // At or below: dead, to be relocated
  double growth = 0.05;      // Share of the count added per round, within maxGaussians
  std::size_t maxGaussians = ~std::size_t{0};
};

// 3DGS-MCMC: dead Gaussians move onto live ones drawn in proportion to opacity,
// then the count grows by `growth` the same way. A target that becomes N copies
// gets opacity o' = 1 - (1 - o)^(1/N) and scale s * o / D, with
// D = sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1), as in
// gsplat. Every copy and its target start with fresh optimiser moments.
// `cloned` counts the relocated dead, `seeded` the growth.
[[nodiscard]] DensifyResult relocate_cloud(const GaussianCloud &cloud, const RelocateOptions &options, std::mt19937 &rng);

} // namespace splat
