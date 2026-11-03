#include "reference_helpers.hpp"
#include "sfm/test_views.hpp"
#include "test_support.hpp"
#include "train/densify.hpp"
#include "train/mip.hpp"
#include "train/dense_points.hpp"
#include "train/reference.hpp"
#include "train/schedule.hpp"
#include "train/train.hpp"
#include "train/trainer.hpp"
#include "ingest/ingest.hpp"
#include "sfm/sfm.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <print>
#include <random>

using namespace splat;
using Catch::Matchers::WithinAbs;

namespace {

// Gaussians of known size and opacity at distinct x positions.
GaussianCloud line_of(std::size_t count, float scale, float opacity = 0.5f, int degree = 0) {
  GaussianCloud cloud;
  cloud.resize(count, degree);
  for (std::size_t i = 0; i < count; ++i) {
    cloud.positions[i * 3] = static_cast<float>(i);
    for (std::size_t a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = std::log(scale);
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = std::log(opacity / (1.f - opacity));
    cloud.sh[i * static_cast<std::size_t>(cloud.shCoeffs()) * 3] = static_cast<float>(i);
  }
  return cloud;
}

DensifyStats stats_of(std::size_t count, float gradient) {
  DensifyStats stats;
  stats.gradientSum.assign(count, gradient);
  stats.visits.assign(count, 1.f);
  stats.maxRadius.assign(count, 1.f);
  return stats;
}

DensifyOptions default_options() {
  DensifyOptions options;
  options.gradientThreshold = 1.f;
  options.denseScale = 0.1f;
  options.maxWorldScale = 10.f;
  return options;
}

} // namespace

TEST_CASE("Low gradients leave the cloud alone", "[densify]") {
  std::mt19937 rng(1);
  const GaussianCloud cloud = line_of(4, 0.05f);
  const DensifyResult r = densify_cloud(cloud, stats_of(4, 0.5f), default_options(), rng);
  CHECK(r.cloud.size() == 4);
  CHECK(r.cloned == 0);
  CHECK(r.split == 0);
  CHECK(r.cloud.positions == cloud.positions);
  CHECK(r.source == std::vector<std::int64_t>{0, 1, 2, 3});
}

TEST_CASE("Small Gaussians with high gradients are cloned", "[densify]") {
  std::mt19937 rng(1);
  const GaussianCloud cloud = line_of(3, 0.05f);
  DensifyStats stats = stats_of(3, 0.f);
  stats.gradientSum[1] = 4.f;
  stats.visits[1] = 2.f; // Average 2, above the threshold of 1
  const DensifyResult r = densify_cloud(cloud, stats, default_options(), rng);
  CHECK(r.cloned == 1);
  CHECK(r.split == 0);
  REQUIRE(r.cloud.size() == 4);
  CHECK(r.source == std::vector<std::int64_t>{0, 1, 2, -1});
  CHECK(r.cloud.positions[3 * 3] == cloud.positions[1 * 3]);
  CHECK(r.cloud.scales[3 * 3] == cloud.scales[1 * 3]);
  r.cloud.validate();
}

TEST_CASE("The average decides, not the sum", "[densify]") {
  std::mt19937 rng(1);
  DensifyStats stats = stats_of(1, 10.f);
  stats.visits[0] = 100.f;
  CHECK(densify_cloud(line_of(1, 0.05f), stats, default_options(), rng).cloned == 0);
  // Never seen: no division by zero, no densification.
  stats = stats_of(1, 0.f);
  stats.visits[0] = 0.f;
  CHECK(densify_cloud(line_of(1, 0.05f), stats, default_options(), rng).cloned == 0);
}

TEST_CASE("Large Gaussians with high gradients are split", "[densify]") {
  std::mt19937 rng(7);
  const GaussianCloud cloud = line_of(2, 1.f);
  DensifyStats stats = stats_of(2, 0.f);
  stats.gradientSum[0] = 2.f;
  const DensifyResult r = densify_cloud(cloud, stats, default_options(), rng);
  CHECK(r.split == 1);
  REQUIRE(r.cloud.size() == 3);
  CHECK(r.source == std::vector<std::int64_t>{1, -1, -1});
  // The children are smaller by 1.6, and their position comes from the parent's distribution.
  CHECK_THAT(std::exp(r.cloud.scales[1 * 3]), WithinAbs(1.f / 1.6f, 1e-5));
  CHECK_THAT(std::exp(r.cloud.scales[2 * 3]), WithinAbs(1.f / 1.6f, 1e-5));
  CHECK(r.cloud.positions[1 * 3] != r.cloud.positions[2 * 3]);
  CHECK(std::abs(r.cloud.positions[1 * 3]) < 6.f);
  CHECK(r.cloud.sh[1 * 3] == cloud.sh[0]);

  // Over many splits the samples have the parent's mean and spread, along its rotated axes.
  GaussianCloud stretched = line_of(1, 0.5f);
  stretched.scales[0] = std::log(2.f);        // Long along local x
  const float s = std::sqrt(0.5f);
  stretched.rotations[0] = s;
  stretched.rotations[3] = s;                 // 90 degrees around z: local x is world y
  stretched.positions[1] = 3.f;
  double sumY = 0.0, squareY = 0.0, squareX = 0.0;
  constexpr int kSamples = 4000;
  for (int i = 0; i < kSamples; ++i) {
    const DensifyResult child = densify_cloud(stretched, stats_of(1, 2.f), default_options(), rng);
    const double y = child.cloud.positions[1] - 3.f;
    const double x = child.cloud.positions[0];
    sumY += y;
    squareY += y * y;
    squareX += x * x;
  }
  CHECK_THAT(sumY / kSamples, WithinAbs(0.0, 0.1));
  CHECK_THAT(std::sqrt(squareY / kSamples), WithinAbs(2.0, 0.1));
  CHECK_THAT(std::sqrt(squareX / kSamples), WithinAbs(0.5, 0.05));
}

TEST_CASE("Pruning removes the transparent, and the oversized when asked", "[densify]") {
  std::mt19937 rng(1);
  GaussianCloud cloud = line_of(4, 0.05f);
  cloud.opacities[0] = std::log(0.001f / 0.999f);
  cloud.scales[1 * 3 + 2] = std::log(20.f);
  DensifyStats stats = stats_of(4, 0.f);
  stats.maxRadius[2] = 50.f;

  DensifyOptions options = default_options();
  options.maxScreenRadius = 20.f;
  DensifyResult r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.pruned == 1);
  CHECK(r.source == std::vector<std::int64_t>{1, 2, 3});

  options.pruneLarge = true;
  r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.pruned == 3);
  CHECK(r.source == std::vector<std::int64_t>{3});
}

TEST_CASE("Pruning removes Gaussians with NaN or infinite values", "[densify]") {
  GaussianCloud cloud = line_of(4, 0.01f);
  cloud.positions[3] = std::nanf("");
  cloud.sh[2 * 3 + 1] = std::numeric_limits<float>::infinity();
  std::mt19937 rng(1);
  const DensifyResult result = densify_cloud(cloud, stats_of(4, 0.f), DensifyOptions{}, rng);
  CHECK(result.pruned == 2);
  REQUIRE(result.source.size() == 2);
  CHECK(result.source[0] == 0);
  CHECK(result.source[1] == 3);
}

TEST_CASE("The limit is kept, with the highest gradients first", "[densify]") {
  std::mt19937 rng(1);
  const GaussianCloud cloud = line_of(6, 0.05f);
  DensifyStats stats = stats_of(6, 0.f);
  stats.gradientSum = {2.f, 9.f, 3.f, 8.f, 0.f, 4.f};
  DensifyOptions options = default_options();
  options.maxGaussians = 8; // 6 kept + room for 2
  DensifyResult r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.cloud.size() == 8);
  CHECK(r.cloned == 2);
  CHECK(r.heldBack == 3);
  CHECK(r.cloud.positions[6 * 3] == 1.f); // The ones with gradient 9 and 8, in order
  CHECK(r.cloud.positions[7 * 3] == 3.f);

  // Already above the limit: nothing is added, and nothing good is thrown away.
  options.maxGaussians = 3;
  r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.cloud.size() == 6);
  CHECK(r.heldBack == 5);
}

TEST_CASE("Large Gaussians can split on the gradient that does not cancel", "[densify]") {
  std::mt19937 rng(1);
  GaussianCloud cloud = line_of(4, 1.f);
  for (std::size_t a = 0; a < 3; ++a) {
    cloud.scales[2 * 3 + a] = std::log(0.05f);
    cloud.scales[3 * 3 + a] = std::log(0.05f);
  }
  DensifyStats stats = stats_of(4, 0.f);
  stats.absoluteGradientSum = {9.f, 0.f, 9.f, 0.f};
  stats.gradientSum = {0.f, 9.f, 0.f, 9.f};
  DensifyOptions options = default_options();
  options.splitThreshold = 4.f;
  const DensifyResult r = densify_cloud(cloud, stats, options, rng);
  // Large: only the first, by the absolute sum. Small: only the last, by the signed one.
  CHECK(r.split == 1);
  CHECK(r.cloned == 1);
  CHECK(r.source == std::vector<std::int64_t>{1, 2, 3, -1, -1, -1});
}

TEST_CASE("With scored views only Gaussians in high-error pixels densify", "[densify]") {
  std::mt19937 rng(1);
  const GaussianCloud cloud = line_of(3, 0.05f);
  DensifyStats stats = stats_of(3, 2.f);
  stats.absoluteGradientSum.assign(3, 0.f);
  stats.markedPixels = {60.f, 49.f, 0.f};
  stats.markedLoss = {0.f, 0.f, 0.f};
  DensifyOptions options = default_options();
  options.scoredViews = 10;
  options.minMarkedPixels = 5.f; // More than 5 per view
  const DensifyResult r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.cloned == 1);
  CHECK(r.cloud.positions[3 * 3] == cloud.positions[0]);
}

TEST_CASE("A share of the transparent goes, the least useful first", "[densify]") {
  std::mt19937 rng(1);
  GaussianCloud cloud = line_of(5, 0.05f);
  for (std::size_t i = 0; i < 4; ++i) cloud.opacities[i] = std::log(0.001f / 0.999f);
  DensifyStats stats = stats_of(5, 0.f);
  stats.contribution = {4.f, 1.f, 3.f, 2.f, 0.f};
  DensifyOptions options = default_options();
  options.transparentShare = 0.5;
  DensifyResult r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.pruned == 2);
  CHECK(r.source == std::vector<std::int64_t>{0, 2, 4});

  // With scored views, those in the worst pixels instead.
  stats.markedPixels.assign(5, 0.f);
  stats.markedLoss = {4.f, 1.f, 3.f, 2.f, 9.f};
  options.scoredViews = 1;
  r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.source == std::vector<std::int64_t>{1, 3, 4});
}

TEST_CASE("Exploration splits large Gaussians and seeds new ones within the limit", "[densify]") {
  std::mt19937 rng(3);
  GaussianCloud cloud = line_of(8, 1.f, 0.5f, 1);
  for (std::size_t a = 0; a < 3; ++a) {
    cloud.scales[0 * 3 + a] = std::log(0.01f); // Too small to split
    cloud.scales[1 * 3 + a] = std::log(4.f);
  }
  const DensifyStats stats = stats_of(8, 0.f);
  DensifyOptions options = default_options();
  options.exploreSplits = 2;
  options.exploreSeeds = 3;
  DensifyResult r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.split == 2);
  CHECK(r.seeded == 3);
  REQUIRE(r.cloud.size() == 8 + 2 + 3);
  r.cloud.validate();
  CHECK(std::find(r.source.begin(), r.source.end(), std::int64_t{0}) != r.source.end());
  // The last three are the seeds: within the box, with one colour from all sides.
  const auto floats = static_cast<std::size_t>(3 * r.cloud.shCoeffs());
  for (std::size_t i = 10; i < 13; ++i) {
    CHECK(r.source[i] == -1);
    CHECK(r.cloud.positions[i * 3] >= 0.f);
    CHECK(r.cloud.positions[i * 3] <= 7.f);
    CHECK(r.cloud.sh[i * floats + 3] == 0.f);
    CHECK_THAT(1.f / (1.f + std::exp(-r.cloud.opacities[i])), WithinAbs(0.1f, 1e-5));
  }

  // At the limit, nothing is added.
  options.maxGaussians = 9;
  r = densify_cloud(cloud, stats, options, rng);
  CHECK(r.cloud.size() == 9);
  CHECK(r.seeded == 0);
}

TEST_CASE("The image sizes follow the photos' spectra", "[densify]") {
  constexpr int size = 64;
  std::mt19937 rng(9);
  std::vector<std::uint8_t> noise(size * size * 4), smooth(size * size * 4);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const auto value = static_cast<std::uint8_t>(127.5 * (1.0 + std::sin(0.2 * x) * std::cos(0.1 * y)));
      for (int c = 0; c < 3; ++c) {
        noise[static_cast<std::size_t>(y * size + x) * 4 + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(rng() & 255u);
        smooth[static_cast<std::size_t>(y * size + x) * 4 + static_cast<std::size_t>(c)] = value;
      }
    }
  }
  const auto switches = [&](const std::vector<std::uint8_t> &image) {
    const std::vector<double> x = frequency_energy(image, size, size, {1, 2, 4});
    CHECK(x[0] > x[1]);
    CHECK(x[1] > x[2]);
    return resolution_switches(x[0], x[1], x[2]);
  };
  // Noise is all high frequencies, so full size is needed at once. A smooth
  // image holds its energy low, so small images serve it well for long.
  const auto fromNoise = switches(noise), fromSmooth = switches(smooth);
  CHECK(fromNoise[0] < 0.25);
  CHECK(fromSmooth[0] > 0.7);
  CHECK(fromSmooth[1] > 0.4);
  CHECK(fromSmooth[1] <= fromSmooth[0]);

  // The natural case of a spectrum falling as 1/f: half the energy per halving.
  const auto natural = resolution_switches(4.0, 2.0, 1.0);
  CHECK_THAT(natural[0], WithinAbs(0.5, 1e-9));
  CHECK_THAT(natural[1], WithinAbs(0.0, 1e-9));
}

TEST_CASE("Gaussians are allowed to grow with the image size", "[densify]") {
  GrowthBudget budget(1000);
  CHECK(budget.final_count() == 5000.0);
  CHECK(budget.limit(1.0, 0.0) == 5000);
  CHECK(budget.limit(1.0, 0.9) == 5000);
  CHECK(budget.limit(4.0, 0.0) == 1000 + 4000 / 16);
  CHECK(budget.limit(2.0, 0.5) < budget.limit(1.0, 0.5));
  // Asking for much raises what the scene needs; asking for little does not lower it.
  budget.observe(1000);
  CHECK(budget.final_count() == 0.98 * 5000.0 + 1000.0);
  budget.observe(0);
  CHECK(budget.final_count() == 0.98 * 5000.0 + 1000.0);
}

TEST_CASE("A growth factor sets the final count", "[densify]") {
  GrowthBudget budget(1000, 8.0);
  CHECK(budget.final_count() == 8000.0);
  CHECK(budget.limit(1.0, 1.0) == 8000);
}

TEST_CASE("A growth factor of zero never caps", "[densify]") {
  GrowthBudget budget(1000, 0.0);
  CHECK(budget.limit(4.0, 0.0) == std::numeric_limits<std::size_t>::max());
  budget.observe(5);
  CHECK(budget.limit(1.0, 1.0) == std::numeric_limits<std::size_t>::max());
  CHECK(std::isinf(budget.final_count()));
}

namespace {
float logit(float p) { return std::log(p / (1.f - p)); }
// 3DGS-MCMC's scale divisor for a Gaussian that becomes n copies of opacity o.
double relocation_divisor(int n, double o) {
  double d = 0.0;
  for (int i = 1; i <= n; ++i) {
    for (int k = 0; k < i; ++k) {
      double binom = 1.0;
      for (int j = 1; j <= k; ++j) binom = binom * static_cast<double>(i - 1 - k + j) / j;
      d += binom * (k % 2 == 0 ? 1.0 : -1.0) * std::pow(o, k + 1) / std::sqrt(static_cast<double>(k + 1));
    }
  }
  return d;
}
} // namespace

TEST_CASE("A dead Gaussian is relocated onto a live one", "[densify]") {
  GaussianCloud cloud = line_of(2, 0.1f);
  cloud.opacities[0] = logit(0.9f);
  cloud.opacities[1] = logit(0.001f);
  std::mt19937 rng(1);
  RelocateOptions options;
  options.maxGaussians = 2;
  const DensifyResult result = relocate_cloud(cloud, options, rng);
  REQUIRE(result.cloud.size() == 2);
  CHECK(result.cloned == 1);
  CHECK(result.seeded == 0);
  const double newOpacity = 1.0 - std::sqrt(1.0 - 0.9);
  const double newScale = 0.1 * 0.9 / relocation_divisor(2, newOpacity);
  for (std::size_t i = 0; i < 2; ++i) {
    INFO("Gaussian " << i);
    CHECK(result.cloud.positions[i * 3] == 0.f);
    CHECK_THAT(1.0 / (1.0 + std::exp(-result.cloud.opacities[i])), WithinAbs(newOpacity, 1e-5));
    CHECK_THAT(std::exp(result.cloud.scales[i * 3]), WithinAbs(newScale, 1e-5));
    CHECK(result.source[i] == -1);
  }
}

TEST_CASE("Relocation targets are drawn in proportion to opacity", "[densify]") {
  GaussianCloud cloud = line_of(1002, 0.1f);
  cloud.opacities[0] = logit(0.9f);
  cloud.opacities[1] = logit(0.09f);
  for (std::size_t i = 2; i < 1002; ++i) cloud.opacities[i] = logit(0.001f);
  std::mt19937 rng(1);
  RelocateOptions options;
  options.maxGaussians = 1002;
  const DensifyResult result = relocate_cloud(cloud, options, rng);
  REQUIRE(result.cloud.size() == 1002);
  std::size_t onFirst = 0;
  for (std::size_t i = 2; i < 1002; ++i) onFirst += result.cloud.positions[i * 3] == 0.f ? 1 : 0;
  CHECK(onFirst > 850);
  CHECK(onFirst < 950);
}

TEST_CASE("Growth adds five percent within the cap", "[densify]") {
  const GaussianCloud cloud = line_of(1000, 0.1f);
  std::mt19937 rng(1);
  RelocateOptions options;
  options.maxGaussians = 1030;
  const DensifyResult grown = relocate_cloud(cloud, options, rng);
  CHECK(grown.cloud.size() == 1030);
  CHECK(grown.seeded == 30);
  CHECK(grown.cloned == 0);
  options.maxGaussians = 1000;
  const DensifyResult capped = relocate_cloud(cloud, options, rng);
  CHECK(capped.cloud.size() == 1000);
  CHECK(capped.seeded == 0);
}

TEST_CASE("Relocation with no live Gaussian leaves the cloud alone", "[densify]") {
  const GaussianCloud cloud = line_of(5, 0.1f, 0.001f);
  std::mt19937 rng(1);
  const DensifyResult result = relocate_cloud(cloud, RelocateOptions{}, rng);
  CHECK(result.cloud.positions == cloud.positions);
  CHECK(result.cloud.opacities == cloud.opacities);
  CHECK(result.cloned == 0);
}

TEST_CASE("Long-axis split places two children along the longest axis", "[densify]") {
  GaussianCloud cloud = line_of(1, 1.f);
  cloud.scales[1] = cloud.scales[2] = std::log(0.2f);
  DensifyStats stats = stats_of(1, 2.f);
  stats.edgeScore.assign(1, 1.f);
  DensifyOptions options = default_options();
  options.edgeSplit = true;
  std::mt19937 rng(1);
  const DensifyResult result = densify_cloud(cloud, stats, options, rng);
  REQUIRE(result.cloud.size() == 2);
  CHECK(result.split == 1);
  std::array<float, 2> xs{result.cloud.positions[0], result.cloud.positions[3]};
  std::ranges::sort(xs);
  CHECK_THAT(xs[0], WithinAbs(-0.45, 1e-6));
  CHECK_THAT(xs[1], WithinAbs(0.45, 1e-6));
  for (std::size_t i = 0; i < 2; ++i) {
    INFO("child " << i);
    CHECK_THAT(result.cloud.scales[i * 3], WithinAbs(std::log(0.55), 1e-6));
    CHECK_THAT(result.cloud.scales[i * 3 + 1], WithinAbs(std::log(0.2 * std::sqrt(1.0 - 0.45 * 0.45)), 1e-6));
    CHECK_THAT(result.cloud.scales[i * 3 + 2], WithinAbs(std::log(0.2 * std::sqrt(1.0 - 0.45 * 0.45)), 1e-6));
    CHECK_THAT(1.0 / (1.0 + std::exp(-result.cloud.opacities[i])), WithinAbs(0.3, 1e-6));
    CHECK(result.source[i] == -1);
  }
}

TEST_CASE("Edge scores set the split probability", "[densify]") {
  const GaussianCloud cloud = line_of(1000, 1.f);
  DensifyStats stats = stats_of(1000, 2.f);
  stats.edgeScore.assign(1000, 0.1f);
  for (std::size_t i = 0; i < 500; ++i) stats.edgeScore[i] = 1.f;
  DensifyOptions options = default_options();
  options.edgeSplit = true;
  std::mt19937 rng(1);
  const DensifyResult result = densify_cloud(cloud, stats, options, rng);
  CHECK(result.split >= 530);
  CHECK(result.split <= 570);
  // Every Gaussian with the top score split: none of them survives unchanged.
  std::size_t keptOfTop = 0;
  for (std::size_t i = 0; i < result.cloud.size(); ++i) {
    if (result.source[i] >= 0 && result.source[i] < 500) ++keptOfTop;
  }
  CHECK(keptOfTop == 0);
}

TEST_CASE("All edge scores zero splits every candidate", "[densify]") {
  const GaussianCloud cloud = line_of(100, 1.f);
  DensifyStats stats = stats_of(100, 2.f);
  stats.edgeScore.assign(100, 0.f);
  DensifyOptions options = default_options();
  options.edgeSplit = true;
  std::mt19937 rng(1);
  const DensifyResult result = densify_cloud(cloud, stats, options, rng);
  CHECK(result.split == 100);
  CHECK(result.cloud.size() == 200);
}

TEST_CASE("The smoothing filter is the smallest footprint of the cameras that see a Gaussian", "[densify]") {
  const auto camera_at = [](float z) {
    Camera camera = Camera::look_at(glm::vec3(0.f, 0.f, z), glm::vec3(0.f), glm::vec3(0.f, -1.f, 0.f), 60.f, 64, 48);
    camera.fx = camera.fy = 100.f;
    camera.cx = 32.f;
    camera.cy = 24.f;
    return camera;
  };
  const std::array<Camera, 2> cameras{camera_at(-2.f), camera_at(-4.f)};
  GaussianCloud cloud = line_of(2, 0.1f);
  cloud.positions[0] = 0.f;    // At the origin: depth 2 from the nearer camera
  cloud.positions[3] = 0.f;    // Behind both cameras
  cloud.positions[5] = -10.f;
  const std::vector<float> filter = smoothing_filter(cloud, cameras);
  REQUIRE(filter.size() == 2);
  CHECK_THAT(filter[0], WithinAbs(std::pow(0.2 * 2.0 / 100.0, 2.0), 1e-9));
  CHECK(filter[1] == 0.f);
}

TEST_CASE("Baking the filter keeps the Gaussian's integral", "[densify]") {
  GaussianCloud cloud = line_of(2, 0.1f);
  const GaussianCloud before = cloud;
  bake_filter(cloud, std::vector<float>{0.03f, 0.f});
  for (std::size_t a = 0; a < 3; ++a) CHECK_THAT(cloud.scales[a], WithinAbs(0.5 * std::log(0.01 + 0.03), 1e-6));
  CHECK_THAT(1.0 / (1.0 + std::exp(-cloud.opacities[0])), WithinAbs(0.5 * std::pow(0.01 / 0.04, 1.5), 1e-6));
  CHECK(cloud.scales[3] == before.scales[3]);
  CHECK(cloud.opacities[1] == before.opacities[1]);
}

TEST_CASE("Patch matching finds points on a textured plane", "[densify]") {
  // Blobs on the plane z = 5, seen by cameras shifted sideways.
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> where(-4.f, 4.f), shade(0.f, 1.f);
  struct Blob {
    float x, y, value;
  };
  std::vector<Blob> blobs(1800);
  for (Blob &b : blobs) b = {where(rng), where(rng), shade(rng)};
  constexpr int width = 200, height = 160;
  std::vector<std::vector<std::uint8_t>> images;
  std::vector<DensePointsView> views;
  std::vector<Camera> cameras;
  for (int v = 0; v < 5; ++v) {
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = camera.fy = 200.f;
    camera.cx = 100.f;
    camera.cy = 80.f;
    camera.position = glm::vec3(0.25f * static_cast<float>(v), 0.f, 0.f);
    camera.view = glm::mat4(1.f);
    camera.view[3] = glm::vec4(-camera.position, 1.f);
    cameras.push_back(camera);
    std::vector<std::uint8_t> rgba(width * height * 4);
    std::vector<float> sum(width * height, 0.f);
    for (const Blob &b : blobs) {
      const float px = 200.f * (b.x - camera.position.x) / 5.f + 100.f, py = 200.f * b.y / 5.f + 80.f;
      for (int y = std::max(0, static_cast<int>(py) - 6); y < std::min(height, static_cast<int>(py) + 7); ++y) {
        for (int x = std::max(0, static_cast<int>(px) - 6); x < std::min(width, static_cast<int>(px) + 7); ++x) {
          const float dx = static_cast<float>(x) + 0.5f - px, dy = static_cast<float>(y) + 0.5f - py;
          sum[static_cast<std::size_t>(y * width + x)] += (b.value - 0.5f) * std::exp(-(dx * dx + dy * dy) / 4.f);
        }
      }
    }
    for (std::size_t i = 0; i < sum.size(); ++i) {
      const auto value = static_cast<std::uint8_t>(std::clamp(128.f + 110.f * sum[i], 0.f, 255.f));
      rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = value;
      rgba[i * 4 + 3] = 255;
    }
    images.push_back(std::move(rgba));
  }
  for (int v = 0; v < 5; ++v) views.push_back({cameras[static_cast<std::size_t>(v)], images[static_cast<std::size_t>(v)]});
  std::vector<ColmapPoint> sparse(40);
  for (std::size_t i = 0; i < sparse.size(); ++i) sparse[i].position = glm::dvec3(where(rng) * 0.3f, where(rng) * 0.3f, 3.5f + 3.f * shade(rng));

  const std::vector<ColmapPoint> found = dense_points(views, sparse, 100000);
  std::println("  patch matching found {} points", found.size());
  REQUIRE(found.size() > 500);
  std::size_t onPlane = 0;
  for (const ColmapPoint &p : found) {
    if (std::abs(p.position.z - 5.0) < 0.15) ++onPlane;
  }
  CHECK(static_cast<double>(onPlane) > 0.95 * static_cast<double>(found.size()));

  // The limit holds, and the same points come back each time.
  const std::vector<ColmapPoint> few = dense_points(views, sparse, 100), again = dense_points(views, sparse, 100);
  CHECK(few.size() == 100);
  CHECK(std::ranges::equal(few, again, [](const ColmapPoint &a, const ColmapPoint &b) { return a.position == b.position; }));
  // Too few views to check anything.
  CHECK(dense_points(std::span(views).first(2), sparse, 100).empty());
}

TEST_CASE("The GPU collects densification statistics", "[densify][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test::train_camera(72, 56);
  const auto target = test::smooth_target(camera.width, camera.height, 11);
  GaussianCloud cloud = test::random_cloud(40, 0, 3);
  cloud.positions[0 * 3 + 2] = -5.f; // Behind the camera: never seen

  Trainer trainer(*ctx, cloud);
  trainer.compute_gradients(camera, target);
  trainer.compute_gradients(camera, target);
  const DensifyStats stats = trainer.densify_stats();
  CHECK(stats.visits[0] == 0.f);
  CHECK(stats.gradientSum[0] == 0.f);
  CHECK(stats.maxRadius[0] == 0.f);
  int seen = 0;
  for (std::size_t i = 1; i < cloud.size(); ++i) {
    if (stats.visits[i] == 0.f) continue;
    ++seen;
    CHECK(stats.visits[i] == 2.f);
    CHECK(stats.gradientSum[i] > 0.f);
    CHECK(stats.maxRadius[i] > 0.f);
  }
  CHECK(seen > 20);

  // Densifying resets them, keeps the optimiser state, and training carries on.
  LearningRates rates;
  trainer.apply_gradients(rates);
  const std::size_t before = trainer.gaussian_count();
  std::mt19937 rng(5);
  DensifyOptions options = default_options();
  options.gradientThreshold = 0.f; // Everything seen
  options.denseScale = 10.f;       // Clones only, which are easy to count
  const DensifyResult result = trainer.densify(options, rng);
  CHECK(result.cloned >= static_cast<std::size_t>(seen));
  CHECK(trainer.gaussian_count() == before + result.cloned);
  for (const float v : trainer.densify_stats().visits) CHECK(v == 0.f);
  const GaussianCloud grown = trainer.cloud();
  CHECK(grown.size() == trainer.gaussian_count());
  const StepStats stats2 = trainer.step(camera, target, rates);
  CHECK(std::isfinite(stats2.loss));

  // Capping opacity leaves the low ones as they are.
  trainer.reset_opacity(0.01f);
  const GaussianCloud capped = trainer.cloud();
  for (const float v : capped.opacities) CHECK(1.f / (1.f + std::exp(-v)) <= 0.0101f);
}

TEST_CASE("Only the active SH degrees render and train", "[densify][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test::train_camera(72, 56);
  const auto target = test::smooth_target(camera.width, camera.height, 2);
  const GaussianCloud full = test::random_cloud(40, 3, 9);

  // The same cloud cut down to degree 1, for the CPU reference.
  GaussianCloud low = full;
  low.shDegree = 1;
  low.sh.clear();
  for (std::size_t i = 0; i < full.size(); ++i) {
    low.sh.insert(low.sh.end(), full.sh.begin() + static_cast<std::ptrdiff_t>(i * 48), full.sh.begin() + static_cast<std::ptrdiff_t>(i * 48 + 12));
  }

  Trainer trainer(*ctx, full);
  trainer.set_active_sh_degree(1);
  const StepStats stats = trainer.compute_gradients(camera, target);
  const ReferenceLoss cpuLoss = reference_loss(reference_render(low, camera, {}), target, 0.2);
  CHECK_THAT(stats.loss, WithinAbs(cpuLoss.loss, 1e-3));

  const ParameterGradients cpu = reference_gradients(low, camera, cpuLoss.adjoint, {});
  const auto gpu = trainer.gradients();
  double worst = 0.0, inactive = 0.0;
  for (std::size_t i = 0; i < full.size(); ++i) {
    for (std::size_t k = 0; k < 48; ++k) {
      const double value = gpu.sh[i * 48 + k];
      if (k < 12) worst = std::max(worst, std::abs(value - cpu.sh[i * 12 + k]));
      else inactive = std::max(inactive, std::abs(value));
    }
  }
  CHECK(worst < 1e-5);
  CHECK(inactive == 0.0);

  // An inactive coefficient does not move, and the output keeps all of them.
  trainer.apply_gradients({});
  const GaussianCloud after = trainer.cloud();
  CHECK(after.shDegree == 3);
  for (std::size_t i = 0; i < full.size(); ++i) {
    for (std::size_t k = 12; k < 48; ++k) REQUIRE(after.sh[i * 48 + k] == full.sh[i * 48 + k]);
  }

  // At full degree, the GPU agrees with the reference too.
  trainer.set_active_sh_degree(3);
  Trainer fresh(*ctx, full);
  fresh.compute_gradients(camera, target);
  const ReferenceLoss fullLoss = reference_loss(reference_render(full, camera, {}), target, 0.2);
  const ParameterGradients fullGradients = reference_gradients(full, camera, fullLoss.adjoint, {});
  const auto gpuFull = fresh.gradients();
  double worstFull = 0.0;
  for (std::size_t i = 0; i < fullGradients.sh.size(); ++i) worstFull = std::max(worstFull, std::abs(gpuFull.sh[i] - fullGradients.sh[i]));
  CHECK(worstFull < 1e-5);
}

TEST_CASE("The memory budget is never exceeded", "[densify][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  CHECK(ctx->free_device_memory() > 0);

  const Camera camera = test::train_camera(72, 56);
  const auto target = test::smooth_target(camera.width, camera.height, 4);
  GaussianCloud cloud = test::random_cloud(200, 0, 8);
  for (std::size_t i = 0; i < cloud.size(); ++i) cloud.scales[i * 3] = std::log(2.f); // Large: they cover many tiles

  // Measure what this takes, then allow less than that.
  std::uint64_t needed = 0;
  {
    Trainer open(*ctx, cloud);
    open.compute_gradients(camera, target);
    needed = open.gpu_bytes();
  }
  TrainerOptions options;
  // Half of what depends on Gaussians. The per-pixel buffers have to fit in any case.
  const std::uint64_t fixedBytes = static_cast<std::uint64_t>(camera.width) * camera.height * Trainer::kBytesPerPixel + 1024;
  options.memoryBudget = fixedBytes + (needed - fixedBytes) / 2;
  Trainer trainer(*ctx, cloud, options);
  bool refused = false;
  int removals = 0;
  for (;;) {
    try {
      trainer.compute_gradients(camera, target);
      break;
    } catch (const BudgetExceeded &) {
      refused = true;
      trainer.remove_largest_gaussians(0.05);
      REQUIRE(++removals < 100);
    }
  }
  CHECK(refused);
  CHECK(trainer.gaussian_count() < cloud.size());
  CHECK(trainer.gpu_bytes() <= options.memoryBudget);

  // Densification under a limit stays under it, however eager.
  std::mt19937 rng(2);
  DensifyOptions densifyOptions = default_options();
  densifyOptions.gradientThreshold = 0.f;
  densifyOptions.denseScale = 0.01f;
  densifyOptions.maxGaussians = trainer.gaussian_count() + 5;
  trainer.compute_gradients(camera, target);
  const DensifyResult result = trainer.densify(densifyOptions, rng);
  CHECK(trainer.gaussian_count() <= densifyOptions.maxGaussians);
  CHECK(result.heldBack > 0);
}

TEST_CASE("Removing the largest keeps the rest and their state", "[densify]") {
  GaussianCloud cloud = line_of(10, 0.1f);
  cloud.scales[3 * 3] = std::log(5.f);
  cloud.scales[7 * 3 + 1] = std::log(3.f);
  const DensifyResult r = remove_largest(cloud, 0.2);
  CHECK(r.pruned == 2);
  CHECK(r.source == std::vector<std::int64_t>{0, 1, 2, 4, 5, 6, 8, 9});
  CHECK(remove_largest(line_of(1, 0.1f), 0.5).cloud.size() == 1); // Never down to nothing
}

TEST_CASE("Pruning by contribution removes those adding least", "[densify]") {
  const GaussianCloud cloud = line_of(10, 0.1f);
  std::vector<float> contribution{5.f, 0.f, 9.f, 0.5f, 7.f, 3.f, 0.1f, 8.f, 2.f, 6.f};
  const DensifyResult r = remove_least_contributing(cloud, contribution, 0.3);
  CHECK(r.pruned == 3);
  CHECK(r.source == std::vector<std::int64_t>{0, 2, 4, 5, 7, 8, 9});
  CHECK(r.cloud.size() == 7);
  CHECK_THROWS(remove_least_contributing(cloud, std::vector<float>(3, 1.f), 0.3));
}

TEST_CASE("Densification improves a fit that is short of Gaussians", "[densify][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");

  const auto views = make_test_views(1, 128, 96, 100.0);
  const Camera camera = test::camera_of(views[0], 128, 96, 100.f);
  std::vector<std::uint8_t> target(128u * 96u * 4u, 255);
  for (std::size_t i = 0; i < 128u * 96u; ++i) std::copy_n(views[0].image.pixels.data() + i * 3, 3, target.data() + i * 4);

  const GaussianCloud start = test::grid_in_front(camera, 10);
  const auto fit = [&](bool densify) {
    Trainer trainer(*ctx, start);
    LearningRates rates;
    rates.position = 1e-3f;
    std::mt19937 rng(3);
    StepStats last;
    for (int i = 1; i <= 800; ++i) {
      last = trainer.step(camera, target, rates);
      if (densify && i >= 100 && i <= 600 && i % 50 == 0) {
        DensifyOptions options;
        options.gradientThreshold = 2e-4f;
        options.denseScale = 0.03f;
        options.maxGaussians = 4000;
        trainer.densify(options, rng);
      }
    }
    return std::pair{last.loss, trainer.gaussian_count()};
  };
  const auto [fixedLoss, fixedCount] = fit(false);
  const auto [denseLoss, denseCount] = fit(true);
  std::println("  without densification: loss {:.4f}, {} Gaussians; with: {:.4f}, {}", fixedLoss, fixedCount, denseLoss, denseCount);
  CHECK(fixedCount == start.size());
  CHECK(denseCount > fixedCount * 2);
  CHECK(denseCount <= 4000);
  CHECK(denseLoss < fixedLoss * 0.8f);
}

// Hidden, as it takes minutes: run with `splat_tests "[.long]"`. It goes through
// the whole schedule, opacity resets and large-Gaussian pruning included, which
// short runs never reach. A bad pruning limit once passed every other test here.
TEST_CASE("A full schedule keeps the quality it reached", "[.long][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "splat_long_training";
  fs::remove_all(dir);
  fs::create_directories(dir / "input");
  const auto views = make_test_views(24, 320, 240, 260.0);
  for (std::size_t i = 0; i < views.size(); ++i) {
    write_image(views[i].image, dir / "input" / ("view_" + std::to_string(i) + ".png"), 95);
  }
  IngestOptions ingest;
  ingest.inputs = {dir / "input"};
  ingest.project = dir / "project";
  ingest.manual.focalPx = 260.0;
  run_ingest(ingest);
  SfmOptions sfm;
  sfm.project = ingest.project;
  run_sfm(sfm);

  TrainOptions train;
  train.project = ingest.project;
  train.out = dir / "splat.ply";
  train.iterations = 8000;
  train.holdout = 8;
  train.reportEvery = 1000;
  train.saveEvery = 0;
  const TrainReport report = run_train(*ctx, train);
  // With views 0, 8 and 16 held out this lands at 27.8-28.5 dB from run to run.
  CHECK(report.holdoutPsnr > 27.0);
  CHECK(report.lastLoss < report.firstLoss * 0.3f);
  // The progress reports measure the cloud before an opacity reset, never
  // the nearly transparent one just after it. The reset at iteration 3000
  // once took 6 dB off that report.
  REQUIRE(report.progressPsnr.size() == 8);
  for (std::size_t i = 1; i < report.progressPsnr.size(); ++i) {
    const auto [iteration, psnr] = report.progressPsnr[i];
    const double previous = report.progressPsnr[i - 1].second;
    INFO("iteration " << iteration << ": " << psnr << " dB after " << previous << " dB");
    CHECK(psnr > previous - 1.0);
  }
  fs::remove_all(dir);
}
