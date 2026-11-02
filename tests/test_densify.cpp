#include "reference_helpers.hpp"
#include "sfm/test_views.hpp"
#include "test_support.hpp"
#include "train/densify.hpp"
#include "train/reference.hpp"
#include "train/train.hpp"
#include "train/trainer.hpp"
#include "ingest/ingest.hpp"
#include "sfm/sfm.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
  CHECK(report.holdoutPsnr > 28.0);
  CHECK(report.lastLoss < report.firstLoss * 0.3f);
  fs::remove_all(dir);
}
