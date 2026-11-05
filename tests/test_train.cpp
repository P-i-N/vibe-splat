#include "reference_helpers.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "sfm/test_views.hpp"
#include "test_support.hpp"
#include "train/evaluate.hpp"
#include "train/reference.hpp"
#include "train/train.hpp"
#include "train/trainer.hpp"
#include "ingest/image.hpp"
#include "sfm/colmap_model.hpp"

#include <catch2/catch_test_macros.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <print>
#include <random>
#include <string>

using namespace splat;
using Catch::Matchers::WithinAbs;

namespace {

using test::random_cloud;

Camera test_camera(int width = 64, int height = 48) { return test::train_camera(width, height); }

std::vector<std::uint8_t> noise_target(int width, int height, std::uint32_t seed) { return test::smooth_target(width, height, seed); }

// How well two gradient vectors agree: direction and length.
void check_close(const std::vector<double> &a, const std::vector<double> &b, const char *what, double minCosine, double maxRatioError) {
  REQUIRE(a.size() == b.size());
  double ab = 0.0, aa = 0.0, bb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  INFO(what);
  REQUIRE(aa > 0.0);
  REQUIRE(bb > 0.0);
  const double cosine = ab / std::sqrt(aa * bb), ratio = std::sqrt(aa / bb);
  std::println("  {:12} cosine {:.6f}, length ratio {:.5f}", what, cosine, ratio);
  CHECK(cosine > minCosine);
  CHECK(std::abs(ratio - 1.0) < maxRatioError);
}

std::vector<double> to_double(const std::vector<float> &v) { return {v.begin(), v.end()}; }

double functional(const GaussianCloud &cloud, const Camera &camera, const std::vector<glm::dvec3> &adjoint, ReferenceOptions options = {}) {
  options.noCutoffs = true;
  const ReferenceImage image = reference_render(cloud, camera, options);
  double sum = 0.0;
  for (std::size_t i = 0; i < adjoint.size(); ++i) sum += glm::dot(adjoint[i], image.rgb[i]);
  return sum;
}

// Central differences of the functional over one parameter array.
std::vector<double> finite_differences(GaussianCloud cloud, std::vector<float> GaussianCloud::*member, const Camera &camera,
                                       const std::vector<glm::dvec3> &adjoint, float step, const ReferenceOptions &options = {}) {
  std::vector<float> &values = cloud.*member;
  std::vector<double> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float original = values[i];
    values[i] = original + step;
    const float up = values[i];
    const double high = functional(cloud, camera, adjoint, options);
    values[i] = original - step;
    const float down = values[i];
    const double low = functional(cloud, camera, adjoint, options);
    values[i] = original;
    out[i] = (high - low) / static_cast<double>(up - down);
  }
  return out;
}

} // namespace

TEST_CASE("Analytic gradients match finite differences on the CPU", "[train][reference]") {
  const Camera camera = test_camera(40, 32);
  std::mt19937 rng(9);
  std::normal_distribution<double> n(0.0, 1.0);
  std::vector<glm::dvec3> adjoint(static_cast<std::size_t>(camera.width) * camera.height);
  for (auto &a : adjoint) a = glm::dvec3(n(rng), n(rng), n(rng));

  // Degree 0 for position: its colour has no view dependence, which the
  // gradient leaves out on purpose.
  const GaussianCloud cloud = random_cloud(12, 0, 3);
  ReferenceOptions smooth;
  smooth.noCutoffs = true;
  const ParameterGradients g = reference_gradients(cloud, camera, adjoint, smooth);
  check_close(g.positions, finite_differences(cloud, &GaussianCloud::positions, camera, adjoint, 2e-3f), "positions", 0.999, 0.01);
  check_close(g.scales, finite_differences(cloud, &GaussianCloud::scales, camera, adjoint, 2e-3f), "scales", 0.999, 0.01);
  check_close(g.rotations, finite_differences(cloud, &GaussianCloud::rotations, camera, adjoint, 2e-3f), "rotations", 0.999, 0.01);
  check_close(g.opacities, finite_differences(cloud, &GaussianCloud::opacities, camera, adjoint, 2e-3f), "opacities", 0.999, 0.01);

  // Degree 3, with position: this is where the view direction term matters.
  const GaussianCloud degree3 = random_cloud(12, 3, 6);
  const ParameterGradients g3 = reference_gradients(degree3, camera, adjoint, smooth);
  check_close(g3.positions, finite_differences(degree3, &GaussianCloud::positions, camera, adjoint, 2e-3f), "positions, degree 3", 0.999, 0.01);
  check_close(g3.sh, finite_differences(degree3, &GaussianCloud::sh, camera, adjoint, 2e-3f), "sh degree 3", 0.999, 0.01);

  const GaussianCloud degree2 = random_cloud(12, 2, 4);
  const ParameterGradients g2 = reference_gradients(degree2, camera, adjoint, smooth);
  check_close(g2.sh, finite_differences(degree2, &GaussianCloud::sh, camera, adjoint, 2e-3f), "sh degree 2", 0.999, 0.01);
}

TEST_CASE("The loss derivative matches finite differences", "[train][reference]") {
  ReferenceImage image;
  image.width = 18;
  image.height = 14;
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> u(0.1, 0.9);
  image.rgb.resize(static_cast<std::size_t>(image.width) * image.height);
  for (auto &p : image.rgb) p = glm::dvec3(u(rng), u(rng), u(rng));
  const auto target = noise_target(image.width, image.height, 5);

  for (const double lambda : {1.0, 0.2}) {
    const ReferenceLoss base = reference_loss(image, target, lambda);
    std::vector<double> analytic, numeric;
    for (std::size_t i = 0; i < image.rgb.size(); i += 7) {
      for (int c = 0; c < 3; ++c) {
        ReferenceImage changed = image;
        constexpr double kStep = 1e-5;
        changed.rgb[i][c] += kStep;
        const double high = reference_loss(changed, target, lambda).loss;
        changed.rgb[i][c] -= 2.0 * kStep;
        const double low = reference_loss(changed, target, lambda).loss;
        numeric.push_back((high - low) / (2.0 * kStep));
        analytic.push_back(base.adjoint[i][c]);
      }
    }
    check_close(analytic, numeric, lambda == 1.0 ? "pure SSIM" : "L1 + SSIM", 0.9999, 0.001);
  }
}

TEST_CASE("The 2D Mip filter's gradients match finite differences on the CPU", "[train][reference]") {
  const Camera camera = test_camera(40, 32);
  std::mt19937 rng(21);
  std::normal_distribution<double> n(0.0, 1.0);
  std::vector<glm::dvec3> adjoint(static_cast<std::size_t>(camera.width) * camera.height);
  for (auto &a : adjoint) a = glm::dvec3(n(rng), n(rng), n(rng));
  const GaussianCloud cloud = random_cloud(12, 0, 3);
  ReferenceOptions smooth;
  smooth.noCutoffs = true;
  smooth.mipFilter2d = true;
  const ParameterGradients g = reference_gradients(cloud, camera, adjoint, smooth);
  check_close(g.positions, finite_differences(cloud, &GaussianCloud::positions, camera, adjoint, 2e-3f, smooth), "positions", 0.999, 0.01);
  check_close(g.scales, finite_differences(cloud, &GaussianCloud::scales, camera, adjoint, 2e-3f, smooth), "scales", 0.999, 0.01);
  check_close(g.rotations, finite_differences(cloud, &GaussianCloud::rotations, camera, adjoint, 2e-3f, smooth), "rotations", 0.999, 0.01);
  check_close(g.opacities, finite_differences(cloud, &GaussianCloud::opacities, camera, adjoint, 2e-3f, smooth), "opacities", 0.999, 0.01);
  // The filter changes the picture: an opaque, sub-pixel Gaussian renders fainter.
  CHECK(functional(cloud, camera, adjoint, smooth) != functional(cloud, camera, adjoint));
}

TEST_CASE("The 3D smoothing filter's gradients match finite differences on the CPU", "[train][reference]") {
  const Camera camera = test_camera(40, 32);
  std::mt19937 rng(23);
  std::normal_distribution<double> n(0.0, 1.0);
  std::vector<glm::dvec3> adjoint(static_cast<std::size_t>(camera.width) * camera.height);
  for (auto &a : adjoint) a = glm::dvec3(n(rng), n(rng), n(rng));
  const GaussianCloud cloud = random_cloud(12, 0, 3);
  ReferenceOptions smooth;
  smooth.noCutoffs = true;
  smooth.mipFilter2d = true;
  for (std::size_t i = 0; i < cloud.size(); ++i) smooth.filter3d.push_back(0.002f * static_cast<float>(i + 1));
  const ParameterGradients g = reference_gradients(cloud, camera, adjoint, smooth);
  check_close(g.positions, finite_differences(cloud, &GaussianCloud::positions, camera, adjoint, 2e-3f, smooth), "positions", 0.999, 0.01);
  check_close(g.scales, finite_differences(cloud, &GaussianCloud::scales, camera, adjoint, 2e-3f, smooth), "scales", 0.999, 0.01);
  check_close(g.rotations, finite_differences(cloud, &GaussianCloud::rotations, camera, adjoint, 2e-3f, smooth), "rotations", 0.999, 0.01);
  check_close(g.opacities, finite_differences(cloud, &GaussianCloud::opacities, camera, adjoint, 2e-3f, smooth), "opacities", 0.999, 0.01);
  ReferenceOptions unfiltered = smooth;
  unfiltered.filter3d.clear();
  CHECK(functional(cloud, camera, adjoint, smooth) != functional(cloud, camera, adjoint, unfiltered));
}

TEST_CASE("GPU forward and gradients match the CPU reference with the Mip filters", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 11);
  const GaussianCloud cloud = random_cloud(60, 1, 13);
  ReferenceOptions referenceOptions;
  referenceOptions.mipFilter2d = true;
  for (std::size_t i = 0; i < cloud.size(); ++i) referenceOptions.filter3d.push_back(0.001f * static_cast<float>(i % 7));
  const ReferenceImage image = reference_render(cloud, camera, referenceOptions);
  const ReferenceLoss cpuLoss = reference_loss(image, target, 0.2);
  const ParameterGradients cpu = reference_gradients(cloud, camera, cpuLoss.adjoint, referenceOptions);

  TrainerOptions options;
  options.mipFilter2d = true;
  Trainer trainer(*ctx, cloud, options);
  trainer.set_filter3d(referenceOptions.filter3d);
  const std::vector<float> gpu = trainer.render(camera);
  double worst = 0.0;
  for (std::size_t i = 0; i < image.rgb.size(); ++i) {
    for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(gpu[i * 4 + static_cast<std::size_t>(c)] - image.rgb[i][c]));
  }
  std::println("  forward with the 2D Mip filter: worst difference from CPU {:.6f}", worst);
  CHECK(worst < 1e-3);

  const StepStats stats = trainer.compute_gradients(camera, target);
  CHECK_THAT(stats.loss, WithinAbs(cpuLoss.loss, 1e-3));
  const auto g = trainer.gradients();
  check_close(to_double(g.positions), cpu.positions, "positions", 0.999, 0.02);
  check_close(to_double(g.scales), cpu.scales, "scales", 0.999, 0.02);
  check_close(to_double(g.rotations), cpu.rotations, "rotations", 0.999, 0.02);
  check_close(to_double(g.opacities), cpu.opacities, "opacities", 0.999, 0.02);
}

TEST_CASE("GPU forward matches the CPU reference and the graphics renderer", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(100, 75);
  const GaussianCloud cloud = random_cloud(300, 1, 7);

  Trainer trainer(*ctx, cloud);
  const std::vector<float> gpu = trainer.render(camera);
  const ReferenceImage cpu = reference_render(cloud, camera, {});
  double worst = 0.0, worstLog = 0.0;
  for (std::size_t i = 0; i < cpu.rgb.size(); ++i) {
    for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(gpu[i * 4 + static_cast<std::size_t>(c)] - cpu.rgb[i][c]));
    worstLog = std::max(worstLog, std::abs(gpu[i * 4 + 3] - cpu.logTransmittance[i]));
  }
  std::println("  forward: worst difference from CPU {:.6f}, log transmittance {:.6f}", worst, worstLog);
  CHECK(worst < 1e-3);
  CHECK(worstLog < 1e-2);

  // Fixed-function blending gets to the same picture, up to 8-bit rounding.
  SplatRenderer renderer(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const RenderedImage graphics = render_offscreen(*ctx, renderer, camera);
  // The graphics path rounds to 8 bits after every blend, so errors can build up
  // with depth complexity: compare on average, with a limit on any single pixel.
  double sum = 0.0, peak = 0.0;
  for (std::size_t i = 0; i < cpu.rgb.size(); ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      const double d = std::abs(static_cast<double>(gpu[i * 4 + c]) * 255.0 - graphics.rgba[i * 4 + c]);
      sum += d;
      peak = std::max(peak, d);
    }
  }
  const double mean = sum / (3.0 * static_cast<double>(cpu.rgb.size()));
  std::println("  forward vs graphics: mean {:.3f}, worst {:.2f} of 255", mean, peak);
  CHECK(mean < 1.0);
  CHECK(peak < 8.0);
}

TEST_CASE("GPU loss and gradients match the CPU reference", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 11);
  const GaussianCloud cloud = random_cloud(60, 1, 13);

  ReferenceOptions referenceOptions;
  referenceOptions.opacityReg = 0.01f; // The penalties ride along, so that the twin is pinned too
  referenceOptions.scaleReg = 0.01f;
  const ReferenceImage image = reference_render(cloud, camera, {});
  const ReferenceLoss cpuLoss = reference_loss(image, target, 0.2);
  const ParameterGradients cpu = reference_gradients(cloud, camera, cpuLoss.adjoint, referenceOptions);

  for (const bool fixedPoint : {false, true}) {
    TrainerOptions options;
    options.forceFixedPoint = fixedPoint;
    options.opacityReg = referenceOptions.opacityReg;
    options.scaleReg = referenceOptions.scaleReg;
    if (!fixedPoint && !ctx->info().floatAtomicAdd) continue;
    Trainer trainer(*ctx, cloud, options);
    CHECK(trainer.uses_fixed_point() == fixedPoint);
    const StepStats stats = trainer.compute_gradients(camera, target);
    INFO("fixed point: " << fixedPoint);
    CHECK_THAT(stats.loss, WithinAbs(cpuLoss.loss, 1e-3));
    CHECK_THAT(stats.l1, WithinAbs(cpuLoss.l1, 1e-3));
    CHECK_THAT(stats.ssim, WithinAbs(cpuLoss.ssim, 1e-3));

    const auto gpu = trainer.gradients();
    std::println("  {} atomics:", fixedPoint ? "fixed-point" : "float");
    const double cosine = fixedPoint ? 0.97 : 0.999, ratio = fixedPoint ? 0.1 : 0.02;
    check_close(to_double(gpu.positions), cpu.positions, "positions", cosine, ratio);
    check_close(to_double(gpu.scales), cpu.scales, "scales", cosine, ratio);
    check_close(to_double(gpu.rotations), cpu.rotations, "rotations", cosine, ratio);
    check_close(to_double(gpu.opacities), cpu.opacities, "opacities", cosine, ratio);
    check_close(to_double(gpu.sh), cpu.sh, "sh", cosine, ratio);
  }
}

TEST_CASE("Opacity and scale penalties add their gradient to every Gaussian", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(64, 48);
  const auto target = noise_target(camera.width, camera.height, 5);
  GaussianCloud cloud = random_cloud(20, 1, 9);
  // Behind the camera: no pixel touches them, so the penalties are the whole gradient.
  for (std::size_t i = 0; i < cloud.size(); ++i) cloud.positions[i * 3 + 2] = -5.f;
  const auto n = static_cast<double>(cloud.size());
  for (const float reg : {0.f, 0.5f}) {
    TrainerOptions options;
    options.opacityReg = reg;
    options.scaleReg = reg * 0.5f;
    Trainer trainer(*ctx, cloud, options);
    trainer.compute_gradients(camera, target);
    const auto g = trainer.gradients();
    INFO("reg " << reg);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      const double o = 1.0 / (1.0 + std::exp(-static_cast<double>(cloud.opacities[i])));
      CHECK_THAT(g.opacities[i], WithinAbs(reg / n * o * (1.0 - o), 1e-6));
      for (std::size_t a = 0; a < 3; ++a) {
        CHECK_THAT(g.scales[i * 3 + a], WithinAbs(options.scaleReg / n * std::exp(static_cast<double>(cloud.scales[i * 3 + a])), 1e-6));
      }
      CHECK(g.positions[i * 3] == 0.f);
    }
  }
}

TEST_CASE("Position noise moves transparent Gaussians by their covariance and leaves opaque ones", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  constexpr std::size_t kCount = 4000;
  GaussianCloud cloud;
  cloud.resize(kCount, 0);
  for (std::size_t i = 0; i < kCount; ++i) {
    cloud.rotations[i * 4] = 1.f;                       // Identity, scales log 0 = 1: the covariance is the identity
    cloud.opacities[i] = i < kCount / 2 ? -20.f : 20.f; // Transparent first, then opaque
  }
  cloud.scales[(kCount - 1) * 3] = -30.f; // A degenerate axis must still give a finite position
  Trainer trainer(*ctx, cloud);
  trainer.add_position_noise(0.5f, 7);
  const GaussianCloud moved = trainer.cloud();
  double sum = 0.0, sumSquares = 0.0;
  for (std::size_t i = 0; i < kCount / 2; ++i) {
    sum += moved.positions[i * 3];
    sumSquares += static_cast<double>(moved.positions[i * 3]) * moved.positions[i * 3];
  }
  const double mean = sum / (kCount / 2), deviation = std::sqrt(sumSquares / (kCount / 2) - mean * mean);
  std::println("  noise on transparent Gaussians: mean {:.4f}, standard deviation {:.4f}", mean, deviation);
  CHECK_THAT(mean, WithinAbs(0.0, 0.05));
  // The gate is a logistic in 1 - opacity, steep at 0.995: fully transparent gives 1 / (1 + e^-0.5).
  CHECK_THAT(deviation, WithinAbs(0.5 / (1.0 + std::exp(-0.5)), 0.03));
  for (std::size_t i = kCount / 2; i < kCount; ++i) {
    for (std::size_t a = 0; a < 3; ++a) CHECK(moved.positions[i * 3 + a] == 0.f);
  }
  for (const float p : moved.positions) REQUIRE(std::isfinite(p));
}

TEST_CASE("GPU forward and gradients match the CPU reference where pixels stop early", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(80, 60);
  const auto target = noise_target(camera.width, camera.height, 19);
  GaussianCloud cloud = random_cloud(400, 1, 17);
  for (float &opacity : cloud.opacities) opacity = 5.f; // Nearly opaque, so most pixels stop early

  const ReferenceImage image = reference_render(cloud, camera, {});
  const auto stopped = std::count_if(image.stop.begin(), image.stop.end(), [](std::size_t stop) { return stop > 0; });
  std::println("  {} of {} pixels stop early", stopped, image.stop.size());
  REQUIRE(stopped > static_cast<std::ptrdiff_t>(image.stop.size() / 10));

  Trainer trainer(*ctx, cloud);
  const std::vector<float> gpu = trainer.render(camera);
  double worst = 0.0, worstLog = 0.0;
  for (std::size_t i = 0; i < image.rgb.size(); ++i) {
    for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(gpu[i * 4 + static_cast<std::size_t>(c)] - image.rgb[i][c]));
    worstLog = std::max(worstLog, std::abs(gpu[i * 4 + 3] - image.logTransmittance[i]));
  }
  std::println("  forward: worst difference from CPU {:.6f}, log transmittance {:.6f}", worst, worstLog);
  CHECK(worst < 1e-3);
  CHECK(worstLog < 1e-2);

  const ReferenceLoss cpuLoss = reference_loss(image, target, 0.2);
  const ParameterGradients cpu = reference_gradients(cloud, camera, cpuLoss.adjoint, {});
  for (const bool fixedPoint : {false, true}) {
    TrainerOptions options;
    options.forceFixedPoint = fixedPoint;
    if (!fixedPoint && !ctx->info().floatAtomicAdd) continue;
    Trainer gradients(*ctx, cloud, options);
    const StepStats stats = gradients.compute_gradients(camera, target);
    INFO("fixed point: " << fixedPoint);
    CHECK_THAT(stats.loss, WithinAbs(cpuLoss.loss, 1e-3));
    const auto g = gradients.gradients();
    std::println("  {} atomics:", fixedPoint ? "fixed-point" : "float");
    const double cosine = fixedPoint ? 0.97 : 0.999, ratio = fixedPoint ? 0.1 : 0.02;
    const DensifyStats densifyStats = gradients.densify_stats();
    check_close(to_double(densifyStats.contribution), reference_contributions(cloud, camera, {}), "contribution", cosine, ratio);
    std::vector<double> absolute(cloud.size());
    for (std::size_t i = 0; i < absolute.size(); ++i) {
      absolute[i] = std::hypot(cpu.absoluteCentre[i * 2] * 0.5 * camera.width, cpu.absoluteCentre[i * 2 + 1] * 0.5 * camera.height);
    }
    check_close(to_double(densifyStats.absoluteGradientSum), absolute, "absolute", cosine, ratio);
    check_close(to_double(g.positions), cpu.positions, "positions", cosine, ratio);
    check_close(to_double(g.scales), cpu.scales, "scales", cosine, ratio);
    check_close(to_double(g.rotations), cpu.rotations, "rotations", cosine, ratio);
    check_close(to_double(g.opacities), cpu.opacities, "opacities", cosine, ratio);
    check_close(to_double(g.sh), cpu.sh, "sh", cosine, ratio);
  }
}

TEST_CASE("The exposure transform's gradients match the CPU", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 23);
  const GaussianCloud cloud = random_cloud(60, 1, 29);
  const Exposure exposure{1.1f, 0.05f, -0.02f, 0.03f, 0.02f, 0.9f, 0.04f, -0.01f, -0.03f, 0.06f, 1.05f, 0.02f};

  const ReferenceImage image = reference_render(cloud, camera, {});
  const auto exposed = [&](const Exposure &e) {
    ReferenceImage out = image;
    for (glm::dvec3 &rgb : out.rgb) {
      const glm::dvec3 c = rgb;
      for (int r = 0; r < 3; ++r) rgb[r] = e[r * 4] * c.r + e[r * 4 + 1] * c.g + e[r * 4 + 2] * c.b + e[r * 4 + 3];
    }
    return out;
  };
  // Central differences of the loss over the 12 values.
  std::vector<double> numeric(12);
  for (std::size_t k = 0; k < 12; ++k) {
    Exposure up = exposure, down = exposure;
    up[k] += 1e-3f;
    down[k] -= 1e-3f;
    numeric[k] = (reference_loss(exposed(up), target, 0.2).loss - reference_loss(exposed(down), target, 0.2).loss) / (static_cast<double>(up[k]) - down[k]);
  }
  // The Gaussians see the adjoint taken back through the transform.
  ReferenceLoss loss = reference_loss(exposed(exposure), target, 0.2);
  for (glm::dvec3 &g : loss.adjoint) {
    const glm::dvec3 a = g;
    for (int c = 0; c < 3; ++c) g[c] = a.r * exposure[c] + a.g * exposure[4 + c] + a.b * exposure[8 + c];
  }
  const ParameterGradients cpu = reference_gradients(cloud, camera, loss.adjoint, {});

  Trainer trainer(*ctx, cloud);
  trainer.set_exposure(exposure);
  const StepStats stats = trainer.compute_gradients(camera, target);
  CHECK_THAT(stats.loss, WithinAbs(loss.loss, 1e-3));
  std::println("  exposure:");
  check_close(to_double(std::vector<float>(stats.exposureGradient.begin(), stats.exposureGradient.end())), numeric, "transform", 0.999, 0.02);
  const auto gpu = trainer.gradients();
  const double cosine = trainer.uses_fixed_point() ? 0.97 : 0.999, ratio = trainer.uses_fixed_point() ? 0.1 : 0.02;
  check_close(to_double(gpu.positions), cpu.positions, "positions", cosine, ratio);
  check_close(to_double(gpu.sh), cpu.sh, "sh", cosine, ratio);
}

TEST_CASE("Adam matches a hand-computed step", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(40, 32);
  const auto target = noise_target(camera.width, camera.height, 2);
  const GaussianCloud cloud = random_cloud(8, 1, 5);

  Trainer trainer(*ctx, cloud);
  trainer.compute_gradients(camera, target);
  const auto g = trainer.gradients();
  LearningRates rates;
  rates.sh = 0.02f;
  rates.shRestScale = 0.05f;
  trainer.apply_gradients(rates);
  const GaussianCloud after = trainer.cloud();

  // In the first step m / (1 - beta1) = g and sqrt(v / (1 - beta2)) = |g|, so
  // every parameter moves by exactly its rate, against the gradient's sign.
  // Above degree 0 SH is kept in half floats, rounded to nearest on the way in
  // and at random after the step: within two of their steps, |v| / 512.
  for (std::size_t i = 0; i < cloud.sh.size(); ++i) {
    if (g.sh[i] == 0.f) continue;
    const bool dc = i % 12 < 3;
    const float rate = dc ? rates.sh : rates.sh * rates.shRestScale;
    const float expected = cloud.sh[i] - (g.sh[i] > 0.f ? rate : -rate);
    CHECK_THAT(after.sh[i], WithinAbs(expected, dc ? 2e-5 : std::max(2e-5, std::abs(expected) / 512.0)));
  }
  for (std::size_t i = 0; i < cloud.opacities.size(); ++i) {
    if (g.opacities[i] == 0.f) continue;
    CHECK_THAT(after.opacities[i], WithinAbs(cloud.opacities[i] - (g.opacities[i] > 0.f ? rates.opacity : -rates.opacity), 2e-5));
  }

  // A view that sees none of them moves nothing, despite the momentum.
  Camera away = camera;
  away.view = glm::rotate(glm::mat4(1.f), glm::pi<float>(), glm::vec3(0.f, 1.f, 0.f)) * camera.view;
  trainer.compute_gradients(away, target);
  trainer.apply_gradients(rates);
  const GaussianCloud unseen = trainer.cloud();
  CHECK(unseen.positions == after.positions);
  CHECK(unseen.opacities == after.opacities);
  CHECK(unseen.sh == after.sh);
}

TEST_CASE("Training overfits a single rendered view", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");

  const auto views = make_test_views(1, 128, 96, 100.0);
  const Camera camera = test::camera_of(views[0], 128, 96, 100.f);
  std::vector<std::uint8_t> target(static_cast<std::size_t>(128) * 96 * 4, 255);
  for (std::size_t i = 0; i < 128u * 96u; ++i) std::copy_n(views[0].image.pixels.data() + i * 3, 3, target.data() + i * 4);

  const GaussianCloud cloud = test::grid_in_front(camera, 40);

  Trainer trainer(*ctx, cloud);
  LearningRates rates;
  rates.position = 1e-3f;
  float first = 0.f, last = 0.f;
  for (int iteration = 0; iteration < 400; ++iteration) {
    const StepStats stats = trainer.step(camera, target, rates);
    REQUIRE(std::isfinite(stats.loss));
    if (iteration == 0) first = stats.loss;
    last = stats.loss;
  }
  std::println("  overfitting one view: loss {:.4f} -> {:.4f}", first, last);
  CHECK(last < first * 0.5f);
  const GaussianCloud trained = trainer.cloud();
  CHECK(trained.size() == cloud.size());
  CHECK(trained.positions != cloud.positions);
  CHECK(trained.sh != cloud.sh);
}

TEST_CASE("Fitting an exposure undoes a colour transform, on the half it did not see", "[train]") {
  constexpr int width = 32, height = 16;
  const auto target = noise_target(width, height, 31);
  std::vector<float> render(static_cast<std::size_t>(width) * height * 4);
  for (std::size_t i = 0; i < render.size() / 4; ++i) {
    // The photo is 1.25 x render + 0.1, per channel with another gain.
    for (std::size_t c = 0; c < 3; ++c) render[i * 4 + c] = (target[i * 4 + c] / 255.f - 0.1f) / (1.25f + 0.1f * static_cast<float>(c));
  }
  CHECK(psnr_of(render, target) < 30.0);
  const Exposure fitted = fit_exposure(render, target, width, height, 0, width / 2);
  CHECK_THAT(fitted[0], WithinAbs(1.25, 1e-2));
  CHECK_THAT(fitted[5], WithinAbs(1.35, 1e-2));
  CHECK_THAT(fitted[10], WithinAbs(1.45, 1e-2));
  CHECK_THAT(fitted[3], WithinAbs(0.1, 1e-2));
  CHECK(cross_fitted_psnr(render, target, width, height) > 45.0);

  // A neutral render gets a neutral fit, and applying a fit gives what it scored.
  std::vector<float> exposed = render;
  apply_exposure(exposed, fitted);
  CHECK(psnr_of(exposed, target) > 45.0);
}

TEST_CASE("Degenerate Gaussians are not drawn and do not spread", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 37);
  GaussianCloud healthy = random_cloud(40, 1, 41);
  Trainer clean(*ctx, healthy);
  const std::vector<float> expected = clean.render(camera);

  GaussianCloud cloud = healthy;
  const std::size_t base = cloud.size();
  cloud.resize(base + 5, cloud.shDegree);
  const auto floats = static_cast<std::size_t>(3 * cloud.shCoeffs());
  for (std::size_t i = 0; i < base; ++i) {
    std::copy_n(healthy.positions.data() + i * 3, 3, cloud.positions.data() + i * 3);
    std::copy_n(healthy.scales.data() + i * 3, 3, cloud.scales.data() + i * 3);
    std::copy_n(healthy.rotations.data() + i * 4, 4, cloud.rotations.data() + i * 4);
    cloud.opacities[i] = healthy.opacities[i];
    std::copy_n(healthy.sh.data() + i * floats, floats, cloud.sh.data() + i * floats);
  }
  for (std::size_t i = base; i < base + 5; ++i) {
    cloud.positions[i * 3 + 2] = 4.f;
    for (std::size_t a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = std::log(0.3f);
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = 2.f;
  }
  const float nan = std::nanf("");
  cloud.positions[base * 3] = nan;
  cloud.scales[(base + 1) * 3] = 200.f; // exp() overflows, and inf * 0 is NaN
  cloud.rotations[(base + 2) * 4] = 1e-5f; // Too short to normalise
  cloud.opacities[base + 3] = nan;
  cloud.sh[(base + 4) * floats] = nan;

  Trainer trainer(*ctx, cloud);
  const std::vector<float> rendered = trainer.render(camera);
  REQUIRE(rendered.size() == expected.size());
  for (std::size_t i = 0; i < rendered.size(); ++i) REQUIRE(rendered[i] == expected[i]);
  const StepStats stats = trainer.compute_gradients(camera, target);
  CHECK(std::isfinite(stats.loss));
  const auto gradients = trainer.gradients();
  for (const float g : gradients.positions) REQUIRE(std::isfinite(g));
  for (const float g : gradients.sh) REQUIRE(std::isfinite(g));
}

TEST_CASE("Scoring a view counts the high-error pixels each Gaussian reaches", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 43);
  const GaussianCloud cloud = random_cloud(60, 1, 47);
  const std::vector<double> cpu = reference_marked_pixels(cloud, camera, target, 0.5, {});
  double cpuTotal = 0.0;
  for (const double c : cpu) cpuTotal += c;
  REQUIRE(cpuTotal > 0.0);

  for (const bool fixedPoint : {false, true}) {
    TrainerOptions options;
    options.forceFixedPoint = fixedPoint;
    if (!fixedPoint && !ctx->info().floatAtomicAdd) continue;
    INFO("fixed point: " << fixedPoint);
    Trainer trainer(*ctx, cloud, options);
    // An ordinary iteration counts nothing.
    trainer.compute_gradients(camera, target);
    for (const float c : trainer.densify_stats().markedPixels) REQUIRE(c == 0.f);

    const StepStats stats = trainer.score_view(camera, target, 0.5f, 2.f);
    const DensifyStats scored = trainer.densify_stats();
    // A pixel either way: the GPU keeps the worst error to 1/65535, which moves the limit a little.
    for (std::size_t i = 0; i < cpu.size(); ++i) {
      CHECK_THAT(scored.markedPixels[i], WithinAbs(2.0 * cpu[i], 3.0));
      CHECK_THAT(scored.markedLoss[i], WithinAbs(2.0 * cpu[i] * stats.loss, 3.0));
    }
    // Back to counting nothing, and the same again adds up.
    trainer.compute_gradients(camera, target);
    const DensifyStats after = trainer.densify_stats();
    for (std::size_t i = 0; i < cpu.size(); ++i) CHECK(after.markedPixels[i] == scored.markedPixels[i]);
  }
}

TEST_CASE("Scoring a view accumulates the edge score", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(96, 64); // Identity pose, focal 60: pixel x = 30 x + 48 at depth 2
  // One vertical edge down the middle: dark left, bright right.
  std::vector<std::uint8_t> target(static_cast<std::size_t>(camera.width) * camera.height * 4, 255);
  for (int y = 0; y < camera.height; ++y) {
    for (int x = 0; x < camera.width; ++x) {
      const std::size_t at = (static_cast<std::size_t>(y) * camera.width + x) * 4;
      target[at] = target[at + 1] = target[at + 2] = x < camera.width / 2 ? 40 : 220;
    }
  }
  // Eight small Gaussians on the edge, eight 24 pixels to its right.
  GaussianCloud cloud;
  cloud.resize(16, 0);
  for (std::size_t i = 0; i < 16; ++i) {
    cloud.positions[i * 3] = i < 8 ? 0.f : 0.8f;
    cloud.positions[i * 3 + 1] = (static_cast<float>(i % 8) - 3.5f) * 0.1f;
    cloud.positions[i * 3 + 2] = 2.f;
    for (std::size_t a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = std::log(0.02f);
    cloud.rotations[i * 4] = 1.f;
  }
  Trainer trainer(*ctx, cloud);
  trainer.score_view(camera, target, 0.5f, 1.f);
  const DensifyStats scored = trainer.densify_stats();
  REQUIRE(scored.edgeScore.size() == cloud.size());
  double nearSum = 0.0, farSum = 0.0;
  for (std::size_t i = 0; i < 8; ++i) nearSum += scored.edgeScore[i];
  for (std::size_t i = 8; i < 16; ++i) farSum += scored.edgeScore[i];
  std::println("  edge score on the edge {:.3f} per Gaussian, 24 pixels away {:.3f}", nearSum / 8, farSum / 8);
  CHECK(nearSum > 0.0);
  CHECK(farSum == 0.0);
  // The high-error count is untouched by the edge weight: the whole image is wrong.
  for (std::size_t i = 0; i < 16; ++i) CHECK(scored.markedPixels[i] > 0.f);
}

TEST_CASE("SH dropout trains with the retained degrees only and never renders with them", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 17);
  const GaussianCloud cloud = random_cloud(300, 3, 7);
  const auto coefficients = static_cast<std::size_t>(cloud.shCoeffs());
  const ReferenceImage full = reference_render(cloud, camera, {});
  const auto worst_against = [&](const std::vector<float> &gpu, const ReferenceImage &cpu) {
    double worst = 0.0;
    for (std::size_t i = 0; i < cpu.rgb.size(); ++i) {
      for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(gpu[i * 4 + static_cast<std::size_t>(c)] - cpu.rgb[i][c]));
    }
    return worst;
  };
  // Gaussians whose coefficients above degree 0 are nonzero in `values`.
  const auto higher_degrees_nonzero = [&](const std::vector<float> &values) {
    std::size_t touched = 0;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
      bool any = false;
      for (std::size_t k = 1; k < coefficients; ++k) {
        for (int c = 0; c < 3; ++c) any = any || values[(i * coefficients + k) * 3 + static_cast<std::size_t>(c)] != 0.f;
      }
      touched += any ? 1 : 0;
    }
    return touched;
  };
  // Those whose coefficients above degree 0 a training step moved.
  Trainer trainer(*ctx, cloud);
  const auto higher_degrees_trained = [&](const LearningRates &rates) {
    const std::vector<float> before = trainer.cloud().sh;
    trainer.step(camera, target, rates);
    std::vector<float> moved = trainer.cloud().sh;
    for (std::size_t i = 0; i < moved.size(); ++i) moved[i] -= before[i];
    return higher_degrees_nonzero(moved);
  };

  trainer.set_active_sh_degree(3);
  LearningRates rates;
  rates.position = 0.f; // Only SH moves
  rates.scale = rates.rotation = rates.opacity = 0.f;
  rates.sh = 0.01f;

  // Dropout on every Gaussian: a training step trains degree 0 only, but
  // rendering, as the reports and the final evaluation do, still shows everything.
  trainer.set_sh_dropout(1.f, 0, 1);
  CHECK(worst_against(trainer.render(camera), full) < 1e-3);
  CHECK(higher_degrees_trained(rates) == 0);
  trainer.compute_gradients(camera, target); // Statistics for densification: not training, not dropped
  CHECK(higher_degrees_nonzero(trainer.gradients().sh) > 250);

  // Half the Gaussians: about half keep their higher degrees in a step.
  trainer.set_sh_dropout(0.5f, 0, 9);
  const std::size_t kept = higher_degrees_trained(rates);
  std::println("  SH dropout at 0.5: {} of {} Gaussians kept their higher degrees in a step", kept, cloud.size());
  CHECK(kept > 100);
  CHECK(kept < 200);
  trainer.set_sh_dropout(0.f, 0, 1);
  CHECK(higher_degrees_trained(rates) > 250);
}

TEST_CASE("Pose gradients match finite differences of the loss", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 53);
  const GaussianCloud cloud = random_cloud(300, 0, 59);
  const auto loss_at = [&](const std::array<float, 6> &move) {
    return reference_loss(reference_render(cloud, refined_pose(camera, move), {.noCutoffs = true}), target, 0.2).loss;
  };
  std::vector<double> numeric(6);
  for (std::size_t k = 0; k < 6; ++k) {
    std::array<float, 6> up{}, down{};
    up[k] = 1e-3f;
    down[k] = -1e-3f;
    numeric[k] = (loss_at(up) - loss_at(down)) / 2e-3;
  }
  TrainerOptions options;
  options.poseGradients = true;
  Trainer trainer(*ctx, cloud, options);
  const StepStats stats = trainer.compute_gradients(camera, target);
  std::println("  pose:");
  // Not exact: it leaves out how the camera turns each Gaussian's shape.
  check_close(to_double(std::vector<float>(stats.poseGradient.begin(), stats.poseGradient.end())), numeric, "pose", 0.95, 0.3);

  // The move is the one the gradient was taken for: a point in camera space turns and shifts.
  const Camera moved = refined_pose(camera, {0.f, 0.1f, 0.f, 0.2f, 0.f, 0.f});
  const glm::vec3 point = glm::vec3(moved.view * glm::vec4(0.f, 0.f, 3.f, 1.f));
  CHECK_THAT(point.x, WithinAbs(3.f * std::sin(0.1f) + 0.2f, 1e-5));
  CHECK_THAT(point.z, WithinAbs(3.f * std::cos(0.1f), 1e-5));
  const glm::vec3 eye = glm::vec3(moved.view * glm::vec4(moved.position, 1.f));
  CHECK_THAT(glm::length(eye), WithinAbs(0.f, 1e-5));
}

TEST_CASE("Second-order steps fit a view within their trust region", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  if (!ctx->info().floatAtomicAdd) SKIP("Needs float atomics");
  const Camera camera = test_camera(72, 56);
  const auto target = noise_target(camera.width, camera.height, 61);
  const GaussianCloud cloud = random_cloud(120, 1, 67);
  Trainer trainer(*ctx, cloud);
  trainer.enable_second_order();
  LearningRates rates;
  rates.trustRegion = 1e-5f;

  // The curvature of J^T J is never negative, and only exists for what is in view.
  trainer.compute_gradients(camera, target);
  trainer.update_curvature();
  const GaussianCloud before = trainer.cloud();
  const StepStats first = trainer.step(camera, target, rates);
  const GaussianCloud after = trainer.cloud();
  // The trust region holds: opacity at 0.5 moves no more than sqrt(4 * 0.5 * e) / 0.25 in logit.
  for (std::size_t i = 0; i < after.size(); ++i) {
    const float a = std::max(1.f / (1.f + std::exp(-before.opacities[i])), 1e-3f);
    const float radius = std::sqrt(4.f * a * rates.trustRegion) / std::max(a * (1.f - a), 1e-3f);
    REQUIRE(std::abs(after.opacities[i] - before.opacities[i]) <= radius * 1.001f);
  }
  float loss = first.loss;
  for (int i = 0; i < 300; ++i) {
    loss = trainer.step(camera, target, rates).loss;
    if (i % 10 == 0) trainer.update_curvature();
  }
  REQUIRE(std::isfinite(loss));
  CHECK(loss < first.loss);
}

TEST_CASE("A prefetched image of a smaller size does not spoil a larger one after it", "[train][vulkan]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) SKIP("No usable Vulkan device");
  const Camera small = test_camera(64, 48), large = test_camera(1024, 768);
  const auto smallTarget = noise_target(small.width, small.height, 71);
  const auto largeTarget = noise_target(large.width, large.height, 73), otherTarget = noise_target(large.width, large.height, 79);
  const GaussianCloud cloud = random_cloud(60, 0, 83);
  Trainer reference(*ctx, cloud);
  const float expected = reference.compute_gradients(large, largeTarget).loss;

  LearningRates still;
  still.position = still.scale = still.rotation = still.opacity = still.sh = 0.f;
  Trainer trainer(*ctx, cloud);
  // Large enough image buffers come first, so that the size change below reallocates nothing.
  (void)trainer.render(large);
  // The prefetch of the small image becomes the staging buffer.
  trainer.prefetch_target(smallTarget);
  (void)trainer.compute_gradients(small, smallTarget);
  // Overflowing would show only by chance, in what lies next to it, so the size itself is checked.
  CHECK_THAT(trainer.step(large, largeTarget, still, otherTarget).loss, WithinAbs(expected, 1e-4));
  CHECK(trainer.staging_bytes() >= largeTarget.size());
  // A prefetch is used, by the step after the one that started it.
  (void)trainer.step(large, largeTarget, still, largeTarget);
  CHECK_THAT(trainer.step(large, largeTarget, still).loss, WithinAbs(expected, 1e-4));
}

TEST_CASE("Training holds out views in file name order and samples the sparse points", "[train]") {
  namespace fs = std::filesystem;
  const fs::path project = fs::temp_directory_path() / "splat_train_order";
  fs::remove_all(project);
  fs::create_directories(project / "images");
  const auto views = make_test_views(9, 64, 48, 60.0);
  ColmapModel model;
  ColmapCamera camera;
  camera.id = 1;
  camera.model = 1; // PINHOLE
  camera.width = 64;
  camera.height = 48;
  camera.params = {60.0, 60.0, 32.0, 24.0};
  model.cameras[camera.id] = camera;
  for (std::size_t i = 0; i < views.size(); ++i) {
    const std::string name = std::format("images/view_{}.png", i);
    write_image(views[i].image, project / name, 95);
    ColmapImage image;
    image.id = static_cast<std::uint32_t>(views.size() - i); // Numbered against name order, as COLMAP may do
    const Eigen::Quaterniond q(views[i].pose.R);
    image.rotation = {q.w(), q.x(), q.y(), q.z()};
    image.translation = {views[i].pose.t.x(), views[i].pose.t.y(), views[i].pose.t.z()};
    image.cameraId = camera.id;
    image.name = name;
    model.images[image.id] = image;
  }
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
  for (std::uint64_t p = 0; p < 500; ++p) {
    ColmapPoint point;
    point.id = p + 1;
    point.position = {coordinate(rng), coordinate(rng), coordinate(rng)};
    point.color = {100, 150, 200};
    model.points.push_back(point);
  }
  save_colmap_model(model, project / "sparse" / "0");

  TrainOptions options;
  options.project = project;
  options.holdout = 4;
  options.densePoints = 0;
  options.sparsePoints = 100;
  const TrainData data = load_train_data(options);
  REQUIRE(data.holdoutViews.size() == 3);
  CHECK(data.holdoutViews[0].name == "images/view_0.png");
  CHECK(data.holdoutViews[1].name == "images/view_4.png");
  CHECK(data.holdoutViews[2].name == "images/view_8.png");
  CHECK(data.trainViews.size() == 6);
  CHECK(data.startingPoints.size() == 100);

  options.sparsePoints = 0;
  CHECK(load_train_data(options).startingPoints.size() == 500);
  fs::remove_all(project);
}

TEST_CASE("The growth cap scales with the length of training", "[train]") {
  TrainOptions options;
  options.iterations = 7000;
  CHECK(growth_factor(options) == 4.0);
  options.iterations = 3000;
  CHECK(growth_factor(options) == 4.0);
  options.iterations = 35000;
  CHECK(growth_factor(options) == 20.0);
  options.maxGrowth = 6.0;
  CHECK(growth_factor(options) == 6.0);
  options.maxGrowth = 0.0;
  CHECK(growth_factor(options) == 0.0);
}
