#include "train/train.hpp"

#include "ingest/image.hpp"
#include "render/camera.hpp"
#include "sfm/colmap_model.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/densify.hpp"
#include "train/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <tuple>

namespace splat {

namespace fs = std::filesystem;

namespace {

constexpr float kInitialOpacity = 0.1f;
constexpr float kResetOpacity = 0.01f;
constexpr double kDenseFraction = 0.01;      // Of the scene's extent: larger Gaussians split, smaller clone
// Larger than either of these and a Gaussian is removed. They only catch true
// outliers. Tight limits take big Gaussians out of walls and floors faster than
// new ones can fill the holes, and quality collapses: 20 pixels on screen cost
// 10 dB in a run of 7000 iterations at 800x600.
constexpr double kMaxWorldFraction = 1.0;    // Of the scene's extent
constexpr double kMaxScreenFraction = 0.25;  // Of the image's longer side
constexpr int kDensifyInterval = 100;
constexpr int kOpacityResetInterval = 3000;
constexpr std::uint64_t kBudgetReserve = 256ull * 1024 * 1024; // Sort scratch, staging, the driver's share
constexpr float kShC0 = 0.28209479177387814f;

struct View {
  Camera camera;
  std::vector<std::uint8_t> rgba;
  std::string name;
};

glm::dvec2 focal_of(const ColmapCamera &c) {
  // PINHOLE, OPENCV and FULL_OPENCV list fx and fy; the others share one f.
  if (c.model == 1 || c.model == 4 || c.model == 6) return {c.params[0], c.params[1]};
  return {c.params[0], c.params[0]};
}

glm::dvec2 principal_of(const ColmapCamera &c) {
  if (c.model == 1 || c.model == 4 || c.model == 6) return {c.params[2], c.params[3]};
  return {c.params[1], c.params[2]};
}

// Resamples an image as a pinhole camera would have seen it. The renderer has
// no lens distortion, so the training images must not have any either.
RgbImage undistort(const RgbImage &image, const ColmapCamera &camera) {
  const glm::dvec2 f = focal_of(camera), c = principal_of(camera);
  RgbImage out;
  out.width = image.width;
  out.height = image.height;
  out.pixels.assign(image.pixels.size(), 0);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      const glm::dvec3 ray((x + 0.5 - c.x) / f.x, (y + 0.5 - c.y) / f.y, 1.0);
      glm::dvec2 source;
      if (!colmap_project(camera, ray, source)) {
        throw std::runtime_error(std::format("training does not support camera model {}", colmap_model_name(camera.model)));
      }
      const double sx = source.x - 0.5, sy = source.y - 0.5;
      const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
      if (x0 < 0 || y0 < 0 || x0 + 1 >= image.width || y0 + 1 >= image.height) continue;
      const double u = sx - x0, v = sy - y0;
      const std::uint8_t *row0 = image.pixels.data() + (static_cast<std::size_t>(y0) * image.width + x0) * 3;
      const std::uint8_t *row1 = row0 + static_cast<std::size_t>(image.width) * 3;
      for (int k = 0; k < 3; ++k) {
        const double top = row0[k] * (1.0 - u) + row0[k + 3] * u;
        const double bottom = row1[k] * (1.0 - u) + row1[k + 3] * u;
        out.pixels[(static_cast<std::size_t>(y) * out.width + x) * 3 + k] = static_cast<std::uint8_t>(std::lround(top * (1.0 - v) + bottom * v));
      }
    }
  }
  return out;
}

bool has_distortion(const ColmapCamera &camera) {
  const std::size_t first = camera.model == 0 ? 3 : camera.model == 1 ? 4 : camera.model == 2 || camera.model == 3 ? 3 : 4;
  return std::any_of(camera.params.begin() + static_cast<std::ptrdiff_t>(std::min(first, camera.params.size())), camera.params.end(),
                     [](double p) { return p != 0.0; });
}

std::vector<std::uint8_t> read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

View load_view(const fs::path &project, const ColmapImage &image, const ColmapCamera &camera, int maxSize) {
  const fs::path file = project / image.name;
  RgbImage rgb = decode_image(read_file(file));
  if (static_cast<std::uint64_t>(rgb.width) != camera.width || static_cast<std::uint64_t>(rgb.height) != camera.height) {
    throw std::runtime_error(std::format("{} is {}x{} but its camera says {}x{}", file.string(), rgb.width, rgb.height, camera.width, camera.height));
  }
  if (has_distortion(camera)) rgb = undistort(rgb, camera);
  const int original = std::max(rgb.width, rgb.height);
  rgb = downscale_to_fit(std::move(rgb), maxSize);
  const double scale = static_cast<double>(std::max(rgb.width, rgb.height)) / original;

  View view;
  view.name = image.name;
  const glm::dvec2 f = focal_of(camera) * scale, c = principal_of(camera) * scale;
  view.camera.fx = static_cast<float>(f.x);
  view.camera.fy = static_cast<float>(f.y);
  view.camera.cx = static_cast<float>(c.x);
  view.camera.cy = static_cast<float>(c.y);
  view.camera.width = rgb.width;
  view.camera.height = rgb.height;

  // World to camera, from the quaternion (w, x, y, z).
  const double w = image.rotation[0], x = image.rotation[1], y = image.rotation[2], z = image.rotation[3];
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  const double qw = w / n, qx = x / n, qy = y / n, qz = z / n;
  const glm::dmat3 R(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy),
                     2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx),
                     2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
  glm::dmat4 viewMatrix(R);
  viewMatrix[3] = glm::dvec4(image.translation, 1.0);
  view.camera.view = glm::mat4(viewMatrix);
  view.camera.position = glm::vec3(image.center());

  view.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
  for (std::size_t i = 0; i < static_cast<std::size_t>(rgb.width) * rgb.height; ++i) {
    std::copy_n(rgb.pixels.data() + i * 3, 3, view.rgba.data() + i * 4);
    view.rgba[i * 4 + 3] = 255;
  }
  return view;
}

// The distance to each point's 3 nearest neighbours, as a starting size. A
// uniform grid keeps this close to linear in the number of points.
std::vector<float> neighbour_distances(const std::vector<ColmapPoint> &points) {
  glm::dvec3 low(std::numeric_limits<double>::max()), high(std::numeric_limits<double>::lowest());
  for (const ColmapPoint &p : points) {
    low = glm::min(low, p.position);
    high = glm::max(high, p.position);
  }
  const glm::dvec3 size = high - low;
  const double cell = std::max(std::cbrt(std::max(size.x * size.y * size.z, 1e-30) / static_cast<double>(points.size()) * 8.0),
                               std::max({size.x, size.y, size.z}) / 512.0);
  using Key = std::tuple<int, int, int>;
  const auto key_of = [&](const glm::dvec3 &p) {
    return Key{static_cast<int>((p.x - low.x) / cell), static_cast<int>((p.y - low.y) / cell), static_cast<int>((p.z - low.z) / cell)};
  };
  std::map<Key, std::vector<std::size_t>> grid;
  for (std::size_t i = 0; i < points.size(); ++i) grid[key_of(points[i].position)].push_back(i);

  std::vector<float> out(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto [cx, cy, cz] = key_of(points[i].position);
    double best[3] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for (int r = 1; r <= 4 && best[2] == std::numeric_limits<double>::max(); ++r) {
      best[0] = best[1] = best[2] = std::numeric_limits<double>::max();
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            const auto it = grid.find({cx + dx, cy + dy, cz + dz});
            if (it == grid.end()) continue;
            for (const std::size_t j : it->second) {
              if (j == i) continue;
              const glm::dvec3 d = points[j].position - points[i].position;
              const double dist = glm::dot(d, d);
              if (dist < best[2]) {
                best[2] = dist;
                if (best[2] < best[1]) std::swap(best[1], best[2]);
                if (best[1] < best[0]) std::swap(best[0], best[1]);
              }
            }
          }
        }
      }
    }
    const double mean = best[2] == std::numeric_limits<double>::max()
                            ? cell
                            : (std::sqrt(best[0]) + std::sqrt(best[1]) + std::sqrt(best[2])) / 3.0;
    out[i] = static_cast<float>(std::max(mean, 1e-7));
  }
  return out;
}

GaussianCloud initial_cloud(const ColmapModel &model, int shDegree) {
  GaussianCloud cloud;
  cloud.resize(model.points.size(), shDegree);
  const std::vector<float> distances = neighbour_distances(model.points);
  const auto coefficients = static_cast<std::size_t>(cloud.shCoeffs());
  const float logitOpacity = std::log(kInitialOpacity / (1.f - kInitialOpacity));
  for (std::size_t i = 0; i < model.points.size(); ++i) {
    const ColmapPoint &p = model.points[i];
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.positions[i * 3 + a] = static_cast<float>(p.position[static_cast<int>(a)]);
      cloud.scales[i * 3 + a] = std::log(distances[i]);
      cloud.sh[i * coefficients * 3 + a] = (static_cast<float>(p.color[a]) / 255.f - 0.5f) / kShC0;
    }
    cloud.rotations[i * 4] = 1.f;
    cloud.opacities[i] = logitOpacity;
  }
  return cloud;
}

double psnr_of(const std::vector<float> &render, const std::vector<std::uint8_t> &target) {
  double sum = 0.0;
  const std::size_t pixels = target.size() / 4;
  for (std::size_t i = 0; i < pixels; ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      const double d = std::clamp(static_cast<double>(render[i * 4 + c]), 0.0, 1.0) - target[i * 4 + c] / 255.0;
      sum += d * d;
    }
  }
  const double mse = sum / (3.0 * static_cast<double>(pixels));
  return mse > 0.0 ? -10.0 * std::log10(mse) : 100.0;
}

void write_preview(const std::vector<float> &render, const View &view, const fs::path &path) {
  // Rendered on the left, the real image on the right.
  RgbImage out;
  out.width = view.camera.width * 2;
  out.height = view.camera.height;
  out.pixels.resize(static_cast<std::size_t>(out.width) * out.height * 3);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < view.camera.width; ++x) {
      const std::size_t at = static_cast<std::size_t>(y) * view.camera.width + x;
      for (std::size_t c = 0; c < 3; ++c) {
        const std::size_t base = static_cast<std::size_t>(y) * out.width;
        out.pixels[(base + x) * 3 + c] = static_cast<std::uint8_t>(std::clamp(render[at * 4 + c], 0.f, 1.f) * 255.f + 0.5f);
        out.pixels[(base + view.camera.width + x) * 3 + c] = view.rgba[at * 4 + c];
      }
    }
  }
  write_image(out, path, 95);
}

} // namespace

TrainReport run_train(vk::Context &ctx, const TrainOptions &options) {
  const ColmapModel model = load_colmap_model(options.project / "sparse" / "0");
  if (model.points.empty() || model.images.empty()) throw std::runtime_error("the sparse model has no points or no images");

  std::vector<View> trainViews, holdoutViews;
  int number = 0;
  for (const auto &[id, image] : model.images) {
    if (options.maxViews > 0 && number >= options.maxViews) break;
    View view = load_view(options.project, image, model.cameras.at(image.cameraId), options.maxSize);
    if (options.holdout > 0 && number % options.holdout == options.holdout - 1) {
      holdoutViews.push_back(std::move(view));
    } else {
      trainViews.push_back(std::move(view));
    }
    ++number;
  }
  if (trainViews.empty()) throw std::runtime_error("no training views");

  // The scene's size sets the position learning rate and the near plane.
  glm::dvec3 middle(0.0);
  for (const auto &[id, image] : model.images) middle += image.center();
  middle /= static_cast<double>(model.images.size());
  double extent = 0.0;
  for (const auto &[id, image] : model.images) extent = std::max(extent, glm::length(image.center() - middle));
  extent = std::max(extent * 1.1, 1e-6);
  std::vector<double> depths;
  for (const ColmapPoint &p : model.points) depths.push_back(glm::length(p.position - middle));
  std::nth_element(depths.begin(), depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2), depths.end());
  const double typicalDepth = std::max(depths[depths.size() / 2] + extent, 1e-6);

  TrainerOptions trainerOptions;
  trainerOptions.lambdaSsim = options.lambdaSsim;
  trainerOptions.nearPlane = static_cast<float>(typicalDepth * 0.01);
  // Asked for before anything is allocated, so that it is all ours to spend.
  double budgetGiB = options.vramBudgetGiB;
  if (budgetGiB <= 0.0) budgetGiB = 0.9 * static_cast<double>(ctx.free_device_memory()) / (1024.0 * 1024.0 * 1024.0);
  const auto budgetBytes = static_cast<std::uint64_t>(budgetGiB * 1024.0 * 1024.0 * 1024.0);
  const std::uint64_t reserve = std::max(kBudgetReserve, budgetBytes / 20);
  trainerOptions.memoryBudget = budgetBytes > reserve ? budgetBytes - reserve : 0;
  std::uint64_t largestView = 0;
  for (const View &view : trainViews) {
    largestView = std::max(largestView, static_cast<std::uint64_t>(view.camera.width) * view.camera.height);
  }
  if (largestView * Trainer::kBytesPerPixel >= trainerOptions.memoryBudget) {
    throw std::runtime_error(std::format("{:.2f} GiB of VRAM is not enough for images of this size; try --max-size",
                                         budgetGiB));
  }
  Trainer trainer(ctx, initial_cloud(model, options.shDegree), trainerOptions);

  // A view that does not fit gives up its largest Gaussians, rather than the
  // run ending. They grow back where they are needed, within the budget.
  const auto fitting = [&](auto &&work) {
    for (;;) {
      try {
        return work();
      } catch (const BudgetExceeded &e) {
        const std::size_t removed = trainer.remove_largest_gaussians(0.05);
        std::println("  {}: removed the {} largest Gaussians", e.what(), removed);
        if (removed == 0) throw;
      }
    }
  };

  // The schedule of the paper for 30000 iterations, proportional when shorter.
  const int densifyFrom = std::min(500, options.iterations / 20 + 1);
  const int densifyUntil = options.iterations / 2;
  const int shInterval = std::max(1, std::min(1000, options.iterations / 10));
  trainer.set_active_sh_degree(0);
  std::mt19937 densifyRng(options.seed + 1);
  double entriesPerVisible = 16.0;
  std::println("Training from {} Gaussians on {} views ({} held out) at {}x{} for {} iterations, SH degree {}, VRAM budget {:.1f} GiB{}",
               model.points.size(), trainViews.size(), holdoutViews.size(), trainViews[0].camera.width, trainViews[0].camera.height,
               options.iterations, options.shDegree, budgetGiB,
               trainer.uses_fixed_point() ? " (fixed-point gradients: this GPU has no float atomics)" : "");

  std::mt19937 rng(options.seed);
  std::vector<std::size_t> order;
  TrainReport report;
  LearningRates rates;
  const double positionStart = 1.6e-4 * extent, positionEnd = 1.6e-6 * extent;
  double lossSum = 0.0;
  int lossCount = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 1; iteration <= options.iterations; ++iteration) {
    if (order.empty()) {
      order.resize(trainViews.size());
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::shuffle(order.begin(), order.end(), rng);
    }
    const View &view = trainViews[order.back()];
    order.pop_back();

    const double progress = static_cast<double>(iteration - 1) / std::max(options.iterations - 1, 1);
    rates.position = static_cast<float>(positionStart * std::pow(positionEnd / positionStart, progress));
    trainer.set_active_sh_degree((iteration - 1) / shInterval);
    const StepStats stats = fitting([&] { return trainer.step(view.camera, view.rgba, rates); });
    if (!std::isfinite(stats.loss)) throw std::runtime_error(std::format("the loss became {} at iteration {}", stats.loss, iteration));
    if (iteration == 1) report.firstLoss = stats.loss;
    report.lastLoss = stats.loss;
    lossSum += stats.loss;
    ++lossCount;
    if (stats.visibleSplats > 0) {
      entriesPerVisible = 0.98 * entriesPerVisible + 0.02 * stats.tileEntries / stats.visibleSplats;
    }
    report.peakGpuBytes = std::max(report.peakGpuBytes, trainer.gpu_bytes());

    if (options.densify && iteration >= densifyFrom && iteration <= densifyUntil && iteration % kDensifyInterval == 0) {
      // What a Gaussian costs, with the tile entries it brings and room for them to grow.
      const double perGaussian = static_cast<double>(trainer.bytes_per_gaussian()) + 1.5 * Trainer::kBytesPerTileEntry * entriesPerVisible;
      const std::uint64_t available = trainerOptions.memoryBudget;
      const std::uint64_t pixelBytes = static_cast<std::uint64_t>(view.camera.width) * view.camera.height * Trainer::kBytesPerPixel;
      DensifyOptions densifyOptions;
      densifyOptions.gradientThreshold = options.gradientThreshold;
      densifyOptions.denseScale = static_cast<float>(kDenseFraction * extent);
      densifyOptions.maxWorldScale = static_cast<float>(kMaxWorldFraction * extent);
      densifyOptions.maxScreenRadius = static_cast<float>(kMaxScreenFraction * std::max(view.camera.width, view.camera.height));
      densifyOptions.pruneLarge = iteration > kOpacityResetInterval;
      densifyOptions.maxGaussians = available > pixelBytes ? static_cast<std::size_t>(static_cast<double>(available - pixelBytes) / perGaussian) : 0;
      if (options.maxGaussians > 0) densifyOptions.maxGaussians = std::min(densifyOptions.maxGaussians, options.maxGaussians);
      const DensifyResult result = trainer.densify(densifyOptions, densifyRng);
      report.heldBack += result.heldBack;
      if (iteration % kOpacityResetInterval == 0) trainer.reset_opacity(kResetOpacity);
    }

    if (iteration % options.reportEvery == 0 || iteration == options.iterations) {
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const View &probe = holdoutViews.empty() ? trainViews[0] : holdoutViews[0];
      const double probePsnr = psnr_of(fitting([&] { return trainer.render(probe.camera); }), probe.rgba);
      std::println("  iteration {:6}: loss {:.4f}, {} PSNR {:.2f} dB, {} Gaussians, {} visible, {} tile entries, {:.2f} GiB, {:.1f} it/s", iteration,
                   lossSum / lossCount, holdoutViews.empty() ? "training view" : "held-out view", probePsnr, trainer.gaussian_count(), stats.visibleSplats, stats.tileEntries,
                   static_cast<double>(trainer.gpu_bytes()) / (1024.0 * 1024.0 * 1024.0), iteration / seconds);
      lossSum = 0.0;
      lossCount = 0;
    }
    if (options.saveEvery > 0 && iteration % options.saveEvery == 0 && iteration < options.iterations) {
      // Through a temporary file, so that a crash never leaves half a checkpoint.
      const fs::path temporary = fs::path(options.out).concat(".tmp");
      save_ply(trainer.cloud(), temporary);
      fs::rename(temporary, options.out);
    }
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  report.iterationsPerSecond = options.iterations / seconds;

  trainer.set_active_sh_degree(options.shDegree);
  const auto mean_psnr = [&](const std::vector<View> &views) {
    double sum = 0.0;
    for (const View &view : views) sum += psnr_of(fitting([&] { return trainer.render(view.camera); }), view.rgba);
    return views.empty() ? 0.0 : sum / static_cast<double>(views.size());
  };
  report.trainPsnr = mean_psnr(trainViews);
  report.holdoutPsnr = mean_psnr(holdoutViews);

  trainer.set_active_sh_degree(options.shDegree);
  report.gaussians = trainer.gaussian_count();
  save_ply(trainer.cloud(), options.out);
  write_preview(trainer.render(trainViews[0].camera), trainViews[0], options.project / "preview_train.png");
  if (!holdoutViews.empty()) {
    write_preview(trainer.render(holdoutViews[0].camera), holdoutViews[0], options.project / "preview_holdout.png");
    std::println("PSNR {:.2f} dB on training views, {:.2f} dB on held-out views", report.trainPsnr, report.holdoutPsnr);
  } else {
    std::println("PSNR {:.2f} dB on training views", report.trainPsnr);
  }
  std::println("{} Gaussians, peak {:.2f} GiB of VRAM{}", report.gaussians, static_cast<double>(report.peakGpuBytes) / (1024.0 * 1024.0 * 1024.0),
               report.heldBack > 0 ? std::format(" ({} densifications held back by the budget)", report.heldBack) : "");
  std::string profile;
  double profileTotal = 0.0;
  for (const auto &[stage, ms] : trainer.timings()) {
    profile += std::format("{} {:.2f}, ", stage, ms);
    profileTotal += ms;
  }
  std::println("GPU time per iteration: {:.1f} ms ({})", profileTotal, profile.substr(0, profile.size() > 2 ? profile.size() - 2 : 0));
  std::println("Finished in {:.1f} s ({:.1f} it/s) -> {}", seconds, report.iterationsPerSecond, options.out.string());
  return report;
}

} // namespace splat
