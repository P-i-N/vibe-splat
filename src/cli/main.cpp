// splat: Gaussian splat generator and viewer.
//
// One program, one subcommand per pipeline stage. Each stage reads and writes
// a project folder, so any stage can be rerun on its own.

#include "cli/args.hpp"
#include "core/progress.hpp"
#include "core/selftest.hpp"
#include "core/vk_check.hpp"
#include "core/vk_context.hpp"
#include "ingest/ingest.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "sfm/sfm.hpp"
#include "sfm/test_views.hpp"
#include "splat/cgc.hpp"
#include "splat/gaussian_cloud.hpp"
#include "train/train.hpp"
#ifdef SPLAT_WITH_VIEWER
#include "viewer/viewer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <numbers>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kUsage = R"(splat {} - Gaussian splat generator and viewer (Vulkan)

Usage: splat <command> [options]

Commands:
  view <scene.ply|cgc>          Open the real-time viewer
  render <scene.ply|cgc> -o <png>  Render one image without a window
  make-test-scene <out.ply|cgc> Write a synthetic scene for trying things out
  convert <in> <out>            Convert between .ply and compressed .cgc
  info                          List Vulkan devices and whether splat can use them
  selftest                      Run a compute kernel on the GPU and check the result
  ingest <inputs...> -o <dir>   Import photos and videos into a project folder
  sfm <project>                 Estimate camera poses from an ingested project
  make-test-views <dir>         Render images with known poses, for trying sfm
  train <project>               Train Gaussians from a project with camera poses

render options:
  -o, --out <file.png>    Output image (required)
  --width <px>            Default 1280
  --height <px>           Default 720
  --yaw <deg>             Orbit angle around the up axis (default 0)
  --pitch <deg>           Elevation above the target (default 25)
  --distance <x>          Distance from the target (default: fit the scene)
  --fov <deg>             Vertical field of view (default 60)
  --up <axis>             World up: -y (default, COLMAP), +y, +z, -z

make-test-scene options:
  --count <n>             Number of Gaussians (default 200000)

convert options (both pruning tests are off unless given; either enables it):
  --prune-sigma <units>   Drop floaters: Gaussians whose largest axis (one standard
                          deviation) is longer than this many world units
  --prune-radius <factor> Drop Gaussians farther from the scene centre than this times
                          the radius that holds 90% of them

ingest options (inputs are image files, video files or folders of them):
  -o, --out <dir>         Project folder (required); gets images/ and ingest.json
  --max-size <px>         Longest side of stored images, 0 keeps it (default 1600)
  --png                   Store PNG instead of JPEG
  --jpeg-quality <1-100>  Default 95
  --fps <n>               Video frames kept per second, 0 keeps all (default 2)
  --max-frames <n>        Frames kept per video, lowering --fps to fit
  --threads <n>           Photo threads, 0 uses all cores (default 0)
  --force                 Replace a previous ingest in this project
Camera information, overriding EXIF for every input (video has no EXIF, so give
it here for a better start; sizes are for the input after rotation):
  --focal-35mm <mm>       35 mm equivalent focal length
  --focal-px <px>         Focal length in pixels
  --focal-mm <mm>         Real focal length, needs --sensor-width-mm
  --sensor-width-mm <mm>  Sensor size along the image's longer side
  --fov <deg>             Horizontal field of view
  --cx <px>, --cy <px>    Principal point (default: image centre)

sfm options (writes project/sparse/0 in COLMAP's format, plus preview.ply):
  --matcher <m>           auto (default), exhaustive or sequential
  --window <n>            Neighbours per image when sequential (default 8)
  --max-features <n>      Per image (default 4000)
  --threads <n>           0 uses all cores (default 0)
  --cpu                   Match features on the CPU even when a GPU is available
  --import <dir>          Instead of estimating, import a COLMAP sparse model

train options (writes project/splat.ply and project/preview_*.png):
  -o, --out <file>        Output file, .ply or compressed .cgc
  --iterations <n>        Default 30000
  --lambda-ssim <x>       Weight of D-SSIM against L1 (default 0.2)
  --sh-degree <0-3>       Default 3
  --max-size <px>         Shrink training images to this (default: keep size)
  --max-views <n>         Use only the first n views
  --holdout <n>           Keep every nth view out of training, to measure quality
  --seed <n>              Default 1
  --report-every <n>      Default 500
  --save-every <n>        Save the output this often, 0 only at the end (default 5000)
  --no-densify            Keep the initial Gaussians, without adding or removing any
  --gradient-threshold <x>  Lower adds more Gaussians (default 0.0002)
  --vram-budget <GiB>     Default: 90% of the GPU's free memory
  --max-gaussians <n>     Default: what the budget allows

make-test-views options:
  --count <n>             Number of images (default 12)
  --width <px>, --height <px>, --focal <px>   Default 640, 480, 500

view options:
  --up <axis>             As for render

Viewer controls:
  left mouse drag  orbit        right/middle drag  pan       wheel  zoom
  W A S D / Q E    move         R  reset view                Esc    quit
  gamepad: left stick move, right stick orbit, triggers zoom, A reset

Global options:
  --device <n>   Use Vulkan device n (see 'splat info')
  --validate     Enable the Vulkan validation layer, if installed
  -h, --help     Show this help
  --version      Show the version
)";

struct GlobalOptions {
  splat::vk::ContextOptions context;
  std::string_view command;
  std::vector<std::string_view> rest;
};

std::optional<std::uint32_t> parse_uint(std::string_view text) {
  std::uint32_t value = 0;
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

int cmd_info() {
  const auto devices = splat::vk::Context::enumerate_devices();
  if (devices.empty()) {
    std::println("No Vulkan devices found.");
    return 1;
  }
  for (const auto &d : devices) {
    std::println("[{}] {}{}", d.index, d.name, d.meetsRequirements ? "" : "  (not usable)");
    std::println("    vendor      {} ({})", splat::vk::vendor_name(d.vendorId), splat::vk::to_string(d.type));
    std::println("    driver      {}", d.driver);
    std::println("    Vulkan      {}.{}.{}", VK_API_VERSION_MAJOR(d.apiVersion),
                 VK_API_VERSION_MINOR(d.apiVersion), VK_API_VERSION_PATCH(d.apiVersion));
    std::println("    VRAM        {:.1f} GiB", gib(d.deviceLocalBytes));
    std::println("    subgroup    {}", d.subgroupSize);
    std::println("    float atomic add  {}", d.floatAtomicAdd ? "yes" : "no");
  }
  return 0;
}

int cmd_selftest(const GlobalOptions &opts) {
  splat::vk::Context ctx(opts.context);
  std::println("Device: {} ({})", ctx.info().name, splat::vk::vendor_name(ctx.info().vendorId));
  const auto result = splat::run_selftest(ctx);
  std::println("Selftest {}: {} ({:.2f} ms)", result.passed ? "PASSED" : "FAILED", result.message,
               result.gpuMilliseconds);
  return result.passed ? 0 : 1;
}

glm::vec3 parse_up(const std::string &axis) {
  if (axis == "-y") return {0.f, -1.f, 0.f};
  if (axis == "+y" || axis == "y") return {0.f, 1.f, 0.f};
  if (axis == "+z" || axis == "z") return {0.f, 0.f, 1.f};
  if (axis == "-z") return {0.f, 0.f, -1.f};
  throw std::runtime_error(std::format("--up must be one of -y, +y, +z, -z (got '{}')", axis));
}

splat::GaussianCloud load_scene(const std::string &path) {
  const auto start = std::chrono::steady_clock::now();
  auto cloud = splat::load_cloud(path);
  const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::println("Loaded {} Gaussians (SH degree {}) from {} in {:.0f} ms", cloud.size(), cloud.shDegree, path, ms);
  return cloud;
}

int cmd_render(const GlobalOptions &opts) {
  std::vector<std::string_view> rest = opts.rest;
  for (auto &a : rest) {
    if (a == "-o") a = "--out";
  }
  const splat::cli::Args args(rest, {});
  args.allow_only({"out", "width", "height", "yaw", "pitch", "distance", "fov", "up"});
  if (args.positionals().size() != 1 || !args.has("out")) {
    std::println(stderr, "Usage: splat render <scene.ply> -o <out.png> [options]  (see splat --help)");
    return 2;
  }
  const auto cloud = load_scene(args.positionals()[0]);
  const int width = std::clamp(args.number("width", 1280), 16, 16384);
  const int height = std::clamp(args.number("height", 720), 16, 16384);

  auto orbit = splat::default_orbit(cloud, parse_up(args.get("up", "-y")));
  orbit.yawDeg = args.number("yaw", orbit.yawDeg);
  orbit.pitchDeg = args.number("pitch", orbit.pitchDeg);
  orbit.distance = args.number("distance", orbit.distance);
  orbit.fovDeg = args.number("fov", orbit.fovDeg);

  splat::vk::Context ctx(opts.context);
  splat::SplatRenderer renderer(ctx, VK_FORMAT_R8G8B8A8_UNORM);
  renderer.set_scene(cloud);
  const auto image = splat::render_offscreen(ctx, renderer, orbit.camera(width, height));
  splat::write_png(image, args.get("out", ""));
  std::println("Rendered {} of {} splats at {}x{} on {} in {:.1f} ms -> {}", image.visibleSplats, cloud.size(),
               width, height, ctx.info().name, image.milliseconds, args.get("out", ""));
  return 0;
}

int cmd_make_test_scene(const GlobalOptions &opts) {
  const splat::cli::Args args(opts.rest, {});
  args.allow_only({"count"});
  if (args.positionals().size() != 1) {
    std::println(stderr, "Usage: splat make-test-scene <out.ply> [--count N]");
    return 2;
  }
  const auto count = args.number<std::size_t>("count", 200'000);
  const auto cloud = splat::make_test_scene(std::max<std::size_t>(count, 16));
  splat::save_cloud(cloud, args.positionals()[0]);
  std::println("Wrote {} Gaussians to {}", cloud.size(), args.positionals()[0]);
  return 0;
}

int cmd_convert(const GlobalOptions &opts) {
  const splat::cli::Args args(opts.rest, {});
  args.allow_only({"prune-sigma", "prune-radius"});
  if (args.positionals().size() != 2) {
    std::println(stderr, "Usage: splat convert <in.ply|cgc> <out.ply|cgc> [--prune-sigma <units>] [--prune-radius <factor>]");
    return 2;
  }
  const std::filesystem::path in(args.positionals()[0]);
  const std::filesystem::path out(args.positionals()[1]);
  splat::PruneOptions prune;
  prune.maxSigma = args.number("prune-sigma", 0.f);
  prune.maxRadiusFactor = args.number("prune-radius", 0.f);
  for (const auto [name, value] : {std::pair{"prune-sigma", prune.maxSigma}, std::pair{"prune-radius", prune.maxRadiusFactor}}) {
    if (args.has(name) && !(value > 0.f)) {
      throw std::runtime_error(std::format("--{} must be greater than 0", name));
    }
  }
  const bool pruning = prune.maxSigma > 0.f || prune.maxRadiusFactor > 0.f;
  const bool toCgc = splat::cloud_format(out) == splat::CloudFormat::Cgc;

  auto cloud = load_scene(in.string());
  const auto start = std::chrono::steady_clock::now();
  // CGC cannot store NaNs, and the floater tests need finite values to compare.
  if (toCgc || pruning) {
    if (const std::size_t removed = splat::remove_non_finite(cloud); removed > 0) {
      std::println("Left out {} Gaussians with NaN or infinite values", removed);
    }
  }
  if (pruning) {
    const std::size_t removed = splat::prune_floaters(cloud, prune);
    std::println("Pruned {} floaters, {} Gaussians left", removed, cloud.size());
  }
  if (toCgc) {
    const auto stats = splat::save_cgc(cloud, out);
    std::println("SH degree per Gaussian: 0: {}, 1: {}, 2: {}, 3: {}", stats.degreeCounts[0], stats.degreeCounts[1],
                 stats.degreeCounts[2], stats.degreeCounts[3]);
  } else {
    splat::save_ply(cloud, out);
  }
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const auto inBytes = std::filesystem::file_size(in);
  const auto outBytes = std::filesystem::file_size(out);
  std::println("Wrote {} in {:.1f} s: {:.2f} MB -> {:.2f} MB ({:.1f}x smaller)", out.string(), seconds, inBytes / 1e6,
               outBytes / 1e6, static_cast<double>(inBytes) / static_cast<double>(std::max<std::uintmax_t>(outBytes, 1)));
  return 0;
}

int cmd_view([[maybe_unused]] const GlobalOptions &opts) {
#ifdef SPLAT_WITH_VIEWER
  const splat::cli::Args args(opts.rest, {});
  args.allow_only({"up"});
  if (args.positionals().size() != 1) {
    std::println(stderr, "Usage: splat view <scene.ply> [--up -y|+y|+z|-z]");
    return 2;
  }
  const auto cloud = load_scene(args.positionals()[0]);
  splat::ViewerOptions vo;
  vo.context = opts.context;
  vo.up = parse_up(args.get("up", "-y"));
  vo.title = args.positionals()[0];
  return splat::run_viewer(cloud, vo);
#else
  std::println(stderr, "This build of splat has no window viewer (built without SDL3). Use 'splat render'.");
  return 2;
#endif
}

int cmd_ingest(const GlobalOptions &opts) {
  std::vector<std::string_view> rest = opts.rest;
  for (auto &a : rest) {
    if (a == "-o") a = "--out";
  }
  const splat::cli::Args args(rest, {"--png", "--force"});
  args.allow_only({"out", "max-size", "png", "jpeg-quality", "fps", "max-frames", "threads", "force", "focal-35mm",
                   "focal-px", "focal-mm", "sensor-width-mm", "fov", "cx", "cy"});
  if (args.positionals().empty() || !args.has("out")) {
    std::println(stderr, "Usage: splat ingest <images, videos or folders...> -o <project folder> [options]  (see splat --help)");
    return 2;
  }

  const auto optional_number = [&](const std::string &name) -> std::optional<double> {
    if (!args.has(name)) return std::nullopt;
    const double value = args.number(name, 0.0);
    if (!(value > 0.0)) throw std::runtime_error(std::format("--{} must be positive", name));
    return value;
  };

  splat::IngestOptions options;
  options.inputs.assign(args.positionals().begin(), args.positionals().end());
  options.project = args.get("out", "");
  options.maxSize = std::max(args.number("max-size", options.maxSize), 0);
  options.jpegQuality = std::clamp(args.number("jpeg-quality", options.jpegQuality), 1, 100);
  options.png = args.has("png");
  options.force = args.has("force");
  options.threads = std::max(args.number("threads", 0), 0);
  options.video.fps = std::max(args.number("fps", options.video.fps), 0.0);
  options.video.maxFrames = std::max(args.number("max-frames", 0), 0);
  options.manual.focal35mm = optional_number("focal-35mm");
  options.manual.focalPx = optional_number("focal-px");
  options.manual.focalMm = optional_number("focal-mm");
  options.manual.sensorWidthMm = optional_number("sensor-width-mm");
  options.manual.fovDeg = optional_number("fov");
  if (options.manual.fovDeg && *options.manual.fovDeg >= 180.0) throw std::runtime_error("--fov must be below 180");
  options.manual.cx = optional_number("cx");
  options.manual.cy = optional_number("cy");

  const auto start = std::chrono::steady_clock::now();
  const auto report = splat::run_ingest(options);
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::println("Ingested {} photos and {} video frames ({} cameras, {} skipped) in {:.1f} s -> {}", report.images,
               report.videoFrames, report.cameras, report.failed, seconds, options.project.string());

  const auto manifest = splat::load_manifest(options.project / "ingest.json");
  bool guessed = false;
  for (const auto &camera : manifest.cameras) {
    std::println("  camera {}: {}x{}, focal {:.1f} px ({}){}{}", camera.id, camera.width, camera.height, camera.fx,
                 splat::to_string(camera.focalSource), camera.make.empty() ? "" : ", ",
                 camera.make.empty() ? "" : camera.make + " " + camera.model);
    guessed = guessed || camera.focalSource == splat::FocalSource::Default;
  }
  if (guessed) {
    std::println("  default: no EXIF focal length in the files, so the focal is guessed as 1.2 x the longest side.");
    std::println("  splat sfm refines it; pass --focal-35mm, --focal-px or --fov to start from the real value.");
  }
  return 0;
}

int cmd_sfm(const GlobalOptions &opts) {
  const splat::cli::Args args(opts.rest, {"--cpu"});
  args.allow_only({"import", "matcher", "window", "max-features", "threads", "cpu"});
  if (args.positionals().size() != 1) {
    std::println(stderr, "Usage: splat sfm <project> [options]  (see splat --help)");
    return 2;
  }
  const std::string project = args.positionals()[0];
  const auto print_stats = [&](const splat::ColmapStats &stats, std::size_t ingested, std::size_t unregistered) {
    std::println("Posed {} of {} images, {} points -> {}/sparse/0", stats.images, ingested, stats.points, project);
    std::println("  mean track length {:.2f}, {:.0f} observations per image, reprojection error {:.3f} px",
                 stats.meanTrackLength, stats.meanObservationsPerImage, stats.meanReprojectionError);
    if (unregistered > 0) std::println("  {} images have no pose and will not be used", unregistered);
  };

  if (args.has("import")) {
    const auto report = splat::import_colmap_model(project, args.get("import", ""));
    print_stats(report.stats, report.ingested, report.unregistered);
    return 0;
  }

  splat::SfmOptions options;
  options.project = project;
  options.maxFeatures = std::max(args.number("max-features", options.maxFeatures), 16);
  options.window = std::max(args.number("window", options.window), 1);
  options.threads = std::max(args.number("threads", 0), 0);
  const std::string matcher = args.get("matcher", "auto");
  if (matcher == "exhaustive") options.matcher = splat::Matcher::Exhaustive;
  else if (matcher == "sequential") options.matcher = splat::Matcher::Sequential;
  else if (matcher != "auto") throw std::runtime_error(std::format("--matcher must be auto, exhaustive or sequential (got '{}')", matcher));

  // Matching runs on the GPU when there is one; the rest is CPU work.
  std::optional<splat::vk::Context> gpu;
  if (!args.has("cpu")) {
    try {
      gpu.emplace(opts.context);
      options.gpu = &*gpu;
    } catch (const std::exception &e) {
      std::println("No usable GPU, matching on the CPU: {}", e.what());
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const auto report = splat::run_sfm(options);
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  print_stats(report.stats, report.ingested, report.unregistered);
  std::println("  {} of {} image pairs matched, {:.1f} s in total", report.pairsVerified, report.pairsTried, seconds);
  for (const auto &camera : report.cameras) {
    constexpr double kRadToDeg = 180.0 / std::numbers::pi;
    const double fovDeg = 2.0 * std::atan(0.5 * camera.width / camera.focal) * kRadToDeg;
    if (camera.focalSource == splat::FocalSource::Manual) {
      std::println("  camera {}: focal {:.1f} px kept as given, {:.1f} degree horizontal field of view", camera.id, camera.focal, fovDeg);
    } else {
      std::println("  camera {}: focal refined {:.1f} -> {:.1f} px ({} start), {:.1f} degree horizontal field of view", camera.id,
                   camera.initialFocal, camera.focal, splat::to_string(camera.focalSource), fovDeg);
    }
  }
  return 0;
}

int cmd_train(const GlobalOptions &opts) {
  std::vector<std::string_view> rest = opts.rest;
  for (auto &a : rest) {
    if (a == "-o") a = "--out";
  }
  const splat::cli::Args args(rest, {"--no-densify"});
  args.allow_only({"out", "iterations", "lambda-ssim", "sh-degree", "max-size", "max-views", "holdout", "seed", "report-every", "save-every",
                   "no-densify", "gradient-threshold", "vram-budget", "max-gaussians"});
  if (args.positionals().size() != 1) {
    std::println(stderr, "Usage: splat train <project> [options]  (see splat --help)");
    return 2;
  }
  splat::TrainOptions options;
  options.project = args.positionals()[0];
  options.out = args.get("out", (options.project / "splat.ply").string());
  options.iterations = std::max(args.number("iterations", options.iterations), 1);
  options.lambdaSsim = std::clamp(args.number("lambda-ssim", options.lambdaSsim), 0.f, 1.f);
  options.shDegree = std::clamp(args.number("sh-degree", options.shDegree), 0, 3);
  options.maxSize = std::max(args.number("max-size", 0), 0);
  options.maxViews = std::max(args.number("max-views", 0), 0);
  options.holdout = std::max(args.number("holdout", 0), 0);
  options.seed = args.number("seed", options.seed);
  options.reportEvery = std::max(args.number("report-every", options.reportEvery), 1);
  options.saveEvery = std::max(args.number("save-every", options.saveEvery), 0);
  options.densify = !args.has("no-densify");
  options.gradientThreshold = args.number("gradient-threshold", options.gradientThreshold);
  options.vramBudgetGiB = args.number("vram-budget", 0.0);
  options.maxGaussians = args.number<std::size_t>("max-gaussians", 0);
  splat::vk::Context ctx(opts.context);
  std::println("Device: {}", ctx.info().name);
  splat::run_train(ctx, options);
  return 0;
}

int cmd_make_test_views(const GlobalOptions &opts) {
  const splat::cli::Args args(opts.rest, {});
  args.allow_only({"count", "width", "height", "focal"});
  if (args.positionals().size() != 1) {
    std::println(stderr, "Usage: splat make-test-views <dir> [--count N] [--width W] [--height H] [--focal F]");
    return 2;
  }
  const std::filesystem::path dir = args.positionals()[0];
  std::filesystem::create_directories(dir);
  const auto views = splat::make_test_views(std::max(args.number("count", 12), 2), std::clamp(args.number("width", 640), 64, 8192),
                                            std::clamp(args.number("height", 480), 64, 8192), args.number("focal", 500.0));
  for (std::size_t i = 0; i < views.size(); ++i) {
    splat::write_image(views[i].image, dir / std::format("view_{:03}.png", i), 95);
  }
  std::println("Wrote {} images to {}", views.size(), dir.string());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::span args(argv + 1, static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  GlobalOptions opts;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "-h" || arg == "--help") {
      std::print(kUsage, SPLAT_VERSION);
      return 0;
    }
    if (arg == "--version") {
      std::println("splat {}", SPLAT_VERSION);
      return 0;
    }
    if (arg == "--validate") {
      opts.context.validation = true;
    } else if (arg == "--device") {
      const auto value = i + 1 < args.size() ? parse_uint(args[i + 1]) : std::nullopt;
      if (!value) {
        std::println(stderr, "--device needs a device number (see 'splat info').");
        return 2;
      }
      opts.context.deviceIndex = value;
      ++i;
    } else if (opts.command.empty()) {
      opts.command = arg;
    } else {
      opts.rest.push_back(arg);
    }
  }

  if (opts.command.empty()) {
    std::print(kUsage, SPLAT_VERSION);
    return 2;
  }

  try {
    if (opts.command == "info") return cmd_info();
    if (opts.command == "selftest") return cmd_selftest(opts);
    if (opts.command == "ingest") return cmd_ingest(opts);
    if (opts.command == "sfm") return cmd_sfm(opts);
    if (opts.command == "make-test-views") return cmd_make_test_views(opts);
    if (opts.command == "train") return cmd_train(opts);
    if (opts.command == "view") return cmd_view(opts);
    if (opts.command == "render") return cmd_render(opts);
    if (opts.command == "make-test-scene") return cmd_make_test_scene(opts);
    if (opts.command == "convert") return cmd_convert(opts);
    std::println(stderr, "Unknown command '{}'. Run 'splat --help'.", opts.command);
    return 2;
  } catch (const splat::vk::Error &e) {
    std::println(stderr, "{} {}", splat::console::paint(splat::console::Color::Red, "Vulkan error:", stderr), e.what());
  } catch (const std::exception &e) {
    std::println(stderr, "{} {}", splat::console::paint(splat::console::Color::Red, "Error:", stderr), e.what());
  }
  return 1;
}
