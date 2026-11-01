// splat: Gaussian splat generator and viewer.
//
// One program, one subcommand per pipeline stage. Each stage reads and writes
// a project folder, so any stage can be rerun on its own.

#include "cli/args.hpp"
#include "core/selftest.hpp"
#include "core/vk_check.hpp"
#include "core/vk_context.hpp"
#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "splat/gaussian_cloud.hpp"
#ifdef SPLAT_WITH_VIEWER
#include "viewer/viewer.hpp"
#endif

#include <algorithm>
#include <chrono>

#include <charconv>
#include <cstdint>
#include <exception>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kUsage = R"(splat {} - Gaussian splat generator and viewer (Vulkan)

Usage: splat <command> [options]

Commands:
  view <scene.ply>              Open the real-time viewer
  render <scene.ply> -o <png>   Render one image without a window
  make-test-scene <out.ply>     Write a synthetic scene for trying things out
  info                          List Vulkan devices and whether splat can use them
  selftest                      Run a compute kernel on the GPU and check the result
  ingest                        Import photos or a video        (milestone M2)
  sfm                           Estimate camera poses           (milestone M3)
  train                         Train Gaussian splats           (milestone M4)

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
  auto cloud = splat::load_ply(path);
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
  splat::save_ply(cloud, args.positionals()[0]);
  std::println("Wrote {} Gaussians to {}", cloud.size(), args.positionals()[0]);
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

int cmd_not_yet(std::string_view command, std::string_view milestone) {
  std::println(stderr, "'splat {}' is not implemented yet (planned for milestone {}).", command, milestone);
  return 2;
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
    if (opts.command == "ingest") return cmd_not_yet(opts.command, "M2");
    if (opts.command == "sfm") return cmd_not_yet(opts.command, "M3");
    if (opts.command == "train") return cmd_not_yet(opts.command, "M4");
    if (opts.command == "view") return cmd_view(opts);
    if (opts.command == "render") return cmd_render(opts);
    if (opts.command == "make-test-scene") return cmd_make_test_scene(opts);
    std::println(stderr, "Unknown command '{}'. Run 'splat --help'.", opts.command);
    return 2;
  } catch (const splat::vk::Error &e) {
    std::println(stderr, "Vulkan error: {}", e.what());
  } catch (const std::exception &e) {
    std::println(stderr, "Error: {}", e.what());
  }
  return 1;
}
