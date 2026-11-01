// splat: Gaussian splat generator and viewer.
//
// One program, one subcommand per pipeline stage. Each stage reads and writes
// a project folder, so any stage can be rerun on its own.

#include "core/selftest.hpp"
#include "core/vk_check.hpp"
#include "core/vk_context.hpp"

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
  info        List Vulkan devices and whether splat can use them
  selftest    Run a compute kernel on the GPU and check the result
  ingest      Import photos or a video, read camera info      (milestone M2)
  sfm         Estimate camera poses                           (milestone M3)
  train       Train Gaussian splats                           (milestone M4)
  view        Open the real-time splat viewer                 (milestone M1)

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
    if (opts.command == "view") return cmd_not_yet(opts.command, "M1");
    std::println(stderr, "Unknown command '{}'. Run 'splat --help'.", opts.command);
    return 2;
  } catch (const splat::vk::Error &e) {
    std::println(stderr, "Vulkan error: {}", e.what());
  } catch (const std::exception &e) {
    std::println(stderr, "Error: {}", e.what());
  }
  return 1;
}
