#include "test_support.hpp"

#include "core/vk_check.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace splat::test {

vk::Context *context() {
  static std::unique_ptr<vk::Context> ctx = [] {
    try {
      return std::make_unique<vk::Context>(vk::ContextOptions{.validation = true});
    } catch (const vk::Error &e) {
      WARN("No usable Vulkan device: " << e.what());
      return std::unique_ptr<vk::Context>{};
    }
  }();
  return ctx.get();
}

double psnr(const RenderedImage &a, const RenderedImage &b) {
  double sum = 0.0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.rgba.size(); ++i) {
    if (i % 4 == 3) continue;
    const double d = static_cast<double>(a.rgba[i]) - static_cast<double>(b.rgba[i]);
    sum += d * d;
    ++n;
  }
  const double mse = sum / static_cast<double>(n);
  return mse == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

} // namespace splat::test
