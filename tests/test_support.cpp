#include "test_support.hpp"

#include "core/vk_check.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

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

} // namespace splat::test
