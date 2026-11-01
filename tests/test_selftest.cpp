#include "core/selftest.hpp"
#include "core/vk_check.hpp"
#include "core/vk_context.hpp"

#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("device enumeration reports at least one device", "[vulkan]") {
  std::vector<splat::vk::DeviceInfo> devices;
  try {
    devices = splat::vk::Context::enumerate_devices();
  } catch (const splat::vk::Error &e) {
    SKIP("No Vulkan loader: " << e.what());
  }
  if (devices.empty()) {
    SKIP("No Vulkan devices");
  }
  CHECK_FALSE(devices.front().name.empty());
}

TEST_CASE("compute selftest produces correct results", "[vulkan]") {
  splat::vk::Context *ctx = splat::test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  SECTION("one full workgroup") {
    const auto r = splat::run_selftest(*ctx, 256);
    CHECK(r.mismatches == 0);
    CHECK(r.subgroupCount == 256);
  }
  SECTION("partial last workgroup") {
    const auto r = splat::run_selftest(*ctx, 1000);
    CHECK(r.mismatches == 0);
    CHECK(r.subgroupCount == 1000);
  }
  SECTION("one million elements") {
    const auto r = splat::run_selftest(*ctx);
    CHECK(r.passed);
  }
}
