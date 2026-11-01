#include "core/radix_sort.hpp"
#include "core/vk_check.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <span>

using namespace splat::vk;

TEST_CASE("GPU radix sort matches std::stable_sort", "[vulkan][sort]") {
  Context *ctx = splat::test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  const std::uint32_t count = GENERATE(2u, 255u, 256u, 4096u, 4097u, 100'000u, (1u << 20) + 7u);
  const std::uint32_t keyBits = GENERATE(8u, 32u);

  std::mt19937 rng(count * 31 + keyBits);
  std::vector<std::uint32_t> keys(count);
  std::vector<std::uint32_t> values(count);
  // Few distinct key values in the 8-bit case: stresses stability.
  const std::uint32_t mask = keyBits == 32 ? 0xFFFFFFFFu : 0x0Fu;
  for (std::uint32_t i = 0; i < count; ++i) {
    keys[i] = static_cast<std::uint32_t>(rng()) & mask;
    values[i] = i;
  }

  constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  const VkDeviceSize bytes = VkDeviceSize{count} * 4;
  Buffer k(*ctx, bytes, kStorage, MemoryUsage::GpuOnly);
  Buffer v(*ctx, bytes, kStorage, MemoryUsage::GpuOnly);
  Buffer tk(*ctx, bytes, kStorage, MemoryUsage::GpuOnly);
  Buffer tv(*ctx, bytes, kStorage, MemoryUsage::GpuOnly);
  k.upload(*ctx, std::span<const std::uint32_t>(keys));
  v.upload(*ctx, std::span<const std::uint32_t>(values));

  RadixSorter sorter(*ctx);
  ctx->submit_and_wait([&](VkCommandBuffer cmd) { sorter.record(cmd, k, v, tk, tv, count, keyBits); });

  const auto gotKeys = k.download_as<std::uint32_t>(*ctx);
  const auto gotValues = v.download_as<std::uint32_t>(*ctx);

  std::vector<std::uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return keys[a] < keys[b]; });

  std::uint32_t mismatches = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (gotKeys[i] != keys[order[i]] || gotValues[i] != order[i]) {
      ++mismatches;
    }
  }
  INFO("count " << count << ", key bits " << keyBits);
  CHECK(mismatches == 0);
}
