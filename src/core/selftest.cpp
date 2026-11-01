#include "core/selftest.hpp"

#include "core/vk_compute.hpp"
#include "splat_shaders/shaders.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <format>

namespace splat {

SelfTestResult run_selftest(vk::Context &ctx, std::uint32_t elements) {
  using vk::Buffer;
  using vk::MemoryUsage;

  struct Push {
    std::uint32_t count;
    float scale;
  } push{elements, 2.5f};

  const VkDeviceSize bytes = VkDeviceSize{elements} * sizeof(float);
  constexpr VkBufferUsageFlags kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  Buffer a(ctx, bytes, kStorage, MemoryUsage::Upload);
  Buffer b(ctx, bytes, kStorage, MemoryUsage::Upload);
  Buffer out(ctx, bytes, kStorage, MemoryUsage::Readback);
  Buffer sums(ctx, sizeof(std::uint32_t), kStorage, MemoryUsage::Readback);

  auto av = a.as<float>();
  auto bv = b.as<float>();
  for (std::uint32_t i = 0; i < elements; ++i) {
    av[i] = static_cast<float>(i % 1000) * 0.5f;
    bv[i] = static_cast<float>(i % 7);
  }
  a.flush();
  b.flush();
  sums.as<std::uint32_t>()[0] = 0;
  sums.flush();

  vk::ComputePipeline pipeline(ctx, shaders::selftest_comp, 4, sizeof(Push));
  const std::uint32_t groups = (elements + 255) / 256;

  const auto start = std::chrono::steady_clock::now();
  ctx.submit_and_wait([&](VkCommandBuffer cmd) {
    pipeline.dispatch(cmd, {&a, &b, &out, &sums}, std::as_bytes(std::span{&push, 1}), groups);
    vk::compute_barrier(cmd);
  });
  const auto end = std::chrono::steady_clock::now();

  out.invalidate();
  sums.invalidate();

  SelfTestResult result;
  result.elements = elements;
  result.gpuMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
  const auto ov = out.as<float>();
  for (std::uint32_t i = 0; i < elements; ++i) {
    // Recompute inputs instead of reading back write-combined upload memory.
    const float expected = static_cast<float>(i % 1000) * 0.5f * push.scale + static_cast<float>(i % 7);
    if (std::fabs(ov[i] - expected) > 1e-4f * std::fabs(expected) + 1e-6f) {
      ++result.mismatches;
    }
  }
  result.subgroupCount = sums.as<std::uint32_t>()[0];
  result.passed = result.mismatches == 0 && result.subgroupCount == elements;
  result.message = result.passed
                       ? std::format("{} elements OK", elements)
                       : std::format("{} mismatches, subgroup count {} (expected {})", result.mismatches,
                                     result.subgroupCount, elements);
  return result;
}

} // namespace splat
