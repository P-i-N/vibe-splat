#pragma once

#include "core/vk_context.hpp"

#include <cstdint>
#include <string>

namespace splat {

struct SelfTestResult {
  bool passed = false;
  std::uint32_t elements = 0;
  std::uint32_t mismatches = 0;
  std::uint32_t subgroupCount = 0; // What the GPU counted with subgroupAdd
  double gpuMilliseconds = 0.0;    // Wall time of the submit, incl. sync
  std::string message;
};

// Runs a small compute kernel (out = a * scale + b, plus a subgroup reduction)
// and checks the result on the CPU. Proves the device, shader embedding,
// buffers and dispatch path all work.
[[nodiscard]] SelfTestResult run_selftest(vk::Context &ctx, std::uint32_t elements = 1u << 20);

} // namespace splat
