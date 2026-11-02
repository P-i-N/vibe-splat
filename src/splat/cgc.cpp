#include "splat/cgc.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace splat {

namespace {

constexpr float kRotationRange = 0.70710678f; // 1/sqrt(2): bound of the three smallest components

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// Spreads the low 21 bits of v so that bit i lands on bit 3i.
std::uint64_t spread3(std::uint32_t v) {
  std::uint64_t x = v & 0x1FFFFF;
  x = (x | x << 32) & 0x1F00000000FFFFull;
  x = (x | x << 16) & 0x1F0000FF0000FFull;
  x = (x | x << 8) & 0x100F00F00F00F00Full;
  x = (x | x << 4) & 0x10C30C30C30C30C3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

// Symmetric quantization over +-r with 2^bits - 1 levels, so 0 is exact.
std::uint32_t quantize_symmetric(float v, float r, int bits) {
  const int h = (1 << (bits - 1)) - 1;
  if (r <= 0.f) return static_cast<std::uint32_t>(h);
  const float q = std::clamp(std::round(v / r * static_cast<float>(h)), static_cast<float>(-h), static_cast<float>(h));
  return static_cast<std::uint32_t>(static_cast<int>(q) + h);
}

float dequantize_symmetric(std::uint32_t q, float r, int bits) {
  const int h = (1 << (bits - 1)) - 1;
  return static_cast<float>(static_cast<int>(q) - h) / static_cast<float>(h) * r;
}

} // namespace

namespace cgc_detail {

std::uint64_t morton_code(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return spread3(x) | spread3(y) << 1 | spread3(z) << 2;
}

std::uint32_t pack_rotation(std::array<float, 4> q) {
  const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(norm > 0.f)) q = {1.f, 0.f, 0.f, 0.f};
  else for (float &v : q) v /= norm;

  int largest = 0;
  for (int i = 1; i < 4; ++i) {
    if (std::abs(q[i]) > std::abs(q[largest])) largest = i;
  }
  const float sign = q[largest] < 0.f ? -1.f : 1.f;
  std::uint32_t packed = static_cast<std::uint32_t>(largest) << 30;
  int shift = 0;
  for (int i = 0; i < 4; ++i) {
    if (i == largest) continue;
    packed |= quantize_symmetric(q[i] * sign, kRotationRange, 10) << shift;
    shift += 10;
  }
  return packed;
}

std::array<float, 4> unpack_rotation(std::uint32_t packed) {
  const int largest = static_cast<int>(packed >> 30);
  std::array<float, 4> q{};
  float sum = 0.f;
  int shift = 0;
  for (int i = 0; i < 4; ++i) {
    if (i == largest) continue;
    q[i] = dequantize_symmetric((packed >> shift) & 0x3FF, kRotationRange, 10);
    sum += q[i] * q[i];
    shift += 10;
  }
  q[largest] = std::sqrt(std::max(0.f, 1.f - sum));
  // Rounding can leave the vector slightly off unit length.
  const float norm = std::sqrt(sum + q[largest] * q[largest]);
  for (float &v : q) v /= norm;
  return q;
}

int choose_sh_degree(std::span<const float> sh, int degree, float opacityLogit, float threshold) {
  const float alpha = sigmoid(opacityLogit);
  constexpr float kInvFourPi = 1.f / (4.f * std::numbers::pi_v<float>);
  for (int band = degree; band > 0; --band) {
    float worst = 0.f;
    for (int c = 0; c < 3; ++c) {
      float sum = 0.f;
      for (int k = band * band; k < (band + 1) * (band + 1); ++k) sum += sh[k * 3 + c] * sh[k * 3 + c];
      worst = std::max(worst, std::sqrt(sum * kInvFourPi) * alpha);
    }
    if (worst >= threshold) return band;
  }
  return 0;
}

void append_planes(std::vector<std::byte> &out, std::span<const std::uint32_t> values) {
  for (int b = 0; b < 4; ++b) {
    for (const std::uint32_t v : values) out.push_back(static_cast<std::byte>(v >> (8 * b)));
  }
}

std::vector<std::uint32_t> read_planes(std::span<const std::byte> in, std::size_t count) {
  std::vector<std::uint32_t> values(count, 0);
  for (std::size_t b = 0; b < 4; ++b) {
    for (std::size_t i = 0; i < count; ++i) {
      values[i] |= static_cast<std::uint32_t>(in[b * count + i]) << (8 * b);
    }
  }
  return values;
}

} // namespace cgc_detail

} // namespace splat
