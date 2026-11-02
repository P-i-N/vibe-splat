#include "splat/cgc.hpp"
#include "splat/lzma.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using namespace splat;

namespace {

std::vector<std::byte> sample_data() {
  std::vector<std::byte> data;
  for (int i = 0; i < 90'000; ++i) data.push_back(static_cast<std::byte>(i % 7 * 31));
  std::mt19937 rng(7);
  for (int i = 0; i < 10'000; ++i) data.push_back(static_cast<std::byte>(rng() & 0xFF));
  return data;
}

} // namespace

TEST_CASE("LZMA round-trips data", "[cgc]") {
  const auto data = sample_data();
  const auto packed = lzma_compress(data);
  CHECK(packed.size() < data.size());
  CHECK(lzma_decompress(packed, data.size()) == data);
}

TEST_CASE("LZMA round-trips empty input and a fast preset", "[cgc]") {
  const std::vector<std::byte> empty;
  CHECK(lzma_decompress(lzma_compress(empty), 0).empty());

  const auto data = sample_data();
  CHECK(lzma_decompress(lzma_compress(data, 1, false), data.size()) == data);
}

TEST_CASE("LZMA rejects bad input", "[cgc]") {
  const auto data = sample_data();
  const auto packed = lzma_compress(data);

  auto truncated = packed;
  truncated.resize(truncated.size() - 10);
  CHECK_THROWS(lzma_decompress(truncated, data.size()));

  auto badProperties = packed;
  badProperties[0] = std::byte{0xFF};
  CHECK_THROWS(lzma_decompress(badProperties, data.size()));

  CHECK_THROWS(lzma_decompress(packed, data.size() + 1));
}

TEST_CASE("Morton codes interleave x, y, z bits", "[cgc]") {
  using cgc_detail::morton_code;
  CHECK(morton_code(1, 0, 0) == 1);
  CHECK(morton_code(0, 1, 0) == 2);
  CHECK(morton_code(0, 0, 1) == 4);
  CHECK(morton_code(3, 0, 0) == 9);
  CHECK(morton_code(0x1FFFFF, 0x1FFFFF, 0x1FFFFF) == (1ull << 63) - 1);
}

namespace {

std::array<float, 4> normalized(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (float &v : q) v /= n;
  return q;
}

float dot4(const std::array<float, 4> &a, const std::array<float, 4> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

} // namespace

TEST_CASE("Rotations survive smallest-three packing", "[cgc]") {
  using namespace cgc_detail;
  std::vector<std::array<float, 4>> inputs = {
      {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 0.f, -1.f}, {-0.5f, 0.5f, -0.5f, 0.5f}, {0.3f, -1.5f, 2.4f, 0.9f}};
  std::mt19937 rng(3);
  std::normal_distribution<float> normal;
  for (int i = 0; i < 1000; ++i) inputs.push_back(normalized({normal(rng), normal(rng), normal(rng), normal(rng)}));

  for (const auto &input : inputs) {
    const auto decoded = unpack_rotation(pack_rotation(input));
    INFO("q = " << input[0] << " " << input[1] << " " << input[2] << " " << input[3]);
    CHECK(std::abs(dot4(decoded, decoded) - 1.f) < 1e-5f);
    CHECK(std::abs(dot4(normalized(input), decoded)) > 0.999f);
  }
  CHECK(unpack_rotation(pack_rotation({0.f, 0.f, 0.f, 0.f})) == std::array<float, 4>{1.f, 0.f, 0.f, 0.f});
}

TEST_CASE("SH degree drops bands that barely change the colour", "[cgc]") {
  using cgc_detail::choose_sh_degree;
  constexpr float threshold = 1.5f / 255.f;
  // Degree 3: 16 coefficients x RGB. Band l starts at coefficient l*l.
  const auto cloud_sh = [](float band1, float band2, float band3) {
    std::vector<float> sh(48, 0.f);
    for (int k = 1; k < 16; ++k) {
      const float v = k < 4 ? band1 : k < 9 ? band2 : band3;
      for (int c = 0; c < 3; ++c) sh[k * 3 + c] = v;
    }
    return sh;
  };
  CHECK(choose_sh_degree(cloud_sh(0.f, 0.f, 0.f), 3, 5.f, threshold) == 0);
  CHECK(choose_sh_degree(cloud_sh(0.f, 0.f, 0.5f), 3, 5.f, threshold) == 3);
  CHECK(choose_sh_degree(cloud_sh(0.5f, 1e-4f, 1e-4f), 3, 5.f, threshold) == 1);
  CHECK(choose_sh_degree(cloud_sh(0.05f, 0.f, 0.f), 3, 5.f, threshold) == 1);
  CHECK(choose_sh_degree(cloud_sh(0.05f, 0.f, 0.f), 3, -8.f, threshold) == 0);
  CHECK(choose_sh_degree(std::vector<float>{0.1f, 0.2f, 0.3f}, 0, 5.f, threshold) == 0);
}

TEST_CASE("Byte planes split 32-bit values by byte", "[cgc]") {
  using namespace cgc_detail;
  const std::vector<std::uint32_t> values = {0x04030201, 0x08070605};
  std::vector<std::byte> out;
  append_planes(out, values);
  const std::vector<std::byte> expected = {std::byte{1}, std::byte{5}, std::byte{2}, std::byte{6},
                                           std::byte{3}, std::byte{7}, std::byte{4}, std::byte{8}};
  CHECK(out == expected);
  CHECK(read_planes(out, 2) == values);
}
