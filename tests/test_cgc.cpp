#include "splat/lzma.hpp"

#include <catch2/catch_test_macros.hpp>

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
