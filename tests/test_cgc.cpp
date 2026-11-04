#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "splat/cgc.hpp"
#include "splat/lzma.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
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

namespace {

// The covariance of a Gaussian: R diag(exp(2 s)) R^T, as 6 unique entries.
std::array<float, 6> covariance(const float *s, const float *q) {
  const auto n = normalized({q[0], q[1], q[2], q[3]});
  const float w = n[0], x = n[1], y = n[2], z = n[3];
  const float r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                         {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                         {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
  std::array<float, 6> c{};
  int out = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      for (int a = 0; a < 3; ++a) c[out] += r[i][a] * r[j][a] * std::exp(2.f * s[a]);
      ++out;
    }
  }
  return c;
}

} // namespace

TEST_CASE("Canonical axes describe the same ellipsoid", "[cgc]") {
  std::mt19937 rng(3);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> uniform(-5.f, 1.f);
  for (int i = 0; i < 2000; ++i) {
    std::array<float, 3> s = {uniform(rng), uniform(rng), uniform(rng)};
    std::array<float, 4> q = {normal(rng), normal(rng), normal(rng), normal(rng)};
    const auto before = covariance(s.data(), q.data());
    cgc_detail::canonicalize_axes(s, q);
    const auto after = covariance(s.data(), q.data());
    CHECK(s[0] >= s[1]);
    CHECK(s[1] >= s[2]);
    CHECK(q[0] >= 0.f);
    for (int a = 1; a < 4; ++a) CHECK(q[0] >= std::abs(q[a]) - 1e-5f);
    const float biggest = std::exp(2.f * s[0]);
    for (int a = 0; a < 6; ++a) CHECK(std::abs(before[a] - after[a]) <= 1e-4f * biggest);
  }
  std::array<float, 3> s = {1.f, 2.f, 3.f};
  std::array<float, 4> zero = {0.f, 0.f, 0.f, 0.f};
  cgc_detail::canonicalize_axes(s, zero);
  CHECK(s == std::array<float, 3>{3.f, 2.f, 1.f});
  CHECK(std::abs(dot4(zero, zero) - 1.f) < 1e-5f);
}

namespace {

std::filesystem::path temp_file(const char *name) { return std::filesystem::temp_directory_path() / name; }

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

GaussianCloud random_cloud(std::size_t count, int degree, std::uint32_t seed) {
  GaussianCloud cloud;
  cloud.resize(count, degree);
  std::mt19937 rng(seed);
  const auto uniform = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); };
  std::normal_distribution<float> normal;
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  for (std::size_t i = 0; i < count; ++i) {
    for (int a = 0; a < 3; ++a) cloud.positions[i * 3 + a] = uniform(0.f, 10.f);
    for (int a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = uniform(-6.f, -1.f);
    const auto q = normalized({normal(rng), normal(rng), normal(rng), normal(rng)});
    for (int a = 0; a < 4; ++a) cloud.rotations[i * 4 + a] = q[a];
    cloud.opacities[i] = uniform(-4.f, 4.f);
    for (std::size_t k = 0; k < coeffs * 3; ++k) cloud.sh[i * coeffs * 3 + k] = k < 3 ? uniform(-1.5f, 1.5f) : uniform(-0.3f, 0.3f);
  }
  return cloud;
}

GaussianCloud round_trip(const GaussianCloud &cloud, const char *name, CgcStats *stats = nullptr) {
  const auto path = temp_file(name);
  const CgcStats s = save_cgc(cloud, path);
  if (stats != nullptr) *stats = s;
  GaussianCloud loaded = load_cgc(path);
  std::filesystem::remove(path);
  return loaded;
}

// Index of the splat in `cloud` nearest to `p`.
std::size_t nearest(const GaussianCloud &cloud, const float *p) {
  std::size_t best = 0;
  float bestDistance = INFINITY;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    float d = 0.f;
    for (int a = 0; a < 3; ++a) d += (cloud.positions[i * 3 + a] - p[a]) * (cloud.positions[i * 3 + a] - p[a]);
    if (d < bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return best;
}

constexpr float kShC0 = 0.28209479177387814f;

// Checks every attribute of `loaded` against `original` within the coarsest
// layer's quantization error. Splats come back reordered, so each is matched to
// the nearest original. SH goes through a codebook, so it is compared on
// average. Returns how many DC colours missed the tolerance.
int check_round_trip(const GaussianCloud &original, const GaussianCloud &loaded, std::size_t skip = SIZE_MAX) {
  REQUIRE(loaded.size() == original.size());
  REQUIRE(loaded.shDegree == original.shDegree);
  const std::size_t coeffs = static_cast<std::size_t>(original.shCoeffs());
  int colourMisses = 0;
  for (std::size_t j = 0; j < loaded.size(); ++j) {
    const std::size_t i = nearest(original, &loaded.positions[j * 3]);
    if (i == skip) continue;
    for (int a = 0; a < 3; ++a) CHECK(std::abs(loaded.positions[j * 3 + a] - original.positions[i * 3 + a]) <= 0.02f);
    const float biggest = std::exp(2.f * std::max({original.scales[i * 3], original.scales[i * 3 + 1], original.scales[i * 3 + 2]}));
    const auto a = covariance(&original.scales[i * 3], &original.rotations[i * 4]);
    const auto b = covariance(&loaded.scales[j * 3], &loaded.rotations[j * 4]);
    for (int k = 0; k < 6; ++k) CHECK(std::abs(a[k] - b[k]) <= 0.25f * biggest);
    CHECK(std::abs(sigmoid(loaded.opacities[j]) - sigmoid(original.opacities[i])) <= 1.f / 32.f + 1e-5f);
    for (int c = 0; c < 3; ++c) {
      const float want = kShC0 * original.sh[i * coeffs * 3 + c] + 0.5f;
      const float got = kShC0 * loaded.sh[j * coeffs * 3 + c] + 0.5f;
      if (std::abs(want - got) > 0.06f) ++colourMisses;
    }
  }
  return colourMisses;
}

// RMS difference of the higher SH bands, matching splats by nearest position.
float sh_rms_error(const GaussianCloud &original, const GaussianCloud &loaded) {
  const std::size_t coeffs = static_cast<std::size_t>(original.shCoeffs());
  double sum = 0.0;
  std::size_t count = 0;
  for (std::size_t j = 0; j < loaded.size(); ++j) {
    const std::size_t i = nearest(original, &loaded.positions[j * 3]);
    for (std::size_t k = 3; k < coeffs * 3; ++k) {
      const double d = original.sh[i * coeffs * 3 + k] - loaded.sh[j * coeffs * 3 + k];
      sum += d * d;
      ++count;
    }
  }
  return count == 0 ? 0.f : static_cast<float>(std::sqrt(sum / static_cast<double>(count)));
}

} // namespace

TEST_CASE("CGC round-trips every attribute within its quantization step", "[cgc]") {
  for (int degree = 0; degree <= 3; ++degree) {
    INFO("SH degree " << degree);
    const GaussianCloud cloud = random_cloud(1000, degree, 11 + degree);
    CgcStats stats;
    const GaussianCloud loaded = round_trip(cloud, "splat_cgc_roundtrip.cgc", &stats);
    CHECK(loaded.shDegree == degree);
    CHECK(stats.bytes > 0);
    CHECK(stats.layerCounts[0] + stats.layerCounts[1] + stats.layerCounts[2] == 1000);
    CHECK(check_round_trip(cloud, loaded) <= 10);
    for (const float v : loaded.sh) CHECK(std::isfinite(v));
  }
}

TEST_CASE("The SH codebook reproduces a few distinct patterns", "[cgc]") {
  GaussianCloud cloud = random_cloud(4000, 3, 5);
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  std::mt19937 rng(6);
  std::normal_distribution<float> normal(0.f, 0.2f);
  std::vector<float> patterns(8 * 45);
  for (float &v : patterns) v = normal(rng);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    std::copy_n(&patterns[(i % 8) * 45], 45, &cloud.sh[i * coeffs * 3 + 3]);
  }
  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_patterns.cgc");
  CHECK(sh_rms_error(cloud, loaded) < 0.03f);
}

TEST_CASE("Empty higher SH bands stay nearly empty", "[cgc]") {
  GaussianCloud cloud = random_cloud(500, 3, 5);
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    for (std::size_t k = 3; k < coeffs * 3; ++k) cloud.sh[i * coeffs * 3 + k] = 0.f;
  }
  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_dc_only.cgc");
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    for (std::size_t k = 3; k < coeffs * 3; ++k) CHECK(loaded.sh[i * coeffs * 3 + k] == 0.f);
  }
}

TEST_CASE("A colour outlier does not coarsen everyone else's colour", "[cgc]") {
  GaussianCloud cloud = random_cloud(1000, 1, 21);
  // Give every splat the same importance class as the outlier, so it has company.
  for (float &v : cloud.opacities) v = 2.f;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    for (int a = 0; a < 3; ++a) cloud.scales[i * 3 + a] = -3.f;
  }
  cloud.positions.insert(cloud.positions.end(), {5.f, 5.f, 11.f});
  cloud.scales.insert(cloud.scales.end(), {-3.f, -3.f, -3.f});
  cloud.rotations.insert(cloud.rotations.end(), {1.f, 0.f, 0.f, 0.f});
  cloud.opacities.push_back(2.f);
  cloud.sh.insert(cloud.sh.end(), {100.f, 100.f, 100.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_outlier.cgc");
  CHECK(check_round_trip(cloud, loaded, cloud.size() - 1) <= 10);
  for (const float v : loaded.sh) CHECK(std::isfinite(v));
}

TEST_CASE("CGC handles small and degenerate clouds", "[cgc]") {
  for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{256}, std::size_t{257}}) {
    INFO(count << " splats");
    const GaussianCloud cloud = random_cloud(count, 2, 31);
    const GaussianCloud loaded = round_trip(cloud, "splat_cgc_sizes.cgc");
    REQUIRE(loaded.size() == count);
    if (count > 0) check_round_trip(cloud, loaded);
  }

  GaussianCloud same = random_cloud(300, 1, 41);
  for (std::size_t i = 0; i < same.size(); ++i) {
    same.positions[i * 3] = 1.f;
    same.positions[i * 3 + 1] = 2.f;
    same.positions[i * 3 + 2] = 3.f;
    for (int a = 0; a < 3; ++a) same.scales[i * 3 + a] = -2.f;
  }
  const GaussianCloud loaded = round_trip(same, "splat_cgc_flat.cgc");
  REQUIRE(loaded.size() == same.size());
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    CHECK(std::abs(loaded.positions[i * 3] - 1.f) < 1e-4f);
    CHECK(std::abs(loaded.positions[i * 3 + 1] - 2.f) < 1e-4f);
    CHECK(std::abs(loaded.positions[i * 3 + 2] - 3.f) < 1e-4f);
    CHECK(std::abs(loaded.scales[i * 3] - -2.f) < 0.1f);
  }
}

TEST_CASE("A far-away floater does not coarsen the scene's positions", "[cgc]") {
  // Positions live on a symlog grid, so precision falls with distance from the centre.
  GaussianCloud cloud = random_cloud(1000, 0, 73);
  cloud.positions[0] = 1e5f;
  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_floater.cgc");
  REQUIRE(loaded.size() == cloud.size());
  for (std::size_t j = 0; j < loaded.size(); ++j) {
    if (loaded.positions[j * 3] > 1000.f) continue;
    const std::size_t i = nearest(cloud, &loaded.positions[j * 3]);
    for (int a = 0; a < 3; ++a) CHECK(std::abs(loaded.positions[j * 3 + a] - cloud.positions[i * 3 + a]) <= 0.2f);
  }
}

TEST_CASE("CGC keeps extreme opacities and scales finite", "[cgc]") {
  GaussianCloud cloud = random_cloud(4, 0, 51);
  cloud.opacities = {50.f, -50.f, 0.f, 3.f};
  cloud.scales[0] = -30.f;
  cloud.scales[1] = 10.f;
  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_extreme.cgc");
  for (const float v : loaded.opacities) CHECK(std::isfinite(v));
  for (const float v : loaded.scales) CHECK(std::isfinite(v));
}

TEST_CASE("The Small preset makes a smaller file", "[cgc]") {
  const GaussianCloud cloud = random_cloud(20'000, 3, 74);
  const auto path = temp_file("splat_cgc_preset.cgc");
  const auto balanced = save_cgc(cloud, path);
  const auto small = save_cgc(cloud, path, {.preset = CgcPreset::Small});
  std::filesystem::remove(path);
  CHECK(small.bytes < balanced.bytes);
  CHECK(small.layerCounts[0] < balanced.layerCounts[0]);
}

TEST_CASE("CGC refuses to save non-finite values", "[cgc]") {
  const auto path = temp_file("splat_cgc_nan.cgc");
  std::filesystem::remove(path);
  GaussianCloud nanPosition = random_cloud(10, 1, 61);
  nanPosition.positions[4] = NAN;
  CHECK_THROWS(save_cgc(nanPosition, path));
  CHECK_FALSE(std::filesystem::exists(path));

  GaussianCloud infSh = random_cloud(10, 1, 62);
  infSh.sh[7] = INFINITY;
  CHECK_THROWS(save_cgc(infSh, path));
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("Loading a malformed CGC file throws", "[cgc]") {
  const auto ply = temp_file("splat_cgc_not.cgc");
  save_ply(random_cloud(10, 0, 71), ply);
  CHECK_THROWS_AS(load_cgc(ply), std::runtime_error);
  std::filesystem::remove(ply);

  const auto path = temp_file("splat_cgc_bad.cgc");
  save_cgc(random_cloud(2000, 3, 72), path);
  std::vector<char> good(std::filesystem::file_size(path));
  std::ifstream(path, std::ios::binary).read(good.data(), static_cast<std::streamsize>(good.size()));
  const auto write = [&](const std::vector<char> &bytes) {
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  };

  auto half = good;
  half.resize(good.size() / 2);
  write(half);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  auto trailing = good;
  trailing.push_back(0);
  write(trailing);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  auto badVersion = good;
  badVersion[4] = 3;
  write(badVersion);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  auto badCount = good;
  badCount[8] = static_cast<char>(badCount[8] + 1);
  write(badCount);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  // First stream's LZMA properties byte: after the 36-byte header, 3 layer
  // headers of 92 bytes and the stream's u32 size.
  auto badProperties = good;
  badProperties[36 + 3 * 92 + 4] = static_cast<char>(0xFF);
  write(badProperties);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  std::filesystem::remove(path);
}

TEST_CASE("A CGC scene renders like the original", "[vulkan][cgc]") {
  vk::Context *ctx = test::context();
  if (ctx == nullptr) {
    SKIP("No usable Vulkan device");
  }
  const GaussianCloud original = make_test_scene(50'000);
  const GaussianCloud loaded = round_trip(original, "splat_cgc_render.cgc");
  const Camera camera = default_orbit(original).camera(320, 180);

  SplatRenderer a(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  a.set_scene(original);
  const RenderedImage imageA = render_offscreen(*ctx, a, camera);
  SplatRenderer b(*ctx, VK_FORMAT_R8G8B8A8_UNORM);
  b.set_scene(loaded);
  const RenderedImage imageB = render_offscreen(*ctx, b, camera);

  const double value = test::psnr(imageA, imageB);
  INFO("PSNR " << value << " dB");
  CHECK(value > 35.0);
}

TEST_CASE("The file extension picks PLY or CGC", "[cgc]") {
  CHECK(cloud_format("a.cgc") == CloudFormat::Cgc);
  CHECK(cloud_format("a.CGC") == CloudFormat::Cgc);
  CHECK(cloud_format("dir.x/b.Cgc") == CloudFormat::Cgc);
  CHECK(cloud_format("a.ply") == CloudFormat::Ply);
  CHECK(cloud_format("a") == CloudFormat::Ply);
  CHECK(cloud_format("a.cgc.tmp") == CloudFormat::Ply);
}

TEST_CASE("save_cloud and load_cloud follow the extension", "[cgc]") {
  const GaussianCloud cloud = random_cloud(10, 1, 81);

  const auto ply = temp_file("splat_dispatch.ply");
  save_cloud(cloud, ply);
  const GaussianCloud fromPly = load_cloud(ply);
  CHECK(fromPly.positions == cloud.positions);
  CHECK(fromPly.sh == cloud.sh);
  std::filesystem::remove(ply);

  const auto cgc = temp_file("splat_dispatch.CGC");
  save_cloud(cloud, cgc);
  check_round_trip(cloud, load_cloud(cgc));
  std::filesystem::remove(cgc);

  const auto temporary = temp_file("splat_dispatch.cgc.tmp");
  save_cloud(cloud, temporary, CloudFormat::Cgc, CgcOptions{.fast = true});
  check_round_trip(cloud, load_cgc(temporary));
  std::filesystem::remove(temporary);
}

TEST_CASE("remove_non_finite drops only the broken Gaussians", "[cgc]") {
  GaussianCloud cloud = random_cloud(6, 1, 91);
  const GaussianCloud original = cloud;
  cloud.positions[1 * 3 + 2] = NAN;
  cloud.sh[4 * 12 + 5] = INFINITY;
  CHECK(remove_non_finite(cloud) == 2);
  REQUIRE(cloud.size() == 4);
  cloud.validate();
  for (const std::size_t kept : {0, 2, 3, 5}) {
    const std::size_t j = kept < 1 ? 0 : kept < 4 ? kept - 1 : kept - 2;
    CHECK(cloud.opacities[j] == original.opacities[kept]);
  }
  CHECK_NOTHROW(save_cgc(cloud, temp_file("splat_cgc_cleaned.cgc")));
  std::filesystem::remove(temp_file("splat_cgc_cleaned.cgc"));
  CHECK(remove_non_finite(cloud) == 0);
}

namespace {

bool same_gaussian(const GaussianCloud &a, std::size_t i, const GaussianCloud &b, std::size_t j) {
  const std::size_t floats = static_cast<std::size_t>(3 * a.shCoeffs());
  for (int k = 0; k < 3; ++k) {
    if (a.positions[i * 3 + k] != b.positions[j * 3 + k] || a.scales[i * 3 + k] != b.scales[j * 3 + k]) return false;
  }
  for (int k = 0; k < 4; ++k) {
    if (a.rotations[i * 4 + k] != b.rotations[j * 4 + k]) return false;
  }
  if (a.opacities[i] != b.opacities[j]) return false;
  for (std::size_t k = 0; k < floats; ++k) {
    if (a.sh[i * floats + k] != b.sh[j * floats + k]) return false;
  }
  return true;
}

// Checks that `pruned` is exactly `original` with the Gaussians in `removed` taken out, in the original order.
void check_pruned(const GaussianCloud &original, const GaussianCloud &pruned, std::vector<std::size_t> removed) {
  REQUIRE(pruned.size() + removed.size() == original.size());
  pruned.validate();
  std::size_t j = 0;
  for (std::size_t i = 0; i < original.size(); ++i) {
    if (std::ranges::find(removed, i) != removed.end()) continue;
    INFO("original Gaussian " << i << " should be pruned Gaussian " << j);
    CHECK(same_gaussian(original, i, pruned, j));
    ++j;
  }
}

} // namespace

TEST_CASE("prune_floaters removes Gaussians whose largest axis is too long", "[prune]") {
  // random_cloud scales are log-uniform in [-6, -1], so every axis is below exp(-1) = 0.37.
  GaussianCloud cloud = random_cloud(8, 1, 92);
  cloud.scales[2 * 3 + 1] = std::log(2.0f);
  cloud.scales[5 * 3 + 0] = std::log(0.8f);
  cloud.scales[6 * 3 + 2] = std::log(0.45f); // below the limit: stays
  const GaussianCloud original = cloud;

  PruneOptions options;
  options.maxSigma = 0.5f;
  CHECK(prune_floaters(cloud, options) == 2);
  check_pruned(original, cloud, {2, 5});
}

TEST_CASE("prune_floaters removes Gaussians far from the scene centre", "[prune]") {
  // random_cloud positions lie in [0, 10]^3: the centre is near (5, 5, 5) and the 90% radius below 9.
  GaussianCloud cloud = random_cloud(100, 2, 93);
  cloud.positions[1 * 3 + 0] = 500.f;
  cloud.positions[60 * 3 + 0] = -300.f;
  cloud.positions[60 * 3 + 1] = -300.f;
  cloud.positions[60 * 3 + 2] = -300.f;
  const GaussianCloud original = cloud;

  PruneOptions options;
  options.maxRadiusFactor = 3.f;
  CHECK(prune_floaters(cloud, options) == 2);
  check_pruned(original, cloud, {1, 60});
}

TEST_CASE("prune_floaters applies both tests and counts a Gaussian once", "[prune]") {
  GaussianCloud cloud = random_cloud(100, 1, 94);
  cloud.positions[7 * 3 + 2] = 400.f;   // far away
  cloud.scales[7 * 3 + 0] = std::log(3.f); // and oversized
  cloud.scales[42 * 3 + 1] = std::log(3.f); // oversized only
  cloud.positions[90 * 3 + 1] = -400.f; // far away only
  const GaussianCloud original = cloud;

  PruneOptions options;
  options.maxSigma = 1.f;
  options.maxRadiusFactor = 3.f;
  CHECK(prune_floaters(cloud, options) == 3);
  check_pruned(original, cloud, {7, 42, 90});
}

TEST_CASE("prune_floaters leaves a cloud with nothing to prune untouched", "[prune]") {
  GaussianCloud cloud = random_cloud(50, 3, 95);
  const GaussianCloud original = cloud;

  PruneOptions options;
  options.maxSigma = 1.f;
  options.maxRadiusFactor = 3.f;
  CHECK(prune_floaters(cloud, options) == 0);
  check_pruned(original, cloud, {});

  // Neither limit set: nothing is tested, even for an outlandish cloud.
  cloud.positions[0] = 1e6f;
  cloud.scales[0] = std::log(1e3f);
  CHECK(prune_floaters(cloud, PruneOptions{}) == 0);
  CHECK(cloud.size() == 50);
}

namespace {

template <typename T> void append(std::vector<std::byte> &out, T value) {
  const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
  out.insert(out.end(), bytes.begin(), bytes.end());
}

void append_planes(std::vector<std::byte> &out, std::initializer_list<std::uint32_t> values) {
  for (int b = 0; b < 4; ++b) {
    for (const std::uint32_t v : values) out.push_back(static_cast<std::byte>(v >> (8 * b)));
  }
}

} // namespace

// Version 1 had no writer left, so this builds a file by hand from its old spec.
TEST_CASE("Version 1 CGC files stay readable", "[cgc]") {
  std::vector<std::byte> payload;
  append<std::uint32_t>(payload, 2);
  for (const float v : {0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.5f, 0.f, 0.f}) append(payload, v); // colour ranges, SH ranges
  const auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };
  append_planes(payload, {bits(0.f), bits(0.f), bits(0.f), bits(1.f), bits(1.f), bits(1.f),
                          bits(-3.f), bits(-3.f), bits(-3.f), bits(-1.f), bits(-1.f), bits(-1.f)});
  append_planes(payload, {0u, 0x7FFu | 0x3FFu << 11 | 0x7FFu << 21});
  const std::uint32_t identity = 511u | 511u << 10 | 511u << 20;
  append_planes(payload, {identity, identity});
  for (const int v : {0, 255, 0, 255, 0, 255}) payload.push_back(static_cast<std::byte>(v)); // scales
  for (const int v : {0, 255, 255, 0, 128, 128, 128, 128}) payload.push_back(static_cast<std::byte>(v)); // RGBA
  for (const int v : {0, 1}) payload.push_back(static_cast<std::byte>(v)); // degrees
  for (int k = 0; k < 9; ++k) payload.push_back(std::byte{30}); // band 1 of the second splat: all +0.5

  std::vector<std::byte> file;
  for (const char c : {'C', 'G', 'C', '\0'}) file.push_back(static_cast<std::byte>(c));
  append<std::uint32_t>(file, 1);
  append<std::uint64_t>(file, payload.size());
  const auto stream = lzma_compress(payload);
  file.insert(file.end(), stream.begin(), stream.end());
  const auto path = temp_file("splat_cgc_v1.cgc");
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
  const GaussianCloud cloud = load_cgc(path);
  std::filesystem::remove(path);

  REQUIRE(cloud.size() == 2);
  CHECK(cloud.shDegree == 1);
  CHECK(cloud.positions[0] == 0.f);
  CHECK(cloud.positions[3] == 1.f);
  CHECK(std::abs(cloud.rotations[0] - 1.f) < 1e-5f);
  CHECK(cloud.scales[0] == -3.f);
  CHECK(cloud.scales[3] == -1.f);
  CHECK(std::abs(kShC0 * cloud.sh[0] + 0.5f) < 1e-5f);
  CHECK(std::abs(kShC0 * cloud.sh[12] + 0.5f - 1.f) < 1e-5f);
  CHECK(std::abs(cloud.sh[15] - 0.5f) < 1e-5f);
  CHECK(cloud.sh[3] == 0.f);
}

// Pins the byte layout from the spec, independently of load_cgc, so that a
// change on both the encoder and decoder side cannot slip through unnoticed.
TEST_CASE("CGC file bytes follow the spec layout", "[cgc]") {
  GaussianCloud cloud;
  cloud.resize(1, 0);
  cloud.positions = {1.f, 2.f, 3.f};
  cloud.scales = {-1.f, -2.f, -3.f}; // already sorted, so the identity stays
  cloud.rotations = {1.f, 0.f, 0.f, 0.f};
  cloud.opacities = {0.f};
  cloud.sh = {0.f, 0.f, 0.f};

  const auto path = temp_file("splat_cgc_layout.cgc");
  save_cgc(cloud, path);
  std::vector<char> file(std::filesystem::file_size(path));
  std::ifstream(path, std::ios::binary).read(file.data(), static_cast<std::streamsize>(file.size()));
  std::filesystem::remove(path);

  const auto u32_at = [&](std::size_t at) {
    std::uint32_t v = 0;
    std::memcpy(&v, file.data() + at, 4);
    return v;
  };
  const auto f32_at = [&](std::size_t at) { return std::bit_cast<float>(u32_at(at)); };
  CHECK(std::vector<char>(file.begin(), file.begin() + 4) == std::vector<char>{'C', 'G', 'C', '\0'});
  CHECK(u32_at(4) == 2);  // version
  CHECK(u32_at(8) == 1);  // count
  CHECK(u32_at(12) == 0); // SH degree
  CHECK(u32_at(16) == 3); // layers
  CHECK(f32_at(20) == 1.f); // centre x

  // The one splat lands in the first layer, at 15 bits for positions.
  constexpr std::size_t h = 36;
  CHECK(u32_at(h) == 1);
  CHECK(u32_at(h + 4) == 15);
  CHECK(u32_at(h + 8) == 8);
  CHECK(u32_at(h + 12) == 7);
  CHECK(u32_at(h + 16) == 7);
  CHECK(u32_at(h + 28) == 0); // no codebook at SH degree 0
  CHECK(u32_at(h + 92) == 0); // the other layers are empty
  CHECK(u32_at(h + 2 * 92) == 0);

  // 6 position planes, 3 rotation, 3 scale, 3 colour and 1 alpha column.
  std::size_t at = h + 3 * 92;
  std::vector<std::vector<std::byte>> columns;
  for (int c = 0; c < 16; ++c) {
    const std::uint32_t size = u32_at(at);
    columns.push_back(lzma_decompress(std::as_bytes(std::span(file)).subspan(at + 4, size), 1));
    at += 4 + size;
  }
  CHECK(at == file.size());
  const auto value = [&](int c) { return std::to_integer<int>(columns[static_cast<std::size_t>(c)][0]); };
  for (int c = 0; c < 6; ++c) CHECK(value(c) == 0); // alone in its grid
  for (int c = 6; c < 9; ++c) CHECK(value(c) == 0); // identity: x = y = z = 0
  CHECK(u32_at(h + 48) == std::bit_cast<std::uint32_t>(-9.f));
  CHECK(value(9) == 160);  // (-1 - -9) / 0.05
  CHECK(value(10) == 20);  // gap of 1 at step 0.05
  CHECK(value(11) == 10);  // gap of 1 at step 0.1
  CHECK(value(15) == 64);  // alpha 0.5 at 7 bits
}
