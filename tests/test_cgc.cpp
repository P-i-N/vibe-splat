#include "render/offscreen.hpp"
#include "render/splat_renderer.hpp"
#include "splat/cgc.hpp"
#include "splat/lzma.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// Checks every attribute of `loaded` against `original` within CGC's
// quantization error. Splats come back in Morton order, so each is matched to
// the nearest original. Returns, per channel, how many DC colours missed the
// tolerance (allowed for the few clamped to the percentile range).
std::array<int, 3> check_round_trip(const GaussianCloud &original, const GaussianCloud &loaded, std::size_t skip = SIZE_MAX) {
  REQUIRE(loaded.size() == original.size());
  const int degree = original.shDegree;
  const std::size_t coeffs = static_cast<std::size_t>(original.shCoeffs());
  const std::size_t loadedCoeffs = static_cast<std::size_t>(loaded.shCoeffs());
  std::array<int, 3> colourMisses{};
  for (std::size_t j = 0; j < loaded.size(); ++j) {
    const std::size_t i = nearest(original, &loaded.positions[j * 3]);
    if (i == skip) continue;
    for (int a = 0; a < 3; ++a) CHECK(std::abs(loaded.positions[j * 3 + a] - original.positions[i * 3 + a]) <= 0.01f);
    for (int a = 0; a < 3; ++a) CHECK(std::abs(loaded.scales[j * 3 + a] - original.scales[i * 3 + a]) <= 0.05f);
    std::array<float, 4> qa{}, qb{};
    for (int a = 0; a < 4; ++a) {
      qa[a] = original.rotations[i * 4 + a];
      qb[a] = loaded.rotations[j * 4 + a];
    }
    CHECK(std::abs(dot4(normalized(qa), qb)) > 0.999f);
    CHECK(std::abs(sigmoid(loaded.opacities[j]) - sigmoid(original.opacities[i])) <= 1.f / 256.f);
    for (int c = 0; c < 3; ++c) {
      const float want = kShC0 * original.sh[i * coeffs * 3 + c] + 0.5f;
      const float got = kShC0 * loaded.sh[j * loadedCoeffs * 3 + c] + 0.5f;
      if (std::abs(want - got) > 0.004f) ++colourMisses[c];
    }
    const int kept = cgc_detail::choose_sh_degree(std::span(&original.sh[i * coeffs * 3], coeffs * 3), degree,
                                                  original.opacities[i], 1.5f / 255.f);
    for (int band = 1; band <= degree; ++band) {
      const float tolerance = band == 1 ? 0.3f / 15.f : 0.3f / 7.f;
      for (std::size_t k = static_cast<std::size_t>(band * band); k < static_cast<std::size_t>((band + 1) * (band + 1)); ++k) {
        for (int c = 0; c < 3; ++c) {
          const float got = loaded.sh[(j * loadedCoeffs + k) * 3 + c];
          if (band <= kept) {
            CHECK(std::abs(got - original.sh[(i * coeffs + k) * 3 + c]) <= tolerance);
          } else {
            CHECK(got == 0.f);
          }
        }
      }
    }
  }
  return colourMisses;
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
    CHECK(stats.degreeCounts[0] + stats.degreeCounts[1] + stats.degreeCounts[2] + stats.degreeCounts[3] == 1000);
    for (const int misses : check_round_trip(cloud, loaded)) CHECK(misses <= 2);
  }
}

TEST_CASE("CGC stores degree 0 when higher SH bands are empty", "[cgc]") {
  GaussianCloud cloud = random_cloud(500, 3, 5);
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    for (std::size_t k = 3; k < coeffs * 3; ++k) cloud.sh[i * coeffs * 3 + k] = 0.f;
  }
  CgcStats stats;
  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_dc_only.cgc", &stats);
  CHECK(stats.degreeCounts[0] == cloud.size());
  CHECK(loaded.shDegree == 0);
}

TEST_CASE("A colour outlier does not coarsen everyone else's colour", "[cgc]") {
  GaussianCloud cloud = random_cloud(1000, 1, 21);
  cloud.positions.insert(cloud.positions.end(), {5.f, 5.f, 11.f});
  cloud.scales.insert(cloud.scales.end(), {-3.f, -3.f, -3.f});
  cloud.rotations.insert(cloud.rotations.end(), {1.f, 0.f, 0.f, 0.f});
  cloud.opacities.push_back(2.f);
  cloud.sh.insert(cloud.sh.end(), {100.f, 100.f, 100.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

  const GaussianCloud loaded = round_trip(cloud, "splat_cgc_outlier.cgc");
  for (const int misses : check_round_trip(cloud, loaded, cloud.size() - 1)) CHECK(misses <= 2);
  for (const float v : loaded.sh) CHECK(std::isfinite(v));
}

TEST_CASE("CGC handles small, chunk-boundary and degenerate clouds", "[cgc]") {
  for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{256}, std::size_t{257}}) {
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
    CHECK(loaded.positions[i * 3] == 1.f);
    CHECK(loaded.positions[i * 3 + 1] == 2.f);
    CHECK(loaded.positions[i * 3 + 2] == 3.f);
    CHECK(loaded.scales[i * 3] == -2.f);
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

  auto badProperties = good;
  badProperties[16] = static_cast<char>(0xFF);
  write(badProperties);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  auto badVersion = good;
  badVersion[4] = 2;
  write(badVersion);
  CHECK_THROWS_AS(load_cgc(path), std::runtime_error);

  std::filesystem::remove(path);
}

namespace {

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

} // namespace

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

  const double value = psnr(imageA, imageB);
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
  save_cloud(cloud, temporary, CloudFormat::Cgc, CgcOptions{.lzmaLevel = 1, .lzmaExtreme = false});
  check_round_trip(cloud, load_cgc(temporary));
  std::filesystem::remove(temporary);
}
