#include "splat/cgc.hpp"

#include "splat/lzma.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace splat {

namespace {

static_assert(std::endian::native == std::endian::little, "CGC reading and writing assumes a little-endian host");

constexpr std::array<char, 4> kMagic = {'C', 'G', 'C', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kFileHeaderSize = 16; // magic, version, payload size
constexpr std::size_t kChunkSize = 256;
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kRotationRange = 0.70710678f; // 1/sqrt(2): bound of the three smallest components
constexpr std::array<int, 4> kShBits = {0, 5, 4, 4}; // per band
// Log scales below this (0.12 mm) are stored as this. Training leaves many
// splats with an axis only microns thin; it looks the same at any width below
// a pixel, but would stretch the chunk's 8-bit scale range over its neighbours.
constexpr float kMinLogScale = -9.f;

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// Unsigned quantization of min..max onto 0 .. 2^bits - 1.
std::uint32_t quantize_unsigned(float v, float lo, float hi, int bits) {
  if (!(hi > lo)) return 0;
  const float top = static_cast<float>((1u << bits) - 1);
  return static_cast<std::uint32_t>(std::clamp(std::round((v - lo) / (hi - lo) * top), 0.f, top));
}

float dequantize_unsigned(std::uint32_t q, float lo, float hi, int bits) {
  if (!(hi > lo)) return lo;
  return lo + static_cast<float>(q) / static_cast<float>((1u << bits) - 1) * (hi - lo);
}

// The value below which `fraction` of `values` lie (nearest rank). Reorders `values`.
float percentile(std::vector<float> &values, double fraction) {
  if (values.empty()) return 0.f;
  const auto k = static_cast<std::size_t>(std::round(fraction * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
  return values[k];
}

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

int choose_sh_degree(std::span<const float> sh, int degree, float threshold) {
  constexpr float kInvFourPi = 1.f / (4.f * std::numbers::pi_v<float>);
  for (int band = degree; band > 0; --band) {
    float worst = 0.f;
    for (int c = 0; c < 3; ++c) {
      float sum = 0.f;
      for (int k = band * band; k < (band + 1) * (band + 1); ++k) sum += sh[k * 3 + c] * sh[k * 3 + c];
      worst = std::max(worst, std::sqrt(sum * kInvFourPi));
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

namespace {

using namespace cgc_detail;

template <typename T> void put(std::vector<std::byte> &out, T value) {
  const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
  out.insert(out.end(), bytes.begin(), bytes.end());
}

void put_floats(std::vector<std::byte> &out, std::span<const float> values) {
  for (const float v : values) put(out, v);
}

// Bounds-checked sequential reads over the decompressed payload.
class Reader {
public:
  explicit Reader(std::span<const std::byte> data) : data_(data) {}

  std::span<const std::byte> take(std::size_t n) {
    if (n > data_.size() - pos_) throw std::runtime_error("payload is shorter than its header says");
    const auto out = data_.subspan(pos_, n);
    pos_ += n;
    return out;
  }
  template <typename T> T get() {
    std::array<std::byte, sizeof(T)> bytes;
    std::ranges::copy(take(sizeof(T)), bytes.begin());
    return std::bit_cast<T>(bytes);
  }
  std::vector<std::uint32_t> planes(std::size_t count) { return read_planes(take(count * 4), count); }
  std::size_t remaining() const { return data_.size() - pos_; }

private:
  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

bool all_finite(std::span<const float> values) {
  return std::ranges::all_of(values, [](float v) { return std::isfinite(v); });
}

} // namespace

CgcStats save_cgc(const GaussianCloud &cloud, const std::filesystem::path &path, const CgcOptions &options) {
  cloud.validate();
  if (!all_finite(cloud.positions) || !all_finite(cloud.scales) || !all_finite(cloud.rotations) ||
      !all_finite(cloud.opacities) || !all_finite(cloud.sh)) {
    throw std::runtime_error(std::format("{}: cannot save a cloud with NaN or infinite values", path.string()));
  }
  const std::size_t n = cloud.size();
  if (n > UINT32_MAX) throw std::runtime_error(std::format("{}: too many Gaussians for CGC", path.string()));
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  const auto sh_of = [&](std::size_t i) { return std::span(cloud.sh).subspan(i * coeffs * 3, coeffs * 3); };
  const auto scale_of = [&](std::size_t i, int a) { return std::max(cloud.scales[i * 3 + a], kMinLogScale); };

  CgcStats stats;
  std::vector<std::uint8_t> degrees(n);
  for (std::size_t i = 0; i < n; ++i) {
    degrees[i] = static_cast<std::uint8_t>(choose_sh_degree(sh_of(i), cloud.shDegree, options.shThreshold));
    ++stats.degreeCounts[degrees[i]];
  }

  // Morton order over the scene's bounding box.
  std::array<float, 3> lo{}, hi{};
  for (int a = 0; a < 3; ++a) {
    lo[a] = n > 0 ? INFINITY : 0.f;
    hi[a] = n > 0 ? -INFINITY : 0.f;
    for (std::size_t i = 0; i < n; ++i) {
      lo[a] = std::min(lo[a], cloud.positions[i * 3 + a]);
      hi[a] = std::max(hi[a], cloud.positions[i * 3 + a]);
    }
  }
  std::vector<std::uint64_t> codes(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::array<std::uint32_t, 3> q{};
    for (int a = 0; a < 3; ++a) {
      const float extent = hi[a] - lo[a];
      if (extent > 0.f) {
        const double t = std::floor(static_cast<double>(cloud.positions[i * 3 + a] - lo[a]) / extent * (1 << 21));
        q[a] = static_cast<std::uint32_t>(std::clamp(t, 0.0, static_cast<double>((1 << 21) - 1)));
      }
    }
    codes[i] = morton_code(q[0], q[1], q[2]);
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return codes[a] < codes[b]; });

  // Scene-wide ranges: DC colour per channel, and SH per band.
  std::array<float, 3> colorMin{}, colorMax{};
  for (int c = 0; c < 3; ++c) {
    std::vector<float> colours(n);
    for (std::size_t i = 0; i < n; ++i) colours[i] = kShC0 * cloud.sh[i * coeffs * 3 + c] + 0.5f;
    colorMin[c] = percentile(colours, 0.001);
    colorMax[c] = percentile(colours, 0.999);
  }
  std::array<float, 3> shRange{};
  for (int band = 1; band <= cloud.shDegree; ++band) {
    std::vector<float> magnitudes;
    for (std::size_t i = 0; i < n; ++i) {
      if (degrees[i] < band) continue;
      for (int k = band * band; k < (band + 1) * (band + 1); ++k) {
        for (int c = 0; c < 3; ++c) magnitudes.push_back(std::abs(cloud.sh[(i * coeffs + k) * 3 + c]));
      }
    }
    shRange[band - 1] = percentile(magnitudes, 0.999);
  }

  std::vector<std::byte> payload;
  put(payload, static_cast<std::uint32_t>(n));
  put_floats(payload, colorMin);
  put_floats(payload, colorMax);
  put_floats(payload, shRange);

  // Chunk table: position and log-scale ranges per 256 splats.
  const std::size_t chunks = (n + kChunkSize - 1) / kChunkSize;
  std::vector<float> table(chunks * 12);
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    float *t = &table[chunk * 12];
    for (int a = 0; a < 3; ++a) {
      t[a] = t[6 + a] = INFINITY;
      t[3 + a] = t[9 + a] = -INFINITY;
    }
    for (std::size_t j = chunk * kChunkSize; j < std::min(n, (chunk + 1) * kChunkSize); ++j) {
      const std::size_t i = order[j];
      for (int a = 0; a < 3; ++a) {
        t[a] = std::min(t[a], cloud.positions[i * 3 + a]);
        t[3 + a] = std::max(t[3 + a], cloud.positions[i * 3 + a]);
        t[6 + a] = std::min(t[6 + a], scale_of(i, a));
        t[9 + a] = std::max(t[9 + a], scale_of(i, a));
      }
    }
  }
  std::vector<std::uint32_t> words(table.size());
  std::ranges::transform(table, words.begin(), [](float v) { return std::bit_cast<std::uint32_t>(v); });
  append_planes(payload, words);

  std::vector<std::uint32_t> positions(n), rotations(n);
  std::array<std::vector<std::uint8_t>, 3> scales;
  std::array<std::vector<std::uint8_t>, 4> rgba;
  for (auto &s : scales) s.resize(n);
  for (auto &c : rgba) c.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t i = order[j];
    const float *t = &table[j / kChunkSize * 12];
    positions[j] = quantize_unsigned(cloud.positions[i * 3], t[0], t[3], 11) |
                   quantize_unsigned(cloud.positions[i * 3 + 1], t[1], t[4], 10) << 11 |
                   quantize_unsigned(cloud.positions[i * 3 + 2], t[2], t[5], 11) << 21;
    rotations[j] = pack_rotation({cloud.rotations[i * 4], cloud.rotations[i * 4 + 1], cloud.rotations[i * 4 + 2],
                                  cloud.rotations[i * 4 + 3]});
    for (int a = 0; a < 3; ++a) {
      scales[a][j] = static_cast<std::uint8_t>(quantize_unsigned(scale_of(i, a), t[6 + a], t[9 + a], 8));
    }
    for (int c = 0; c < 3; ++c) {
      const float colour = kShC0 * cloud.sh[i * coeffs * 3 + c] + 0.5f;
      rgba[c][j] = static_cast<std::uint8_t>(quantize_unsigned(colour, colorMin[c], colorMax[c], 8));
    }
    rgba[3][j] = static_cast<std::uint8_t>(std::clamp(std::floor(sigmoid(cloud.opacities[i]) * 256.f), 0.f, 255.f));
  }
  append_planes(payload, positions);
  append_planes(payload, rotations);
  for (const auto &column : scales) {
    for (const std::uint8_t v : column) payload.push_back(std::byte{v});
  }
  for (const auto &column : rgba) {
    for (const std::uint8_t v : column) payload.push_back(std::byte{v});
  }
  for (std::size_t j = 0; j < n; ++j) payload.push_back(std::byte{degrees[order[j]]});

  // Higher SH bands, only for the splats that keep them.
  for (int band = 1; band <= cloud.shDegree; ++band) {
    for (int k = band * band; k < (band + 1) * (band + 1); ++k) {
      for (int c = 0; c < 3; ++c) {
        for (std::size_t j = 0; j < n; ++j) {
          const std::size_t i = order[j];
          if (degrees[i] < band) continue;
          const float v = cloud.sh[(i * coeffs + static_cast<std::size_t>(k)) * 3 + c];
          payload.push_back(static_cast<std::byte>(quantize_symmetric(v, shRange[band - 1], kShBits[band])));
        }
      }
    }
  }

  const std::vector<std::byte> stream = lzma_compress(payload, options.lzmaLevel, options.lzmaExtreme);
  std::vector<std::byte> header;
  for (const char ch : kMagic) header.push_back(static_cast<std::byte>(ch));
  put(header, kVersion);
  put(header, static_cast<std::uint64_t>(payload.size()));

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error(std::format("Cannot write '{}'", path.string()));
  out.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char *>(stream.data()), static_cast<std::streamsize>(stream.size()));
  if (!out) throw std::runtime_error(std::format("Failed writing '{}'", path.string()));
  stats.bytes = header.size() + stream.size();
  return stats;
}

GaussianCloud load_cgc(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("Cannot open '{}'", path.string()));
  const std::vector<char> file{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  const auto bytes = std::as_bytes(std::span(file));
  if (bytes.size() < kFileHeaderSize || !std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
    throw std::runtime_error(std::format("{}: not a CGC file", path.string()));
  }
  Reader header(bytes.first(kFileHeaderSize));
  header.take(4);
  const auto version = header.get<std::uint32_t>();
  if (version != kVersion) throw std::runtime_error(std::format("{}: unsupported CGC version {}", path.string(), version));
  const auto payloadSize = header.get<std::uint64_t>();

  try {
    const std::vector<std::byte> payload = lzma_decompress(bytes.subspan(kFileHeaderSize), payloadSize);
    Reader r(payload);
    const std::size_t n = r.get<std::uint32_t>();
    std::array<float, 3> colorMin{}, colorMax{}, shRange{};
    for (float &v : colorMin) v = r.get<float>();
    for (float &v : colorMax) v = r.get<float>();
    for (float &v : shRange) v = r.get<float>();

    const std::size_t chunks = (n + kChunkSize - 1) / kChunkSize;
    const std::vector<std::uint32_t> table = r.planes(chunks * 12);
    const auto bound = [&](std::size_t j, int field) { return std::bit_cast<float>(table[j / kChunkSize * 12 + field]); };
    const std::vector<std::uint32_t> positions = r.planes(n);
    const std::vector<std::uint32_t> rotations = r.planes(n);
    std::array<std::span<const std::byte>, 3> scales;
    for (auto &column : scales) column = r.take(n);
    std::array<std::span<const std::byte>, 4> rgba;
    for (auto &column : rgba) column = r.take(n);
    const std::span<const std::byte> degrees = r.take(n);
    int maxDegree = 0;
    for (const std::byte d : degrees) {
      if (static_cast<int>(d) > 3) throw std::runtime_error("SH degree above 3");
      maxDegree = std::max(maxDegree, static_cast<int>(d));
    }

    GaussianCloud cloud;
    cloud.resize(n, maxDegree);
    const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
    for (std::size_t j = 0; j < n; ++j) {
      cloud.positions[j * 3] = dequantize_unsigned(positions[j] & 0x7FF, bound(j, 0), bound(j, 3), 11);
      cloud.positions[j * 3 + 1] = dequantize_unsigned(positions[j] >> 11 & 0x3FF, bound(j, 1), bound(j, 4), 10);
      cloud.positions[j * 3 + 2] = dequantize_unsigned(positions[j] >> 21, bound(j, 2), bound(j, 5), 11);
      const auto q = unpack_rotation(rotations[j]);
      std::ranges::copy(q, cloud.rotations.begin() + static_cast<std::ptrdiff_t>(j * 4));
      for (int a = 0; a < 3; ++a) {
        cloud.scales[j * 3 + a] = dequantize_unsigned(static_cast<std::uint32_t>(scales[a][j]), bound(j, 6 + a), bound(j, 9 + a), 8);
      }
      for (int c = 0; c < 3; ++c) {
        const float colour = dequantize_unsigned(static_cast<std::uint32_t>(rgba[c][j]), colorMin[c], colorMax[c], 8);
        cloud.sh[j * coeffs * 3 + c] = (colour - 0.5f) / kShC0;
      }
      const float alpha = (static_cast<float>(rgba[3][j]) + 0.5f) / 256.f;
      cloud.opacities[j] = std::log(alpha / (1.f - alpha));
    }
    for (int band = 1; band <= maxDegree; ++band) {
      std::vector<std::size_t> members;
      for (std::size_t j = 0; j < n; ++j) {
        if (static_cast<int>(degrees[j]) >= band) members.push_back(j);
      }
      for (int k = band * band; k < (band + 1) * (band + 1); ++k) {
        for (int c = 0; c < 3; ++c) {
          const auto column = r.take(members.size());
          for (std::size_t m = 0; m < members.size(); ++m) {
            cloud.sh[(members[m] * coeffs + static_cast<std::size_t>(k)) * 3 + c] =
                dequantize_symmetric(static_cast<std::uint32_t>(column[m]), shRange[band - 1], kShBits[band]);
          }
        }
      }
    }
    if (r.remaining() != 0) throw std::runtime_error("payload has unexpected trailing data");
    return cloud;
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(std::format("{}: {}", path.string(), e.what()));
  }
}

} // namespace splat
