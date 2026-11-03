#include "splat/cgc.hpp"

#include "splat/lzma.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

namespace splat {

namespace {

static_assert(std::endian::native == std::endian::little, "CGC reading and writing assumes a little-endian host");

constexpr std::array<char, 4> kMagic = {'C', 'G', 'C', '\0'};
constexpr std::uint32_t kVersion = 2;
constexpr float kShC0 = 0.28209479177387814f;
constexpr float kRotationRange = 0.70710678f; // 1/sqrt(2): bound of the three smallest components
// Log scales below this (0.12 mm) are stored as this. Training leaves many
// splats with an axis only microns thin; it looks the same at any width below
// a pixel.
constexpr float kMinLogScale = -9.f;
constexpr float kCodebookStep = 0.02f;
constexpr std::size_t kMaxCodebook = 65536;

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

void parallel_for(std::size_t n, const std::function<void(std::size_t)> &body) {
  const std::size_t grain = std::clamp<std::size_t>(n / 64, 1, 4096);
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (;;) {
        const std::size_t begin = next.fetch_add(grain);
        if (begin >= n) return;
        for (std::size_t i = begin; i < std::min(n, begin + grain); ++i) body(i);
      }
    });
  }
  for (auto &t : pool) t.join();
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

std::uint32_t compact3(std::uint64_t x) {
  x &= 0x1249249249249249ull;
  x = (x | x >> 2) & 0x10C30C30C30C30C3ull;
  x = (x | x >> 4) & 0x100F00F00F00F00Full;
  x = (x | x >> 8) & 0x1F0000FF0000FFull;
  x = (x | x >> 16) & 0x1F00000000FFFFull;
  x = (x | x >> 32) & 0x1FFFFF;
  return static_cast<std::uint32_t>(x);
}

std::uint32_t zigzag(int v) { return static_cast<std::uint32_t>(v >= 0 ? v * 2 : -v * 2 - 1); }
int unzigzag(std::uint32_t z) { return (z & 1) != 0 ? -static_cast<int>((z + 1) / 2) : static_cast<int>(z / 2); }

float symlog(float t) { return std::copysign(std::log1p(std::abs(t)), t); }
float symexp(float y) { return std::copysign(std::expm1(std::abs(y)), y); }

// YCoCg with simple inverses. Most SH energy is in Y, which is why the codebook
// clusters better here than in RGB.
std::array<float, 3> to_ycocg(float r, float g, float b) { return {0.25f * r + 0.5f * g + 0.25f * b, r - b, g - 0.5f * (r + b)}; }
std::array<float, 3> from_ycocg(float y, float co, float cg) {
  const float t = y - 0.5f * cg;
  return {t + 0.5f * co, y + 0.5f * cg, t - 0.5f * co};
}

// One importance layer's precision. Importance is opacity times cross-section,
// so a layer of faint, small splats can be stored far more coarsely unnoticed.
struct LayerParams {
  double fromPercentile; // the layer holds splats of at least this importance percentile
  std::uint32_t posBits, rotBits, alphaBits;
  std::array<std::uint32_t, 3> colorBits;
  float scaleStep, gap1Step, gap2Step;
  std::size_t splatsPerCode; // codebook size: layer size divided by this
};

std::vector<LayerParams> layer_params(CgcPreset preset) {
  const double top = preset == CgcPreset::Small ? 95.0 : 80.0;
  const double middle = preset == CgcPreset::Small ? 70.0 : 40.0;
  return {{top, 15, 8, 7, {7, 7, 7}, 0.05f, 0.05f, 0.1f, 16},
          {middle, 14, 6, 6, {6, 6, 6}, 0.1f, 0.1f, 0.2f, 120},
          {0.0, 13, 5, 5, {6, 5, 5}, 0.15f, 0.15f, 0.3f, 500}};
}

// k-means with a two-level search, so that tens of thousands of centres over
// millions of vectors stay affordable. Points may be weighted.
struct Codebook {
  std::vector<float> centers;
  std::vector<std::uint32_t> labels;
  std::size_t size = 0;
};

float distance2(const float *a, const float *b, std::size_t d) {
  float sum = 0.f;
  for (std::size_t i = 0; i < d; ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
  return sum;
}

std::uint32_t nearest_center(const float *v, const float *centers, std::size_t k, std::size_t d) {
  float best = INFINITY;
  std::uint32_t label = 0;
  for (std::size_t j = 0; j < k; ++j) {
    const float e = distance2(v, centers + j * d, d);
    if (e < best) {
      best = e;
      label = static_cast<std::uint32_t>(j);
    }
  }
  return label;
}

std::vector<float> lloyd(const std::vector<float> &x, std::span<const float> weights, std::span<const std::uint32_t> ids, std::size_t d,
                         std::size_t k, int iterations, std::mt19937 &rng, bool threaded) {
  std::vector<float> centers(k * d);
  const auto seed = [&](std::size_t j) { std::copy_n(&x[ids[rng() % ids.size()] * d], d, &centers[j * d]); };
  for (std::size_t j = 0; j < k; ++j) seed(j);
  std::vector<std::uint32_t> labels(ids.size());
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto assign = [&](std::size_t n) { labels[n] = nearest_center(&x[ids[n] * d], centers.data(), k, d); };
    if (threaded) parallel_for(ids.size(), assign);
    else for (std::size_t n = 0; n < ids.size(); ++n) assign(n);
    std::vector<double> sum(k * d, 0.0), total(k, 0.0);
    for (std::size_t n = 0; n < ids.size(); ++n) {
      const double w = weights[ids[n]];
      total[labels[n]] += w;
      for (std::size_t a = 0; a < d; ++a) sum[labels[n] * d + a] += w * x[ids[n] * d + a];
    }
    for (std::size_t j = 0; j < k; ++j) {
      if (total[j] <= 0.0) {
        seed(j);
      } else {
        for (std::size_t a = 0; a < d; ++a) centers[j * d + a] = static_cast<float>(sum[j * d + a] / total[j]);
      }
    }
  }
  return centers;
}

Codebook build_codebook(const std::vector<float> &x, std::span<const float> weights, std::size_t d, std::size_t k, bool fast) {
  const std::size_t n = x.size() / d;
  std::mt19937 rng(12345);
  std::vector<std::uint32_t> sample(n);
  std::iota(sample.begin(), sample.end(), 0u);
  std::ranges::shuffle(sample, rng);
  sample.resize(std::min(n, fast ? std::size_t{50'000} : std::max<std::size_t>(200'000, k * 16)));

  const std::size_t top = std::clamp<std::size_t>(static_cast<std::size_t>(std::sqrt(static_cast<double>(k))), 1, 512);
  const auto topSample = std::span(sample).first(std::min<std::size_t>(sample.size(), 100'000));
  const std::vector<float> topCenters = lloyd(x, weights, topSample, d, top, fast ? 3 : 8, rng, true);

  std::vector<std::uint32_t> cellOf(sample.size());
  parallel_for(sample.size(), [&](std::size_t i) { cellOf[i] = nearest_center(&x[sample[i] * d], topCenters.data(), top, d); });
  std::vector<std::vector<std::uint32_t>> cells(top);
  for (std::size_t i = 0; i < sample.size(); ++i) cells[cellOf[i]].push_back(sample[i]);

  std::vector<std::size_t> cellSize(top), base(top + 1, 0);
  for (std::size_t t = 0; t < top; ++t) {
    const double share = static_cast<double>(k) * static_cast<double>(cells[t].size()) / static_cast<double>(sample.size());
    cellSize[t] = cells[t].empty() ? 0 : std::clamp<std::size_t>(static_cast<std::size_t>(std::round(share)), 1, cells[t].size());
    base[t + 1] = base[t] + cellSize[t];
  }
  // Rounding can push the total over the 16-bit limit of the labels.
  while (base[top] > kMaxCodebook) {
    const std::size_t largest = static_cast<std::size_t>(std::ranges::max_element(cellSize) - cellSize.begin());
    --cellSize[largest];
    for (std::size_t t = 0; t < top; ++t) base[t + 1] = base[t] + cellSize[t];
  }

  Codebook book;
  book.size = base[top];
  book.centers.resize(book.size * d);
  parallel_for(top, [&](std::size_t t) {
    if (cellSize[t] == 0) return;
    std::mt19937 local(static_cast<std::uint32_t>(t) + 99);
    const std::vector<float> c = lloyd(x, weights, cells[t], d, cellSize[t], fast ? 2 : 6, local, false);
    std::ranges::copy(c, book.centers.begin() + static_cast<std::ptrdiff_t>(base[t] * d));
  });

  // Assign everything by probing the three nearest cells, update the centres
  // once over all the data, and assign again.
  book.labels.resize(n);
  const auto assign_all = [&] {
    parallel_for(n, [&](std::size_t i) {
      const float *v = &x[i * d];
      std::array<float, 3> nearest = {INFINITY, INFINITY, INFINITY};
      std::array<std::size_t, 3> cell = {top, top, top};
      for (std::size_t t = 0; t < top; ++t) {
        if (cellSize[t] == 0) continue;
        const float e = distance2(v, &topCenters[t * d], d);
        for (std::size_t s = 0; s < 3; ++s) {
          if (e < nearest[s]) {
            for (std::size_t m = 2; m > s; --m) {
              nearest[m] = nearest[m - 1];
              cell[m] = cell[m - 1];
            }
            nearest[s] = e;
            cell[s] = t;
            break;
          }
        }
      }
      float best = INFINITY;
      for (const std::size_t t : cell) {
        if (t == top) continue;
        const std::uint32_t j = nearest_center(v, &book.centers[base[t] * d], cellSize[t], d);
        const float e = distance2(v, &book.centers[(base[t] + j) * d], d);
        if (e < best) {
          best = e;
          book.labels[i] = static_cast<std::uint32_t>(base[t] + j);
        }
      }
    });
  };
  assign_all();
  std::vector<double> sum(book.size * d, 0.0), total(book.size, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    total[book.labels[i]] += weights[i];
    for (std::size_t a = 0; a < d; ++a) sum[book.labels[i] * d + a] += static_cast<double>(weights[i]) * x[i * d + a];
  }
  for (std::size_t j = 0; j < book.size; ++j) {
    if (total[j] <= 0.0) continue;
    for (std::size_t a = 0; a < d; ++a) book.centers[j * d + a] = static_cast<float>(sum[j * d + a] / total[j]);
  }
  assign_all();
  return book;
}

template <typename T> void put(std::vector<std::byte> &out, T value) {
  const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
  out.insert(out.end(), bytes.begin(), bytes.end());
}

// The fixed-size header of one layer. Every field is 4 bytes, little endian.
struct LayerHeader {
  std::uint32_t count = 0, posBits = 0, rotBits = 0, alphaBits = 0;
  std::array<std::uint32_t, 3> colorBits{};
  std::uint32_t codebookSize = 0;
  std::array<float, 3> posLo{};
  float posStep = 1.f, scaleLo = kMinLogScale, scaleStep = 1.f, gap1Step = 1.f, gap2Step = 1.f;
  std::array<float, 3> colorLo{}, colorStep{};
  float codebookStep = kCodebookStep;
};
static_assert(sizeof(LayerHeader) == 23 * 4 && std::is_trivially_copyable_v<LayerHeader>);

using Column = std::vector<std::byte>;

// The r-th smallest value after skipping the most extreme ones at each end
// (at least one in clouds big enough), so that an outlier cannot coarsen a
// whole colour channel.
std::array<float, 2> robust_range(std::vector<float> values) {
  const std::size_t n = values.size();
  const std::size_t skip = n < 64 ? 0 : std::max<std::size_t>(1, n / 10'000);
  const auto low = values.begin() + static_cast<std::ptrdiff_t>(skip);
  std::nth_element(values.begin(), low, values.end());
  const float lo = *low;
  const auto high = values.begin() + static_cast<std::ptrdiff_t>(n - 1 - skip);
  std::nth_element(values.begin(), high, values.end());
  return {lo, *high};
}

// Encodes one layer into its columns, in file order.
std::vector<Column> encode_layer(const GaussianCloud &cloud, std::vector<std::uint32_t> members, const LayerParams &params,
                                 const std::array<float, 3> &center, float radius, bool fast, LayerHeader &h) {
  const std::size_t n = members.size();
  h.count = static_cast<std::uint32_t>(n);
  h.posBits = params.posBits;
  h.rotBits = params.rotBits;
  h.alphaBits = params.alphaBits;
  h.colorBits = params.colorBits;
  std::vector<Column> columns;
  if (n == 0) return columns;

  // Positions: a cube grid in symlog space around the scene centre, so that
  // precision falls with distance. Sorted along a Morton curve; only the gaps
  // between codes are stored, which saves the entropy of the splats' order.
  const auto warped = [&](std::uint32_t i, int a) { return symlog((cloud.positions[i * 3 + static_cast<std::size_t>(a)] - center[static_cast<std::size_t>(a)]) / radius); };
  std::array<float, 3> hi{};
  for (std::size_t a = 0; a < 3; ++a) {
    h.posLo[a] = INFINITY;
    hi[a] = -INFINITY;
    for (const std::uint32_t i : members) {
      const float v = warped(i, static_cast<int>(a));
      h.posLo[a] = std::min(h.posLo[a], v);
      hi[a] = std::max(hi[a], v);
    }
  }
  const float extent = std::max({hi[0] - h.posLo[0], hi[1] - h.posLo[1], hi[2] - h.posLo[2]});
  h.posStep = extent > 0.f ? extent / static_cast<float>((1u << params.posBits) - 1) : 1.f;
  std::vector<std::uint64_t> codes(cloud.size());
  parallel_for(n, [&](std::size_t j) {
    std::array<std::uint32_t, 3> q{};
    for (std::size_t a = 0; a < 3; ++a) q[a] = static_cast<std::uint32_t>(std::round((warped(members[j], static_cast<int>(a)) - h.posLo[a]) / h.posStep));
    codes[members[j]] = cgc_detail::morton_code(q[0], q[1], q[2]);
  });
  std::ranges::sort(members, [&](std::uint32_t a, std::uint32_t b) { return codes[a] != codes[b] ? codes[a] < codes[b] : a < b; });
  const std::size_t planes = (params.posBits * 3 + 7) / 8;
  for (std::size_t p = 0; p < planes; ++p) {
    Column column(n);
    std::uint64_t previous = 0;
    for (std::size_t j = 0; j < n; ++j) {
      const std::uint64_t code = codes[members[j]];
      column[j] = static_cast<std::byte>((code - previous) >> (8 * p));
      previous = code;
    }
    columns.push_back(std::move(column));
  }

  // Rotations and scales: axes sorted by size, canonical quaternion.
  std::array<Column, 3> rotation, scale;
  for (auto &c : rotation) c.resize(n);
  for (auto &c : scale) c.resize(n);
  float largest = kMinLogScale;
  for (const std::uint32_t i : members) largest = std::max({largest, cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]});
  h.scaleLo = kMinLogScale;
  h.scaleStep = std::max(params.scaleStep, (largest - kMinLogScale) / 255.f);
  h.gap1Step = params.gap1Step;
  h.gap2Step = params.gap2Step;
  const int rotationTop = (1 << (params.rotBits - 1)) - 1;
  parallel_for(n, [&](std::size_t j) {
    const std::uint32_t i = members[j];
    std::array<float, 3> s = {cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]};
    std::array<float, 4> q = {cloud.rotations[i * 4], cloud.rotations[i * 4 + 1], cloud.rotations[i * 4 + 2], cloud.rotations[i * 4 + 3]};
    cgc_detail::canonicalize_axes(s, q);
    for (std::size_t a = 0; a < 3; ++a) {
      const int v = static_cast<int>(std::round(std::clamp(q[a + 1] / kRotationRange, -1.f, 1.f) * static_cast<float>(rotationTop)));
      rotation[a][j] = static_cast<std::byte>(zigzag(v));
    }
    const float s0 = std::max(s[0], kMinLogScale), s1 = std::max(s[1], kMinLogScale), s2 = std::max(s[2], kMinLogScale);
    const int q0 = std::clamp(static_cast<int>(std::round((s0 - h.scaleLo) / h.scaleStep)), 0, 255);
    const float r0 = h.scaleLo + static_cast<float>(q0) * h.scaleStep;
    const int q1 = std::clamp(static_cast<int>(std::round((r0 - s1) / h.gap1Step)), 0, 255);
    const float r1 = r0 - static_cast<float>(q1) * h.gap1Step;
    const int q2 = std::clamp(static_cast<int>(std::round((r1 - s2) / h.gap2Step)), 0, 255);
    scale[0][j] = static_cast<std::byte>(q0);
    scale[1][j] = static_cast<std::byte>(q1);
    scale[2][j] = static_cast<std::byte>(q2);
  });
  for (auto &c : rotation) columns.push_back(std::move(c));
  for (auto &c : scale) columns.push_back(std::move(c));

  // Colour in YCoCg.
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  std::array<std::vector<float>, 3> ycocg;
  for (auto &c : ycocg) c.resize(n);
  parallel_for(n, [&](std::size_t j) {
    const float *sh = &cloud.sh[members[j] * coeffs * 3];
    const auto c = to_ycocg(kShC0 * sh[0] + 0.5f, kShC0 * sh[1] + 0.5f, kShC0 * sh[2] + 0.5f);
    for (std::size_t a = 0; a < 3; ++a) ycocg[a][j] = c[a];
  });
  for (std::size_t a = 0; a < 3; ++a) {
    const auto range = robust_range(ycocg[a]);
    const float top = static_cast<float>((1u << params.colorBits[a]) - 1);
    h.colorLo[a] = range[0];
    h.colorStep[a] = range[1] > range[0] ? (range[1] - range[0]) / top : 1.f;
    Column column(n);
    for (std::size_t j = 0; j < n; ++j) {
      column[j] = static_cast<std::byte>(static_cast<int>(std::clamp(std::round((ycocg[a][j] - h.colorLo[a]) / h.colorStep[a]), 0.f, top)));
    }
    columns.push_back(std::move(column));
  }

  Column alpha(n);
  const float levels = static_cast<float>(1u << params.alphaBits);
  for (std::size_t j = 0; j < n; ++j) {
    alpha[j] = static_cast<std::byte>(static_cast<int>(std::clamp(std::floor(sigmoid(cloud.opacities[members[j]]) * levels), 0.f, levels - 1.f)));
  }
  columns.push_back(std::move(alpha));

  // Higher SH bands: one codebook over all of them, in YCoCg. Centres are
  // weighted by how much each splat can show, its opacity times its area.
  const std::size_t dims = (coeffs - 1) * 3;
  if (dims == 0) return columns;
  std::vector<float> x(n * dims), weights(n);
  parallel_for(n, [&](std::size_t j) {
    const std::uint32_t i = members[j];
    for (std::size_t k = 1; k < coeffs; ++k) {
      const float *sh = &cloud.sh[(i * coeffs + k) * 3];
      const auto c = to_ycocg(sh[0], sh[1], sh[2]);
      std::ranges::copy(c, &x[j * dims + (k - 1) * 3]);
    }
    std::array<float, 3> s = {cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]};
    std::ranges::sort(s);
    weights[j] = sigmoid(cloud.opacities[i]) * std::exp(std::clamp(s[2] + s[1], -30.f, 30.f)) + 1e-12f;
  });
  // Small layers get relatively more codes: their SH shows, and files that size have bytes to spare.
  const std::size_t wanted = std::min({std::max({n / params.splatsPerCode, std::min<std::size_t>(n / 4, 1024), std::size_t{1}}), n, kMaxCodebook});
  Codebook book = build_codebook(x, weights, dims, fast ? std::min<std::size_t>(wanted, 1024) : wanted, fast);
  h.codebookSize = static_cast<std::uint32_t>(book.size);
  h.codebookStep = kCodebookStep;

  // Most frequent entries first, which makes the high label byte cheap.
  std::vector<std::uint32_t> uses(book.size, 0), byUse(book.size), rank(book.size);
  for (const std::uint32_t label : book.labels) ++uses[label];
  std::iota(byUse.begin(), byUse.end(), 0u);
  std::ranges::sort(byUse, [&](std::uint32_t a, std::uint32_t b) { return uses[a] != uses[b] ? uses[a] > uses[b] : a < b; });
  for (std::size_t t = 0; t < book.size; ++t) rank[byUse[t]] = static_cast<std::uint32_t>(t);
  Column labelLow(n), labelHigh(n);
  for (std::size_t j = 0; j < n; ++j) {
    const std::uint32_t t = rank[book.labels[j]];
    labelLow[j] = static_cast<std::byte>(t);
    labelHigh[j] = static_cast<std::byte>(t >> 8);
  }
  columns.push_back(std::move(labelLow));
  columns.push_back(std::move(labelHigh));
  Column low(book.size * dims), high(book.size * dims);
  for (std::size_t a = 0; a < dims; ++a) {
    for (std::size_t t = 0; t < book.size; ++t) {
      const float v = book.centers[byUse[t] * dims + a];
      const std::uint32_t z = zigzag(std::clamp(static_cast<int>(std::round(v / kCodebookStep)), -32767, 32767));
      low[a * book.size + t] = static_cast<std::byte>(z);
      high[a * book.size + t] = static_cast<std::byte>(z >> 8);
    }
  }
  columns.push_back(std::move(low));
  columns.push_back(std::move(high));
  return columns;
}

bool all_finite(std::span<const float> values) {
  return std::ranges::all_of(values, [](float v) { return std::isfinite(v); });
}

// Bounds-checked sequential reads.
class Reader {
public:
  explicit Reader(std::span<const std::byte> data) : data_(data) {}

  std::span<const std::byte> take(std::size_t n) {
    if (n > data_.size() - pos_) throw std::runtime_error("file is shorter than its header says");
    const auto out = data_.subspan(pos_, n);
    pos_ += n;
    return out;
  }
  template <typename T> T get() {
    std::array<std::byte, sizeof(T)> bytes;
    std::ranges::copy(take(sizeof(T)), bytes.begin());
    return std::bit_cast<T>(bytes);
  }
  std::size_t remaining() const { return data_.size() - pos_; }

private:
  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

// Column sizes of one layer, in file order.
std::vector<std::size_t> column_sizes(const LayerHeader &h, std::size_t dims) {
  std::vector<std::size_t> sizes;
  if (h.count == 0) return sizes;
  const std::size_t n = h.count;
  sizes.assign((h.posBits * 3 + 7) / 8 + 3 + 3 + 3 + 1, n);
  if (dims > 0) {
    sizes.insert(sizes.end(), {n, n});
    const std::size_t book = std::size_t{h.codebookSize} * dims;
    sizes.insert(sizes.end(), {book, book});
  }
  return sizes;
}

void decode_layer(const LayerHeader &h, const std::vector<std::vector<std::byte>> &columns, const std::array<float, 3> &center, float radius,
                  std::size_t first, GaussianCloud &cloud) {
  const std::size_t n = h.count;
  if (n == 0) return;
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  const std::size_t planes = (h.posBits * 3 + 7) / 8;
  const auto at = [&](std::size_t column, std::size_t j) { return std::to_integer<std::uint32_t>(columns[column][j]); };
  std::uint64_t code = 0;
  for (std::size_t j = 0; j < n; ++j) {
    std::uint64_t gap = 0;
    for (std::size_t p = 0; p < planes; ++p) gap |= static_cast<std::uint64_t>(at(p, j)) << (8 * p);
    code += gap;
    const std::array<std::uint32_t, 3> q = {compact3(code), compact3(code >> 1), compact3(code >> 2)};
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.positions[(first + j) * 3 + a] = symexp(h.posLo[a] + static_cast<float>(q[a]) * h.posStep) * radius + center[a];
    }
  }
  std::size_t c = planes;
  const float rotationStep = kRotationRange / static_cast<float>((1 << (h.rotBits - 1)) - 1);
  const float levels = static_cast<float>(1u << h.alphaBits);
  const std::size_t rotColumn = c, scaleColumn = c + 3, colorColumn = c + 6, alphaColumn = c + 9;
  parallel_for(n, [&](std::size_t j) {
    const std::size_t out = first + j;
    float sum = 0.f;
    for (std::size_t a = 0; a < 3; ++a) {
      const float v = static_cast<float>(unzigzag(at(rotColumn + a, j))) * rotationStep;
      cloud.rotations[out * 4 + a + 1] = v;
      sum += v * v;
    }
    cloud.rotations[out * 4] = std::sqrt(std::max(0.f, 1.f - sum));
    const float s0 = h.scaleLo + static_cast<float>(at(scaleColumn, j)) * h.scaleStep;
    const float s1 = s0 - static_cast<float>(at(scaleColumn + 1, j)) * h.gap1Step;
    const float s2 = s1 - static_cast<float>(at(scaleColumn + 2, j)) * h.gap2Step;
    cloud.scales[out * 3] = s0;
    cloud.scales[out * 3 + 1] = s1;
    cloud.scales[out * 3 + 2] = s2;
    std::array<float, 3> ycocg{};
    for (std::size_t a = 0; a < 3; ++a) ycocg[a] = h.colorLo[a] + static_cast<float>(at(colorColumn + a, j)) * h.colorStep[a];
    const auto rgb = from_ycocg(ycocg[0], ycocg[1], ycocg[2]);
    for (std::size_t a = 0; a < 3; ++a) cloud.sh[out * coeffs * 3 + a] = (rgb[a] - 0.5f) / kShC0;
    const float alpha = (static_cast<float>(at(alphaColumn, j)) + 0.5f) / levels;
    cloud.opacities[out] = std::log(alpha / (1.f - alpha));
  });
  const std::size_t dims = (coeffs - 1) * 3;
  if (dims == 0) return;
  const std::size_t labelColumn = alphaColumn + 1, book = h.codebookSize;
  std::vector<float> rgbBook(book * dims);
  parallel_for(book, [&](std::size_t t) {
    const auto entry = [&](std::size_t a) {
      const std::uint32_t z = at(labelColumn + 2, a * book + t) | at(labelColumn + 3, a * book + t) << 8;
      return static_cast<float>(unzigzag(z)) * h.codebookStep;
    };
    for (std::size_t a = 0; a < dims; a += 3) {
      const auto rgb = from_ycocg(entry(a), entry(a + 1), entry(a + 2));
      std::ranges::copy(rgb, &rgbBook[t * dims + a]);
    }
  });
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t label = at(labelColumn, j) | at(labelColumn + 1, j) << 8;
    if (label >= book) throw std::runtime_error("SH label outside the codebook");
    std::copy_n(&rgbBook[label * dims], dims, &cloud.sh[((first + j) * coeffs + 1) * 3]);
  }
}

GaussianCloud load_v1(std::span<const std::byte> bytes);

} // namespace

namespace cgc_detail {

std::uint64_t morton_code(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return spread3(x) | spread3(y) << 1 | spread3(z) << 2;
}

void canonicalize_axes(std::array<float, 3> &logScales, std::array<float, 4> &q) {
  const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(norm > 0.f)) q = {1.f, 0.f, 0.f, 0.f};
  else for (float &v : q) v /= norm;
  const auto [w, x, y, z] = q;
  // Rotation matrix columns are the ellipsoid's axes.
  const std::array<std::array<float, 3>, 3> r = {{{1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)},
                                                  {2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)},
                                                  {2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)}}};
  std::array<std::size_t, 3> order = {0, 1, 2};
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return logScales[a] > logScales[b]; });
  logScales = {logScales[order[0]], logScales[order[1]], logScales[order[2]]};
  std::array<std::array<float, 3>, 3> m = {r[order[0]], r[order[1]], r[order[2]]};
  const float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
                    m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
  if (det < 0.f) for (float &v : m[2]) v = -v;

  // Back to a quaternion (Shepperd's method).
  const float trace = m[0][0] + m[1][1] + m[2][2];
  std::array<float, 4> t{};
  if (trace > 0.f) {
    const float s = std::sqrt(trace + 1.f) * 2.f;
    t = {s / 4.f, (m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s, (m[0][1] - m[1][0]) / s};
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const float s = std::sqrt(1.f + m[0][0] - m[1][1] - m[2][2]) * 2.f;
    t = {(m[1][2] - m[2][1]) / s, s / 4.f, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s};
  } else if (m[1][1] > m[2][2]) {
    const float s = std::sqrt(1.f + m[1][1] - m[0][0] - m[2][2]) * 2.f;
    t = {(m[2][0] - m[0][2]) / s, (m[1][0] + m[0][1]) / s, s / 4.f, (m[2][1] + m[1][2]) / s};
  } else {
    const float s = std::sqrt(1.f + m[2][2] - m[0][0] - m[1][1]) * 2.f;
    t = {(m[0][1] - m[1][0]) / s, (m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, s / 4.f};
  }
  // Flipping two axes leaves an ellipsoid unchanged, and multiplies the
  // quaternion by i, j or k, which moves any component into w. So w can always
  // be the largest component, and needs neither storing nor an index.
  std::size_t largest = 0;
  for (std::size_t a = 1; a < 4; ++a) {
    if (std::abs(t[a]) > std::abs(t[largest])) largest = a;
  }
  const auto [tw, tx, ty, tz] = t;
  switch (largest) {
  case 0: q = {tw, tx, ty, tz}; break;
  case 1: q = {-tx, tw, tz, -ty}; break;
  case 2: q = {-ty, -tz, tw, tx}; break;
  default: q = {-tz, ty, -tx, tw}; break;
  }
  if (q[0] < 0.f) for (float &v : q) v = -v;
}

} // namespace cgc_detail

CgcStats save_cgc(const GaussianCloud &cloud, const std::filesystem::path &path, const CgcOptions &options) {
  cloud.validate();
  if (!all_finite(cloud.positions) || !all_finite(cloud.scales) || !all_finite(cloud.rotations) ||
      !all_finite(cloud.opacities) || !all_finite(cloud.sh)) {
    throw std::runtime_error(std::format("{}: cannot save a cloud with NaN or infinite values", path.string()));
  }
  const std::size_t n = cloud.size();
  if (n > UINT32_MAX) throw std::runtime_error(std::format("{}: too many Gaussians for CGC", path.string()));

  std::array<float, 3> center{};
  float radius = 1.f;
  if (n > 0) {
    const Bounds bounds = compute_bounds(cloud);
    for (std::size_t a = 0; a < 3; ++a) center[a] = bounds.center[a];
    if (bounds.radius > 0.f) radius = bounds.radius;
  }

  // Split into layers by importance, most important first.
  const std::vector<LayerParams> params = layer_params(options.preset);
  std::vector<float> importance(n);
  parallel_for(n, [&](std::size_t i) {
    std::array<float, 3> s = {cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]};
    std::ranges::sort(s);
    importance[i] = std::log(std::max(sigmoid(cloud.opacities[i]), 1e-30f)) + s[2] + s[1];
  });
  std::vector<float> sorted = importance;
  std::ranges::sort(sorted);
  std::vector<std::vector<std::uint32_t>> members(params.size());
  std::vector<float> cuts;
  for (const LayerParams &p : params) {
    cuts.push_back(n > 0 ? sorted[static_cast<std::size_t>(p.fromPercentile / 100.0 * static_cast<double>(n - 1))] : 0.f);
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    std::size_t l = 0;
    while (l + 1 < params.size() && importance[i] < cuts[l]) ++l;
    members[l].push_back(i);
  }

  CgcStats stats;
  std::vector<LayerHeader> headers(params.size());
  std::vector<Column> columns;
  for (std::size_t l = 0; l < params.size(); ++l) {
    stats.layerCounts[l] = members[l].size();
    std::vector<Column> c = encode_layer(cloud, std::move(members[l]), params[l], center, radius, options.fast, headers[l]);
    std::ranges::move(c, std::back_inserter(columns));
  }

  // Each column is its own stream: lc = lp = pb = 0 suits byte columns best,
  // and a viewer may decode them in parallel.
  std::vector<std::vector<std::byte>> streams(columns.size());
  std::atomic<std::size_t> nextColumn{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t) {
    pool.emplace_back([&] {
      for (std::size_t i; (i = nextColumn.fetch_add(1)) < columns.size();) {
        streams[i] = lzma_compress(columns[i], options.fast ? 1 : 6, false, 0, 0, 0);
      }
    });
  }
  for (auto &t : pool) t.join();

  std::vector<std::byte> file;
  for (const char ch : kMagic) file.push_back(static_cast<std::byte>(ch));
  put(file, kVersion);
  put(file, static_cast<std::uint32_t>(n));
  put(file, static_cast<std::uint32_t>(cloud.shDegree));
  put(file, static_cast<std::uint32_t>(headers.size()));
  for (const float v : center) put(file, v);
  put(file, radius);
  for (const LayerHeader &h : headers) put(file, h);
  for (const auto &stream : streams) {
    put(file, static_cast<std::uint32_t>(stream.size()));
    file.insert(file.end(), stream.begin(), stream.end());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error(std::format("Cannot write '{}'", path.string()));
  out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
  if (!out) throw std::runtime_error(std::format("Failed writing '{}'", path.string()));
  stats.bytes = file.size();
  return stats;
}

GaussianCloud load_cgc(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("Cannot open '{}'", path.string()));
  const std::vector<char> file{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  const auto bytes = std::as_bytes(std::span(file));
  if (bytes.size() < 8 || !std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
    throw std::runtime_error(std::format("{}: not a CGC file", path.string()));
  }
  try {
    Reader r(bytes);
    r.take(4);
    const auto version = r.get<std::uint32_t>();
    if (version == 1) return load_v1(bytes);
    if (version != kVersion) throw std::runtime_error(std::format("unsupported CGC version {}", version));
    const std::size_t n = r.get<std::uint32_t>();
    const auto degree = r.get<std::uint32_t>();
    const auto layers = r.get<std::uint32_t>();
    if (degree > 3) throw std::runtime_error("SH degree above 3");
    if (layers > 16) throw std::runtime_error("too many layers");
    std::array<float, 3> center{};
    for (float &v : center) v = r.get<float>();
    const auto radius = r.get<float>();
    std::vector<LayerHeader> headers(layers);
    const std::size_t dims = 3 * ((degree + 1) * (degree + 1) - 1);
    std::size_t total = 0;
    for (LayerHeader &h : headers) {
      h = r.get<LayerHeader>();
      total += h.count;
      if (h.count == 0) continue;
      const bool colorOk = std::ranges::all_of(h.colorBits, [](std::uint32_t b) { return b >= 1 && b <= 8; });
      if (h.posBits < 1 || h.posBits > 21 || h.rotBits < 2 || h.rotBits > 8 || h.alphaBits < 1 || h.alphaBits > 8 || !colorOk ||
          h.codebookSize > kMaxCodebook || (dims > 0 && h.codebookSize == 0)) {
        throw std::runtime_error("invalid layer header");
      }
    }
    if (total != n) throw std::runtime_error("layer sizes do not add up");

    struct Job {
      std::span<const std::byte> stream;
      std::size_t size;
      std::vector<std::byte> *out;
    };
    std::vector<std::vector<std::vector<std::byte>>> columns(layers);
    std::vector<Job> jobs;
    for (std::size_t l = 0; l < layers; ++l) {
      const std::vector<std::size_t> sizes = column_sizes(headers[l], dims);
      columns[l].resize(sizes.size());
      for (std::size_t c = 0; c < sizes.size(); ++c) {
        const auto compressed = r.get<std::uint32_t>();
        jobs.push_back({r.take(compressed), sizes[c], &columns[l][c]});
      }
    }
    if (r.remaining() != 0) throw std::runtime_error("file has unexpected trailing data");

    std::atomic<std::size_t> next{0};
    std::mutex failureMutex;
    std::string failure;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t) {
      pool.emplace_back([&] {
        for (std::size_t i; (i = next.fetch_add(1)) < jobs.size();) {
          try {
            *jobs[i].out = lzma_decompress(jobs[i].stream, jobs[i].size);
          } catch (const std::runtime_error &e) {
            const std::scoped_lock lock(failureMutex);
            failure = e.what();
          }
        }
      });
    }
    for (auto &t : pool) t.join();
    if (!failure.empty()) throw std::runtime_error(failure);

    GaussianCloud cloud;
    cloud.resize(n, static_cast<int>(degree));
    std::size_t first = 0;
    for (std::size_t l = 0; l < layers; ++l) {
      decode_layer(headers[l], columns[l], center, radius, first, cloud);
      first += headers[l].count;
    }
    return cloud;
  } catch (const std::runtime_error &e) {
    throw std::runtime_error(std::format("{}: {}", path.string(), e.what()));
  }
}

namespace {

// Version 1 files (chunk boxes, byte planes, one LZMA stream) stay readable.
float dequantize_unsigned(std::uint32_t q, float lo, float hi, int bits) {
  if (!(hi > lo)) return lo;
  return lo + static_cast<float>(q) / static_cast<float>((1u << bits) - 1) * (hi - lo);
}

float dequantize_symmetric(std::uint32_t q, float r, int bits) {
  const int h = (1 << (bits - 1)) - 1;
  return static_cast<float>(static_cast<int>(q) - h) / static_cast<float>(h) * r;
}

std::array<float, 4> unpack_rotation_v1(std::uint32_t packed) {
  const int largest = static_cast<int>(packed >> 30);
  std::array<float, 4> q{};
  float sum = 0.f;
  int shift = 0;
  for (int i = 0; i < 4; ++i) {
    if (i == largest) continue;
    q[static_cast<std::size_t>(i)] = dequantize_symmetric((packed >> shift) & 0x3FF, kRotationRange, 10);
    sum += q[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i)];
    shift += 10;
  }
  q[static_cast<std::size_t>(largest)] = std::sqrt(std::max(0.f, 1.f - sum));
  const float norm = std::sqrt(sum + q[static_cast<std::size_t>(largest)] * q[static_cast<std::size_t>(largest)]);
  for (float &v : q) v /= norm;
  return q;
}

std::vector<std::uint32_t> read_planes(std::span<const std::byte> in, std::size_t count) {
  std::vector<std::uint32_t> values(count, 0);
  for (std::size_t b = 0; b < 4; ++b) {
    for (std::size_t i = 0; i < count; ++i) values[i] |= static_cast<std::uint32_t>(in[b * count + i]) << (8 * b);
  }
  return values;
}

GaussianCloud load_v1(std::span<const std::byte> bytes) {
  constexpr std::size_t kHeaderSize = 16, kChunkSize = 256;
  constexpr std::array<int, 4> kShBits = {0, 5, 4, 4};
  if (bytes.size() < kHeaderSize) throw std::runtime_error("file is shorter than its header says");
  Reader header(bytes.first(kHeaderSize));
  header.take(8);
  const auto payloadSize = header.get<std::uint64_t>();
  const std::vector<std::byte> payload = lzma_decompress(bytes.subspan(kHeaderSize), payloadSize);
  Reader r(payload);
  const std::size_t n = r.get<std::uint32_t>();
  std::array<float, 3> colorMin{}, colorMax{}, shRange{};
  for (float &v : colorMin) v = r.get<float>();
  for (float &v : colorMax) v = r.get<float>();
  for (float &v : shRange) v = r.get<float>();
  const auto planes = [&](std::size_t count) { return read_planes(r.take(count * 4), count); };

  const std::size_t chunks = (n + kChunkSize - 1) / kChunkSize;
  const std::vector<std::uint32_t> table = planes(chunks * 12);
  const auto bound = [&](std::size_t j, std::size_t field) { return std::bit_cast<float>(table[j / kChunkSize * 12 + field]); };
  const std::vector<std::uint32_t> positions = planes(n);
  const std::vector<std::uint32_t> rotations = planes(n);
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
    const auto q = unpack_rotation_v1(rotations[j]);
    std::ranges::copy(q, cloud.rotations.begin() + static_cast<std::ptrdiff_t>(j * 4));
    for (std::size_t a = 0; a < 3; ++a) {
      cloud.scales[j * 3 + a] = dequantize_unsigned(static_cast<std::uint32_t>(scales[a][j]), bound(j, 6 + a), bound(j, 9 + a), 8);
    }
    for (std::size_t c = 0; c < 3; ++c) {
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
      for (std::size_t c = 0; c < 3; ++c) {
        const auto column = r.take(members.size());
        for (std::size_t m = 0; m < members.size(); ++m) {
          cloud.sh[(members[m] * coeffs + static_cast<std::size_t>(k)) * 3 + c] =
              dequantize_symmetric(static_cast<std::uint32_t>(column[m]), shRange[static_cast<std::size_t>(band - 1)], kShBits[static_cast<std::size_t>(band)]);
        }
      }
    }
  }
  if (r.remaining() != 0) throw std::runtime_error("payload has unexpected trailing data");
  return cloud;
}

} // namespace

} // namespace splat
