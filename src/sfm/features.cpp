#include "sfm/features.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>

namespace splat {

namespace {

constexpr int kScalesPerOctave = 3;
constexpr float kBaseSigma = 1.6f;
// A sixth of Lowe's 0.03 (per scale step, on a [0, 1] image). Only the
// strongest extrema are kept, so on a busy image this changes nothing; it
// matters where there is little contrast to begin with, such as a close-up of a
// smooth, glossy object against a blurred background.
constexpr float kContrastThreshold = 0.005f / kScalesPerOctave;
constexpr float kEdgeRatio = 10.f;
constexpr int kBorder = 5;
constexpr int kOrientationBins = 36;
constexpr int kDescriptorGrid = 4;
constexpr int kDescriptorBins = 8;
constexpr float kTwoPi = 2.f * std::numbers::pi_v<float>;

// atan2 to within 2e-4 rad (a minimax polynomial on [0, 1] plus octant
// folding). Histogram bins are 0.17 rad wide at the finest, so this is plenty,
// and it is several times faster than the library call.
float fast_atan2(float y, float x) {
  const float ax = std::abs(x), ay = std::abs(y);
  const float a = std::min(ax, ay) / (std::max(ax, ay) + 1e-30f);
  const float s = a * a;
  float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
  if (ay > ax) r = 0.5f * std::numbers::pi_v<float> - r;
  if (x < 0.f) r = std::numbers::pi_v<float> - r;
  return y < 0.f ? -r : r;
}

// exp(-x) for x >= 0 to about 1e-4 relative: 2^-t split into integer and
// fraction, the fraction by a quartic. Used for the Gaussian weights of samples.
float fast_exp_neg(float x) {
  const float t = std::min(x * std::numbers::log2e_v<float>, 126.f);
  const float whole = std::floor(t), f = t - whole; // 2^-t = 2^-(whole + f)
  const float p = 0.9999961f + f * (-0.6930294f + f * (0.2393854f + f * (-0.0531868f + f * 0.0068383f))); // 2^-f, fitted
  float scale;
  const auto bits = static_cast<std::uint32_t>((127 - static_cast<int>(whole)) << 23);
  std::memcpy(&scale, &bits, sizeof scale);
  return p * scale;
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<float> values;

  Plane() = default;
  Plane(int w, int h) : width(w), height(h), values(static_cast<std::size_t>(w) * h) {}
  [[nodiscard]] float at(int x, int y) const { return values[static_cast<std::size_t>(y) * width + x]; }
  [[nodiscard]] float &at(int x, int y) { return values[static_cast<std::size_t>(y) * width + x]; }
};

// Separable Gaussian blur, edges clamped. Both passes are written as a sum of
// scaled, shifted rows so the compiler vectorises them: the horizontal pass
// reads from a row padded with the edge values, so the inner loop has no
// bounds checks.
Plane blur(const Plane &source, float sigma) {
  const int radius = std::max(1, static_cast<int>(std::ceil(3.f * sigma)));
  std::vector<float> kernel(static_cast<std::size_t>(radius) * 2 + 1);
  float total = 0.f;
  for (int i = -radius; i <= radius; ++i) {
    const float value = std::exp(-0.5f * static_cast<float>(i * i) / (sigma * sigma));
    kernel[static_cast<std::size_t>(i + radius)] = value;
    total += value;
  }
  for (float &k : kernel) k /= total;

  const int w = source.width, h = source.height;
  Plane horizontal(w, h);
  std::vector<float> padded(static_cast<std::size_t>(w + 2 * radius));
  for (int y = 0; y < h; ++y) {
    const float *in = source.values.data() + static_cast<std::size_t>(y) * w;
    std::copy_n(in, w, padded.begin() + radius);
    std::fill_n(padded.begin(), radius, in[0]);
    std::fill_n(padded.begin() + radius + w, radius, in[w - 1]);
    float *out = horizontal.values.data() + static_cast<std::size_t>(y) * w;
    std::fill_n(out, w, 0.f);
    for (int i = 0; i < 2 * radius + 1; ++i) {
      const float k = kernel[static_cast<std::size_t>(i)];
      const float *shifted = padded.data() + i;
      for (int x = 0; x < w; ++x) out[x] += k * shifted[x];
    }
  }
  Plane out(w, h);
  for (int y = 0; y < h; ++y) {
    float *column = out.values.data() + static_cast<std::size_t>(y) * w;
    for (int i = -radius; i <= radius; ++i) {
      const float k = kernel[static_cast<std::size_t>(i + radius)];
      const float *row = horizontal.values.data() + static_cast<std::size_t>(std::clamp(y + i, 0, h - 1)) * w;
      for (int x = 0; x < w; ++x) column[x] += k * row[x];
    }
  }
  return out;
}

Plane half_size(const Plane &source) {
  Plane out(source.width / 2, source.height / 2);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) out.at(x, y) = source.at(x * 2, y * 2);
  }
  return out;
}

struct Octave {
  std::vector<Plane> gaussians;
  std::vector<Plane> dogs;
};

struct Extremum {
  int octave;
  int scale;      // Index of the nearest Gaussian level
  float x;        // In octave pixels
  float y;
  float sigma;    // In octave pixels
  float response;
};

// Fits a quadratic around a DoG extremum (Lowe section 4). False if it is
// unstable, low in contrast or on an edge.
bool refine(const Octave &octave, int &x, int &y, int &s, float &offsetX, float &offsetY, float &offsetS, float &response) {
  for (int iteration = 0; iteration < 5; ++iteration) {
    const Plane &d0 = octave.dogs[static_cast<std::size_t>(s - 1)];
    const Plane &d1 = octave.dogs[static_cast<std::size_t>(s)];
    const Plane &d2 = octave.dogs[static_cast<std::size_t>(s + 1)];
    const float v = d1.at(x, y);
    const Eigen::Vector3f gradient(0.5f * (d1.at(x + 1, y) - d1.at(x - 1, y)), 0.5f * (d1.at(x, y + 1) - d1.at(x, y - 1)),
                                   0.5f * (d2.at(x, y) - d0.at(x, y)));
    const float dxx = d1.at(x + 1, y) + d1.at(x - 1, y) - 2.f * v;
    const float dyy = d1.at(x, y + 1) + d1.at(x, y - 1) - 2.f * v;
    const float dss = d2.at(x, y) + d0.at(x, y) - 2.f * v;
    const float dxy = 0.25f * (d1.at(x + 1, y + 1) - d1.at(x - 1, y + 1) - d1.at(x + 1, y - 1) + d1.at(x - 1, y - 1));
    const float dxs = 0.25f * (d2.at(x + 1, y) - d2.at(x - 1, y) - d0.at(x + 1, y) + d0.at(x - 1, y));
    const float dys = 0.25f * (d2.at(x, y + 1) - d2.at(x, y - 1) - d0.at(x, y + 1) + d0.at(x, y - 1));
    Eigen::Matrix3f hessian;
    hessian << dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss;
    const Eigen::Vector3f offset = -hessian.partialPivLu().solve(gradient);
    if (!offset.allFinite()) return false;

    if (offset.cwiseAbs().maxCoeff() < 0.5f) {
      offsetX = offset.x();
      offsetY = offset.y();
      offsetS = offset.z();
      response = v + 0.5f * gradient.dot(offset);
      if (std::abs(response) < kContrastThreshold) return false;
      const float trace = dxx + dyy, det = dxx * dyy - dxy * dxy;
      return det > 0.f && trace * trace * kEdgeRatio < (kEdgeRatio + 1.f) * (kEdgeRatio + 1.f) * det;
    }
    x += static_cast<int>(std::lround(offset.x()));
    y += static_cast<int>(std::lround(offset.y()));
    s += static_cast<int>(std::lround(offset.z()));
    if (s < 1 || s > kScalesPerOctave || x < kBorder || y < kBorder || x >= d1.width - kBorder || y >= d1.height - kBorder) {
      return false;
    }
  }
  return false;
}

std::vector<float> orientations(const Plane &g, float px, float py, float sigma) {
  const int cx = static_cast<int>(std::lround(px)), cy = static_cast<int>(std::lround(py));
  const float weightSigma = 1.5f * sigma;
  const int radius = static_cast<int>(std::lround(3.f * weightSigma));
  float histogram[kOrientationBins]{};
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = cx + dx, y = cy + dy;
      if (x < 1 || y < 1 || x >= g.width - 1 || y >= g.height - 1) continue;
      const float gx = g.at(x + 1, y) - g.at(x - 1, y), gy = g.at(x, y + 1) - g.at(x, y - 1);
      const float weight = fast_exp_neg(0.5f * static_cast<float>(dx * dx + dy * dy) / (weightSigma * weightSigma));
      float angle = fast_atan2(gy, gx);
      if (angle < 0.f) angle += kTwoPi;
      const int bin = std::min(kOrientationBins - 1, static_cast<int>(angle / kTwoPi * kOrientationBins));
      histogram[bin] += weight * std::sqrt(gx * gx + gy * gy);
    }
  }
  float smooth[kOrientationBins];
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < kOrientationBins; ++i) {
      smooth[i] = (histogram[(i + kOrientationBins - 1) % kOrientationBins] + 2.f * histogram[i] +
                   histogram[(i + 1) % kOrientationBins]) * 0.25f;
    }
    std::copy(std::begin(smooth), std::end(smooth), std::begin(histogram));
  }
  const float peak = *std::max_element(std::begin(histogram), std::end(histogram));
  std::vector<float> angles;
  for (int i = 0; i < kOrientationBins; ++i) {
    const float left = histogram[(i + kOrientationBins - 1) % kOrientationBins];
    const float right = histogram[(i + 1) % kOrientationBins];
    if (histogram[i] >= 0.8f * peak && histogram[i] > left && histogram[i] > right) {
      const float offset = 0.5f * (left - right) / (left - 2.f * histogram[i] + right);
      float angle = (static_cast<float>(i) + 0.5f + offset) / kOrientationBins * kTwoPi;
      if (angle < 0.f) angle += kTwoPi;
      if (angle >= kTwoPi) angle -= kTwoPi;
      angles.push_back(angle);
    }
  }
  return angles;
}

void describe(const Plane &g, float px, float py, float sigma, float angle, float *out) {
  std::fill_n(out, kDescriptorSize, 0.f);
  const float cosA = std::cos(angle), sinA = std::sin(angle);
  const float cell = 3.f * sigma;
  const int radius = static_cast<int>(std::lround(cell * std::numbers::sqrt2_v<float> * (kDescriptorGrid + 1) * 0.5f));
  const int cx = static_cast<int>(std::lround(px)), cy = static_cast<int>(std::lround(py));
  const float fracX = px - static_cast<float>(cx), fracY = py - static_cast<float>(cy);
  float histogram[kDescriptorGrid][kDescriptorGrid][kDescriptorBins]{};

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = cx + dx, y = cy + dy;
      if (x < 1 || y < 1 || x >= g.width - 1 || y >= g.height - 1) continue;
      const float rx = static_cast<float>(dx) - fracX, ry = static_cast<float>(dy) - fracY;
      // In descriptor cells, relative to the keypoint's own orientation.
      const float u = (cosA * rx + sinA * ry) / cell, v = (-sinA * rx + cosA * ry) / cell;
      const float col = u + 0.5f * kDescriptorGrid - 0.5f, row = v + 0.5f * kDescriptorGrid - 0.5f;
      if (col <= -1.f || row <= -1.f || col >= kDescriptorGrid || row >= kDescriptorGrid) continue;

      const float gx = g.at(x + 1, y) - g.at(x - 1, y), gy = g.at(x, y + 1) - g.at(x, y - 1);
      float relative = fast_atan2(gy, gx) - angle;
      while (relative < 0.f) relative += kTwoPi;
      while (relative >= kTwoPi) relative -= kTwoPi;
      const float bin = relative / kTwoPi * kDescriptorBins;
      const float weightSigma = 0.5f * kDescriptorGrid;
      const float magnitude = std::sqrt(gx * gx + gy * gy) * fast_exp_neg(0.5f * (u * u + v * v) / (weightSigma * weightSigma));

      // Share each sample between the 8 nearest histogram cells.
      const int r0 = static_cast<int>(std::floor(row)), c0 = static_cast<int>(std::floor(col));
      const int b0 = static_cast<int>(std::floor(bin));
      const float dr = row - static_cast<float>(r0), dc = col - static_cast<float>(c0), db = bin - static_cast<float>(b0);
      for (int ir = 0; ir < 2; ++ir) {
        const int r = r0 + ir;
        if (r < 0 || r >= kDescriptorGrid) continue;
        const float wr = ir == 0 ? 1.f - dr : dr;
        for (int ic = 0; ic < 2; ++ic) {
          const int c = c0 + ic;
          if (c < 0 || c >= kDescriptorGrid) continue;
          const float wc = ic == 0 ? 1.f - dc : dc;
          for (int ib = 0; ib < 2; ++ib) {
            const float wb = ib == 0 ? 1.f - db : db;
            histogram[r][c][(b0 + ib) % kDescriptorBins] += magnitude * wr * wc * wb;
          }
        }
      }
    }
  }

  float *flat = &histogram[0][0][0];
  float norm = 0.f;
  for (int i = 0; i < kDescriptorSize; ++i) norm += flat[i] * flat[i];
  norm = std::sqrt(norm) + 1e-12f;
  float sum = 0.f;
  for (int i = 0; i < kDescriptorSize; ++i) {
    flat[i] = std::min(flat[i] / norm, 0.2f); // Limits the influence of lighting
    sum += flat[i];
  }
  // RootSIFT (Arandjelovic 2012): L2 distance then behaves like the Hellinger
  // distance, which compares histograms better.
  sum += 1e-12f;
  for (int i = 0; i < kDescriptorSize; ++i) out[i] = std::sqrt(flat[i] / sum);
}

} // namespace

ImageFeatures extract_features(const RgbImage &image, int maxFeatures) {
  Plane base(image.width, image.height);
  for (std::size_t i = 0; i < base.values.size(); ++i) {
    const std::uint8_t *p = image.pixels.data() + i * 3;
    base.values[i] = (0.299f * static_cast<float>(p[0]) + 0.587f * static_cast<float>(p[1]) + 0.114f * static_cast<float>(p[2])) / 255.f;
  }
  base = blur(base, std::sqrt(kBaseSigma * kBaseSigma - 0.25f)); // The camera has already blurred by about 0.5

  const float stepRatio = std::pow(2.f, 1.f / kScalesPerOctave);
  std::vector<Octave> octaves;
  std::vector<Extremum> extrema;
  for (int o = 0; std::min(base.width, base.height) >= 32; ++o) {
    Octave octave;
    octave.gaussians.push_back(base);
    float sigma = kBaseSigma;
    for (int i = 1; i < kScalesPerOctave + 3; ++i) {
      const float next = sigma * stepRatio;
      octave.gaussians.push_back(blur(octave.gaussians.back(), std::sqrt(next * next - sigma * sigma)));
      sigma = next;
    }
    for (std::size_t i = 0; i + 1 < octave.gaussians.size(); ++i) {
      Plane dog(base.width, base.height);
      for (std::size_t p = 0; p < dog.values.size(); ++p) {
        dog.values[p] = octave.gaussians[i + 1].values[p] - octave.gaussians[i].values[p];
      }
      octave.dogs.push_back(std::move(dog));
    }

    for (int s = 1; s <= kScalesPerOctave; ++s) {
      const Plane &d0 = octave.dogs[static_cast<std::size_t>(s - 1)];
      const Plane &d1 = octave.dogs[static_cast<std::size_t>(s)];
      const Plane &d2 = octave.dogs[static_cast<std::size_t>(s + 1)];
      for (int y = kBorder; y < base.height - kBorder; ++y) {
        for (int x = kBorder; x < base.width - kBorder; ++x) {
          const float v = d1.at(x, y);
          if (std::abs(v) < 0.5f * kContrastThreshold) continue;
          bool extreme = true;
          for (int dz = 0; dz < 3 && extreme; ++dz) {
            const Plane &d = dz == 0 ? d0 : dz == 1 ? d1 : d2;
            for (int dy = -1; dy <= 1 && extreme; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 1 && dx == 0 && dy == 0) continue;
                const float other = d.at(x + dx, y + dy);
                if (v > 0.f ? other >= v : other <= v) {
                  extreme = false;
                  break;
                }
              }
            }
          }
          if (!extreme) continue;

          int rx = x, ry = y, rs = s;
          float ox = 0.f, oy = 0.f, os = 0.f, response = 0.f;
          if (!refine(octave, rx, ry, rs, ox, oy, os, response)) continue;
          const float level = static_cast<float>(rs) + os;
          extrema.push_back({o, std::clamp(static_cast<int>(std::lround(level)), 0, kScalesPerOctave + 2),
                             static_cast<float>(rx) + ox, static_cast<float>(ry) + oy,
                             kBaseSigma * std::pow(2.f, level / kScalesPerOctave), std::abs(response)});
        }
      }
    }
    octaves.push_back(std::move(octave));
    base = half_size(octaves.back().gaussians[kScalesPerOctave]);
  }

  std::sort(extrema.begin(), extrema.end(), [](const Extremum &a, const Extremum &b) { return a.response > b.response; });

  ImageFeatures features;
  const auto limit = static_cast<std::size_t>(std::max(maxFeatures, 1));
  for (const Extremum &e : extrema) {
    if (features.keypoints.size() >= limit) break;
    const Plane &g = octaves[static_cast<std::size_t>(e.octave)].gaussians[static_cast<std::size_t>(e.scale)];
    const float factor = static_cast<float>(1 << e.octave);
    for (const float angle : orientations(g, e.x, e.y, e.sigma)) {
      if (features.keypoints.size() >= limit) break;
      Keypoint kp;
      kp.x = e.x * factor + 0.5f;
      kp.y = e.y * factor + 0.5f;
      kp.scale = e.sigma * factor;
      kp.angle = angle;
      const int ix = std::clamp(static_cast<int>(kp.x), 0, image.width - 1);
      const int iy = std::clamp(static_cast<int>(kp.y), 0, image.height - 1);
      std::copy_n(image.pixels.data() + (static_cast<std::size_t>(iy) * image.width + ix) * 3, 3, kp.color.data());
      features.keypoints.push_back(kp);
      features.descriptors.resize(features.descriptors.size() + kDescriptorSize);
      describe(g, e.x, e.y, e.sigma, angle, features.descriptors.data() + features.descriptors.size() - kDescriptorSize);
    }
  }
  return features;
}

std::vector<std::pair<int, int>> match_features(const ImageFeatures &a, const ImageFeatures &b, float ratio) {
  const int na = static_cast<int>(a.keypoints.size()), nb = static_cast<int>(b.keypoints.size());
  std::vector<std::pair<int, int>> matches;
  if (na == 0 || nb < 2) return matches;

  // For unit vectors, the nearest neighbour has the largest dot product. All
  // dot products of a block of A's rows against B come from one matrix product,
  // which Eigen vectorises and blocks for the cache; the block keeps the score
  // matrix small.
  using Descriptors = Eigen::Matrix<float, Eigen::Dynamic, kDescriptorSize, Eigen::RowMajor>;
  using Scores = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const Descriptors> da(a.descriptors.data(), na, kDescriptorSize);
  const Eigen::Map<const Descriptors> db(b.descriptors.data(), nb, kDescriptorSize);
  constexpr int kBlockRows = 128;
  Scores scores(std::min(kBlockRows, na), nb);

  std::vector<int> bestOfA(static_cast<std::size_t>(na), -1);
  std::vector<float> bestDotOfB(static_cast<std::size_t>(nb), -2.f);
  std::vector<int> bestOfB(static_cast<std::size_t>(nb), -1);
  const float ratioSquared = ratio * ratio;
  for (int start = 0; start < na; start += kBlockRows) {
    const int rows = std::min(kBlockRows, na - start);
    scores.topRows(rows).noalias() = da.middleRows(start, rows) * db.transpose();
    for (int r = 0; r < rows; ++r) {
      const int i = start + r;
      const float *row = scores.data() + static_cast<std::ptrdiff_t>(r) * nb;
      float first = -2.f, second = -2.f;
      int firstIndex = -1;
      for (int j = 0; j < nb; ++j) {
        const float dot = row[j];
        if (dot > first) {
          second = first;
          first = dot;
          firstIndex = j;
        } else if (dot > second) {
          second = dot;
        }
        if (dot > bestDotOfB[static_cast<std::size_t>(j)]) {
          bestDotOfB[static_cast<std::size_t>(j)] = dot;
          bestOfB[static_cast<std::size_t>(j)] = i;
        }
      }
      // Squared distance is 2 - 2 dot.
      if (2.f - 2.f * first < ratioSquared * (2.f - 2.f * second)) bestOfA[static_cast<std::size_t>(i)] = firstIndex;
    }
  }
  for (int i = 0; i < na; ++i) {
    const int j = bestOfA[static_cast<std::size_t>(i)];
    if (j >= 0 && bestOfB[static_cast<std::size_t>(j)] == i) matches.emplace_back(i, j);
  }
  return matches;
}

} // namespace splat
