#include "train/schedule.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <numbers>

namespace splat {

namespace {

using Complex = std::complex<double>;

// In place, of a power-of-two size, every `stride` elements.
void fft(Complex *values, std::size_t size, std::size_t stride) {
  for (std::size_t i = 1, j = 0; i < size; ++i) {
    std::size_t bit = size >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(values[i * stride], values[j * stride]);
  }
  for (std::size_t length = 2; length <= size; length <<= 1) {
    const Complex step = std::polar(1.0, -2.0 * std::numbers::pi / static_cast<double>(length));
    for (std::size_t start = 0; start < size; start += length) {
      Complex w(1.0);
      for (std::size_t k = 0; k < length / 2; ++k) {
        const Complex a = values[(start + k) * stride], b = values[(start + k + length / 2) * stride] * w;
        values[(start + k) * stride] = a + b;
        values[(start + k + length / 2) * stride] = a - b;
        w *= step;
      }
    }
  }
}

} // namespace

std::vector<double> frequency_energy(const std::vector<std::uint8_t> &rgba, int width, int height, const std::vector<int> &factors) {
  // Mirrored up to a power of two, since a jump at the seam would show up as high frequencies.
  const std::size_t w = std::bit_ceil(static_cast<std::size_t>(width)), h = std::bit_ceil(static_cast<std::size_t>(height));
  const auto mirrored = [](std::size_t at, std::size_t size) { return at < size ? at : 2 * size - 1 - std::min(at, 2 * size - 1); };
  std::vector<Complex> values(w * h);
  for (std::size_t y = 0; y < h; ++y) {
    const std::size_t sy = mirrored(y, static_cast<std::size_t>(height));
    for (std::size_t x = 0; x < w; ++x) {
      const std::uint8_t *p = rgba.data() + (sy * static_cast<std::size_t>(width) + mirrored(x, static_cast<std::size_t>(width))) * 4;
      values[y * w + x] = (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) / 255.0;
    }
  }
  for (std::size_t y = 0; y < h; ++y) fft(values.data() + y * w, w, 1);
  for (std::size_t x = 0; x < w; ++x) fft(values.data() + x, h, w);

  std::vector<double> out(factors.size(), 0.0);
  for (std::size_t y = 0; y < h; ++y) {
    const double fy = static_cast<double>(std::min(y, h - y)) / static_cast<double>(h);
    for (std::size_t x = 0; x < w; ++x) {
      const double fx = static_cast<double>(std::min(x, w - x)) / static_cast<double>(w);
      const double magnitude = std::abs(values[y * w + x]);
      for (std::size_t k = 0; k < factors.size(); ++k) {
        const double limit = 0.5 / factors[k];
        if (fx <= limit && fy <= limit) out[k] += magnitude;
      }
    }
  }
  return out;
}

std::array<double, 2> resolution_switches(double full, double half, double quarter) {
  const double lowest = full / 4.0;
  const auto at = [&](double energy) { return energy > lowest ? std::log(energy / lowest) / std::log(full / lowest) : 0.0; };
  const double untilFull = std::clamp(at(half), 0.0, 1.0);
  return {untilFull, std::clamp(at(quarter), 0.0, untilFull)};
}

std::size_t GrowthBudget::limit(double factor, double progress) const noexcept {
  const double growth = (final_ - initial_) / std::pow(factor, 2.0 - progress);
  return static_cast<std::size_t>(initial_ + growth);
}

void GrowthBudget::observe(std::size_t wanted) noexcept { final_ = std::max(final_, 0.98 * final_ + static_cast<double>(wanted)); }

} // namespace splat
