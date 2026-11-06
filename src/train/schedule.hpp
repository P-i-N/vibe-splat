#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace splat {

// DashGaussian's view of training: fitting ever higher frequencies of the
// photos, so the image size follows how much of a scene's spectrum is low.

// Sum of the magnitudes of the spectrum of an RGBA8 image's brightness, within
// the frequencies an image smaller by each of `factors` can hold.
[[nodiscard]] std::vector<double> frequency_energy(const std::vector<std::uint8_t> &rgba, int width, int height, const std::vector<int> &factors);

// When, as a share of training, to leave quarter size and half size, from the
// mean frequency_energy of the photos at factors 1, 2 and 4: the switch to size
// 1/r comes at ln(X_r / X_m) / ln(X_1 / X_m), with X_m a quarter of X_1. In the
// order of kCoarseUntil in train.cpp: until half size ends, until quarter size ends.
[[nodiscard]] std::array<double, 2> resolution_switches(double full, double half, double quarter);

// How many Gaussians the scene needs and how many it may have yet, growing with
// the image size so that small images are not given more than they can show.
class GrowthBudget {
public:
  // The final count is `factor` times the initial one, raised by what the
  // scene asks for; a factor of 0 or less never caps.
  explicit GrowthBudget(std::size_t initial, double factor = 5.0)
      : initial_(static_cast<double>(initial)),
        final_(factor > 0.0 ? factor * static_cast<double>(initial) : std::numeric_limits<double>::infinity()) {}
  // What the iteration at `progress` of training may have at 1 / factor of the image size.
  [[nodiscard]] std::size_t limit(double factor, double progress) const noexcept;
  // After a densification round, how many Gaussians it asked to add, allowed or not.
  void observe(std::size_t wanted) noexcept;
  [[nodiscard]] double final_count() const noexcept { return final_; }

private:
  double initial_, final_;
};

} // namespace splat
