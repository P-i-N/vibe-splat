#include "train/densify.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>

namespace splat {

namespace {

constexpr float kSplitShrink = 1.6f; // From the paper

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

void append_copy(GaussianCloud &to, const GaussianCloud &from, std::size_t i) {
  const auto floats = static_cast<std::size_t>(3 * from.shCoeffs());
  to.positions.insert(to.positions.end(), from.positions.begin() + static_cast<std::ptrdiff_t>(i * 3),
                      from.positions.begin() + static_cast<std::ptrdiff_t>(i * 3 + 3));
  to.scales.insert(to.scales.end(), from.scales.begin() + static_cast<std::ptrdiff_t>(i * 3),
                   from.scales.begin() + static_cast<std::ptrdiff_t>(i * 3 + 3));
  to.rotations.insert(to.rotations.end(), from.rotations.begin() + static_cast<std::ptrdiff_t>(i * 4),
                      from.rotations.begin() + static_cast<std::ptrdiff_t>(i * 4 + 4));
  to.opacities.push_back(from.opacities[i]);
  to.sh.insert(to.sh.end(), from.sh.begin() + static_cast<std::ptrdiff_t>(i * floats),
               from.sh.begin() + static_cast<std::ptrdiff_t>((i + 1) * floats));
}

// NaN compares false with every limit, so without this such a Gaussian would stay for good.
bool all_finite(const GaussianCloud &cloud, std::size_t i) {
  const auto floats = static_cast<std::size_t>(3 * cloud.shCoeffs());
  const auto finite = [](std::span<const float> values) { return std::ranges::all_of(values, [](float v) { return std::isfinite(v); }); };
  return finite(std::span(cloud.positions).subspan(i * 3, 3)) && finite(std::span(cloud.scales).subspan(i * 3, 3)) &&
         finite(std::span(cloud.rotations).subspan(i * 4, 4)) && std::isfinite(cloud.opacities[i]) && finite(std::span(cloud.sh).subspan(i * floats, floats));
}

// A sample from the Gaussian itself: rotated standard normal, scaled.
void sample_position(const GaussianCloud &cloud, std::size_t i, std::mt19937 &rng, float out[3]) {
  std::normal_distribution<float> normal(0.f, 1.f);
  const float local[3] = {normal(rng) * std::exp(cloud.scales[i * 3]), normal(rng) * std::exp(cloud.scales[i * 3 + 1]),
                          normal(rng) * std::exp(cloud.scales[i * 3 + 2])};
  float w = cloud.rotations[i * 4], x = cloud.rotations[i * 4 + 1], y = cloud.rotations[i * 4 + 2], z = cloud.rotations[i * 4 + 3];
  const float length = std::sqrt(w * w + x * x + y * y + z * z);
  if (length > 0.f) {
    w /= length;
    x /= length;
    y /= length;
    z /= length;
  } else {
    w = 1.f;
    x = y = z = 0.f;
  }
  const float R[3][3] = {{1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y)},
                         {2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x)},
                         {2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)}};
  for (std::size_t r = 0; r < 3; ++r) {
    out[r] = cloud.positions[i * 3 + r] + R[r][0] * local[0] + R[r][1] * local[1] + R[r][2] * local[2];
  }
}

// Removes the given share of Gaussians, those with the highest keys first,
// always keeping at least one.
DensifyResult remove_highest(const GaussianCloud &cloud, const std::vector<float> &keys, double share) {
  const std::size_t n = cloud.size();
  const auto removed = std::min(n - 1, static_cast<std::size_t>(std::ceil(static_cast<double>(n) * share)));
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] > keys[b]; });
  std::vector<char> gone(n, 0);
  for (std::size_t i = 0; i < removed; ++i) gone[order[i]] = 1;

  DensifyResult result;
  result.cloud.shDegree = cloud.shDegree;
  result.pruned = removed;
  for (std::size_t i = 0; i < n; ++i) {
    if (gone[i]) continue;
    append_copy(result.cloud, cloud, i);
    result.source.push_back(static_cast<std::int64_t>(i));
  }
  return result;
}

} // namespace

DensifyResult remove_largest(const GaussianCloud &cloud, double share) {
  const std::size_t n = cloud.size();
  std::vector<float> sizes(n);
  for (std::size_t i = 0; i < n; ++i) sizes[i] = std::max({cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]});
  return remove_highest(cloud, sizes, share);
}

DensifyResult remove_least_contributing(const GaussianCloud &cloud, const std::vector<float> &contribution, double share) {
  if (contribution.size() != cloud.size()) throw std::runtime_error("prune: contributions do not match the cloud");
  std::vector<float> keys(contribution.size());
  for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = -contribution[i];
  return remove_highest(cloud, keys, share);
}

DensifyResult densify_cloud(const GaussianCloud &cloud, const DensifyStats &stats, const DensifyOptions &options, std::mt19937 &rng) {
  const std::size_t n = cloud.size();
  if (stats.gradientSum.size() != n || stats.visits.size() != n || stats.maxRadius.size() != n) {
    throw std::runtime_error("densify: statistics do not match the cloud");
  }

  DensifyResult result;
  enum class Action : std::uint8_t { Keep, Prune, Clone, Split };
  std::vector<Action> actions(n, Action::Keep);
  std::vector<float> average(n, 0.f);
  std::vector<std::size_t> candidates;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float largest = std::exp(std::max({cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]}));
    const bool transparent = sigmoid(cloud.opacities[i]) < options.pruneOpacity;
    const bool oversized = options.pruneLarge && (stats.maxRadius[i] > options.maxScreenRadius || largest > options.maxWorldScale);
    if (transparent || oversized || !all_finite(cloud, i)) {
      actions[i] = Action::Prune;
      ++result.pruned;
      continue;
    }
    ++kept;
    average[i] = stats.visits[i] > 0.f ? stats.gradientSum[i] / stats.visits[i] : 0.f;
    if (average[i] >= options.gradientThreshold) candidates.push_back(i);
  }

  // Each candidate adds one Gaussian, whichever it does: a clone keeps the
  // original and a split replaces it with two.
  const std::size_t room = options.maxGaussians > kept ? options.maxGaussians - kept : 0;
  if (candidates.size() > room) {
    std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) { return average[a] > average[b]; });
    result.heldBack = candidates.size() - room;
    candidates.resize(room);
    std::sort(candidates.begin(), candidates.end());
  }
  for (const std::size_t i : candidates) {
    const float largest = std::exp(std::max({cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]}));
    if (largest > options.denseScale) {
      actions[i] = Action::Split;
      ++result.split;
    } else {
      actions[i] = Action::Clone;
      ++result.cloned;
    }
  }

  GaussianCloud &out = result.cloud;
  out.shDegree = cloud.shDegree;
  // Survivors first, in their order; then what is new.
  for (std::size_t i = 0; i < n; ++i) {
    if (actions[i] == Action::Keep || actions[i] == Action::Clone) {
      append_copy(out, cloud, i);
      result.source.push_back(static_cast<std::int64_t>(i));
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (actions[i] == Action::Clone) {
      append_copy(out, cloud, i);
      result.source.push_back(-1);
    } else if (actions[i] == Action::Split) {
      for (int child = 0; child < 2; ++child) {
        const std::size_t at = out.size();
        append_copy(out, cloud, i);
        float position[3];
        sample_position(cloud, i, rng, position);
        for (std::size_t a = 0; a < 3; ++a) {
          out.positions[at * 3 + a] = position[a];
          out.scales[at * 3 + a] -= std::log(kSplitShrink);
        }
        result.source.push_back(-1);
      }
    }
  }
  return result;
}

} // namespace splat
