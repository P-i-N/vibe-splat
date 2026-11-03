#include "train/densify.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

// The Gaussian's rotation as a matrix, identity for a degenerate quaternion.
void rotation_of(const GaussianCloud &cloud, std::size_t i, float R[3][3]) {
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
  const float M[3][3] = {{1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z), 2.f * (x * z + w * y)},
                         {2.f * (x * y + w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x)},
                         {2.f * (x * z - w * y), 2.f * (y * z + w * x), 1.f - 2.f * (x * x + y * y)}};
  std::copy(&M[0][0], &M[0][0] + 9, &R[0][0]);
}

// A sample from the Gaussian itself: rotated standard normal, scaled.
void sample_position(const GaussianCloud &cloud, std::size_t i, std::mt19937 &rng, float out[3]) {
  std::normal_distribution<float> normal(0.f, 1.f);
  const float local[3] = {normal(rng) * std::exp(cloud.scales[i * 3]), normal(rng) * std::exp(cloud.scales[i * 3 + 1]),
                          normal(rng) * std::exp(cloud.scales[i * 3 + 2])};
  float R[3][3];
  rotation_of(cloud, i, R);
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

DensifyResult remove_least_opaque(const GaussianCloud &cloud, double share) {
  std::vector<float> keys(cloud.size());
  for (std::size_t i = 0; i < cloud.size(); ++i) keys[i] = -cloud.opacities[i];
  return remove_highest(cloud, keys, share);
}

DensifyResult densify_cloud(const GaussianCloud &cloud, const DensifyStats &stats, const DensifyOptions &options, std::mt19937 &rng) {
  const std::size_t n = cloud.size();
  if (stats.gradientSum.size() != n || stats.visits.size() != n || stats.maxRadius.size() != n) {
    throw std::runtime_error("densify: statistics do not match the cloud");
  }
  const bool scored = options.scoredViews > 0;
  if ((options.splitThreshold > 0.f && stats.absoluteGradientSum.size() != n) ||
      (scored && (stats.markedPixels.size() != n || stats.markedLoss.size() != n)) ||
      (options.transparentShare < 1.0 && stats.contribution.size() != n)) {
    throw std::runtime_error("densify: statistics do not match the options");
  }

  DensifyResult result;
  enum class Action : std::uint8_t { Keep, Prune, Clone, Split, LongAxis };
  std::vector<Action> actions(n, Action::Keep);
  const auto largest_of = [&](std::size_t i) { return std::exp(std::max({cloud.scales[i * 3], cloud.scales[i * 3 + 1], cloud.scales[i * 3 + 2]})); };

  std::vector<std::size_t> transparent;
  for (std::size_t i = 0; i < n; ++i) {
    const bool oversized = options.pruneLarge && (stats.maxRadius[i] > options.maxScreenRadius || largest_of(i) > options.maxWorldScale);
    if (oversized || !all_finite(cloud, i)) {
      actions[i] = Action::Prune;
      ++result.pruned;
    } else if (sigmoid(cloud.opacities[i]) < options.pruneOpacity) {
      transparent.push_back(i);
    }
  }
  if (options.transparentShare < 1.0) {
    const auto worst_first = [&](std::size_t a, std::size_t b) {
      return scored ? stats.markedLoss[a] > stats.markedLoss[b] : stats.contribution[a] < stats.contribution[b];
    };
    std::stable_sort(transparent.begin(), transparent.end(), worst_first);
    transparent.resize(static_cast<std::size_t>(std::ceil(static_cast<double>(transparent.size()) * options.transparentShare)));
  }
  for (const std::size_t i : transparent) {
    actions[i] = Action::Prune;
    ++result.pruned;
  }

  // How far above its limit each candidate is, as limits differ between the two kinds.
  std::vector<float> priority(n, 0.f);
  std::vector<std::size_t> candidates;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (actions[i] == Action::Prune) continue;
    ++kept;
    if (stats.visits[i] <= 0.f) continue;
    if (scored && stats.markedPixels[i] / static_cast<float>(options.scoredViews) <= options.minMarkedPixels) continue;
    const bool splitsAlone = options.splitThreshold > 0.f && largest_of(i) > options.denseScale;
    const float value = (splitsAlone ? stats.absoluteGradientSum[i] : stats.gradientSum[i]) / stats.visits[i];
    const float limit = splitsAlone ? options.splitThreshold : options.gradientThreshold;
    if (value >= limit) {
      priority[i] = limit > 0.f ? value / limit : value;
      candidates.push_back(i);
    }
  }

  // Each candidate adds one Gaussian, whichever it does: a clone keeps the
  // original and a split replaces it with two.
  const std::size_t room = options.maxGaussians > kept ? options.maxGaussians - kept : 0;
  if (candidates.size() > room) {
    std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) { return priority[a] > priority[b]; });
    result.heldBack = candidates.size() - room;
    candidates.resize(room);
    std::sort(candidates.begin(), candidates.end());
  }
  // With the edge-aware score, a candidate splits with probability
  // score / best, unless no candidate touched an edge at all.
  if (options.edgeSplit) {
    const auto score = [&](std::size_t i) { return i < stats.edgeScore.size() ? stats.edgeScore[i] : 0.f; };
    float best = 0.f;
    for (const std::size_t i : candidates) best = std::max(best, score(i));
    if (best > 0.f) {
      std::uniform_real_distribution<float> draw(0.f, 1.f);
      std::erase_if(candidates, [&](std::size_t i) { return draw(rng) > score(i) / best; });
    }
  }
  for (const std::size_t i : candidates) {
    if (options.edgeSplit) {
      actions[i] = Action::LongAxis;
      ++result.split;
    } else if (largest_of(i) > options.denseScale) {
      actions[i] = Action::Split;
      ++result.split;
    } else {
      actions[i] = Action::Clone;
      ++result.cloned;
    }
  }

  // Exploration, with what room is left. Among the Gaussians nothing asked
  // for, a split goes to one of the larger half.
  std::size_t left = room - candidates.size();
  if (options.exploreSplits > 0 && left > 0) {
    std::vector<std::size_t> large;
    for (std::size_t i = 0; i < n; ++i) {
      if (actions[i] == Action::Keep && largest_of(i) > options.denseScale) large.push_back(i);
    }
    const auto mean_scale = [&](std::size_t i) { return cloud.scales[i * 3] + cloud.scales[i * 3 + 1] + cloud.scales[i * 3 + 2]; };
    const auto middle = large.begin() + static_cast<std::ptrdiff_t>(large.size() / 2);
    std::nth_element(large.begin(), middle, large.end(), [&](std::size_t a, std::size_t b) { return mean_scale(a) > mean_scale(b); });
    large.resize((large.size() + 1) / 2);
    std::shuffle(large.begin(), large.end(), rng);
    for (std::size_t k = 0; k < std::min({options.exploreSplits, large.size(), left}); ++k) {
      actions[large[k]] = Action::Split;
      ++result.split;
    }
    left -= std::min({options.exploreSplits, large.size(), left});
  }

  // Seeds: where to, how large (a typical Gaussian) and which colour (a random one's).
  std::vector<std::size_t> survivors;
  std::array<float, 3> low{}, high{};
  float seedScale = 0.f;
  result.seeded = std::min(options.exploreSeeds, left);
  if (result.seeded > 0) {
    for (std::size_t i = 0; i < n; ++i) {
      if (actions[i] != Action::Prune) survivors.push_back(i);
    }
    if (survivors.empty()) {
      result.seeded = 0;
    } else {
      low.fill(std::numeric_limits<float>::max());
      high.fill(std::numeric_limits<float>::lowest());
      std::vector<float> scales;
      scales.reserve(survivors.size());
      for (const std::size_t i : survivors) {
        for (std::size_t a = 0; a < 3; ++a) {
          low[a] = std::min(low[a], cloud.positions[i * 3 + a]);
          high[a] = std::max(high[a], cloud.positions[i * 3 + a]);
        }
        scales.push_back((cloud.scales[i * 3] + cloud.scales[i * 3 + 1] + cloud.scales[i * 3 + 2]) / 3.f);
      }
      std::nth_element(scales.begin(), scales.begin() + static_cast<std::ptrdiff_t>(scales.size() / 2), scales.end());
      seedScale = scales[scales.size() / 2];
    }
  }

  GaussianCloud &out = result.cloud;
  out.shDegree = cloud.shDegree;
  // Reserved up front: growing the arrays one Gaussian at a time copies them over and over.
  // A clone adds one, and a split replaces one with two.
  const std::size_t total = kept + result.cloned + result.split + result.seeded;
  out.positions.reserve(total * 3);
  out.scales.reserve(total * 3);
  out.rotations.reserve(total * 4);
  out.opacities.reserve(total);
  out.sh.reserve(total * static_cast<std::size_t>(3 * cloud.shCoeffs()));
  result.source.reserve(total);
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
    } else if (actions[i] == Action::LongAxis) {
      // Two children on the longest axis, tangent to the parent at its ends:
      // at 0.45 of its semi-axis from the centre, 0.55 of it long, the other
      // axes shrunk to keep inside, and 0.6 of its opacity each.
      constexpr float kOffset = 0.45f;
      const std::size_t axis = static_cast<std::size_t>(std::max_element(cloud.scales.begin() + static_cast<std::ptrdiff_t>(i * 3),
                                                                         cloud.scales.begin() + static_cast<std::ptrdiff_t>(i * 3 + 3)) -
                                                        (cloud.scales.begin() + static_cast<std::ptrdiff_t>(i * 3)));
      float R[3][3];
      rotation_of(cloud, i, R);
      const float length = std::exp(cloud.scales[i * 3 + axis]);
      const double opacity = 0.6 * sigmoid(cloud.opacities[i]);
      for (const float side : {-1.f, 1.f}) {
        const std::size_t at = out.size();
        append_copy(out, cloud, i);
        for (std::size_t a = 0; a < 3; ++a) {
          out.positions[at * 3 + a] += side * kOffset * length * R[a][axis];
          out.scales[at * 3 + a] += a == axis ? std::log(1.f - kOffset) : 0.5f * std::log(1.f - kOffset * kOffset);
        }
        out.opacities[at] = static_cast<float>(std::log(opacity / (1.0 - opacity)));
        result.source.push_back(-1);
      }
    }
  }
  const auto floats = static_cast<std::size_t>(3 * cloud.shCoeffs());
  for (std::size_t k = 0; k < result.seeded; ++k) {
    const std::size_t at = out.size();
    append_copy(out, cloud, survivors[std::uniform_int_distribution<std::size_t>(0, survivors.size() - 1)(rng)]);
    for (std::size_t a = 0; a < 3; ++a) {
      out.positions[at * 3 + a] = std::uniform_real_distribution<float>(low[a], high[a])(rng);
      out.scales[at * 3 + a] = seedScale;
    }
    std::fill_n(out.rotations.begin() + static_cast<std::ptrdiff_t>(at * 4), 4, 0.f);
    out.rotations[at * 4] = 1.f;
    out.opacities[at] = std::log(0.1f / 0.9f);
    std::fill_n(out.sh.begin() + static_cast<std::ptrdiff_t>(at * floats + 3), floats - 3, 0.f); // Same colour from all sides
    result.source.push_back(-1);
  }
  return result;
}

DensifyResult relocate_cloud(const GaussianCloud &cloud, const RelocateOptions &options, std::mt19937 &rng) {
  const std::size_t n = cloud.size();
  DensifyResult result;
  result.cloud = cloud;
  result.source.resize(n);
  std::iota(result.source.begin(), result.source.end(), std::int64_t{0});
  std::vector<std::size_t> alive, dead;
  for (std::size_t i = 0; i < n; ++i) {
    (sigmoid(cloud.opacities[i]) <= options.minOpacity || !all_finite(cloud, i) ? dead : alive).push_back(i);
  }
  if (alive.empty()) return result;
  GaussianCloud &out = result.cloud;
  const auto floats = static_cast<std::size_t>(3 * cloud.shCoeffs());

  // Turns each drawn target into 1 + (times drawn) copies of equal weight.
  constexpr int kMaxCopies = 51; // Beyond this the alternating sum below is numerically useless (gsplat's limit)
  const auto update_targets = [&](const std::vector<std::size_t> &targets) {
    std::vector<int> times(out.size(), 0);
    for (const std::size_t t : targets) ++times[t];
    for (std::size_t t = 0; t < out.size(); ++t) {
      if (times[t] == 0) continue;
      const int copies = std::min(times[t] + 1, kMaxCopies);
      const double o = sigmoid(out.opacities[t]);
      const double fresh = 1.0 - std::pow(1.0 - o, 1.0 / copies);
      double divisor = 0.0;
      for (int i = 1; i <= copies; ++i) {
        double binom = 1.0; // C(i - 1, k), updated as k grows
        for (int k = 0; k < i; ++k) {
          if (k > 0) binom = binom * static_cast<double>(i - k) / k;
          divisor += binom * (k % 2 == 0 ? 1.0 : -1.0) * std::pow(fresh, k + 1) / std::sqrt(static_cast<double>(k + 1));
        }
      }
      const double clamped = std::clamp(fresh, static_cast<double>(options.minOpacity), 1.0 - 1e-6);
      out.opacities[t] = static_cast<float>(std::log(clamped / (1.0 - clamped)));
      const double factor = divisor > 0.0 ? o / divisor : 1.0;
      for (std::size_t a = 0; a < 3; ++a) out.scales[t * 3 + a] += static_cast<float>(std::log(factor));
      result.source[t] = -1;
    }
  };
  const auto copy_into = [&](std::size_t from, std::size_t to) {
    std::copy_n(out.positions.begin() + static_cast<std::ptrdiff_t>(from * 3), 3, out.positions.begin() + static_cast<std::ptrdiff_t>(to * 3));
    std::copy_n(out.scales.begin() + static_cast<std::ptrdiff_t>(from * 3), 3, out.scales.begin() + static_cast<std::ptrdiff_t>(to * 3));
    std::copy_n(out.rotations.begin() + static_cast<std::ptrdiff_t>(from * 4), 4, out.rotations.begin() + static_cast<std::ptrdiff_t>(to * 4));
    out.opacities[to] = out.opacities[from];
    std::copy_n(out.sh.begin() + static_cast<std::ptrdiff_t>(from * floats), floats, out.sh.begin() + static_cast<std::ptrdiff_t>(to * floats));
  };
  const auto draw = [&](const std::vector<std::size_t> &from, std::size_t count) {
    std::vector<double> weights;
    weights.reserve(from.size());
    for (const std::size_t i : from) weights.push_back(sigmoid(out.opacities[i]));
    std::discrete_distribution<std::size_t> pick(weights.begin(), weights.end());
    std::vector<std::size_t> targets(count);
    for (std::size_t &t : targets) t = from[pick(rng)];
    return targets;
  };

  // The dead move onto live Gaussians.
  if (!dead.empty()) {
    const std::vector<std::size_t> targets = draw(alive, dead.size());
    update_targets(targets);
    for (std::size_t j = 0; j < dead.size(); ++j) {
      copy_into(targets[j], dead[j]);
      result.source[dead[j]] = -1;
    }
    result.cloned = dead.size();
  }

  // Growth, by the same draw over everything now alive.
  const std::size_t room = options.maxGaussians > n ? options.maxGaussians - n : 0;
  const auto wanted = std::min(static_cast<std::size_t>(options.growth * static_cast<double>(n)), room);
  if (wanted > 0) {
    std::vector<std::size_t> everyone(n);
    std::iota(everyone.begin(), everyone.end(), std::size_t{0});
    const std::vector<std::size_t> targets = draw(everyone, wanted);
    update_targets(targets);
    out.positions.reserve((n + wanted) * 3);
    out.scales.reserve((n + wanted) * 3);
    out.rotations.reserve((n + wanted) * 4);
    out.opacities.reserve(n + wanted);
    out.sh.reserve((n + wanted) * floats);
    for (const std::size_t t : targets) {
      const std::size_t at = out.size();
      out.positions.resize(at * 3 + 3);
      out.scales.resize(at * 3 + 3);
      out.rotations.resize(at * 4 + 4);
      out.opacities.resize(at + 1);
      out.sh.resize((at + 1) * floats);
      copy_into(t, at);
      result.source.push_back(-1);
    }
    result.seeded = wanted;
  }
  return result;
}

} // namespace splat
