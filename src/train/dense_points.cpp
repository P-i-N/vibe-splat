#include "train/dense_points.hpp"

#include "core/parallel.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <random>

namespace splat {

namespace {

constexpr int kMaxSide = 512;       // Matching works on images of about this size
constexpr int kGrid = 3;            // A pixel in every this many, each way
constexpr int kPatch = 3;           // Half the size of the patches
constexpr float kMinContrast = 8.f; // Standard deviation of a patch, out of 255
constexpr float kMinCorrelation = 0.9f;
constexpr float kMinThirdView = 0.8f;
constexpr float kMinLead = 0.03f;   // How far above the runner-up elsewhere on the line
constexpr int kMaxSteps = 384;
constexpr float kBaseline = 0.15f; // Neighbours as far from the camera as this share of the depth: nearer is imprecise, farther looks different
constexpr std::size_t kNeighbours = 2;
constexpr std::size_t kMaxReferences = 180; // As EDGS

struct Gray {
  int width = 0, height = 0;
  std::vector<float> pixels;

  float at(int x, int y) const { return pixels[static_cast<std::size_t>(y) * width + x]; }
  float sample(float x, float y) const {
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    const float u = x - static_cast<float>(x0), v = y - static_cast<float>(y0);
    return (at(x0, y0) * (1.f - u) + at(x0 + 1, y0) * u) * (1.f - v) + (at(x0, y0 + 1) * (1.f - u) + at(x0 + 1, y0 + 1) * u) * v;
  }
  bool holds(float x, float y) const {
    return x >= static_cast<float>(kPatch) && y >= static_cast<float>(kPatch) && x < static_cast<float>(width - kPatch - 1) && y < static_cast<float>(height - kPatch - 1);
  }
};

struct Prepared {
  Camera camera; // For the small image
  Gray gray;
  glm::vec3 forward{};
};

Prepared prepare_view(const DensePointsView &view) {
  const int factor = std::max(1, (std::max(view.camera.width, view.camera.height) + kMaxSide - 1) / kMaxSide);
  Prepared out;
  out.camera = view.camera;
  out.gray.width = view.camera.width / factor;
  out.gray.height = view.camera.height / factor;
  out.camera.width = out.gray.width;
  out.camera.height = out.gray.height;
  const float scale = 1.f / static_cast<float>(factor);
  out.camera.fx *= scale;
  out.camera.fy *= scale;
  out.camera.cx *= scale;
  out.camera.cy *= scale;
  out.gray.pixels.resize(static_cast<std::size_t>(out.gray.width) * out.gray.height);
  for (int y = 0; y < out.gray.height; ++y) {
    for (int x = 0; x < out.gray.width; ++x) {
      float sum = 0.f;
      for (int v = 0; v < factor; ++v) {
        for (int u = 0; u < factor; ++u) {
          const std::uint8_t *p = view.rgba.data() + (static_cast<std::size_t>(y * factor + v) * view.camera.width + static_cast<std::size_t>(x * factor + u)) * 4;
          sum += 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        }
      }
      out.gray.pixels[static_cast<std::size_t>(y) * out.gray.width + x] = sum / static_cast<float>(factor * factor);
    }
  }
  const glm::mat3 R(view.camera.view);
  out.forward = glm::transpose(R) * glm::vec3(0.f, 0.f, 1.f);
  return out;
}

using Patch = std::array<float, (2 * kPatch + 1) * (2 * kPatch + 1)>;

// Zero mean and unit length, or nothing if there is too little contrast to tell.
std::optional<Patch> normalised(Patch patch) {
  float mean = 0.f;
  for (const float v : patch) mean += v;
  mean /= static_cast<float>(patch.size());
  float length = 0.f;
  for (float &v : patch) {
    v -= mean;
    length += v * v;
  }
  if (length < kMinContrast * kMinContrast * static_cast<float>(patch.size())) return std::nullopt;
  length = std::sqrt(length);
  for (float &v : patch) v /= length;
  return patch;
}

Patch patch_at(const Gray &gray, float x, float y) {
  Patch out;
  std::size_t k = 0;
  for (int v = -kPatch; v <= kPatch; ++v) {
    for (int u = -kPatch; u <= kPatch; ++u) out[k++] = gray.sample(x + static_cast<float>(u), y + static_cast<float>(v));
  }
  return out;
}

float correlation(const Patch &a, const Patch &b) {
  float sum = 0.f;
  for (std::size_t k = 0; k < a.size(); ++k) sum += a[k] * b[k];
  return sum;
}

glm::vec2 project(const Camera &camera, const glm::vec3 &world, float &depth) {
  const glm::vec3 t = glm::vec3(camera.view * glm::vec4(world, 1.f));
  depth = t.z;
  return {camera.fx * t.x / t.z + camera.cx, camera.fy * t.y / t.z + camera.cy};
}

} // namespace

std::vector<ColmapPoint> dense_points(std::span<const DensePointsView> views, std::span<const ColmapPoint> sparse, std::size_t maxPoints) {
  if (views.size() < 3 || maxPoints == 0) return {};
  std::vector<Prepared> prepared;
  prepared.reserve(views.size());
  for (const DensePointsView &view : views) prepared.push_back(prepare_view(view));

  const std::size_t step = std::max<std::size_t>(views.size() / kMaxReferences, 1);
  std::vector<std::size_t> references;
  for (std::size_t i = 0; i < views.size(); i += step) references.push_back(i);

  std::mutex found;
  std::vector<ColmapPoint> points;
  parallel_for(references.size(), 0, [&](std::size_t r) {
    const std::size_t index = references[r];
    const Prepared &reference = prepared[index];
    // The depths worth searching: where the sparse points lie from here.
    std::vector<float> depths;
    for (const ColmapPoint &p : sparse) {
      float depth = 0.f;
      const glm::vec2 at = project(reference.camera, glm::vec3(p.position), depth);
      if (depth > 0.f && at.x >= 0.f && at.y >= 0.f && at.x < static_cast<float>(reference.gray.width) && at.y < static_cast<float>(reference.gray.height)) depths.push_back(depth);
    }
    if (depths.size() < 10) return;
    std::sort(depths.begin(), depths.end());
    const float inverseFar = 1.f / (depths[depths.size() * 98 / 100] * 1.2f), inverseNear = 1.f / (depths[depths.size() * 2 / 100] * 0.8f);

    // Cameras looking the same way, with a good baseline to triangulate on.
    const float wantedBaseline = kBaseline * depths[depths.size() / 2];
    std::vector<std::pair<float, std::size_t>> nearby;
    for (std::size_t j = 0; j < prepared.size(); ++j) {
      if (j == index || glm::dot(prepared[j].forward, reference.forward) < 0.7f) continue;
      const float distance = glm::length(prepared[j].camera.position - reference.camera.position);
      if (distance > 0.f) nearby.emplace_back(std::abs(distance - wantedBaseline), j);
    }
    if (nearby.size() < kNeighbours) return;
    std::partial_sort(nearby.begin(), nearby.begin() + kNeighbours, nearby.end());

    const Gray &first = prepared[nearby[0].second].gray, &second = prepared[nearby[1].second].gray;
    const Camera &firstCamera = prepared[nearby[0].second].camera, &secondCamera = prepared[nearby[1].second].camera;
    const glm::mat3 toWorld = glm::transpose(glm::mat3(reference.camera.view));
    std::vector<ColmapPoint> local;
    for (int y = kPatch; y < reference.gray.height - kPatch - 1; y += kGrid) {
      for (int x = kPatch; x < reference.gray.width - kPatch - 1; x += kGrid) {
        const auto wanted = normalised(patch_at(reference.gray, static_cast<float>(x), static_cast<float>(y)));
        if (!wanted) continue;
        const glm::vec3 ray = toWorld * glm::vec3((static_cast<float>(x) + 0.5f - reference.camera.cx) / reference.camera.fx,
                                                  (static_cast<float>(y) + 0.5f - reference.camera.cy) / reference.camera.fy, 1.f);
        const auto world_at = [&](float inverseDepth) { return reference.camera.position + ray / inverseDepth; };

        // About half a pixel per step along the line in the neighbour.
        float depthNear = 0.f, depthFar = 0.f;
        const glm::vec2 a = project(firstCamera, world_at(inverseNear), depthNear), b = project(firstCamera, world_at(inverseFar), depthFar);
        if (depthNear <= 0.f || depthFar <= 0.f) continue;
        const int steps = std::clamp(static_cast<int>(2.f * glm::length(a - b)), 2, kMaxSteps);
        std::array<float, kMaxSteps + 1> scores;
        scores.fill(-1.f);
        int best = -1;
        for (int s = 0; s <= steps; ++s) {
          const float inverseDepth = inverseFar + (inverseNear - inverseFar) * static_cast<float>(s) / static_cast<float>(steps);
          float depth = 0.f;
          const glm::vec2 at = project(firstCamera, world_at(inverseDepth), depth);
          if (depth <= 0.f || !first.holds(at.x - 0.5f, at.y - 0.5f)) continue;
          const auto found_patch = normalised(patch_at(first, at.x - 0.5f, at.y - 0.5f));
          if (!found_patch) continue;
          scores[static_cast<std::size_t>(s)] = correlation(*wanted, *found_patch);
          if (best < 0 || scores[static_cast<std::size_t>(s)] > scores[static_cast<std::size_t>(best)]) best = s;
        }
        if (best < 0 || scores[static_cast<std::size_t>(best)] < kMinCorrelation) continue;
        bool unique = true;
        for (int s = 0; s <= steps && unique; ++s) {
          if (std::abs(s - best) > 4 && scores[static_cast<std::size_t>(s)] > scores[static_cast<std::size_t>(best)] - kMinLead) unique = false;
        }
        if (!unique) continue;

        // Between the steps: the top of the parabola through the best and its neighbours.
        float refined = static_cast<float>(best);
        if (best > 0 && best < steps) {
          const float below = scores[static_cast<std::size_t>(best - 1)], top = scores[static_cast<std::size_t>(best)], above = scores[static_cast<std::size_t>(best + 1)];
          const float curve = below - 2.f * top + above;
          if (below > -1.f && above > -1.f && curve < 0.f) refined += std::clamp(0.5f * (below - above) / curve, -0.5f, 0.5f);
        }
        const glm::vec3 world = world_at(inverseFar + (inverseNear - inverseFar) * refined / static_cast<float>(steps));
        float depth = 0.f;
        const glm::vec2 third = project(secondCamera, world, depth);
        if (depth <= 0.f || !second.holds(third.x - 0.5f, third.y - 0.5f)) continue;
        const auto thirdPatch = normalised(patch_at(second, third.x - 0.5f, third.y - 0.5f));
        if (!thirdPatch || correlation(*wanted, *thirdPatch) < kMinThirdView) continue;

        const DensePointsView &view = views[index];
        const auto full_x = static_cast<std::size_t>(std::clamp((static_cast<float>(x) + 0.5f) * static_cast<float>(view.camera.width) / static_cast<float>(reference.gray.width), 0.f, static_cast<float>(view.camera.width - 1)));
        const auto full_y = static_cast<std::size_t>(std::clamp((static_cast<float>(y) + 0.5f) * static_cast<float>(view.camera.height) / static_cast<float>(reference.gray.height), 0.f, static_cast<float>(view.camera.height - 1)));
        const std::uint8_t *colour = view.rgba.data() + (full_y * static_cast<std::size_t>(view.camera.width) + full_x) * 4;
        ColmapPoint point{};
        point.position = glm::dvec3(world);
        point.color = {colour[0], colour[1], colour[2]};
        local.push_back(point);
      }
    }
    const std::lock_guard lock(found);
    points.insert(points.end(), local.begin(), local.end());
  });

  if (points.size() > maxPoints) {
    std::mt19937 rng(1);
    std::shuffle(points.begin(), points.end(), rng);
    points.resize(maxPoints);
  }
  return points;
}

} // namespace splat
