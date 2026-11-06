#include "train/mip.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace splat {

std::vector<float> smoothing_filter(const GaussianCloud &cloud, std::span<const Camera> cameras) {
  std::vector<float> filter(cloud.size(), 0.f);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const glm::vec4 p(cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2], 1.f);
    float smallest = std::numeric_limits<float>::infinity();
    for (const Camera &camera : cameras) {
      const glm::vec3 t = glm::vec3(camera.view * p);
      if (!(t.z > 0.f)) continue;
      const float x = camera.fx * t.x / t.z + camera.cx, y = camera.fy * t.y / t.z + camera.cy;
      if (x < 0.f || y < 0.f || x > static_cast<float>(camera.width) || y > static_cast<float>(camera.height)) continue;
      smallest = std::min(smallest, t.z / std::max(camera.fx, camera.fy));
    }
    if (std::isfinite(smallest)) filter[i] = 0.04f * smallest * smallest;
  }
  return filter;
}

void bake_filter(GaussianCloud &cloud, std::span<const float> filter) {
  for (std::size_t i = 0; i < cloud.size() && i < filter.size(); ++i) {
    const double f = filter[i];
    if (!(f > 0.0)) continue;
    double factor = 1.0;
    for (std::size_t a = 0; a < 3; ++a) {
      const double s2 = std::exp(2.0 * static_cast<double>(cloud.scales[i * 3 + a]));
      factor *= std::sqrt(s2 / (s2 + f));
      cloud.scales[i * 3 + a] = static_cast<float>(0.5 * std::log(s2 + f));
    }
    const double opacity = factor / (1.0 + std::exp(-static_cast<double>(cloud.opacities[i])));
    const double clamped = std::clamp(opacity, 1e-7, 1.0 - 1e-7);
    cloud.opacities[i] = static_cast<float>(std::log(clamped / (1.0 - clamped)));
  }
}

} // namespace splat
