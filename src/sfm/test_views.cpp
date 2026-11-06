#include "sfm/test_views.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace splat {

namespace {

float lattice(int x, int y, int z) {
  auto h = static_cast<std::uint32_t>(x) * 73856093u ^ static_cast<std::uint32_t>(y) * 19349663u ^
           static_cast<std::uint32_t>(z) * 83492791u;
  h ^= h >> 13;
  h *= 1274126177u;
  h ^= h >> 16;
  return static_cast<float>(h & 0xFFFFu) / 65535.f;
}

float value_noise(const Eigen::Vector3d &p) {
  const int x = static_cast<int>(std::floor(p.x())), y = static_cast<int>(std::floor(p.y())), z = static_cast<int>(std::floor(p.z()));
  const auto smooth = [](double t) { return static_cast<float>(t * t * (3.0 - 2.0 * t)); };
  const float fx = smooth(p.x() - x), fy = smooth(p.y() - y), fz = smooth(p.z() - z);
  const auto mix = [](float a, float b, float t) { return a + (b - a) * t; };
  return mix(mix(mix(lattice(x, y, z), lattice(x + 1, y, z), fx), mix(lattice(x, y + 1, z), lattice(x + 1, y + 1, z), fx), fy),
             mix(mix(lattice(x, y, z + 1), lattice(x + 1, y, z + 1), fx), mix(lattice(x, y + 1, z + 1), lattice(x + 1, y + 1, z + 1), fx), fy),
             fz);
}

std::uint8_t texture(const Eigen::Vector3d &p, double seed) {
  float value = 0.f, amplitude = 0.5f;
  double frequency = 2.0;
  for (int octave = 0; octave < 5; ++octave) {
    value += amplitude * value_noise(p * frequency + Eigen::Vector3d::Constant(seed));
    amplitude *= 0.6f;
    frequency *= 2.1;
  }
  return static_cast<std::uint8_t>(std::clamp(value * 1.2f - 0.1f, 0.f, 1.f) * 255.f);
}

struct Box {
  Eigen::Vector3d low;
  Eigen::Vector3d high;
};

// Distances where a ray enters and leaves a box, or false if it misses.
bool slab(const Box &box, const Eigen::Vector3d &origin, const Eigen::Vector3d &direction, double &enter, double &leave) {
  enter = -std::numeric_limits<double>::infinity();
  leave = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    const double a = (box.low(axis) - origin(axis)) / direction(axis);
    const double b = (box.high(axis) - origin(axis)) / direction(axis);
    enter = std::max(enter, std::min(a, b));
    leave = std::min(leave, std::max(a, b));
  }
  return enter <= leave;
}

} // namespace

std::vector<TestView> make_test_views(int count, int width, int height, double focal) {
  // COLMAP's convention: +y points down, so the floor is at large y.
  const Box room{{-7.0, -3.0, -7.0}, {7.0, 2.0, 7.0}};
  const Box boxes[] = {{{-1.2, 0.2, -1.0}, {0.2, 2.0, 0.4}}, {{0.6, -0.4, 0.2}, {1.8, 2.0, 1.4}}, {{-0.4, 1.0, 1.2}, {0.5, 2.0, 2.1}}};

  std::vector<TestView> views;
  for (int v = 0; v < count; ++v) {
    // A wide arc rather than a full circle, with some variation in height.
    const double angle = -1.1 + 2.2 * v / std::max(count - 1, 1);
    const double radius = 4.5 + 0.3 * std::sin(v * 1.3);
    const Eigen::Vector3d center(radius * std::sin(angle), -0.8 + 0.3 * std::sin(v * 2.1), -radius * std::cos(angle));
    const Eigen::Vector3d target(0.0, 0.8, 0.3);
    const Eigen::Vector3d z = (target - center).normalized();
    const Eigen::Vector3d x = Eigen::Vector3d::UnitY().cross(z).normalized();
    const Eigen::Vector3d y = z.cross(x);

    TestView view;
    view.pose.R.row(0) = x;
    view.pose.R.row(1) = y;
    view.pose.R.row(2) = z;
    view.pose.t = -view.pose.R * center;
    view.image.width = width;
    view.image.height = height;
    view.image.pixels.resize(static_cast<std::size_t>(width) * height * 3);

    for (int py = 0; py < height; ++py) {
      for (int px = 0; px < width; ++px) {
        const Eigen::Vector3d ray = x * ((px + 0.5 - width * 0.5) / focal) + y * ((py + 0.5 - height * 0.5) / focal) + z;
        double enter = 0.0, leave = 0.0;
        double nearest = std::numeric_limits<double>::infinity();
        if (slab(room, center, ray, enter, leave)) nearest = leave; // Seen from inside
        for (const Box &box : boxes) {
          if (slab(box, center, ray, enter, leave) && enter > 0.0) nearest = std::min(nearest, enter);
        }
        const Eigen::Vector3d hit = center + ray * nearest;
        std::uint8_t *out = view.image.pixels.data() + (static_cast<std::size_t>(py) * width + px) * 3;
        out[0] = texture(hit, 11.0);
        out[1] = texture(hit, 37.0);
        out[2] = texture(hit, 71.0);
      }
    }
    views.push_back(std::move(view));
  }
  return views;
}

} // namespace splat
