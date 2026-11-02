#include "train/evaluate.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace splat {

namespace {

double squared_error(const std::vector<float> &render, const std::vector<std::uint8_t> &target, std::size_t pixel, const Exposure &exposure) {
  const float *c = render.data() + pixel * 4;
  double sum = 0.0;
  for (std::size_t r = 0; r < 3; ++r) {
    const double value = exposure[r * 4] * c[0] + exposure[r * 4 + 1] * c[1] + exposure[r * 4 + 2] * c[2] + exposure[r * 4 + 3];
    const double d = std::clamp(value, 0.0, 1.0) - target[pixel * 4 + r] / 255.0;
    sum += d * d;
  }
  return sum;
}

double to_psnr(double mse) { return mse > 0.0 ? -10.0 * std::log10(mse) : 100.0; }

} // namespace

double psnr_of(const std::vector<float> &render, const std::vector<std::uint8_t> &target) {
  double sum = 0.0;
  const std::size_t pixels = target.size() / 4;
  for (std::size_t i = 0; i < pixels; ++i) sum += squared_error(render, target, i, kNeutralExposure);
  return to_psnr(sum / (3.0 * static_cast<double>(pixels)));
}

void apply_exposure(std::vector<float> &render, const Exposure &e) {
  if (e == kNeutralExposure) return;
  for (std::size_t i = 0; i + 3 < render.size(); i += 4) {
    const float r = render[i], g = render[i + 1], b = render[i + 2];
    for (std::size_t c = 0; c < 3; ++c) render[i + c] = e[c * 4] * r + e[c * 4 + 1] * g + e[c * 4 + 2] * b + e[c * 4 + 3];
  }
}

Exposure fit_exposure(const std::vector<float> &render, const std::vector<std::uint8_t> &target, int width, int height, int x0, int x1) {
  Eigen::Matrix4d normal = Eigen::Matrix4d::Zero();
  Eigen::Matrix<double, 4, 3> right = Eigen::Matrix<double, 4, 3>::Zero();
  for (int y = 0; y < height; ++y) {
    for (int x = x0; x < x1; ++x) {
      const std::size_t at = static_cast<std::size_t>(y) * width + x;
      const Eigen::Vector4d in(render[at * 4], render[at * 4 + 1], render[at * 4 + 2], 1.0);
      const Eigen::RowVector3d out(target[at * 4] / 255.0, target[at * 4 + 1] / 255.0, target[at * 4 + 2] / 255.0);
      normal.noalias() += in * in.transpose();
      right.noalias() += in * out;
    }
  }
  // A little pull towards neutral, for a half with hardly any colour in it.
  const double pixels = static_cast<double>(height) * (x1 - x0);
  const double damping = 1e-6 * std::max(pixels, 1.0);
  Eigen::Matrix<double, 4, 3> neutral = Eigen::Matrix<double, 4, 3>::Zero();
  neutral.topLeftCorner<3, 3>().setIdentity();
  normal.diagonal().array() += damping;
  const Eigen::Matrix<double, 4, 3> solved = normal.ldlt().solve(right + damping * neutral);
  Exposure out{};
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t k = 0; k < 4; ++k) out[r * 4 + k] = static_cast<float>(solved(static_cast<int>(k), static_cast<int>(r)));
  }
  return out;
}

double cross_fitted_psnr(const std::vector<float> &render, const std::vector<std::uint8_t> &target, int width, int height) {
  const Exposure left = fit_exposure(render, target, width, height, 0, width / 2);
  const Exposure right = fit_exposure(render, target, width, height, width / 2, width);
  double sum = 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Each half through the fit of the other.
      sum += squared_error(render, target, static_cast<std::size_t>(y) * width + x, x < width / 2 ? right : left);
    }
  }
  return to_psnr(sum / (3.0 * static_cast<double>(width) * height));
}

} // namespace splat
