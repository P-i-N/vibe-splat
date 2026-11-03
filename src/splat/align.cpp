#include "splat/align.hpp"

#include <Eigen/Dense>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

namespace splat {

namespace {

using glm::dmat3;
using glm::dvec3;

constexpr dvec3 kUp{0.0, -1.0, 0.0};
constexpr std::size_t kMaxCandidates = 50000;
constexpr int kRansacIterations = 1000;
constexpr double kMaxTiltDeg = 20.0;
// Planes are scored by their inliers times cos(tilt from the cameras' up) to
// this power, so a sloped bank of bushes holding a few more Gaussians than a
// level lawn does not win: 10 degrees costs 14%, 20 degrees 45%.
constexpr double kTiltPenalty = 10.0;
constexpr double kTolerance = 0.015;     // Of the bounds radius: how far from the plane counts as on it
constexpr double kMinInlierShare = 0.02; // Of the candidates, for a plane to count as the floor
// Of the Gaussians within four tolerances of the plane, the share within one.
// A blob with no surface spreads them evenly (1/4); grass in the bicycle scene
// puts 0.44-0.46 there.
constexpr double kMinFlatness = 0.33;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The basis preprocess.comp evaluates colours with.
std::array<double, 16> sh_basis(dvec3 d) {
  const double x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
  constexpr double C1 = 0.4886025119029199;
  return {0.28209479177387814,
          -C1 * y,
          C1 * z,
          -C1 * x,
          1.0925484305920792 * x * y,
          -1.0925484305920792 * y * z,
          0.31539156525252005 * (2.0 * zz - xx - yy),
          -1.0925484305920792 * x * z,
          0.5462742152960396 * (xx - yy),
          -0.5900435899266435 * y * (3.0 * xx - yy),
          2.890611442640554 * x * y * z,
          -0.4570457994644658 * y * (4.0 * zz - xx - yy),
          0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
          -0.4570457994644658 * x * (4.0 * zz - xx - yy),
          1.445305721320277 * z * (xx - yy),
          -0.5900435899266435 * x * (xx - 3.0 * yy)};
}

// Eigenvalues ascending, eigenvectors as the matching columns.
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(const Eigen::Matrix3d &m) { return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(m); }

dvec3 to_glm(const Eigen::Vector3d &v) { return {v.x(), v.y(), v.z()}; }

dvec3 position(const GaussianCloud &cloud, std::size_t i) {
  return {cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2]};
}

// The least-squares plane through points, its normal on the side of `up`.
std::pair<dvec3, double> fit_plane(const std::vector<dvec3> &points, dvec3 up) {
  dvec3 mean(0.0);
  for (const dvec3 &p : points) mean += p;
  mean /= static_cast<double>(points.size());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const dvec3 &p : points) {
    const Eigen::Vector3d d(p.x - mean.x, p.y - mean.y, p.z - mean.z);
    covariance += d * d.transpose();
  }
  dvec3 normal = glm::normalize(to_glm(eigen(covariance).eigenvectors().col(0)));
  if (glm::dot(normal, up) < 0.0) normal = -normal;
  return {normal, glm::dot(normal, mean)};
}

} // namespace

dvec3 estimate_up(std::span<const Camera> cameras) {
  if (cameras.empty()) return kUp;
  dvec3 meanUp(0.0);
  Eigen::Matrix3d rights = Eigen::Matrix3d::Zero();
  for (const Camera &camera : cameras) {
    // Rows of the world-to-camera rotation are the camera axes in the world.
    const dmat3 r(glm::mat3(camera.view));
    const Eigen::Vector3d right(r[0][0], r[1][0], r[2][0]);
    meanUp -= dvec3(r[0][1], r[1][1], r[2][1]); // Camera y points down
    rights += right * right.transpose();
  }
  if (glm::length(meanUp) < 1e-9) return kUp;
  meanUp = glm::normalize(meanUp);
  // The direction the x axes are all perpendicular to. Only trusted if they
  // spread around it, not when every camera faces the same way.
  const auto solver = eigen(rights);
  const Eigen::Vector3d values = solver.eigenvalues();
  dvec3 normal = glm::normalize(to_glm(solver.eigenvectors().col(0)));
  if (glm::dot(normal, meanUp) < 0.0) normal = -normal;
  const bool spread = values[1] > 4.0 * values[0] && values[1] > 0.05 * static_cast<double>(cameras.size());
  return spread && glm::dot(normal, meanUp) > std::cos(std::numbers::pi / 4.0) ? normal : meanUp;
}

FloorEstimate estimate_floor(const GaussianCloud &cloud, std::span<const Camera> cameras) {
  FloorEstimate floor;
  const dvec3 up = estimate_up(cameras);
  floor.normal = up;
  if (cloud.empty()) return floor;

  const Bounds bounds = compute_bounds(cloud);
  const dvec3 center(bounds.center[0], bounds.center[1], bounds.center[2]);
  const double radius = std::max(static_cast<double>(bounds.radius), 1e-6);

  // Opaque Gaussians not far out, where floaters are.
  std::vector<dvec3> points;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const dvec3 p = position(cloud, i);
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    if (sigmoid(cloud.opacities[i]) < 0.5 || glm::length(p - center) > 3.0 * radius) continue;
    points.push_back(p);
  }
  std::mt19937 rng(1);
  if (points.size() > kMaxCandidates) {
    for (std::size_t i = 0; i < kMaxCandidates; ++i) {
      std::swap(points[i], points[std::uniform_int_distribution<std::size_t>(i, points.size() - 1)(rng)]);
    }
    points.resize(kMaxCandidates);
  }
  floor.candidates = points.size();
  if (points.empty()) return floor;

  const double tolerance = kTolerance * radius;
  const double minCos = std::cos(kMaxTiltDeg * std::numbers::pi / 180.0);
  const auto count_inliers = [&](dvec3 n, double d, double within) {
    return static_cast<std::size_t>(std::count_if(points.begin(), points.end(), [&](const dvec3 &p) { return std::abs(glm::dot(n, p) - d) < within; }));
  };
  // Most cameras must be above the floor, which rules out ceilings.
  const auto under_cameras = [&](dvec3 n, double d) {
    std::size_t above = 0;
    for (const Camera &camera : cameras) above += glm::dot(n, dvec3(camera.position)) - d > tolerance;
    return cameras.empty() || 2 * above > cameras.size();
  };

  std::size_t bestCount = 0;
  double bestScore = 0.0;
  dvec3 bestNormal = up;
  double bestOffset = 0.0;
  if (points.size() >= 3) {
    std::uniform_int_distribution<std::size_t> pick(0, points.size() - 1);
    for (int iteration = 0; iteration < kRansacIterations; ++iteration) {
      const dvec3 a = points[pick(rng)], b = points[pick(rng)], c = points[pick(rng)];
      dvec3 n = glm::cross(b - a, c - a);
      if (glm::length(n) < 1e-12) continue;
      n = glm::normalize(n);
      if (glm::dot(n, up) < 0.0) n = -n;
      if (glm::dot(n, up) < minCos) continue;
      const double d = glm::dot(n, a);
      if (!under_cameras(n, d)) continue;
      const std::size_t count = count_inliers(n, d, tolerance);
      if (const double score = static_cast<double>(count) * std::pow(glm::dot(n, up), kTiltPenalty); score > bestScore) {
        bestScore = score;
        bestCount = count;
        bestNormal = n;
        bestOffset = d;
      }
    }
  }

  const std::size_t needed = std::max<std::size_t>(50, static_cast<std::size_t>(kMinInlierShare * static_cast<double>(points.size())));
  if (bestCount >= needed) {
    // Refit to the inliers, which three points only roughly agree on.
    for (int round = 0; round < 2; ++round) {
      std::vector<dvec3> inliers;
      for (const dvec3 &p : points) {
        if (std::abs(glm::dot(bestNormal, p) - bestOffset) < tolerance) inliers.push_back(p);
      }
      if (inliers.size() < 3) break;
      const auto [n, d] = fit_plane(inliers, up);
      if (glm::dot(n, up) < minCos) break;
      bestNormal = n;
      bestOffset = d;
    }
    const std::size_t inliers = count_inliers(bestNormal, bestOffset, tolerance);
    const std::size_t near = count_inliers(bestNormal, bestOffset, 4.0 * tolerance);
    if (inliers >= needed && static_cast<double>(inliers) >= kMinFlatness * static_cast<double>(near)) {
      floor.normal = bestNormal;
      floor.offset = bestOffset;
      floor.inliers = inliers;
      floor.fitted = true;
      return floor;
    }
  }

  // No plane: the cameras' up, under nearly everything.
  std::vector<double> heights;
  heights.reserve(points.size());
  for (const dvec3 &p : points) heights.push_back(glm::dot(up, p));
  const auto low = heights.begin() + static_cast<std::ptrdiff_t>(heights.size() / 50);
  std::nth_element(heights.begin(), low, heights.end());
  floor.offset = *low;
  return floor;
}

RigidTransform floor_alignment(const FloorEstimate &floor, const GaussianCloud &cloud) {
  const dvec3 n = glm::normalize(floor.normal);
  RigidTransform transform;
  // The shortest rotation taking n to kUp (Rodrigues).
  const dvec3 v = glm::cross(n, kUp);
  const double c = glm::dot(n, kUp);
  if (c < -1.0 + 1e-9) {
    transform.rotation = dmat3(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0); // Half a turn about x
  } else {
    const dmat3 k(0.0, v.z, -v.y, -v.z, 0.0, v.x, v.y, -v.x, 0.0); // k * w = v x w
    transform.rotation = dmat3(1.0) + k + k * k * (1.0 / (1.0 + c));
  }
  dvec3 origin(0.0);
  if (!cloud.empty()) {
    const Bounds bounds = compute_bounds(cloud);
    origin = dvec3(bounds.center[0], bounds.center[1], bounds.center[2]);
  }
  origin -= (glm::dot(n, origin) - floor.offset) * n; // Down onto the floor
  transform.translation = -(transform.rotation * origin);
  return transform;
}

void transform_cloud(GaussianCloud &cloud, const RigidTransform &transform) {
  const dmat3 &r = transform.rotation;
  const glm::dquat turn = glm::quat_cast(r);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const dvec3 p = r * position(cloud, i) + transform.translation;
    for (int k = 0; k < 3; ++k) cloud.positions[i * 3 + k] = static_cast<float>(p[k]);
    float *q = &cloud.rotations[i * 4];
    const glm::dquat turned = turn * glm::dquat(q[0], q[1], q[2], q[3]);
    q[0] = static_cast<float>(turned.w);
    q[1] = static_cast<float>(turned.x);
    q[2] = static_cast<float>(turned.y);
    q[3] = static_cast<float>(turned.z);
  }
  if (cloud.shDegree == 0) return;

  // The turned colour seen along d is the old one seen along r^T d. Each band
  // turns into itself, so per band the new coefficients are a fixed linear
  // map of the old ones, found by least squares over many directions.
  constexpr int kDirections = 64;
  Eigen::MatrixXd basis(kDirections, 16), turnedBasis(kDirections, 16);
  for (int k = 0; k < kDirections; ++k) {
    // Fibonacci sphere
    const double z = 1.0 - (2.0 * k + 1.0) / kDirections;
    const double phi = k * std::numbers::pi * (3.0 - std::sqrt(5.0));
    const dvec3 d(std::sqrt(1.0 - z * z) * std::cos(phi), std::sqrt(1.0 - z * z) * std::sin(phi), z);
    const auto b = sh_basis(d), t = sh_basis(glm::transpose(r) * d);
    for (int j = 0; j < 16; ++j) {
      basis(k, j) = b[j];
      turnedBasis(k, j) = t[j];
    }
  }
  const int coeffs = cloud.shCoeffs();
  std::vector<Eigen::MatrixXd> bands;
  for (int l = 1; l <= cloud.shDegree; ++l) {
    const int first = l * l, size = 2 * l + 1;
    bands.push_back(basis.middleCols(first, size).colPivHouseholderQr().solve(turnedBasis.middleCols(first, size)));
  }
  Eigen::VectorXd old(7);
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    float *sh = &cloud.sh[i * static_cast<std::size_t>(coeffs) * 3];
    for (int l = 1; l <= cloud.shDegree; ++l) {
      const int first = l * l, size = 2 * l + 1;
      const Eigen::MatrixXd &m = bands[static_cast<std::size_t>(l - 1)];
      for (int c = 0; c < 3; ++c) {
        for (int j = 0; j < size; ++j) old[j] = sh[(first + j) * 3 + c];
        for (int j = 0; j < size; ++j) sh[(first + j) * 3 + c] = static_cast<float>(m.row(j).dot(old.head(size)));
      }
    }
  }
}

} // namespace splat
