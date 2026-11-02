#include "sfm/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace splat {

namespace {

// Polynomials in x, y, z of total degree at most 3, in the monomial order the
// action matrix relies on: the 10 cubics, then the 10 lower-degree terms.
constexpr int kMonomials = 20;
constexpr int kExponents[kMonomials][3] = {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2}, {0, 3, 0},
                                           {0, 2, 1}, {0, 1, 2}, {0, 0, 3}, {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0},
                                           {0, 1, 1}, {0, 0, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};

struct MonomialTable {
  int index[4][4][4]{};
  constexpr MonomialTable() {
    for (int m = 0; m < kMonomials; ++m) index[kExponents[m][0]][kExponents[m][1]][kExponents[m][2]] = m;
  }
};
constexpr MonomialTable kTable;

struct Poly {
  std::array<double, kMonomials> c{};

  Poly operator+(const Poly &o) const {
    Poly r;
    for (int i = 0; i < kMonomials; ++i) r.c[i] = c[i] + o.c[i];
    return r;
  }
  Poly operator-(const Poly &o) const {
    Poly r;
    for (int i = 0; i < kMonomials; ++i) r.c[i] = c[i] - o.c[i];
    return r;
  }
  Poly operator*(double s) const {
    Poly r;
    for (int i = 0; i < kMonomials; ++i) r.c[i] = c[i] * s;
    return r;
  }
  Poly operator*(const Poly &o) const {
    Poly r;
    for (int i = 0; i < kMonomials; ++i) {
      if (c[i] == 0.0) continue;
      for (int j = 0; j < kMonomials; ++j) {
        if (o.c[j] == 0.0) continue;
        const int ex = kExponents[i][0] + kExponents[j][0];
        const int ey = kExponents[i][1] + kExponents[j][1];
        const int ez = kExponents[i][2] + kExponents[j][2];
        if (ex + ey + ez <= 3) r.c[kTable.index[ex][ey][ez]] += c[i] * o.c[j];
      }
    }
    return r;
  }
};

using PolyMatrix = std::array<std::array<Poly, 3>, 3>;

PolyMatrix multiply(const PolyMatrix &a, const PolyMatrix &b) {
  PolyMatrix r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) r[i][j] = r[i][j] + a[i][k] * b[k][j];
    }
  }
  return r;
}

Eigen::Matrix3d essential_from_pose(const Pose &pose) {
  Eigen::Matrix3d skew;
  skew << 0.0, -pose.t.z(), pose.t.y(), pose.t.z(), 0.0, -pose.t.x(), -pose.t.y(), pose.t.x(), 0.0;
  return skew * pose.R;
}

// A minimal sample only fits its own five points well. This polishes the pose
// against all inliers: Gauss-Newton on the Sampson error, with a numerical
// Jacobian over rotation (3) and the direction of translation (2).
Pose refine_relative_pose(const Pose &start, const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                          const std::vector<int> &inliers) {
  const auto residuals = [&](const Pose &pose) {
    const Eigen::Matrix3d E = essential_from_pose(pose);
    Eigen::VectorXd r(static_cast<Eigen::Index>(inliers.size()));
    for (std::size_t i = 0; i < inliers.size(); ++i) {
      const Eigen::Vector3d &a = x1[static_cast<std::size_t>(inliers[i])], &b = x2[static_cast<std::size_t>(inliers[i])];
      const Eigen::Vector3d Ea = E * a, Etb = E.transpose() * b;
      const double denominator = std::sqrt(Ea.head<2>().squaredNorm() + Etb.head<2>().squaredNorm());
      r(static_cast<Eigen::Index>(i)) = denominator > 0.0 ? b.dot(Ea) / denominator : 0.0;
    }
    return r;
  };
  const auto moved = [](const Pose &pose, const Eigen::Matrix<double, 5, 1> &d, const Eigen::Vector3d &e1, const Eigen::Vector3d &e2) {
    Pose out;
    out.R = rotation_from_vector(d.head<3>()) * pose.R;
    out.t = (pose.t + e1 * d(3) + e2 * d(4)).normalized();
    return out;
  };

  Pose pose = start;
  pose.t.normalize();
  double cost = residuals(pose).squaredNorm();
  for (int iteration = 0; iteration < 10; ++iteration) {
    const Eigen::Vector3d e1 = pose.t.unitOrthogonal(), e2 = pose.t.cross(e1);
    const Eigen::VectorXd r = residuals(pose);
    Eigen::Matrix<double, Eigen::Dynamic, 5> J(r.size(), 5);
    constexpr double kStep = 1e-6;
    for (int k = 0; k < 5; ++k) {
      Eigen::Matrix<double, 5, 1> d = Eigen::Matrix<double, 5, 1>::Zero();
      d(k) = kStep;
      J.col(k) = (residuals(moved(pose, d, e1, e2)) - r) / kStep;
    }
    const Eigen::Matrix<double, 5, 5> H = J.transpose() * J;
    const Eigen::Matrix<double, 5, 1> g = J.transpose() * r;
    bool improved = false;
    // On a plane the cost has a nearly flat direction, so steps need damping.
    for (double damping = 1e-6; damping < 10.0 && !improved; damping *= 10.0) {
      Eigen::Matrix<double, 5, 5> damped = H;
      damped.diagonal() *= 1.0 + damping;
      const Eigen::Matrix<double, 5, 1> step = damped.ldlt().solve(-g);
      if (!step.allFinite()) continue;
      const Pose next = moved(pose, step, e1, e2);
      const double nextCost = residuals(next).squaredNorm();
      if (nextCost < cost) {
        pose = next;
        cost = nextCost;
        improved = true;
      }
    }
    if (!improved) break;
  }
  return pose;
}

double angle_between(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

// The two-view triangulation used for the cheirality check. Returns point in camera 1.
Eigen::Vector3d triangulate_pair(const Pose &pose, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2) {
  static const Pose identity;
  return triangulate({identity, pose}, {x1, x2});
}

} // namespace

std::vector<Eigen::Matrix3d> essential_five_point(const std::array<Eigen::Vector3d, 5> &x1,
                                                  const std::array<Eigen::Vector3d, 5> &x2) {
  Eigen::Matrix<double, 5, 9> constraints;
  for (int i = 0; i < 5; ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) constraints(i, r * 3 + c) = x2[static_cast<std::size_t>(i)](r) * x1[static_cast<std::size_t>(i)](c);
    }
  }
  const Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(constraints, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 4> basis = svd.matrixV().rightCols<4>();

  // E = x X + y Y + z Z + W
  PolyMatrix E;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Poly &p = E[r][c];
      const int at = r * 3 + c;
      p.c[kTable.index[1][0][0]] = basis(at, 0);
      p.c[kTable.index[0][1][0]] = basis(at, 1);
      p.c[kTable.index[0][0][1]] = basis(at, 2);
      p.c[kTable.index[0][0][0]] = basis(at, 3);
    }
  }

  // Ten cubic constraints: det(E) = 0 and 2 E E^T E - trace(E E^T) E = 0.
  Eigen::Matrix<double, 10, kMonomials> M;
  const Poly det = E[0][0] * (E[1][1] * E[2][2] - E[1][2] * E[2][1]) - E[0][1] * (E[1][0] * E[2][2] - E[1][2] * E[2][0]) +
                   E[0][2] * (E[1][0] * E[2][1] - E[1][1] * E[2][0]);
  for (int m = 0; m < kMonomials; ++m) M(0, m) = det.c[m];

  PolyMatrix Et;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) Et[r][c] = E[c][r];
  }
  const PolyMatrix EEt = multiply(E, Et);
  const Poly trace = EEt[0][0] + EEt[1][1] + EEt[2][2];
  const PolyMatrix EEtE = multiply(EEt, E);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const Poly p = EEtE[r][c] * 2.0 - trace * E[r][c];
      for (int m = 0; m < kMonomials; ++m) M(1 + r * 3 + c, m) = p.c[m];
    }
  }

  const Eigen::Matrix<double, 10, 10> left = M.leftCols<10>();
  const Eigen::Matrix<double, 10, 10> B = left.partialPivLu().solve(M.rightCols<10>());
  if (!B.allFinite()) return {};

  // Multiplication by x on the basis (x2, xy, xz, y2, yz, z2, x, y, z, 1).
  Eigen::Matrix<double, 10, 10> action = Eigen::Matrix<double, 10, 10>::Zero();
  action.topRows<6>() = -B.topRows<6>();
  action(6, 0) = 1.0; // x * x = x2
  action(7, 1) = 1.0; // x * y = xy
  action(8, 2) = 1.0; // x * z = xz
  action(9, 6) = 1.0; // x * 1 = x

  const Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigen(action);
  if (eigen.info() != Eigen::Success) return {};
  std::vector<Eigen::Matrix3d> solutions;
  for (int i = 0; i < 10; ++i) {
    if (std::abs(eigen.eigenvalues()(i).imag()) > 1e-9) continue;
    const Eigen::Matrix<double, 10, 1> v = eigen.eigenvectors().col(i).real();
    if (std::abs(v(9)) < 1e-14) continue;
    const double x = v(6) / v(9), y = v(7) / v(9), z = v(8) / v(9);
    const Eigen::Matrix<double, 9, 1> e = basis.col(0) * x + basis.col(1) * y + basis.col(2) * z + basis.col(3);
    Eigen::Matrix3d candidate;
    candidate << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
    solutions.push_back(candidate.normalized());
  }
  return solutions;
}

double sampson_squared(const Eigen::Matrix3d &E, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2) {
  const Eigen::Vector3d Ex1 = E * x1;
  const Eigen::Vector3d Etx2 = E.transpose() * x2;
  const double numerator = x2.dot(Ex1);
  const double denominator = Ex1.head<2>().squaredNorm() + Etx2.head<2>().squaredNorm();
  return denominator > 0.0 ? numerator * numerator / denominator : std::numeric_limits<double>::max();
}

std::optional<EssentialResult> estimate_relative_pose(const std::vector<Eigen::Vector3d> &x1,
                                                      const std::vector<Eigen::Vector3d> &x2, double threshold,
                                                      std::mt19937 &rng, int minInliers) {
  const int n = static_cast<int>(x1.size());
  if (n < 5 || n < minInliers) return std::nullopt;
  const double thresholdSquared = threshold * threshold;
  std::uniform_int_distribution<int> pick(0, n - 1);

  // Iterations for a 99.9% chance of one all-inlier sample at this inlier ratio.
  const auto iterations_for = [](double inlierRatio) {
    constexpr int kMaxIterations = 500;
    const double noFailure = 1.0 - std::pow(inlierRatio, 5);
    if (noFailure <= 0.0) return 1;
    // Clamped before the cast: a low ratio asks for billions of iterations.
    const double needed = std::clamp(std::ceil(std::log(0.001) / std::log(noFailure)), 1.0, static_cast<double>(kMaxIterations));
    return static_cast<int>(needed);
  };

  // On a plane, two different essential matrices fit all points equally well
  // and only the points' side of the cameras tells them apart. So keep every
  // close contender for the check below, not just the top one.
  std::vector<std::pair<int, Eigen::Matrix3d>> contenders;
  int bestCount = 0;
  int maxIterations = iterations_for(static_cast<double>(std::max(minInliers, 5)) / n);
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    std::array<int, 5> sample{};
    for (int i = 0; i < 5; ++i) {
      bool repeated = true;
      while (repeated) {
        sample[static_cast<std::size_t>(i)] = pick(rng);
        repeated = std::find(sample.begin(), sample.begin() + i, sample[static_cast<std::size_t>(i)]) != sample.begin() + i;
      }
    }
    std::array<Eigen::Vector3d, 5> a, b;
    for (std::size_t i = 0; i < 5; ++i) {
      a[i] = x1[static_cast<std::size_t>(sample[i])];
      b[i] = x2[static_cast<std::size_t>(sample[i])];
    }
    for (const Eigen::Matrix3d &E : essential_five_point(a, b)) {
      int count = 0;
      for (int i = 0; i < n; ++i) {
        if (sampson_squared(E, x1[static_cast<std::size_t>(i)], x2[static_cast<std::size_t>(i)]) < thresholdSquared) ++count;
      }
      if (count * 10 >= bestCount * 8 && count >= 5) contenders.emplace_back(count, E);
      if (count > bestCount) {
        bestCount = count;
        maxIterations = std::min(maxIterations, iterations_for(static_cast<double>(count) / n));
      }
    }
  }
  if (bestCount < 5) return std::nullopt;

  std::stable_sort(contenders.begin(), contenders.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
  if (contenders.size() > 16) contenders.resize(16);

  EssentialResult best;
  for (const auto &[count, E] : contenders) {
    // E = U diag(1,1,0) V^T gives four poses; the right one has its points in
    // front of both cameras.
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    if (U.determinant() < 0.0) U.col(2) = -U.col(2);
    if (V.determinant() < 0.0) V.col(2) = -V.col(2);
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    const Eigen::Matrix3d R1 = U * W * V.transpose(), R2 = U * W.transpose() * V.transpose();
    const Eigen::Vector3d t = U.col(2);
    const Pose candidates[4] = {{R1, t}, {R1, -t}, {R2, t}, {R2, -t}};

    std::vector<int> geometric;
    for (int i = 0; i < n; ++i) {
      if (sampson_squared(E, x1[static_cast<std::size_t>(i)], x2[static_cast<std::size_t>(i)]) < thresholdSquared) geometric.push_back(i);
    }

    for (const Pose &pose : candidates) {
      EssentialResult result;
      result.pose = pose;
      std::vector<double> angles;
      const Eigen::Vector3d secondCenter = pose.center();
      for (const int i : geometric) {
        const Eigen::Vector3d &a = x1[static_cast<std::size_t>(i)], &b = x2[static_cast<std::size_t>(i)];
        const Eigen::Vector3d X = triangulate_pair(pose, a, b);
        if (!X.allFinite() || X.z() <= 0.0 || (pose.R * X + pose.t).z() <= 0.0) continue;
        result.inliers.push_back(i);
        angles.push_back(angle_between(X, X - secondCenter));
      }
      if (result.inliers.size() > best.inliers.size()) {
        const auto middle = angles.begin() + static_cast<std::ptrdiff_t>(angles.size() / 2);
        std::nth_element(angles.begin(), middle, angles.end());
        result.medianAngleDeg = *middle * 180.0 / std::numbers::pi;
        best = std::move(result);
      }
    }
  }
  if (best.inliers.empty()) return std::nullopt;
  best.pose = refine_relative_pose(best.pose, x1, x2, best.inliers);
  return best;
}

Eigen::Vector3d triangulate(const std::vector<Pose> &poses, const std::vector<Eigen::Vector3d> &bearings) {
  Eigen::Matrix4d normal = Eigen::Matrix4d::Zero();
  for (std::size_t i = 0; i < poses.size(); ++i) {
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = poses[i].R;
    P.col(3) = poses[i].t;
    const Eigen::Matrix<double, 1, 4> a = bearings[i].x() * P.row(2) - P.row(0);
    const Eigen::Matrix<double, 1, 4> b = bearings[i].y() * P.row(2) - P.row(1);
    normal += a.transpose() * a + b.transpose() * b;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(normal);
  const Eigen::Vector4d h = solver.eigenvectors().col(0);
  return h.head<3>() / h.w();
}

std::vector<Pose> solve_p3p(const std::array<Eigen::Vector3d, 3> &bearings, const std::array<Eigen::Vector3d, 3> &points) {
  const Eigen::Vector3d f1 = bearings[0].normalized(), f2 = bearings[1].normalized(), f3 = bearings[2].normalized();
  const double cosA = f2.dot(f3), cosB = f1.dot(f3), cosC = f1.dot(f2);
  const double a2 = (points[1] - points[2]).squaredNorm();
  const double b2 = (points[0] - points[2]).squaredNorm();
  const double c2 = (points[0] - points[1]).squaredNorm();
  if (a2 < 1e-16 || b2 < 1e-16 || c2 < 1e-16) return {};

  const double p = (a2 - c2) / b2, q = (a2 + c2) / b2;
  const double A4 = (p - 1.0) * (p - 1.0) - 4.0 * c2 / b2 * cosA * cosA;
  const double A3 = 4.0 * (p * (1.0 - p) * cosB - (1.0 - q) * cosA * cosC + 2.0 * c2 / b2 * cosA * cosA * cosB);
  const double A2 = 2.0 * (p * p - 1.0 + 2.0 * p * p * cosB * cosB + 2.0 * (b2 - c2) / b2 * cosA * cosA -
                           4.0 * q * cosA * cosB * cosC + 2.0 * (b2 - a2) / b2 * cosC * cosC);
  const double A1 = 4.0 * (-p * (1.0 + p) * cosB + 2.0 * a2 / b2 * cosC * cosC * cosB - (1.0 - q) * cosA * cosC);
  const double A0 = (1.0 + p) * (1.0 + p) - 4.0 * a2 / b2 * cosC * cosC;
  if (std::abs(A4) < 1e-14) return {};

  Eigen::Matrix4d companion = Eigen::Matrix4d::Zero();
  companion(1, 0) = companion(2, 1) = companion(3, 2) = 1.0;
  companion(0, 3) = -A0 / A4;
  companion(1, 3) = -A1 / A4;
  companion(2, 3) = -A2 / A4;
  companion(3, 3) = -A3 / A4;
  const Eigen::EigenSolver<Eigen::Matrix4d> roots(companion, false);
  if (roots.info() != Eigen::Success) return {};

  std::vector<Pose> poses;
  for (int i = 0; i < 4; ++i) {
    if (std::abs(roots.eigenvalues()(i).imag()) > 1e-6) continue;
    const double v = roots.eigenvalues()(i).real();
    if (v <= 0.0) continue;
    const double denominator = 2.0 * (cosC - v * cosA);
    if (std::abs(denominator) < 1e-12) continue;
    const double u = ((p - 1.0) * v * v - 2.0 * p * cosB * v + 1.0 + p) / denominator;
    if (u <= 0.0) continue;
    const double s1Squared = b2 / (1.0 + v * v - 2.0 * v * cosB);
    if (s1Squared <= 0.0) continue;
    const double s1 = std::sqrt(s1Squared);
    const std::array<Eigen::Vector3d, 3> inCamera = {f1 * s1, f2 * (u * s1), f3 * (v * s1)};

    // Rigid alignment of the world points onto the camera-space points (Kabsch).
    const Eigen::Vector3d worldMean = (points[0] + points[1] + points[2]) / 3.0;
    const Eigen::Vector3d cameraMean = (inCamera[0] + inCamera[1] + inCamera[2]) / 3.0;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (std::size_t k = 0; k < 3; ++k) H += (inCamera[k] - cameraMean) * (points[k] - worldMean).transpose();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) S(2, 2) = -1.0;
    Pose pose;
    pose.R = svd.matrixU() * S * svd.matrixV().transpose();
    pose.t = cameraMean - pose.R * worldMean;
    if (pose.R.allFinite() && pose.t.allFinite()) poses.push_back(pose);
  }
  return poses;
}

std::optional<AbsolutePoseResult> estimate_absolute_pose(const std::vector<Eigen::Vector3d> &bearings,
                                                         const std::vector<Eigen::Vector3d> &points, double threshold,
                                                         std::mt19937 &rng) {
  const int n = static_cast<int>(bearings.size());
  if (n < 4) return std::nullopt;
  const double thresholdSquared = threshold * threshold;
  std::uniform_int_distribution<int> pick(0, n - 1);

  const auto inliers_of = [&](const Pose &pose, std::vector<int> *out) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d c = pose.R * points[static_cast<std::size_t>(i)] + pose.t;
      if (c.z() <= 0.0) continue;
      const Eigen::Vector2d error = c.head<2>() / c.z() - bearings[static_cast<std::size_t>(i)].head<2>();
      if (error.squaredNorm() < thresholdSquared) {
        ++count;
        if (out != nullptr) out->push_back(i);
      }
    }
    return count;
  };

  std::optional<Pose> best;
  int bestCount = 0;
  int maxIterations = 1000;
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    const int i0 = pick(rng);
    int i1 = pick(rng), i2 = pick(rng);
    while (i1 == i0) i1 = pick(rng);
    while (i2 == i0 || i2 == i1) i2 = pick(rng);
    const auto u = [](int i) { return static_cast<std::size_t>(i); };
    for (const Pose &pose : solve_p3p({bearings[u(i0)], bearings[u(i1)], bearings[u(i2)]}, {points[u(i0)], points[u(i1)], points[u(i2)]})) {
      const int count = inliers_of(pose, nullptr);
      if (count > bestCount) {
        bestCount = count;
        best = pose;
        const double ratio = static_cast<double>(count) / n;
        const double noFailure = 1.0 - ratio * ratio * ratio;
        maxIterations = noFailure <= 0.0 ? 0 : std::min(1000, static_cast<int>(std::ceil(std::log(0.001) / std::log(noFailure))));
      }
    }
  }
  if (!best) return std::nullopt;
  AbsolutePoseResult result;
  result.pose = *best;
  inliers_of(*best, &result.inliers);
  return result;
}

Eigen::Matrix3d rotation_from_vector(const Eigen::Vector3d &w) {
  const double angle = w.norm();
  if (angle < 1e-12) return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, w / angle).toRotationMatrix();
}

} // namespace splat
