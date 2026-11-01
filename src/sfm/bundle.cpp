#include "sfm/bundle.hpp"

#include <algorithm>
#include <cmath>

namespace splat {

namespace {

constexpr double kKPriorSigma = 0.3;

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

// The derivative of a pixel with respect to the camera-space point.
Eigen::Matrix<double, 2, 3> projection_jacobian(const Lens &lens, const Eigen::Vector3d &c) {
  const double iz = 1.0 / c.z();
  const Eigen::Vector2d u = c.head<2>() * iz;
  const double r2 = u.squaredNorm();
  const double d = 1.0 + lens.k * r2;
  Eigen::Matrix<double, 2, 3> dudc;
  dudc << iz, 0.0, -u.x() * iz, 0.0, iz, -u.y() * iz;
  const Eigen::Matrix2d dpdu = lens.f * (d * Eigen::Matrix2d::Identity() + 2.0 * lens.k * u * u.transpose());
  return dpdu * dudc;
}

double huber_weight(double norm, double delta) { return norm <= delta ? 1.0 : delta / norm; }

double huber_cost(double norm, double delta) {
  return norm <= delta ? 0.5 * norm * norm : delta * (norm - 0.5 * delta);
}

double total_cost(const std::vector<Lens> &lenses, const std::vector<BundleView> &views,
                  const std::vector<Eigen::Vector3d> &points, const std::vector<BundleObservation> &observations,
                  double delta, bool refineLenses, double *rms) {
  double cost = 0.0, squares = 0.0;
  for (const BundleObservation &o : observations) {
    const BundleView &view = views[static_cast<std::size_t>(o.view)];
    Eigen::Vector2d pixel;
    const Eigen::Vector3d c = view.pose.R * points[static_cast<std::size_t>(o.point)] + view.pose.t;
    // A point behind the camera gets a large, constant error rather than none.
    const double norm = project_point(lenses[static_cast<std::size_t>(view.lens)], c, pixel) ? (pixel - o.pixel).norm() : 1000.0;
    cost += huber_cost(norm, delta);
    squares += norm * norm;
  }
  if (refineLenses) {
    for (const Lens &lens : lenses) {
      const double rf = lens.fixedF ? 0.0 : (lens.f - lens.fPrior) / lens.fPriorSigma;
      const double rk = lens.k / kKPriorSigma;
      cost += 0.5 * (rf * rf + rk * rk);
    }
  }
  if (rms != nullptr) *rms = observations.empty() ? 0.0 : std::sqrt(squares / static_cast<double>(observations.size()));
  return cost;
}

} // namespace

bool project_point(const Lens &lens, const Eigen::Vector3d &c, Eigen::Vector2d &pixel) {
  if (c.z() <= 1e-12) return false;
  const Eigen::Vector2d u = c.head<2>() / c.z();
  const double d = 1.0 + lens.k * u.squaredNorm();
  pixel = {lens.f * d * u.x() + lens.cx, lens.f * d * u.y() + lens.cy};
  return true;
}

Eigen::Vector3d unproject_pixel(const Lens &lens, const Eigen::Vector2d &pixel) {
  const Eigen::Vector2d distorted((pixel.x() - lens.cx) / lens.f, (pixel.y() - lens.cy) / lens.f);
  Eigen::Vector2d u = distorted;
  for (int i = 0; i < 8; ++i) u = distorted / (1.0 + lens.k * u.squaredNorm());
  return {u.x(), u.y(), 1.0};
}

BundleResult bundle_adjust(std::vector<Lens> &lenses, std::vector<BundleView> &views, std::vector<Eigen::Vector3d> &points,
                           const std::vector<BundleObservation> &observations, const BundleOptions &options) {
  // The reduced system holds 6 values per free view, then 2 per lens (f, k).
  std::vector<int> viewOffset(views.size(), -1);
  int size = 0;
  for (std::size_t i = 0; i < views.size(); ++i) {
    if (!views[i].fixed) {
      viewOffset[i] = size;
      size += 6;
    }
  }
  std::vector<int> lensOffset(lenses.size(), -1);
  if (options.refineLenses) {
    for (std::size_t i = 0; i < lenses.size(); ++i) {
      lensOffset[i] = size;
      size += 2;
    }
  }

  std::vector<std::vector<int>> observationsOfPoint(points.size());
  for (std::size_t i = 0; i < observations.size(); ++i) {
    observationsOfPoint[static_cast<std::size_t>(observations[i].point)].push_back(static_cast<int>(i));
  }

  BundleResult result;
  double cost = total_cost(lenses, views, points, observations, options.huberPixels, options.refineLenses, &result.initialRms);
  result.finalRms = result.initialRms;

  struct Block {
    int offset[2];
    Eigen::Matrix<double, 3, 8> W; // Point against view (6) and lens (2)
    Eigen::Matrix<double, 8, 1> g;
  };

  double lambda = 1e-4;
  std::vector<Block> blocks;
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    result.iterations = iteration + 1;
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(size, size);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(size);

    // Everything the point update needs later, stored per point.
    std::vector<Eigen::Matrix3d> Vinv(points.size());
    std::vector<Eigen::Vector3d> gPoint(points.size());
    std::vector<std::vector<Block>> pointBlocks(points.size());

    const auto scatter = [&](int a, int aLen, int b, int bLen, const auto &value, int aFrom, int bFrom) {
      if (a < 0 || b < 0) return;
      S.block(a, b, aLen, bLen).noalias() += value.block(aFrom, bFrom, aLen, bLen);
    };

    for (std::size_t p = 0; p < points.size(); ++p) {
      Eigen::Matrix3d V = Eigen::Matrix3d::Zero();
      Eigen::Vector3d gp = Eigen::Vector3d::Zero();
      blocks.clear();
      for (const int index : observationsOfPoint[p]) {
        const BundleObservation &o = observations[static_cast<std::size_t>(index)];
        const BundleView &view = views[static_cast<std::size_t>(o.view)];
        const Lens &lens = lenses[static_cast<std::size_t>(view.lens)];
        const Eigen::Vector3d rotated = view.pose.R * points[p];
        const Eigen::Vector3d c = rotated + view.pose.t;
        Eigen::Vector2d pixel;
        if (!project_point(lens, c, pixel)) continue;
        const Eigen::Vector2d r = pixel - o.pixel;
        const double w = huber_weight(r.norm(), options.huberPixels);

        const Eigen::Matrix<double, 2, 3> A = projection_jacobian(lens, c);
        const Eigen::Matrix<double, 2, 3> Jp = A * view.pose.R;
        Eigen::Matrix<double, 2, 8> Jc = Eigen::Matrix<double, 2, 8>::Zero();
        Jc.leftCols<3>() = -A * skew(rotated); // Rotation, applied on the left
        Jc.middleCols<3>(3) = A;               // Translation
        const Eigen::Vector2d u = c.head<2>() / c.z();
        const double r2 = u.squaredNorm();
        if (!lens.fixedF) Jc.col(6) = (1.0 + lens.k * r2) * u;
        Jc.col(7) = lens.f * r2 * u;

        V.noalias() += w * Jp.transpose() * Jp;
        gp.noalias() += w * Jp.transpose() * r;

        Block block;
        block.offset[0] = viewOffset[static_cast<std::size_t>(o.view)];
        block.offset[1] = lensOffset[static_cast<std::size_t>(view.lens)];
        block.W.noalias() = w * Jp.transpose() * Jc;
        block.g.noalias() = w * Jc.transpose() * r;
        const Eigen::Matrix<double, 8, 8> Hcc = w * Jc.transpose() * Jc;
        scatter(block.offset[0], 6, block.offset[0], 6, Hcc, 0, 0);
        scatter(block.offset[0], 6, block.offset[1], 2, Hcc, 0, 6);
        scatter(block.offset[1], 2, block.offset[0], 6, Hcc, 6, 0);
        scatter(block.offset[1], 2, block.offset[1], 2, Hcc, 6, 6);
        if (block.offset[0] >= 0) rhs.segment<6>(block.offset[0]) -= block.g.head<6>();
        if (block.offset[1] >= 0) rhs.segment<2>(block.offset[1]) -= block.g.tail<2>();
        blocks.push_back(block);
      }

      // Marginalise this point. Its damping belongs here, before inversion.
      V.diagonal() *= 1.0 + lambda;
      V.diagonal().array() += 1e-12;
      const Eigen::Matrix3d inverse = V.inverse();
      Vinv[p] = inverse;
      gPoint[p] = gp;
      for (const Block &a : blocks) {
        const Eigen::Matrix<double, 8, 3> WaVinv = a.W.transpose() * inverse;
        const Eigen::Matrix<double, 8, 1> toRhs = WaVinv * gp;
        if (a.offset[0] >= 0) rhs.segment<6>(a.offset[0]) += toRhs.head<6>();
        if (a.offset[1] >= 0) rhs.segment<2>(a.offset[1]) += toRhs.tail<2>();
        for (const Block &b : blocks) {
          const Eigen::Matrix<double, 8, 8> product = -(WaVinv * b.W);
          scatter(a.offset[0], 6, b.offset[0], 6, product, 0, 0);
          scatter(a.offset[0], 6, b.offset[1], 2, product, 0, 6);
          scatter(a.offset[1], 2, b.offset[0], 6, product, 6, 0);
          scatter(a.offset[1], 2, b.offset[1], 2, product, 6, 6);
        }
      }
      pointBlocks[p] = blocks;
    }

    // Weak priors keep f and k sensible when the scene cannot determine them.
    if (options.refineLenses) {
      for (std::size_t i = 0; i < lenses.size(); ++i) {
        const int at = lensOffset[i];
        if (!lenses[i].fixedF) {
          const double weight = 1.0 / (lenses[i].fPriorSigma * lenses[i].fPriorSigma);
          S(at, at) += weight;
          rhs(at) -= weight * (lenses[i].f - lenses[i].fPrior);
        }
        const double weight = 1.0 / (kKPriorSigma * kKPriorSigma);
        S(at + 1, at + 1) += weight;
        rhs(at + 1) -= weight * lenses[i].k;
      }
    }

    // Try steps until one lowers the cost.
    bool improved = false;
    for (int attempt = 0; attempt < 8 && !improved; ++attempt) {
      Eigen::MatrixXd damped = S;
      for (int i = 0; i < size; ++i) {
        damped(i, i) = damped(i, i) <= 1e-12 ? 1.0 : damped(i, i) * (1.0 + lambda);
      }
      Eigen::VectorXd step = Eigen::VectorXd::Zero(size);
      if (size > 0) {
        const Eigen::LDLT<Eigen::MatrixXd> solver(damped);
        step = solver.solve(rhs);
      }
      if (!step.allFinite()) {
        lambda *= 10.0;
        continue;
      }

      auto newViews = views;
      auto newLenses = lenses;
      auto newPoints = points;
      for (std::size_t i = 0; i < views.size(); ++i) {
        if (viewOffset[i] < 0) continue;
        const Eigen::Matrix<double, 6, 1> d = step.segment<6>(viewOffset[i]);
        newViews[i].pose.R = rotation_from_vector(d.head<3>()) * views[i].pose.R;
        newViews[i].pose.t = views[i].pose.t + d.tail<3>();
      }
      if (options.refineLenses) {
        for (std::size_t i = 0; i < lenses.size(); ++i) {
          if (!lenses[i].fixedF) newLenses[i].f = std::max(lenses[i].f + step(lensOffset[i]), 1e-3);
          newLenses[i].k = lenses[i].k + step(lensOffset[i] + 1);
        }
      }
      for (std::size_t p = 0; p < points.size(); ++p) {
        Eigen::Vector3d b = -gPoint[p];
        for (const Block &block : pointBlocks[p]) {
          if (block.offset[0] >= 0) b.noalias() -= block.W.leftCols<6>() * step.segment<6>(block.offset[0]);
          if (block.offset[1] >= 0) b.noalias() -= block.W.rightCols<2>() * step.segment<2>(block.offset[1]);
        }
        newPoints[p] = points[p] + Vinv[p] * b;
      }

      double rms = 0.0;
      const double newCost = total_cost(newLenses, newViews, newPoints, observations, options.huberPixels, options.refineLenses, &rms);
      if (std::isfinite(newCost) && newCost < cost) {
        const double relative = (cost - newCost) / std::max(cost, 1e-12);
        views = std::move(newViews);
        lenses = std::move(newLenses);
        points = std::move(newPoints);
        cost = newCost;
        result.finalRms = rms;
        lambda = std::max(lambda * 0.3, 1e-9);
        improved = true;
        if (relative < 1e-9) return result;
      } else {
        lambda *= 10.0;
      }
    }
    if (!improved) break;
  }
  return result;
}

void refine_pose(const Lens &lens, Pose &pose, const std::vector<Eigen::Vector3d> &points,
                 const std::vector<Eigen::Vector2d> &pixels) {
  for (int iteration = 0; iteration < 8; ++iteration) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
      const Eigen::Vector3d rotated = pose.R * points[i];
      const Eigen::Vector3d c = rotated + pose.t;
      Eigen::Vector2d pixel;
      if (!project_point(lens, c, pixel)) continue;
      const Eigen::Vector2d r = pixel - pixels[i];
      const double w = huber_weight(r.norm(), 2.0);
      const Eigen::Matrix<double, 2, 3> A = projection_jacobian(lens, c);
      Eigen::Matrix<double, 2, 6> J;
      J.leftCols<3>() = -A * skew(rotated);
      J.rightCols<3>() = A;
      H.noalias() += w * J.transpose() * J;
      g.noalias() += w * J.transpose() * r;
    }
    H.diagonal() *= 1.0 + 1e-6;
    const Eigen::Matrix<double, 6, 1> step = H.ldlt().solve(-g);
    if (!step.allFinite()) return;
    pose.R = rotation_from_vector(step.head<3>()) * pose.R;
    pose.t += step.tail<3>();
    if (step.norm() < 1e-8) return;
  }
}

} // namespace splat
