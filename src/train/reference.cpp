#include "train/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace splat {

namespace {

constexpr double kWeights[11] = {0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
                                 0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838};
constexpr double kC1 = 0.0001, kC2 = 0.0009;
constexpr double kLogMinTransmittance = -9.210340372; // LOG_MIN_TRANSMITTANCE in train_types.glsl

using glm::dmat2;
using glm::dmat3;
using glm::dmat3x2;
using glm::dvec2;
using glm::dvec3;
using glm::dvec4;

struct Splat {
  std::size_t index = 0;
  dvec2 center{0.0};
  double radius = 0.0, depth = 0.0;
  dvec3 conic{0.0}; // xx, xy, yy
  double opacity = 0.0;
  dvec3 color{0.0};
};

dmat3 quat_to_mat(dvec4 q) {
  const double w = q.x, x = q.y, y = q.z, z = q.w;
  return dmat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
               2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
               2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

std::array<double, 16> sh_basis(dvec3 dir, int degree) {
  const double x = dir.x, y = dir.y, z = dir.z, xx = x * x, yy = y * y, zz = z * z;
  constexpr double C1 = 0.4886025119029199;
  std::array<double, 16> b{};
  b[0] = 0.28209479177387814;
  if (degree > 0) {
    b[1] = -C1 * y;
    b[2] = C1 * z;
    b[3] = -C1 * x;
  }
  if (degree > 1) {
    b[4] = 1.0925484305920792 * x * y;
    b[5] = -1.0925484305920792 * y * z;
    b[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
    b[7] = -1.0925484305920792 * x * z;
    b[8] = 0.5462742152960396 * (xx - yy);
  }
  if (degree > 2) {
    b[9] = -0.5900435899266435 * y * (3.0 * xx - yy);
    b[10] = 2.890611442640554 * x * y * z;
    b[11] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    b[12] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[13] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    b[14] = 1.445305721320277 * z * (xx - yy);
    b[15] = -0.5900435899266435 * x * (xx - 3.0 * yy);
  }
  return b;
}

std::array<dvec3, 16> sh_basis_gradient(dvec3 dir, int degree) {
  constexpr double C1 = 0.4886025119029199, A = 1.0925484305920792, K = 0.31539156525252005, L = 0.5462742152960396;
  constexpr double C = -0.5900435899266435, E = 2.890611442640554, F = -0.4570457994644658, G = 0.3731763325901154, H = 1.445305721320277;
  const double x = dir.x, y = dir.y, z = dir.z, xx = x * x, yy = y * y, zz = z * z;
  std::array<dvec3, 16> g{};
  if (degree > 0) {
    g[1] = {0.0, -C1, 0.0};
    g[2] = {0.0, 0.0, C1};
    g[3] = {-C1, 0.0, 0.0};
  }
  if (degree > 1) {
    g[4] = A * dvec3(y, x, 0.0);
    g[5] = -A * dvec3(0.0, z, y);
    g[6] = K * dvec3(-2.0 * x, -2.0 * y, 4.0 * z);
    g[7] = -A * dvec3(z, 0.0, x);
    g[8] = L * dvec3(2.0 * x, -2.0 * y, 0.0);
  }
  if (degree > 2) {
    g[9] = C * dvec3(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0);
    g[10] = E * dvec3(y * z, x * z, x * y);
    g[11] = F * dvec3(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z);
    g[12] = G * dvec3(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy);
    g[13] = F * dvec3(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z);
    g[14] = H * dvec3(2.0 * x * z, -2.0 * y * z, xx - yy);
    g[15] = C * dvec3(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0);
  }
  return g;
}

dvec3 position_of(const GaussianCloud &c, std::size_t i) {
  return {c.positions[i * 3], c.positions[i * 3 + 1], c.positions[i * 3 + 2]};
}

dvec4 quat_of(const GaussianCloud &c, std::size_t i, double *length = nullptr) {
  dvec4 q(c.rotations[i * 4], c.rotations[i * 4 + 1], c.rotations[i * 4 + 2], c.rotations[i * 4 + 3]);
  const double len = glm::length(q);
  if (length != nullptr) *length = len;
  return len > 0.0 ? q / len : dvec4(1.0, 0.0, 0.0, 0.0);
}

// train_preprocess.comp. Returns splats back to front.
std::vector<Splat> prepare(const GaussianCloud &cloud, const Camera &cam, const ReferenceOptions &options) {
  const glm::dmat4 view(cam.view);
  const dmat3 Rv(view);
  const double fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
  const double limX = 1.3 * 0.5 * cam.width / fx, limY = 1.3 * 0.5 * cam.height / fy;
  std::vector<Splat> splats;
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const dvec3 p = position_of(cloud, i);
    const dvec3 t = dvec3(view * glm::dvec4(p, 1.0));
    if (t.z < options.nearPlane) continue;
    const double opacity = 1.0 / (1.0 + std::exp(-static_cast<double>(cloud.opacities[i])));
    if (opacity < 1.0 / 255.0) continue;

    const dvec3 s(std::exp(static_cast<double>(cloud.scales[i * 3])), std::exp(static_cast<double>(cloud.scales[i * 3 + 1])),
                  std::exp(static_cast<double>(cloud.scales[i * 3 + 2])));
    const dmat3 M = quat_to_mat(quat_of(cloud, i)) * dmat3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z);
    const dmat3 sigma = M * glm::transpose(M);

    const double tx = std::clamp(t.x / t.z, -limX, limX) * t.z, ty = std::clamp(t.y / t.z, -limY, limY) * t.z;
    const dmat3 J(fx / t.z, 0, 0, 0, fy / t.z, 0, -fx * tx / (t.z * t.z), -fy * ty / (t.z * t.z), 0);
    const dmat3 T = J * Rv;
    const dmat3 cov = T * sigma * glm::transpose(T);
    const double a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
    const double det = a * c - b * b;
    if (det <= 0.0) continue;
    const double mid = 0.5 * (a + c);
    const double lambda = mid + std::sqrt(std::max(0.1, mid * mid - det));

    Splat out;
    out.index = i;
    out.radius = std::ceil(3.0 * std::sqrt(lambda));
    out.center = {fx * t.x / t.z + cx, fy * t.y / t.z + cy};
    if (out.center.x + out.radius < 0.0 || out.center.y + out.radius < 0.0 || out.center.x - out.radius > cam.width ||
        out.center.y - out.radius > cam.height) {
      continue;
    }
    out.depth = t.z;
    out.conic = dvec3(c, -b, a) / det;
    out.opacity = opacity;
    const auto basis = sh_basis(glm::normalize(p - dvec3(cam.position)), cloud.shDegree);
    const std::size_t coefficients = static_cast<std::size_t>(cloud.shCoeffs());
    dvec3 color(0.5);
    for (std::size_t k = 0; k < coefficients; ++k) {
      const std::size_t at = (i * coefficients + k) * 3;
      color += basis[k] * dvec3(cloud.sh[at], cloud.sh[at + 1], cloud.sh[at + 2]);
    }
    out.color = glm::max(color, dvec3(0.0));
    splats.push_back(out);
  }
  std::stable_sort(splats.begin(), splats.end(), [](const Splat &a, const Splat &b) { return a.depth > b.depth; });
  return splats;
}

// splat_at() in train_types.glsl.
bool splat_at(const Splat &s, int x, int y, dvec2 &d, double &power, double &alpha, bool noCutoffs) {
  d = dvec2(x + 0.5, y + 0.5) - s.center;
  if (!noCutoffs && (std::abs(d.x) > s.radius || std::abs(d.y) > s.radius)) return false;
  power = -0.5 * (s.conic.x * d.x * d.x + s.conic.z * d.y * d.y) - s.conic.y * d.x * d.y;
  if (power > 0.0) return false;
  alpha = std::min(0.99, s.opacity * std::exp(power));
  return noCutoffs || alpha >= 1.0 / 255.0;
}

double trace_term(const dmat2 &G, const dmat3x2 &A, const dmat3 &X, const dmat3x2 &J) {
  const dmat2 product = G * (A * X * glm::transpose(J));
  return product[0][0] + product[1][1];
}

// train_forward.comp. Adds each splat's weight in each pixel to `contributions`, if given.
ReferenceImage composite(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options, std::vector<double> *contributions) {
  ReferenceImage image;
  image.width = camera.width;
  image.height = camera.height;
  const std::size_t pixels = static_cast<std::size_t>(camera.width) * camera.height;
  image.rgb.assign(pixels, dvec3(0.0));
  image.logTransmittance.assign(pixels, 0.0);
  image.stop.assign(pixels, 0);
  std::vector<bool> done(pixels, false);
  // Front to back, so from the end of the sorted list.
  const std::vector<Splat> splats = prepare(cloud, camera, options);
  for (std::size_t j = splats.size(); j-- > 0;) {
    const Splat &s = splats[j];
    const double reach = options.noCutoffs ? 1e6 : s.radius;
    const int x0 = std::max(0, static_cast<int>(std::floor(s.center.x - reach)) - 1);
    const int x1 = std::min(camera.width - 1, static_cast<int>(std::ceil(s.center.x + reach)) + 1);
    const int y0 = std::max(0, static_cast<int>(std::floor(s.center.y - reach)) - 1);
    const int y1 = std::min(camera.height - 1, static_cast<int>(std::ceil(s.center.y + reach)) + 1);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        dvec2 d;
        double power = 0.0, alpha = 0.0;
        if (!splat_at(s, x, y, d, power, alpha, options.noCutoffs)) continue;
        const std::size_t at = static_cast<std::size_t>(y) * camera.width + x;
        if (done[at]) continue;
        const double logRest = std::log(1.0 - alpha);
        if (!options.noCutoffs && image.logTransmittance[at] + logRest < kLogMinTransmittance) {
          image.stop[at] = j + 1;
          done[at] = true;
          continue;
        }
        image.rgb[at] += std::exp(image.logTransmittance[at]) * alpha * s.color;
        if (contributions != nullptr) (*contributions)[s.index] += std::exp(image.logTransmittance[at]) * alpha;
        image.logTransmittance[at] += logRest;
      }
    }
  }
  const dvec3 background(options.background[0], options.background[1], options.background[2]);
  for (std::size_t at = 0; at < pixels; ++at) image.rgb[at] += std::exp(image.logTransmittance[at]) * background;
  return image;
}

} // namespace

ReferenceImage reference_render(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options) {
  return composite(cloud, camera, options, nullptr);
}

std::vector<double> reference_contributions(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options) {
  std::vector<double> contributions(cloud.size(), 0.0);
  (void)composite(cloud, camera, options, &contributions);
  return contributions;
}

ReferenceLoss reference_loss(const ReferenceImage &image, const std::vector<std::uint8_t> &target, double lambda) {
  const int w = image.width, h = image.height;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;
  const auto target_at = [&](std::size_t at) {
    const std::uint8_t *t = target.data() + at * 4;
    return dvec3(t[0], t[1], t[2]) / 255.0;
  };
  std::vector<dvec3> P(pixels), Q(pixels), R(pixels);
  double ssimSum = 0.0, l1Sum = 0.0;
  for (int row = 0; row < h; ++row) {
    for (int px = 0; px < w; ++px) {
      dvec3 mx(0.0), my(0.0), xx(0.0), yy(0.0), xy(0.0);
      double norm = 0.0;
      for (int v = std::max(row - 5, 0); v <= std::min(row + 5, h - 1); ++v) {
        for (int u = std::max(px - 5, 0); u <= std::min(px + 5, w - 1); ++u) {
          const double weight = kWeights[v - row + 5] * kWeights[u - px + 5];
          const std::size_t at = static_cast<std::size_t>(v) * w + u;
          const dvec3 a = image.rgb[at], b = target_at(at);
          mx += weight * a;
          my += weight * b;
          xx += weight * a * a;
          yy += weight * b * b;
          xy += weight * a * b;
          norm += weight;
        }
      }
      mx /= norm;
      my /= norm;
      const dvec3 vx = xx / norm - mx * mx, vy = yy / norm - my * my, cxy = xy / norm - mx * my;
      const dvec3 n1 = 2.0 * mx * my + kC1, n2 = 2.0 * cxy + kC2, d1 = mx * mx + my * my + kC1, d2 = vx + vy + kC2;
      const dvec3 s = (n1 * n2) / (d1 * d2);
      ssimSum += s.r + s.g + s.b;
      const dvec3 dMu = 2.0 * my * n2 / (d1 * d2) - s * 2.0 * mx / d1, dVar = -s / d2, dCov = 2.0 * n1 / (d1 * d2);
      const std::size_t at = static_cast<std::size_t>(row) * w + px;
      P[at] = (dMu - 2.0 * mx * dVar - my * dCov) / norm;
      Q[at] = dVar / norm;
      R[at] = dCov / norm;
    }
  }

  ReferenceLoss out;
  out.adjoint.resize(pixels);
  const double perValue = 1.0 / (3.0 * static_cast<double>(pixels));
  for (int row = 0; row < h; ++row) {
    for (int px = 0; px < w; ++px) {
      const std::size_t at = static_cast<std::size_t>(row) * w + px;
      const dvec3 x = image.rgb[at], y = target_at(at), difference = x - y;
      l1Sum += std::abs(difference.r) + std::abs(difference.g) + std::abs(difference.b);
      dvec3 p(0.0), q(0.0), r(0.0);
      for (int v = std::max(row - 5, 0); v <= std::min(row + 5, h - 1); ++v) {
        for (int u = std::max(px - 5, 0); u <= std::min(px + 5, w - 1); ++u) {
          const double weight = kWeights[v - row + 5] * kWeights[u - px + 5];
          const std::size_t other = static_cast<std::size_t>(v) * w + u;
          p += weight * P[other];
          q += weight * Q[other];
          r += weight * R[other];
        }
      }
      out.adjoint[at] = ((1.0 - lambda) * glm::sign(difference) - lambda * (p + 2.0 * x * q + y * r)) * perValue;
    }
  }
  out.l1 = l1Sum * perValue;
  out.ssim = ssimSum * perValue;
  out.loss = (1.0 - lambda) * out.l1 + lambda * (1.0 - out.ssim);
  return out;
}

ParameterGradients reference_gradients(const GaussianCloud &cloud, const Camera &cam, const std::vector<dvec3> &adjoint,
                                       const ReferenceOptions &options) {
  ReferenceImage total = reference_render(cloud, cam, options);
  const std::size_t pixels = static_cast<std::size_t>(cam.width) * cam.height;
  std::vector<dvec3> behind(pixels, dvec3(options.background[0], options.background[1], options.background[2]));
  std::vector<double> logBehind(pixels, 0.0);

  const std::size_t n = cloud.size();
  ParameterGradients out;
  out.positions.assign(n * 3, 0.0);
  out.scales.assign(n * 3, 0.0);
  out.rotations.assign(n * 4, 0.0);
  out.opacities.assign(n, 0.0);
  out.sh.assign(cloud.sh.size(), 0.0);

  // train_backward.glsl
  std::vector<std::array<double, 9>> raw(n, std::array<double, 9>{});
  const std::vector<Splat> splats = prepare(cloud, cam, options);
  for (std::size_t j = 0; j < splats.size(); ++j) {
    const Splat &s = splats[j];
    auto &g = raw[s.index];
    const double reach = options.noCutoffs ? 1e6 : s.radius;
    const int x0 = std::max(0, static_cast<int>(std::floor(s.center.x - reach)) - 1);
    const int x1 = std::min(cam.width - 1, static_cast<int>(std::ceil(s.center.x + reach)) + 1);
    const int y0 = std::max(0, static_cast<int>(std::floor(s.center.y - reach)) - 1);
    const int y1 = std::min(cam.height - 1, static_cast<int>(std::ceil(s.center.y + reach)) + 1);
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        dvec2 d;
        double power = 0.0, alpha = 0.0;
        if (!splat_at(s, x, y, d, power, alpha, options.noCutoffs)) continue;
        const std::size_t at = static_cast<std::size_t>(y) * cam.width + x;
        if (j < total.stop[at]) continue;
        const double logRest = std::log(1.0 - alpha);
        const double T = std::exp(total.logTransmittance[at] - logBehind[at] - logRest);
        const dvec3 A = adjoint[at], B = behind[at];
        const dvec3 dColour = T * alpha * A;
        g[6] += dColour.r;
        g[7] += dColour.g;
        g[8] += dColour.b;
        if (s.opacity * std::exp(power) < 0.99) {
          const double dAlpha = T * glm::dot(A, s.color - B);
          g[0] += dAlpha * std::exp(power);
          const double dPower = dAlpha * alpha;
          const dvec2 v(s.conic.x * d.x + s.conic.y * d.y, s.conic.y * d.x + s.conic.z * d.y);
          g[1] += dPower * v.x;
          g[2] += dPower * v.y;
          g[3] += dPower * 0.5 * v.x * v.x;
          g[4] += dPower * 0.5 * v.x * v.y;
          g[5] += dPower * 0.5 * v.y * v.y;
        }
        behind[at] = glm::mix(B, s.color, alpha);
        logBehind[at] += logRest;
      }
    }
  }

  // train_gradient.comp
  const glm::dmat4 view(cam.view);
  const dmat3 Rv(view);
  const double fx = cam.fx, fy = cam.fy;
  const double limX = 1.3 * 0.5 * cam.width / fx, limY = 1.3 * 0.5 * cam.height / fy;
  for (std::size_t i = 0; i < n; ++i) {
    const auto &g = raw[i];
    if (std::all_of(g.begin(), g.end(), [](double v) { return v == 0.0; })) continue;
    const dvec3 p = position_of(cloud, i);
    double quatLength = 1.0;
    const dvec4 quat = quat_of(cloud, i, &quatLength);
    const dvec3 t = dvec3(view * glm::dvec4(p, 1.0));
    const double z = t.z, iz = 1.0 / z, iz2 = iz * iz;
    const dvec2 gCentre(g[1], g[2]);
    const dmat2 G(g[3], g[4], g[4], g[5]);
    const dmat3 R = quat_to_mat(quat);
    const dvec3 s2(std::exp(2.0 * static_cast<double>(cloud.scales[i * 3])), std::exp(2.0 * static_cast<double>(cloud.scales[i * 3 + 1])),
                   std::exp(2.0 * static_cast<double>(cloud.scales[i * 3 + 2])));
    const dmat3 D(s2.x, 0, 0, 0, s2.y, 0, 0, 0, s2.z);
    const dmat3 X = Rv * (R * D * glm::transpose(R)) * glm::transpose(Rv);

    const double rx = t.x * iz, ry = t.y * iz;
    const bool clampedX = std::abs(rx) > limX, clampedY = std::abs(ry) > limY;
    const double tx = std::clamp(rx, -limX, limX) * z, ty = std::clamp(ry, -limY, limY) * z;
    const dmat3x2 J(dvec2(fx * iz, 0.0), dvec2(0.0, fy * iz), dvec2(-fx * tx * iz2, -fy * ty * iz2));
    const dmat3x2 Jc(dvec2(fx * iz, 0.0), dvec2(0.0, fy * iz), dvec2(-fx * t.x * iz2, -fy * t.y * iz2));
    dvec3 gT = glm::transpose(Jc) * gCentre;
    const dmat3x2 dJx(dvec2(0.0), dvec2(0.0), dvec2(clampedX ? 0.0 : -fx * iz2, 0.0));
    const dmat3x2 dJy(dvec2(0.0), dvec2(0.0), dvec2(0.0, clampedY ? 0.0 : -fy * iz2));
    const dmat3x2 dJz(dvec2(-fx * iz2, 0.0), dvec2(0.0, -fy * iz2),
                      dvec2(clampedX ? fx * tx * iz2 * iz : 2.0 * fx * tx * iz2 * iz,
                            clampedY ? fy * ty * iz2 * iz : 2.0 * fy * ty * iz2 * iz));
    gT += 2.0 * dvec3(trace_term(G, dJx, X, J), trace_term(G, dJy, X, J), trace_term(G, dJz, X, J));
    dvec3 gP = glm::transpose(Rv) * gT;

    const dmat3x2 T = J * Rv;
    const dmat3 G3 = glm::transpose(T) * G * T;
    for (int a = 0; a < 3; ++a) out.scales[i * 3 + a] = 2.0 * s2[a] * glm::dot(R[a], G3 * R[a]);

    const dmat3 P = 2.0 * G3 * R * D;
    const double w = quat.x, x = quat.y, y = quat.z, zq = quat.w;
    const dmat3 dR[4] = {dmat3(0.0, 2.0 * zq, -2.0 * y, -2.0 * zq, 0.0, 2.0 * x, 2.0 * y, -2.0 * x, 0.0),
                         dmat3(0.0, 2.0 * y, 2.0 * zq, 2.0 * y, -4.0 * x, 2.0 * w, 2.0 * zq, -2.0 * w, -4.0 * x),
                         dmat3(-4.0 * y, 2.0 * x, -2.0 * w, 2.0 * x, 0.0, 2.0 * zq, 2.0 * w, 2.0 * zq, -4.0 * y),
                         dmat3(-4.0 * zq, 2.0 * w, 2.0 * x, -2.0 * w, -4.0 * zq, 2.0 * y, 2.0 * x, 2.0 * y, 0.0)};
    dvec4 gq(0.0);
    for (int k = 0; k < 4; ++k) gq[k] = glm::dot(P[0], dR[k][0]) + glm::dot(P[1], dR[k][1]) + glm::dot(P[2], dR[k][2]);
    const dvec4 tangent = (gq - quat * glm::dot(quat, gq)) / std::max(quatLength, 1e-12);
    for (int k = 0; k < 4; ++k) out.rotations[i * 4 + k] = tangent[k];

    const double opacity = 1.0 / (1.0 + std::exp(-static_cast<double>(cloud.opacities[i])));
    out.opacities[i] = g[0] * opacity * (1.0 - opacity);

    const dvec3 gColour(g[6], g[7], g[8]);
    const auto basis = sh_basis(glm::normalize(p - dvec3(cam.position)), cloud.shDegree);
    const std::size_t coefficients = static_cast<std::size_t>(cloud.shCoeffs());
    dvec3 colour(0.5);
    for (std::size_t k = 0; k < coefficients; ++k) {
      const std::size_t at = (i * coefficients + k) * 3;
      colour += basis[k] * dvec3(cloud.sh[at], cloud.sh[at + 1], cloud.sh[at + 2]);
    }
    // Position also changes the view direction, and with it the colour.
    if (cloud.shDegree > 0) {
      const dvec3 toSplat = p - dvec3(cam.position);
      const double distance = glm::length(toSplat);
      const dvec3 dir = toSplat / distance;
      const auto basisGradient = sh_basis_gradient(dir, cloud.shDegree);
      const dvec3 unclamped(colour.r > 0.0 ? 1.0 : 0.0, colour.g > 0.0 ? 1.0 : 0.0, colour.b > 0.0 ? 1.0 : 0.0);
      dvec3 gDir(0.0);
      for (std::size_t k = 1; k < coefficients; ++k) {
        const std::size_t at = (i * coefficients + k) * 3;
        gDir += glm::dot(gColour * unclamped, dvec3(cloud.sh[at], cloud.sh[at + 1], cloud.sh[at + 2])) * basisGradient[k];
      }
      gP += (gDir - dir * glm::dot(dir, gDir)) / distance;
    }
    for (int a = 0; a < 3; ++a) out.positions[i * 3 + a] = gP[a];

    for (std::size_t k = 0; k < coefficients; ++k) {
      const std::size_t at = (i * coefficients + k) * 3;
      for (int c = 0; c < 3; ++c) out.sh[at + static_cast<std::size_t>(c)] = colour[c] > 0.0 ? basis[k] * gColour[c] : 0.0;
    }
  }
  return out;
}

} // namespace splat
