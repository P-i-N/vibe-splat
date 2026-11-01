#pragma once

#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace splat {

// CPU versions of every training kernel, written to be read next to the GLSL.
// Slow and in double precision. The tests check these against finite
// differences, then check the GPU against these.

struct ReferenceOptions {
  std::array<float, 3> background{0.f, 0.f, 0.f};
  float nearPlane = 0.01f;
  // Leaves out the cutoffs at 3 sigma and alpha 1/255. They are steps, which
  // finite differences see and true derivatives do not, so checking gradients
  // needs them gone.
  bool noCutoffs = false;
};

struct ReferenceImage {
  int width = 0;
  int height = 0;
  std::vector<glm::dvec3> rgb;
  std::vector<double> logTransmittance;
};

[[nodiscard]] ReferenceImage reference_render(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options);

struct ReferenceLoss {
  double loss = 0.0;
  double l1 = 0.0;   // Mean absolute difference
  double ssim = 0.0; // Mean SSIM
  std::vector<glm::dvec3> adjoint; // dLoss/dImage
};

// L = (1 - lambda) L1 + lambda (1 - SSIM). `target` is RGBA8.
[[nodiscard]] ReferenceLoss reference_loss(const ReferenceImage &image, const std::vector<std::uint8_t> &target, double lambda);

struct ParameterGradients {
  std::vector<double> positions, scales, rotations, opacities, sh;
};

// Gradients of the scalar sum(adjoint * image) with respect to all parameters.
[[nodiscard]] ParameterGradients reference_gradients(const GaussianCloud &cloud, const Camera &camera,
                                                     const std::vector<glm::dvec3> &adjoint, const ReferenceOptions &options);

} // namespace splat
