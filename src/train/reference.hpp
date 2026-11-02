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
  // Leaves out the cutoffs at 3 sigma and alpha 1/255, and the early stop at
  // transmittance 1e-4. They are steps, which finite differences see and true
  // derivatives do not, so checking gradients needs them gone.
  bool noCutoffs = false;
};

struct ReferenceImage {
  int width = 0;
  int height = 0;
  std::vector<glm::dvec3> rgb;
  std::vector<double> logTransmittance;
  // Per pixel, the splats of the back-to-front list below this index were left
  // out by the early stop.
  std::vector<std::size_t> stop;
};

[[nodiscard]] ReferenceImage reference_render(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options);

// Per Gaussian, its blending weight T alpha summed over the pixels, which the
// backward pass adds up for pruning.
[[nodiscard]] std::vector<double> reference_contributions(const GaussianCloud &cloud, const Camera &camera, const ReferenceOptions &options);

// Per Gaussian, how many pixels it reaches whose mean error is at least `share`
// of the image's worst, as Trainer::score_view() counts them. `target` is RGBA8.
[[nodiscard]] std::vector<double> reference_marked_pixels(const GaussianCloud &cloud, const Camera &camera, const std::vector<std::uint8_t> &target,
                                                          double share, const ReferenceOptions &options);

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
  std::vector<double> absoluteCentre; // 2 per Gaussian: |dL/dcentre| summed per pixel, so nothing cancels
};

// Gradients of the scalar sum(adjoint * image) with respect to all parameters.
[[nodiscard]] ParameterGradients reference_gradients(const GaussianCloud &cloud, const Camera &camera,
                                                     const std::vector<glm::dvec3> &adjoint, const ReferenceOptions &options);

} // namespace splat
