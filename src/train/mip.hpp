#pragma once

#include "render/camera.hpp"
#include "splat/gaussian_cloud.hpp"

#include <span>
#include <vector>

namespace splat {

// Mip-Splatting's 3D smoothing filter: for each Gaussian, (0.2 * depth / focal)^2
// at the camera that sees it with the smallest pixel footprint, and 0 if none
// sees it (in front of the camera and inside its image). The focal is the
// larger of fx and fy. Added to the covariance on all three axes, with the
// opacity scaled to keep the integral, it caps the frequencies a Gaussian can
// carry at what the photos sampled.
[[nodiscard]] std::vector<float> smoothing_filter(const GaussianCloud &cloud, std::span<const Camera> cameras);

// Folds the filter into the cloud, for saving: each scale becomes
// 0.5 log(exp(2 s) + f) and the opacity is scaled by prod sqrt(exp(2 s) / (exp(2 s) + f)).
void bake_filter(GaussianCloud &cloud, std::span<const float> filter);

} // namespace splat
