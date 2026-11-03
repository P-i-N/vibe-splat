#pragma once

#include "train/trainer.hpp"

#include <cstdint>
#include <vector>

namespace splat {

// Peak signal to noise ratio of a render from Trainer::render() against an RGBA8 photo.
[[nodiscard]] double psnr_of(const std::vector<float> &render, const std::vector<std::uint8_t> &target);

// Applies an exposure in place to the rgb of a render from Trainer::render().
void apply_exposure(std::vector<float> &render, const Exposure &exposure);

// The exposure that best turns the render's columns [x0, x1) into the photo's, by least squares.
[[nodiscard]] Exposure fit_exposure(const std::vector<float> &render, const std::vector<std::uint8_t> &target, int width, int height, int x0, int x1);

// PSNR with the exposure fitted on one half of the image and scored on the
// other, then the other way round, so that every pixel is scored by a fit it
// took no part in. The usual way (NeRF in the Wild) to score held-out views of
// a model that learns each photo's exposure, since it never saw theirs.
[[nodiscard]] double cross_fitted_psnr(const std::vector<float> &render, const std::vector<std::uint8_t> &target, int width, int height);

} // namespace splat
