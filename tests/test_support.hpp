#pragma once

#include "core/vk_context.hpp"
#include "render/offscreen.hpp"

namespace splat::test {

// One shared Vulkan context for all tests (created on first use, with
// validation when the layer is installed). Returns nullptr when no usable
// device exists, so GPU tests can SKIP. CI uses Mesa's lavapipe.
vk::Context *context();

// PSNR over the RGB channels of two renders of the same size, 99 dB when equal.
double psnr(const RenderedImage &a, const RenderedImage &b);

} // namespace splat::test
