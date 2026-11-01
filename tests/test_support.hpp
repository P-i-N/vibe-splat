#pragma once

#include "core/vk_context.hpp"

namespace splat::test {

// One shared Vulkan context for all tests (created on first use, with
// validation when the layer is installed). Returns nullptr when no usable
// device exists, so GPU tests can SKIP. CI uses Mesa's lavapipe.
vk::Context *context();

} // namespace splat::test
