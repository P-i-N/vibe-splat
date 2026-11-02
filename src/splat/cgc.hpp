#pragma once

#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splat {

// Pieces of the CGC encoder, exposed for unit tests.
namespace cgc_detail {

// Interleaves the low 21 bits of x, y, z: x into bit 0, y into bit 1, z into
// bit 2, and so on.
[[nodiscard]] std::uint64_t morton_code(std::uint32_t x, std::uint32_t y, std::uint32_t z);

// "Smallest three" quaternion packing (w, x, y, z): index of the dropped
// largest component in bits 30-31, the other three in 10 bits each. Input
// need not be normalised; a zero quaternion packs as the identity.
[[nodiscard]] std::uint32_t pack_rotation(std::array<float, 4> wxyz);
[[nodiscard]] std::array<float, 4> unpack_rotation(std::uint32_t packed);

// The lowest SH degree whose dropped bands each change the colour by less
// than `threshold` (RMS over all view directions, scaled by opacity).
// `sh` holds one Gaussian's 3 * (degree + 1)^2 coefficients in
// GaussianCloud order.
[[nodiscard]] int choose_sh_degree(std::span<const float> sh, int degree, float opacityLogit, float threshold);

// Appends 32-bit values as four byte planes, least significant first.
void append_planes(std::vector<std::byte> &out, std::span<const std::uint32_t> values);
[[nodiscard]] std::vector<std::uint32_t> read_planes(std::span<const std::byte> in, std::size_t count);

} // namespace cgc_detail

} // namespace splat
