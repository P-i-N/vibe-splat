#pragma once

#include "splat/gaussian_cloud.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace splat {

// CGC (Compressed Gaussian Cloud): a lossy, compact file format for viewing
// splats over the network. Splats are Morton-ordered into chunks of 256,
// quantized, stored column by column and compressed with LZMA. Each splat
// keeps only the SH bands that visibly change its colour. The byte layout is
// described in docs/superpowers/specs/2026-11-02-cgc-format-design.md.

struct CgcOptions {
  // A band is dropped when it changes the colour by less than this (RMS over
  // view directions, scaled by opacity, in 0..1 colour units).
  float shThreshold = 1.5f / 255.f;
  int lzmaLevel = 9; // liblzma preset 0..9
  bool lzmaExtreme = true;
};

struct CgcStats {
  std::uint64_t bytes = 0;                    // size of the written file
  std::array<std::size_t, 4> degreeCounts{}; // splats stored at SH degree 0..3
};

// Writes `cloud` as CGC. Throws std::runtime_error, writing nothing, if any
// value is NaN or infinite, or if the file cannot be written.
CgcStats save_cgc(const GaussianCloud &cloud, const std::filesystem::path &path, const CgcOptions &options = {});

// Reads a CGC file. The cloud's SH degree is the highest any splat uses;
// splats with fewer bands get zeros. Splats come back in Morton order.
// Throws std::runtime_error on malformed files.
[[nodiscard]] GaussianCloud load_cgc(const std::filesystem::path &path);

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
