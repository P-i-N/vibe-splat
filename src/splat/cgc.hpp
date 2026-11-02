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
// splats over the network. Splats are split into three layers by importance,
// each with its own precision. Within a layer they are Morton-sorted on a global
// grid, rotations are canonicalized, higher SH bands share a codebook, and every
// byte column is its own LZMA stream. The byte layout is described in
// docs/superpowers/specs/2026-11-02-cgc-v2-format.md.

enum class CgcPreset {
  Balanced, // about 29x smaller than PLY, about 42 dB against the original
  Small,    // about 32x smaller, about 41 dB, with weaker close-ups
};

struct CgcOptions {
  CgcPreset preset = CgcPreset::Balanced;
  // Much quicker and a little larger, for training checkpoints.
  bool fast = false;
};

struct CgcStats {
  std::uint64_t bytes = 0;                // size of the written file
  std::array<std::size_t, 3> layerCounts{}; // splats per layer, most important first
};

// Writes `cloud` as CGC. Throws std::runtime_error, writing nothing, if any
// value is NaN or infinite, or if the file cannot be written.
CgcStats save_cgc(const GaussianCloud &cloud, const std::filesystem::path &path, const CgcOptions &options = {});

// Reads a CGC file of version 2, or an older version 1 file. Splats come back
// layer by layer, each layer in Morton order; a splat's axes may be permuted
// and flipped, which describes the same ellipsoid.
// Throws std::runtime_error on malformed files.
[[nodiscard]] GaussianCloud load_cgc(const std::filesystem::path &path);

// Pieces of the CGC encoder, exposed for unit tests.
namespace cgc_detail {

// Interleaves the low 21 bits of x, y, z: x into bit 0, y into bit 1, z into
// bit 2, and so on.
[[nodiscard]] std::uint64_t morton_code(std::uint32_t x, std::uint32_t y, std::uint32_t z);

// Rewrites a splat's axes so that log scales descend, and its quaternion
// (w, x, y, z) so that w is the largest component and not negative. The
// ellipsoid stays the same. A zero quaternion counts as the identity.
void canonicalize_axes(std::array<float, 3> &logScales, std::array<float, 4> &wxyz);

} // namespace cgc_detail

} // namespace splat
