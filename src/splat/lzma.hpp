#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace splat {

// Compresses `data` into a complete .lzma ("alone" format) stream: a 13-byte
// header (properties, dictionary size, size written as unknown) followed by
// LZMA1 data ending with an end marker. `level` is the liblzma preset 0..9.
[[nodiscard]] std::vector<std::byte> lzma_compress(std::span<const std::byte> data, int level = 9, bool extreme = true);

// Decompresses a .lzma stream. Throws std::runtime_error on corrupt or
// truncated input, or when the result is not exactly `expectedSize` bytes.
[[nodiscard]] std::vector<std::byte> lzma_decompress(std::span<const std::byte> stream, std::size_t expectedSize);

} // namespace splat
