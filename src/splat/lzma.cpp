#include "splat/lzma.hpp"

#include <lzma.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace splat {

namespace {

struct Stream {
  lzma_stream s = LZMA_STREAM_INIT;
  ~Stream() { lzma_end(&s); }
};

// Feeds all of `input` through an initialised stream, growing `out` as needed.
void run(lzma_stream &s, std::span<const std::byte> input, std::vector<std::byte> &out, std::size_t limit) {
  s.next_in = reinterpret_cast<const std::uint8_t *>(input.data());
  s.avail_in = input.size();
  for (;;) {
    if (s.avail_out == 0) {
      const std::size_t used = out.size();
      if (used >= limit) {
        throw std::runtime_error("LZMA data is larger than expected");
      }
      out.resize(std::min(limit, std::max<std::size_t>(used * 2, 1 << 16)));
      s.next_out = reinterpret_cast<std::uint8_t *>(out.data()) + used;
      s.avail_out = out.size() - used;
    }
    const lzma_ret ret = lzma_code(&s, LZMA_FINISH);
    if (ret == LZMA_STREAM_END) break;
    if (ret == LZMA_BUF_ERROR) throw std::runtime_error("LZMA data is truncated");
    if (ret != LZMA_OK) throw std::runtime_error(std::format("LZMA error {}", static_cast<int>(ret)));
  }
  out.resize(out.size() - s.avail_out);
}

} // namespace

std::vector<std::byte> lzma_compress(std::span<const std::byte> data, int level, bool extreme) {
  lzma_options_lzma options;
  if (lzma_lzma_preset(&options, static_cast<std::uint32_t>(level) | (extreme ? LZMA_PRESET_EXTREME : 0u))) {
    throw std::runtime_error(std::format("Invalid LZMA preset {}", level));
  }
  Stream stream;
  if (lzma_alone_encoder(&stream.s, &options) != LZMA_OK) {
    throw std::runtime_error("Cannot start the LZMA encoder");
  }
  std::vector<std::byte> out;
  stream.s.avail_out = 0;
  run(stream.s, data, out, SIZE_MAX);
  return out;
}

std::vector<std::byte> lzma_decompress(std::span<const std::byte> stream, std::size_t expectedSize) {
  Stream decoder;
  if (lzma_alone_decoder(&decoder.s, UINT64_MAX) != LZMA_OK) {
    throw std::runtime_error("Cannot start the LZMA decoder");
  }
  std::vector<std::byte> out;
  out.reserve(expectedSize);
  // One byte of headroom, so that data longer than expected is detected.
  run(decoder.s, stream, out, expectedSize + 1);
  if (out.size() != expectedSize) {
    throw std::runtime_error(std::format("LZMA data has {} bytes, expected {}", out.size(), expectedSize));
  }
  return out;
}

} // namespace splat
