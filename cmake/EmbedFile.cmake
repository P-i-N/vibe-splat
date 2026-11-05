# Script mode: turns any file into a C++ source with a byte array.
# Inputs: INPUT (file path), OUTPUT (cpp path), SYMBOL (C++ identifier).
# Declares `extern const std::span<const std::uint8_t> splat::embedded::<SYMBOL>;`.
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "(([^,]*,){16})" "\\1\n  " bytes "${bytes}")

file(WRITE "${OUTPUT}"
"// Generated from ${INPUT}. Do not edit.
#include <cstdint>
#include <span>

namespace splat::embedded {
namespace {
constexpr std::uint8_t data[] = {
  ${bytes}
};
} // namespace
extern const std::span<const std::uint8_t> ${SYMBOL};
const std::span<const std::uint8_t> ${SYMBOL}{data};
} // namespace splat::embedded
")
