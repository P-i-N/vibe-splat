# Script mode: turns a .spv file into a C++ source with a uint32_t array.
# Inputs: INPUT (spv path), OUTPUT (cpp path), SYMBOL (C++ identifier).
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR rem "${hex_len} % 8")
if(NOT rem EQUAL 0 OR hex_len EQUAL 0)
  message(FATAL_ERROR "${INPUT} is not a valid SPIR-V module")
endif()

# SPIR-V is little-endian 32-bit words; reorder bytes per word.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," words "${hex}")
string(REGEX REPLACE "(([^,]*,){8})" "\\1\n  " words "${words}")

file(WRITE "${OUTPUT}"
"// Generated from ${INPUT}. Do not edit.
#include <cstdint>
#include <span>

namespace splat::shaders {
namespace {
alignas(4) constexpr std::uint32_t data[] = {
  ${words}
};
} // namespace
extern const std::span<const std::uint32_t> ${SYMBOL};
const std::span<const std::uint32_t> ${SYMBOL}{data};
} // namespace splat::shaders
")
