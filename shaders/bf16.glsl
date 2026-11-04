// bfloat16: the upper half of a float, so a float's range with 8 bits of
// precision. Stored two to a word, element 2i in the low half of word i.
#ifndef BF16_GLSL
#define BF16_GLSL

float bf16_load(uint word, uint slot) {
  return uintBitsToFloat(slot == 0u ? word << 16 : word & 0xFFFF0000u);
}

// To nearest, ties to even.
uint bf16_round(float x) {
  const uint bits = floatBitsToUint(x);
  return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

// Up or down at random, in proportion to how near each is, so that on average
// it is exact: a value that each step moves by less than its rounding still
// moves. The randomness is PCG's output permutation of the index and seed.
uint bf16_stochastic(float x, uint index, uint seed) {
  const uint state = (index ^ (seed * 0x9E3779B9u)) * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (floatBitsToUint(x) + (((word >> 22u) ^ word) & 0xFFFFu)) >> 16;
}

uint bf16_store(uint word, uint slot, uint bits) {
  return slot == 0u ? (word & 0xFFFF0000u) | bits : (word & 0xFFFFu) | (bits << 16);
}

#endif
