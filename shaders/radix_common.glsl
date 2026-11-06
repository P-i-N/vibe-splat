// Shared constants for the 8-bit-per-pass LSD radix sort.
#ifndef RADIX_COMMON_GLSL
#define RADIX_COMMON_GLSL

#define RADIX_THREADS 256u
#define RADIX_KEYS_PER_THREAD 16u
#define RADIX_TILE (RADIX_THREADS * RADIX_KEYS_PER_THREAD) // 4096 keys per workgroup
#define RADIX_BUCKETS 256u

layout(push_constant) uniform RadixPush {
  uint count;    // Number of keys
  uint shift;    // Bit offset of this pass' digit
  uint numTiles; // ceil(count / RADIX_TILE)
} pc;

#endif
