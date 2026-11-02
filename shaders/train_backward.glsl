// Backward pass, step 1: for each tile, in the same back-to-front order as the
// forward pass, adds up what each splat contributes to dL. At a pixel,
//   C = F + T (alpha c + (1 - alpha) B)
// where B is the composite behind this splat and T the transmittance in front
// of it, which gives
//   dC/dalpha = T (c - B)    dC/dc = T alpha
// B is rebuilt as we go. T comes from the log-transmittances, so it needs no
// division by (1 - alpha), which would amplify rounding errors at every layer.
//
// One workgroup per tile and one thread per pixel, as in the forward pass.
// Each splat's sums over the tile's pixels are reduced within each subgroup,
// and one lane per subgroup adds them to the splat's totals.
//
// Each splat collects 9 values, indexed as in train_gradient.comp:
//   0: dL/dopacity, 1-2: dL/dcentre, 3-5: dL/dcov (xx, xy, yy), 6-8: dL/dcolour
#include "train_types.glsl"

layout(local_size_x = 16, local_size_y = 16) in; // TILE_SIZE

layout(std430, binding = 0) readonly buffer Splats { TrainSplat splats[]; };
layout(std430, binding = 1) readonly buffer EntryValues { uint entryValues[]; };
layout(std430, binding = 2) readonly buffer Ranges { uvec2 ranges[]; };
layout(std430, binding = 3) readonly buffer Image { vec4 image[]; };
layout(std430, binding = 4) readonly buffer Adjoint { vec4 adjoint[]; };

layout(push_constant) uniform Push {
  uint width;
  uint height;
  uint gridW;
  uint tileCount;
  vec4 background;
} pc;

shared TrainSplat batch[TILE_BATCH];
shared uint batchIndex[TILE_BATCH];

// Sums the 9 values over the subgroup and adds them to the splat's totals.
// A butterfly over the lanes that halves the number of values per lane at
// each step (padded to 16), so that it costs 16 shuffles for 32 lanes rather
// than the 45 of nine separate reductions. At the end lane l holds the total
// of value bitreverse(l), and up to 9 lanes add their totals at once. Every
// lane of the subgroup must take part.
void reduce_and_add(uint base, float g[9]) {
  float v[16] = float[16](g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const uint lane = gl_SubgroupInvocationID;
  // Lanes with the bit clear keep the lower half of the list and receive the
  // partner's lower half; the others keep and receive the upper half.
#define HALVE(offset, count)                                                             {                                                                                        const bool upper = (lane & (offset)) != 0u;                                            for (uint i = 0u; i < (count) / 2u; ++i) {                                               const float send = upper ? v[i] : v[i + (count) / 2u];                                 const float keep = upper ? v[i + (count) / 2u] : v[i];                                 v[i] = keep + subgroupShuffleXor(send, (offset));                                    }                                                                                    }
  HALVE(1u, 16u)
  if (gl_SubgroupSize >= 4u) HALVE(2u, 8u)
  if (gl_SubgroupSize >= 8u) HALVE(4u, 4u)
  if (gl_SubgroupSize >= 16u) HALVE(8u, 2u)
#undef HALVE
  if (gl_SubgroupSize >= 32u) v[0] += subgroupShuffleXor(v[0], 16u);
  if (gl_SubgroupSize >= 64u) v[0] += subgroupShuffleXor(v[0], 32u);
  if (gl_SubgroupSize >= 128u) v[0] += subgroupShuffleXor(v[0], 64u);

  // After k halving steps a lane holds 16 >> k values, numbered from the
  // bit-reversed low k bits of the lane shifted up by 4 - k.
  const uint steps = uint(min(findMSB(int(gl_SubgroupSize)), 4));
  const uint perLane = 16u >> steps;
  if (lane < 16u) {
    const uint first = (bitfieldReverse(lane) >> (32u - steps)) << (4u - steps);
    for (uint i = 0u; i < 16u; ++i) {
      if (i < perLane && first + i < 9u) add_gradient(base + first + i, v[i]);
    }
  }
}

void main() {
  const uint tile = gl_WorkGroupID.x;
  const uint tx = tile % pc.gridW, ty = tile / pc.gridW;
  const uvec4 tilePixels = uvec4(tx * TILE_SIZE, ty * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, pc.width),
                                 min(ty * TILE_SIZE + TILE_SIZE, pc.height));
  const uint x = tilePixels.x + gl_LocalInvocationID.x, y = tilePixels.y + gl_LocalInvocationID.y;
  const bool inside = x < pc.width && y < pc.height;
  const uint at = inside ? y * pc.width + x : 0u;
  const float logTotal = inside ? image[at].a : 0.0;
  const vec3 A = inside ? adjoint[at].rgb : vec3(0.0);

  // The composite behind the current splat, and its log transmittance.
  vec3 B = pc.background.rgb;
  float logBehind = 0.0;

  const uvec2 range = ranges[tile];
  for (uint start = range.x; start < range.y; start += TILE_BATCH) {
    const uint count = min(TILE_BATCH, range.y - start);
    barrier();
    if (gl_LocalInvocationIndex < count) {
      const uint index = entryValues[start + gl_LocalInvocationIndex];
      batchIndex[gl_LocalInvocationIndex] = index;
      batch[gl_LocalInvocationIndex] = splats[index];
    }
    barrier();
    for (uint k = 0u; k < count; ++k) {
      const TrainSplat s = batch[k];
      float g[9] = float[9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      bool any = false;
      vec2 d;
      float power, alpha;
      const uvec4 area = splat_pixels(s, tilePixels);
      if (inside && x >= area.x && x < area.z && y >= area.y && y < area.w && splat_at(s, x, y, d, power, alpha)) {
        const vec3 q = s.conicOpacity.xyz;
        const float logRest = log(1.0 - alpha);
        const float T = exp(logTotal - logBehind - logRest);

        const vec3 dColour = T * alpha * A;
        g[6] = dColour.r;
        g[7] = dColour.g;
        g[8] = dColour.b;

        // At the 0.99 clamp, alpha no longer depends on the parameters.
        if (s.conicOpacity.w * exp(power) < 0.99) {
          const float dAlpha = T * dot(A, s.color.rgb - B);
          g[0] = dAlpha * exp(power);
          const float dPower = dAlpha * alpha;
          const vec2 v = vec2(q.x * d.x + q.y * d.y, q.y * d.x + q.z * d.y);
          g[1] = dPower * v.x;
          g[2] = dPower * v.y;
          g[3] = dPower * 0.5 * v.x * v.x;
          g[4] = dPower * 0.5 * v.x * v.y;
          g[5] = dPower * 0.5 * v.y * v.y;
        }
        any = true;

        B = mix(B, s.color.rgb, alpha);
        logBehind += logRest;
      }
      // Uniform within the subgroup, as the reduction must be.
      if (subgroupAny(any)) {
        reduce_and_add(batchIndex[k] * 9u, g);
      }
    }
  }
}
