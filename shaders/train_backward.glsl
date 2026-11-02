// Backward pass, step 1: for each tile, in the same back-to-front order as the
// forward pass, adds up what each splat contributes to dL. At a pixel,
//   C = F + T (alpha c + (1 - alpha) B)
// where B is the composite behind this splat and T the transmittance in front
// of it, which gives
//   dC/dalpha = T (c - B)    dC/dc = T alpha
// B is rebuilt as we go. T comes from the log-transmittances, so it needs no
// division by (1 - alpha), which would amplify rounding errors at every layer.
//
// Each splat collects 9 values, indexed as in train_gradient.comp:
//   0: dL/dopacity, 1-2: dL/dcentre, 3-5: dL/dcov (xx, xy, yy), 6-8: dL/dcolour
#include "train_types.glsl"

layout(local_size_x = 16) in;

layout(std430, binding = 0) readonly buffer Splats { TrainSplat splats[]; };
layout(std430, binding = 1) readonly buffer EntryValues { uint entryValues[]; };
layout(std430, binding = 2) readonly buffer Ranges { uvec2 ranges[]; };
layout(std430, binding = 3) readonly buffer Image { vec4 image[]; };
layout(std430, binding = 4) readonly buffer Adjoint { vec4 adjoint[]; };

// rgb: composite so far, a: its log transmittance. Tiles do not share pixels.
layout(std430, binding = BEHIND_BINDING) buffer Behind { vec4 behind[]; };

layout(push_constant) uniform Push {
  uint width;
  uint height;
  uint gridW;
  uint tileCount;
  vec4 background;
} pc;

void main() {
  const uint tile = gl_GlobalInvocationID.x;
  if (tile >= pc.tileCount) {
    return;
  }
  const uint tx = tile % pc.gridW, ty = tile / pc.gridW;
  const uint x0 = tx * TILE_SIZE, y0 = ty * TILE_SIZE;
  const uint x1 = min(x0 + TILE_SIZE, pc.width), y1 = min(y0 + TILE_SIZE, pc.height);

  for (uint y = y0; y < y1; ++y) {
    for (uint x = x0; x < x1; ++x) {
      behind[y * pc.width + x] = vec4(pc.background.rgb, 0.0);
    }
  }

  const uvec2 range = ranges[tile];
  for (uint e = range.x; e < range.y; ++e) {
    const uint index = entryValues[e];
    const TrainSplat s = splats[index];
    const vec3 q = s.conicOpacity.xyz;
    float g[9] = float[9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    bool any = false;
    const uvec4 area = splat_pixels(s, uvec4(x0, y0, x1, y1));
    for (uint y = area.y; y < area.w; ++y) {
      for (uint x = area.x; x < area.z; ++x) {
        vec2 d;
        float power, alpha;
        if (!splat_at(s, x, y, d, power, alpha)) {
          continue;
        }
        const uint at = y * pc.width + x;
        const vec4 state = behind[at];
        const float logRest = log(1.0 - alpha);
        const float T = exp(image[at].a - state.a - logRest);
        const vec3 A = adjoint[at].rgb;
        const vec3 B = state.rgb;

        const vec3 dColour = T * alpha * A;
        g[6] += dColour.r;
        g[7] += dColour.g;
        g[8] += dColour.b;

        // At the 0.99 clamp, alpha no longer depends on the parameters.
        if (s.conicOpacity.w * exp(power) < 0.99) {
          const float dAlpha = T * dot(A, s.color.rgb - B);
          g[0] += dAlpha * exp(power);
          const float dPower = dAlpha * alpha;
          const vec2 v = vec2(q.x * d.x + q.y * d.y, q.y * d.x + q.z * d.y);
          g[1] += dPower * v.x;
          g[2] += dPower * v.y;
          g[3] += dPower * 0.5 * v.x * v.x;
          g[4] += dPower * 0.5 * v.x * v.y;
          g[5] += dPower * 0.5 * v.y * v.y;
        }
        any = true;

        behind[at] = vec4(mix(B, s.color.rgb, alpha), state.a + logRest);
      }
    }
    if (any) {
      for (uint k = 0u; k < 9u; ++k) {
        add_gradient(index * 9u + k, g[k]);
      }
    }
  }
}
