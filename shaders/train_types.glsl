// Shared by the training shaders. Must match src/train/trainer.cpp and the CPU
// reference in src/train/reference.cpp.
#ifndef TRAIN_TYPES_GLSL
#define TRAIN_TYPES_GLSL

const uint TILE_SIZE = 16u;
const uint TILE_BATCH = TILE_SIZE * TILE_SIZE; // Splats staged through shared memory at a time
const uint INVISIBLE = 0xFFFFFFFFu;
// A pixel takes no more splats once its transmittance would fall below
// 1e-4, as in the 3DGS paper. Compared as logarithms: this is log(1e-4).
const float LOG_MIN_TRANSMITTANCE = -9.210340372;
// Below these a Gaussian is degenerate and is not drawn, as in Faster-GS.
const float MIN_QUATERNION_LENGTH = 1e-4;
const float MIN_CONIC_DETERMINANT = 1e-6;
const uint RAW_PER_SPLAT = 14u; // Sums per splat from the backward pass, listed in train_backward.glsl

// What densification decides on, per Gaussian.
struct DensifyStat {
  vec4 main; // x: gradient sum, y: visits, z: largest radius, w: contribution
  vec4 more; // x: sum of the gradient without cancelling, y: marked pixels, z: those by loss, w: edge score
};

struct TrainSplat {
  vec4 centerRadius; // xy pixel centre, z radius in pixels, w view depth
  vec4 conicOpacity; // xyz inverse 2D covariance (xx, xy, yy), w opacity
  vec4 color;        // rgb, a unused
  vec4 extent;       // xy: half size of the area it can reach, in pixels
  uvec4 tiles;       // Covered tiles, inclusive: x0, y0, x1, y1
};

// The pixels of a tile that a splat can reach, as x0, y0, x1, y1 with the upper
// bounds exclusive.
uvec4 splat_pixels(TrainSplat s, uvec4 tilePixels) {
  const vec2 low = max(s.centerRadius.xy - s.extent.xy - 0.5, vec2(0.0));
  const vec2 high = max(s.centerRadius.xy + s.extent.xy + 0.5, vec2(0.0));
  return uvec4(max(uint(low.x), tilePixels.x), max(uint(low.y), tilePixels.y),
               min(uint(high.x) + 1u, tilePixels.z), min(uint(high.y) + 1u, tilePixels.w));
}

// Whether a splat reaches any pixel centre of a tile, by the rule of
// splat_at(): q = -2 power = d^T conic d must stay within 2 log(255 opacity).
// The bounding box of that ellipse covers many tiles it misses, mostly along
// the diagonal of a thin one, and each of those costs a tile entry that every
// pixel of the tile would look at for nothing. The convex q is least at the
// centre or, outside, on the edge of the tile's square of pixel centres.
bool splat_reaches_tile(TrainSplat s, uint tx, uint ty) {
  const vec3 q = s.conicOpacity.xyz;
  const float reach = 2.0 * log(255.0 * s.conicOpacity.w);
  const vec2 low = vec2(tx, ty) * float(TILE_SIZE) + 0.5 - s.centerRadius.xy;
  const vec2 high = low + float(TILE_SIZE - 1u);
  if (all(lessThanEqual(low, vec2(0.0))) && all(greaterThanEqual(high, vec2(0.0)))) {
    return true;
  }
  float least = 1e30;
  for (uint side = 0u; side < 2u; ++side) {
    // Along x = const: q = q.x x^2 + 2 q.y x y + q.z y^2, least at y = -q.y x / q.z.
    const float x = side == 0u ? low.x : high.x;
    const float y = clamp(-q.y * x / q.z, low.y, high.y);
    least = min(least, q.x * x * x + 2.0 * q.y * x * y + q.z * y * y);
    // Along y = const, likewise.
    const float v = side == 0u ? low.y : high.y;
    const float u = clamp(-q.y * v / q.x, low.x, high.x);
    least = min(least, q.x * u * u + 2.0 * q.y * u * v + q.z * v * v);
  }
  // A little slack, so that rounding never loses a pixel splat_at() would take.
  return least <= reach * 1.001 + 1e-3;
}

// Evaluates a splat at a pixel. False if the pixel gets nothing from it. The
// rules match the graphics path in splat.vert and splat.frag.
bool splat_at(TrainSplat s, uint x, uint y, out vec2 d, out float power, out float alpha) {
  d = vec2(float(x) + 0.5, float(y) + 0.5) - s.centerRadius.xy;
  const float r = s.centerRadius.z;
  if (abs(d.x) > r || abs(d.y) > r) {
    return false;
  }
  const vec3 q = s.conicOpacity.xyz;
  power = -0.5 * (q.x * d.x * d.x + q.z * d.y * d.y) - q.y * d.x * d.y;
  if (power > 0.0) {
    return false;
  }
  alpha = min(0.99, s.conicOpacity.w * exp(power));
  return alpha >= 1.0 / 255.0;
}

// The spherical-harmonic basis functions, in the order of the .ply coefficients.
void sh_basis(vec3 dir, uint degree, out float b[16]) {
  const float C0 = 0.28209479177387814, C1 = 0.4886025119029199;
  const float x = dir.x, y = dir.y, z = dir.z;
  const float xx = x * x, yy = y * y, zz = z * z;
  b[0] = C0;
  if (degree > 0u) {
    b[1] = -C1 * y;
    b[2] = C1 * z;
    b[3] = -C1 * x;
  }
  if (degree > 1u) {
    b[4] = 1.0925484305920792 * x * y;
    b[5] = -1.0925484305920792 * y * z;
    b[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
    b[7] = -1.0925484305920792 * x * z;
    b[8] = 0.5462742152960396 * (xx - yy);
  }
  if (degree > 2u) {
    b[9] = -0.5900435899266435 * y * (3.0 * xx - yy);
    b[10] = 2.890611442640554 * x * y * z;
    b[11] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    b[12] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[13] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    b[14] = 1.445305721320277 * z * (xx - yy);
    b[15] = -0.5900435899266435 * x * (xx - 3.0 * yy);
  }
}

// How each basis function changes with the view direction.
void sh_basis_gradient(vec3 dir, uint degree, out vec3 g[16]) {
  const float C1 = 0.4886025119029199, A = 1.0925484305920792, K = 0.31539156525252005, L = 0.5462742152960396;
  const float C = -0.5900435899266435, E = 2.890611442640554, F = -0.4570457994644658, G = 0.3731763325901154, H = 1.445305721320277;
  const float x = dir.x, y = dir.y, z = dir.z;
  const float xx = x * x, yy = y * y, zz = z * z;
  g[0] = vec3(0.0);
  if (degree > 0u) {
    g[1] = vec3(0.0, -C1, 0.0);
    g[2] = vec3(0.0, 0.0, C1);
    g[3] = vec3(-C1, 0.0, 0.0);
  }
  if (degree > 1u) {
    g[4] = A * vec3(y, x, 0.0);
    g[5] = -A * vec3(0.0, z, y);
    g[6] = K * vec3(-2.0 * x, -2.0 * y, 4.0 * z);
    g[7] = -A * vec3(z, 0.0, x);
    g[8] = L * vec3(2.0 * x, -2.0 * y, 0.0);
  }
  if (degree > 2u) {
    g[9] = C * vec3(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0);
    g[10] = E * vec3(y * z, x * z, x * y);
    g[11] = F * vec3(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z);
    g[12] = G * vec3(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy);
    g[13] = F * vec3(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z);
    g[14] = H * vec3(2.0 * x * z, -2.0 * y * z, xx - yy);
    g[15] = C * vec3(3.0 * xx - 3.0 * yy, -6.0 * x * y, 0.0);
  }
}

// PCG's output permutation over a 32-bit state, as a hash to a uniform in [0, 1).
float train_uniform(uint index, uint seed) {
  const uint state = (index ^ (seed * 0x9E3779B9u)) * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return float((word >> 22u) ^ word) / 4294967296.0;
}

// SH dropout (Dropping Anchor and Spherical Harmonics, CVPR 2026): with the
// given probability a Gaussian keeps only the coefficients up to `retained`
// this iteration, so that its colour settles in the low degrees.
uint train_sh_coefficients(uint index, uint coefficients, float dropout, uint retained, uint seed) {
  if (dropout > 0.0 && train_uniform(index, seed) < dropout) {
    return min(coefficients, (retained + 1u) * (retained + 1u));
  }
  return coefficients;
}

mat3 train_quat_to_mat(vec4 q) {
  const float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
              2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
              2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

#endif
