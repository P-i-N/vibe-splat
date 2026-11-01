// Shared by the training shaders. Must match src/train/trainer.cpp and the CPU
// reference in src/train/reference.cpp.
#ifndef TRAIN_TYPES_GLSL
#define TRAIN_TYPES_GLSL

const uint TILE_SIZE = 16u;
const uint INVISIBLE = 0xFFFFFFFFu;

struct TrainSplat {
  vec4 centerRadius; // xy pixel centre, z radius in pixels, w view depth
  vec4 conicOpacity; // xyz inverse 2D covariance (xx, xy, yy), w opacity
  vec4 color;        // rgb, a unused
  uvec4 tiles;       // Covered tiles, inclusive: x0, y0, x1, y1
};

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

mat3 train_quat_to_mat(vec4 q) {
  const float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
              2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
              2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

#endif
