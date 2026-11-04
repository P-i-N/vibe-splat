// Data layouts shared by the preprocess compute shader and the splat
// rasteriser. Must match src/render/splat_renderer.cpp.
#ifndef SPLAT_TYPES_GLSL
#define SPLAT_TYPES_GLSL

// 24 bytes. Only the position stays a full float.
struct Gaussian {
  float position[3];
  uint rotation;      // Normalised quaternion (w, x, y, z), see pack_rotation()
  uint scaleXY;       // Half floats: standard deviations (already exp'd)
  uint scaleZOpacity; // Half floats: z standard deviation, opacity in [0,1] (already activated)
};

// 24 bytes. The centre stays a full float: halves are 2 px apart beyond 2048.
struct Splat2D {
  vec2 center; // Pixels
  uvec2 shape; // Half floats: radius in pixels, inverse 2D covariance (a, b, c)
  uvec2 color; // Half floats: rgb, opacity
};

// Smallest three: which component is largest in the top 2 bits, then the
// other three in [-sqrt(1/2), sqrt(1/2)] at 10 bits each. q and -q are the
// same rotation, so the largest is made positive and rebuilt from the others.
uint pack_rotation(vec4 q) {
  const vec4 a = abs(q);
  const uint largest = a.x >= max(a.y, max(a.z, a.w)) ? 0u : a.y >= max(a.z, a.w) ? 1u : a.z >= a.w ? 2u : 3u;
  if (q[largest] < 0.0) {
    q = -q;
  }
  uint bits = largest << 30;
  uint shift = 20u;
  for (uint c = 0u; c < 4u; ++c) {
    if (c == largest) {
      continue;
    }
    const float v = clamp(q[c] * 0.70710678 + 0.5, 0.0, 1.0); // [-sqrt(1/2), sqrt(1/2)] to [0, 1]
    bits |= uint(round(v * 1023.0)) << shift;
    shift -= 10u;
  }
  return bits;
}

vec4 unpack_rotation(uint bits) {
  const uint largest = bits >> 30;
  vec4 q;
  float sum = 0.0;
  uint shift = 20u;
  for (uint c = 0u; c < 4u; ++c) {
    if (c == largest) {
      continue;
    }
    q[c] = (float((bits >> shift) & 1023u) / 1023.0 - 0.5) * 1.41421356;
    sum += q[c] * q[c];
    shift -= 10u;
  }
  q[largest] = sqrt(max(0.0, 1.0 - sum));
  return q;
}

// Words per Gaussian of SH packed by pack_sh.comp: the DC term and a scale as
// half floats, then the other coefficients as signed bytes times that scale.
uint packed_sh_words(uint coefficients) {
  return 2u + (3u * (coefficients - 1u) + 3u) / 4u;
}

// OpenCV-style pinhole camera: x right, y down, z forward.
struct CameraParams {
  mat4 view;       // world -> camera
  vec4 position;   // camera centre in world space
  vec4 intrinsics; // fx, fy, cx, cy (pixels)
  vec4 viewport;   // width, height, tan(half fov x), tan(half fov y)
  uvec4 counts;    // gaussian count, SH degree, SH words (floats when not packed) per gaussian, SH coefficients in use (training only)
};

#endif
