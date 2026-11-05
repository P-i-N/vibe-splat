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

// 20 bytes. The centre stays a full float: halves are 2 px apart beyond 2048.
// The quad's radius follows from the conic, see splat_radius().
struct Splat2D {
  float center[2];    // Pixels
  uint conicAB;       // Half floats: inverse 2D covariance a, b
  uint conicCOpacity; // Half floats: its c, and the opacity
  uint color;         // See pack_color()
};

// Colour as Vulkan's E5B9G9R9 format: 9-bit mantissas that share a 5-bit
// exponent, so that colours over 1, which blending sees before the target
// clamps, keep their size. Trained scenes have some up to 4 and more.
uint pack_color(vec3 c) {
  c = clamp(c, 0.0, 65408.0);
  const float largest = max(c.r, max(c.g, c.b));
  if (!(largest > 0.0)) {
    return 0u;
  }
  int exponent = max(-16, int(floor(log2(largest)))) + 16;
  float step = exp2(float(exponent - 24));
  if (floor(largest / step + 0.5) >= 512.0) {
    ++exponent;
    step *= 2.0;
  }
  const uvec3 v = uvec3(floor(c / step + 0.5));
  return v.r | (v.g << 9) | (v.b << 18) | (uint(exponent) << 27);
}
vec3 unpack_color(uint bits) {
  return vec3(uvec3(bits, bits >> 9, bits >> 18) & 511u) * exp2(float(int(bits >> 27) - 24));
}

// 3 standard deviations along the major axis of the 2D covariance (a, b, c)
// = (xx, xy, yy), in whole pixels: the half size of the quad.
float splat_radius(float a, float b, float c) {
  const float mid = 0.5 * (a + c);
  const float lambda = mid + sqrt(max(0.1, mid * mid - (a * c - b * b)));
  return ceil(3.0 * sqrt(lambda));
}

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

// SH packed by pack_sh.comp, per Gaussian: the DC term and each band's
// largest magnitude as half floats, then the coefficients relative to that,
// band 1 as signed bytes and bands 2 and 3 as signed 4 bits, which their
// smaller part of the colour allows. Words:
//   0     DC r, g
//   1     DC b, band 1's scale
//   2     band 2's scale, band 3's (from degree 2)
//   then  band 1's 9 bytes, then the 4-bit values, which never cross a word
uint packed_sh_header(uint coefficients) {
  return coefficients >= 9u ? 3u : 2u;
}
uint packed_sh_words(uint coefficients) {
  const uint bits = coefficients > 1u ? 72u + 12u * (coefficients - 4u) : 0u;
  return packed_sh_header(coefficients) + (bits + 31u) / 32u;
}
// The band of coefficient k > 0, and the bit at which its channel c starts
// after the header, with its width.
uint sh_band(uint k) {
  return k < 4u ? 1u : k < 9u ? 2u : 3u;
}
uvec2 packed_sh_bits(uint k, uint c) {
  const uint j = (k - 1u) * 3u + c;
  return k < 4u ? uvec2(8u * j, 8u) : uvec2(72u + 4u * (j - 9u), 4u);
}

// SH as the trainer keeps it, per Gaussian: the DC term's 3 floats, then the
// other coefficients as half floats, two to a word. As half_sh_words() in
// src/splat/gaussian_cloud.hpp.
uint trainer_sh_words(uint coefficients) {
  return 3u + (3u * (coefficients - 1u) + 1u) / 2u;
}

// The word of a Gaussian's that holds element j of GaussianCloud's sh, and
// that element from it.
uint trainer_sh_word(uint j) {
  return j < 3u ? j : 3u + (j - 3u) / 2u;
}
float trainer_sh_value(uint word, uint j) {
  return j < 3u ? uintBitsToFloat(word) : unpackHalf2x16(word)[(j - 3u) & 1u];
}

// OpenCV-style pinhole camera: x right, y down, z forward.
struct CameraParams {
  mat4 view;       // world -> camera
  vec4 position;   // camera centre in world space
  vec4 intrinsics; // fx, fy, cx, cy (pixels)
  vec4 viewport;   // width, height, tan(half fov x), tan(half fov y)
  uvec4 counts;    // gaussian count, SH degree, SH words per gaussian, SH coefficients (in use, when training)
};

#endif
