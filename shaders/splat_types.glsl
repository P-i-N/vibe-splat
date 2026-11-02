// Data layouts shared by the preprocess compute shader and the splat
// rasteriser. Must match src/render/splat_renderer.cpp.
#ifndef SPLAT_TYPES_GLSL
#define SPLAT_TYPES_GLSL

struct Gaussian {
  vec4 posOpacity; // xyz position, w = opacity in [0,1] (already activated)
  vec4 rotation;   // normalised quaternion (w, x, y, z)
  vec4 scale;      // xyz standard deviations (already exp'd), w unused
};

struct Splat2D {
  vec4 centerRadius;  // xy pixel centre, z radius in pixels, w view depth
  vec4 conicOpacity;  // xyz inverse 2D covariance (a, b, c), w opacity
  vec4 color;         // rgb, a unused
};

// OpenCV-style pinhole camera: x right, y down, z forward.
struct CameraParams {
  mat4 view;       // world -> camera
  vec4 position;   // camera centre in world space
  vec4 intrinsics; // fx, fy, cx, cy (pixels)
  vec4 viewport;   // width, height, tan(half fov x), tan(half fov y)
  uvec4 counts;    // gaussian count, SH degree, SH floats per gaussian, SH coefficients in use (training only)
};

#endif
