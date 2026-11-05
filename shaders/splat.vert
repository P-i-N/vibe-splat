#version 460
// Expands each visible splat (in back-to-front order) into a screen-space
// quad covering its 3-sigma extent.
#extension GL_GOOGLE_include_directive : require
#include "splat_types.glsl"

layout(std430, binding = 0) readonly buffer Splats { Splat2D splats[]; };
layout(std430, binding = 1) readonly buffer Order { uint order[]; };

layout(push_constant) uniform Push {
  vec2 viewport;
  float pointRadius; // Above 0 in point mode, where every quad has this radius
} pc;

layout(location = 0) out vec2 outOffset;            // pixel offset from the splat centre
layout(location = 1) flat out vec4 outConicOpacity;
layout(location = 2) flat out vec3 outColor;

void main() {
  const Splat2D s = splats[order[gl_InstanceIndex]];
  const vec2 conicAB = unpackHalf2x16(s.conicAB), conicCOpacity = unpackHalf2x16(s.conicCOpacity);
  const vec3 conic = vec3(conicAB, conicCOpacity.x);
  // The covariance is the conic's inverse, as preprocess.comp found it.
  const float det = conic.x * conic.z - conic.y * conic.y;
  const float radius = pc.pointRadius > 0.0 ? pc.pointRadius : splat_radius(conic.z / det, -conic.y / det, conic.x / det);
  // Triangle strip corners: (-1,-1), (1,-1), (-1,1), (1,1).
  const vec2 corner = vec2(float(gl_VertexIndex & 1) * 2.0 - 1.0, float(gl_VertexIndex >> 1) * 2.0 - 1.0);
  const vec2 offset = corner * radius;
  const vec2 pixel = vec2(s.center[0], s.center[1]) + offset;
  gl_Position = vec4(pixel / pc.viewport * 2.0 - 1.0, 0.0, 1.0);
  outOffset = offset;
  outConicOpacity = vec4(conic, conicCOpacity.y);
  outColor = unpack_color(s.color);
}
