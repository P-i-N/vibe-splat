#version 460
// Expands each visible splat (in back-to-front order) into a screen-space
// quad covering its 3-sigma extent.
#extension GL_GOOGLE_include_directive : require
#include "splat_types.glsl"

layout(std430, binding = 0) readonly buffer Splats { Splat2D splats[]; };
layout(std430, binding = 1) readonly buffer Order { uint order[]; };

layout(push_constant) uniform Push { vec2 viewport; } pc;

layout(location = 0) out vec2 outOffset;            // pixel offset from the splat centre
layout(location = 1) flat out vec4 outConicOpacity;
layout(location = 2) flat out vec3 outColor;

void main() {
  const Splat2D s = splats[order[gl_InstanceIndex]];
  // Triangle strip corners: (-1,-1), (1,-1), (-1,1), (1,1).
  const vec2 corner = vec2(float(gl_VertexIndex & 1) * 2.0 - 1.0, float(gl_VertexIndex >> 1) * 2.0 - 1.0);
  const vec2 offset = corner * s.centerRadius.z;
  const vec2 pixel = s.centerRadius.xy + offset;
  gl_Position = vec4(pixel / pc.viewport * 2.0 - 1.0, 0.0, 1.0);
  outOffset = offset;
  outConicOpacity = s.conicOpacity;
  outColor = s.color.rgb;
}
