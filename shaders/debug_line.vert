#version 460
// World-space lines (camera frustums, grid, axes) through the same pinhole
// camera as the splats. The clip position is linear in camera space, so lines
// crossing the near plane are clipped correctly.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform Push {
  mat4 view;       // world -> camera
  vec4 intrinsics; // fx, fy, cx, cy
  vec4 viewport;   // width, height, near plane, unused
} pc;

layout(location = 0) out vec4 outColor;

void main() {
  const vec3 c = (pc.view * vec4(inPosition, 1.0)).xyz;
  const float x = pc.intrinsics.x * c.x + pc.intrinsics.z * c.z;
  const float y = pc.intrinsics.y * c.y + pc.intrinsics.w * c.z;
  gl_Position = vec4(x / pc.viewport.x * 2.0 - c.z, y / pc.viewport.y * 2.0 - c.z, c.z - pc.viewport.z, c.z);
  outColor = inColor;
}
