#version 460
// Evaluates the 2D Gaussian and outputs premultiplied colour for
// back-to-front "over" blending.
layout(location = 0) in vec2 inOffset;
layout(location = 1) flat in vec4 inConicOpacity;
layout(location = 2) flat in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main() {
  const vec2 d = inOffset;
  const vec3 conic = inConicOpacity.xyz;
  const float power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
  if (power > 0.0) {
    discard;
  }
  const float alpha = min(0.99, inConicOpacity.w * exp(power));
  if (alpha < 1.0 / 255.0) {
    discard;
  }
  outColor = vec4(inColor * alpha, alpha);
}
