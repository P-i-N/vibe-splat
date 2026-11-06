#include "render/camera.hpp"

#include <algorithm>
#include <cmath>

namespace splat {

Camera Camera::look_at(glm::vec3 eye, glm::vec3 target, glm::vec3 up, float verticalFovDeg, int width, int height) {
  const glm::vec3 forward = glm::normalize(target - eye);
  glm::vec3 right = glm::cross(-up, forward); // x = y * z with y = down
  if (glm::dot(right, right) < 1e-12f) {
    right = glm::vec3(1.f, 0.f, 0.f);
  }
  right = glm::normalize(right);
  const glm::vec3 down = glm::cross(forward, right);

  Camera cam;
  cam.position = eye;
  // Rows of the rotation are the camera axes in world space.
  glm::mat4 view(1.f);
  for (int c = 0; c < 3; ++c) {
    view[c][0] = right[c];
    view[c][1] = down[c];
    view[c][2] = forward[c];
  }
  view[3][0] = -glm::dot(right, eye);
  view[3][1] = -glm::dot(down, eye);
  view[3][2] = -glm::dot(forward, eye);
  cam.view = view;
  cam.width = width;
  cam.height = height;
  cam.fy = 0.5f * static_cast<float>(height) / std::tan(glm::radians(verticalFovDeg) * 0.5f);
  cam.fx = cam.fy;
  cam.cx = 0.5f * static_cast<float>(width);
  cam.cy = 0.5f * static_cast<float>(height);
  return cam;
}

void OrbitCamera::basis(glm::vec3 &forward, glm::vec3 &right, glm::vec3 &down) const {
  const glm::vec3 eyePos = eye();
  forward = glm::normalize(target - eyePos);
  right = glm::normalize(glm::cross(-glm::normalize(up), forward));
  down = glm::cross(forward, right);
}

glm::vec3 OrbitCamera::eye() const {
  // Build an orthonormal frame around `up`.
  const glm::vec3 u = glm::normalize(up);
  glm::vec3 ref = std::abs(u.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
  const glm::vec3 a = glm::normalize(ref - u * glm::dot(ref, u));
  const glm::vec3 b = glm::cross(u, a);
  const float yaw = glm::radians(yawDeg);
  const float pitch = glm::radians(pitchDeg);
  // Direction from target to eye: horizontal component plus `up` for pitch.
  const glm::vec3 horizontal = std::cos(yaw) * a + std::sin(yaw) * b;
  const glm::vec3 dir = std::cos(pitch) * horizontal + std::sin(pitch) * u;
  return target + dir * distance;
}

Camera OrbitCamera::camera(int width, int height) const {
  return Camera::look_at(eye(), target, up, fovDeg, width, height);
}

void OrbitCamera::orbit(float dYawDeg, float dPitchDeg) {
  yawDeg = std::fmod(yawDeg + dYawDeg, 360.f);
  pitchDeg = std::clamp(pitchDeg + dPitchDeg, -89.f, 89.f);
}

void OrbitCamera::look(float dYawDeg, float dPitchDeg) {
  const glm::vec3 eyePos = eye();
  orbit(dYawDeg, dPitchDeg);
  // eye() is target + dir * distance, so moving the target back keeps the eye.
  target += eyePos - eye();
}

void OrbitCamera::zoom(float factor) { distance = std::clamp(distance * factor, 1e-3f, 1e6f); }

void OrbitCamera::pan(float dxScreen, float dyScreen) {
  glm::vec3 f, r, d;
  basis(f, r, d);
  target += (r * dxScreen + d * dyScreen) * distance;
}

void OrbitCamera::move(float forward, float right, float upward) {
  glm::vec3 f, r, d;
  basis(f, r, d);
  target += (f * forward + r * right - d * upward) * distance;
}

void OrbitCamera::look_along(glm::vec3 eye, glm::vec3 forward, float targetDistance) {
  const glm::vec3 u = glm::normalize(up);
  // The same frame as eye() builds.
  glm::vec3 ref = std::abs(u.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
  const glm::vec3 a = glm::normalize(ref - u * glm::dot(ref, u));
  const glm::vec3 b = glm::cross(u, a);
  const glm::vec3 dir = -glm::normalize(forward); // From the target to the eye
  pitchDeg = std::clamp(glm::degrees(std::asin(std::clamp(glm::dot(dir, u), -1.f, 1.f))), -89.f, 89.f);
  yawDeg = glm::degrees(std::atan2(glm::dot(dir, b), glm::dot(dir, a)));
  distance = std::max(targetDistance, 1e-3f);
  target = eye - dir * distance;
}

} // namespace splat
