#pragma once

#include <glm/glm.hpp>

namespace splat {

// Pinhole camera in the OpenCV/COLMAP convention used by 3DGS data:
// camera x right, y down, z forward; pixel (0,0) is the top-left corner.
struct Camera {
  glm::mat4 view{1.f};   // world -> camera
  glm::vec3 position{0.f};
  float fx = 1.f, fy = 1.f, cx = 0.f, cy = 0.f;
  int width = 1, height = 1;

  // Builds a camera at `eye` looking at `target`; `up` is the world up
  // direction (for COLMAP-style data that is -y).
  static Camera look_at(glm::vec3 eye, glm::vec3 target, glm::vec3 up, float verticalFovDeg, int width, int height);
};

// Mouse/gamepad-friendly orbit camera around a target point.
struct OrbitCamera {
  glm::vec3 target{0.f};
  glm::vec3 up{0.f, -1.f, 0.f}; // world up
  float yawDeg = 0.f;    // around `up`
  float pitchDeg = 20.f; // positive = looking down at the target
  float distance = 5.f;
  float fovDeg = 60.f;

  [[nodiscard]] glm::vec3 eye() const;
  [[nodiscard]] Camera camera(int width, int height) const;

  void orbit(float dYawDeg, float dPitchDeg);
  void zoom(float factor); // < 1 moves closer
  // Moves the target in the camera's screen plane / forward, in world units
  // scaled by distance.
  void pan(float dxScreen, float dyScreen);
  void move(float forward, float right, float upward);

private:
  void basis(glm::vec3 &forward, glm::vec3 &right, glm::vec3 &down) const;
};

} // namespace splat
