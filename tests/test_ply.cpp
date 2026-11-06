#include "splat/gaussian_cloud.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace splat;

namespace {

std::filesystem::path temp_file(const char *name) {
  return std::filesystem::temp_directory_path() / name;
}

} // namespace

TEST_CASE("PLY save and load round-trip every attribute", "[ply]") {
  for (int degree = 0; degree <= 3; ++degree) {
    GaussianCloud cloud;
    cloud.resize(37, degree);
    for (std::size_t i = 0; i < cloud.positions.size(); ++i) cloud.positions[i] = static_cast<float>(i) * 0.25f;
    for (std::size_t i = 0; i < cloud.scales.size(); ++i) cloud.scales[i] = -static_cast<float>(i) * 0.01f;
    for (std::size_t i = 0; i < cloud.rotations.size(); ++i) cloud.rotations[i] = static_cast<float>(i % 5);
    for (std::size_t i = 0; i < cloud.opacities.size(); ++i) cloud.opacities[i] = static_cast<float>(i) - 10.f;
    for (std::size_t i = 0; i < cloud.sh.size(); ++i) cloud.sh[i] = std::sin(static_cast<float>(i));

    const auto path = temp_file("splat_roundtrip.ply");
    save_ply(cloud, path);
    const GaussianCloud loaded = load_ply(path);
    std::filesystem::remove(path);

    INFO("SH degree " << degree);
    REQUIRE(loaded.shDegree == degree);
    REQUIRE(loaded.size() == cloud.size());
    CHECK(loaded.positions == cloud.positions);
    CHECK(loaded.scales == cloud.scales);
    CHECK(loaded.rotations == cloud.rotations);
    CHECK(loaded.opacities == cloud.opacities);
    CHECK(loaded.sh == cloud.sh);
  }
}

TEST_CASE("PLY f_rest coefficients are channel-major in the file", "[ply]") {
  // One Gaussian, degree 1: f_rest_0..2 = red coeffs 1..3, 3..5 green, 6..8 blue.
  const auto path = temp_file("splat_ascii.ply");
  {
    std::ofstream out(path);
    out << "ply\nformat ascii 1.0\ncomment written by a test\nelement vertex 1\n";
    for (const char *p : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}) out << "property float " << p << "\n";
    for (int i = 0; i < 9; ++i) out << "property float f_rest_" << i << "\n";
    for (const char *p : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
      out << "property float " << p << "\n";
    out << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
    out << "1 2 3  0.1 0.2 0.3  10 11 12 20 21 22 30 31 32  0.5  -1 -2 -3  1 0 0 0\n";
  }
  const GaussianCloud c = load_ply(path);
  std::filesystem::remove(path);

  REQUIRE(c.size() == 1);
  REQUIRE(c.shDegree == 1);
  CHECK(c.positions == std::vector<float>{1, 2, 3});
  CHECK(c.sh[0 * 3 + 0] == 0.1f); // DC red
  CHECK(c.sh[1 * 3 + 0] == 10.f); // coeff 1 red
  CHECK(c.sh[1 * 3 + 1] == 20.f); // coeff 1 green
  CHECK(c.sh[3 * 3 + 2] == 32.f); // coeff 3 blue
  CHECK(c.opacities[0] == 0.5f);
  CHECK(c.scales == std::vector<float>{-1, -2, -3});
}

TEST_CASE("PLY loader rejects files that are not splats", "[ply]") {
  const auto path = temp_file("splat_mesh.ply");
  {
    std::ofstream out(path);
    out << "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
           "end_header\n0 0 0\n";
  }
  CHECK_THROWS(load_ply(path));
  std::filesystem::remove(path);
}

TEST_CASE("Test scene has sensible bounds", "[ply]") {
  const GaussianCloud scene = make_test_scene(10'000);
  REQUIRE(scene.size() == 10'000);
  scene.validate();
  const Bounds b = compute_bounds(scene);
  CHECK(b.radius > 1.f);
  CHECK(b.radius < 10.f);
}
