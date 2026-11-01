#pragma once

#include "ingest/image.hpp"
#include "sfm/geometry.hpp"

#include <vector>

namespace splat {

struct TestView {
  RgbImage image;
  Pose pose;
};

// Ray-traces a textured room with a few boxes in it from cameras on a circle,
// all looking at the middle. The poses are known exactly, which makes this the
// reference for checking pose estimation. Focal length is in pixels; the
// principal point is the image centre.
[[nodiscard]] std::vector<TestView> make_test_views(int count, int width, int height, double focal);

} // namespace splat
