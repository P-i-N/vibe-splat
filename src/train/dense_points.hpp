#pragma once

#include "render/camera.hpp"
#include "sfm/colmap_model.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splat {

struct DensePointsView {
  Camera camera;
  std::span<const std::uint8_t> rgba;
};

// Many more starting points than structure from motion leaves, as EDGS has,
// which spares the Gaussians much of the growing they would do otherwise. EDGS
// matches pixels with a neural network. This uses plain patch matching: a
// textured pixel of a view is sought along its epipolar line in a neighbouring
// view, between the depths the sparse points suggest, by normalised correlation.
// What it finds must stand out from the rest of the line and also be found in a
// third view. At most maxPoints, spread evenly over what was found.
[[nodiscard]] std::vector<ColmapPoint> dense_points(std::span<const DensePointsView> views, std::span<const ColmapPoint> sparse, std::size_t maxPoints);

} // namespace splat
