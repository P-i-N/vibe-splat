#pragma once

#include "core/vk_context.hpp"

#include <cstdint>
#include <filesystem>

namespace splat {

struct TrainOptions {
  std::filesystem::path project;
  std::filesystem::path out;
  int iterations = 7000;
  float lambdaSsim = 0.2f;
  int shDegree = 0;
  int maxSize = 0;  // Shrink images to this; 0 keeps their size
  int maxViews = 0; // Use only the first n views; 0 uses all
  int holdout = 0;  // Keep every nth view out of training; 0 trains on all
  std::uint32_t seed = 1;
  int reportEvery = 500;
};

struct TrainReport {
  float firstLoss = 0.f;
  float lastLoss = 0.f;
  double trainPsnr = 0.0;
  double holdoutPsnr = 0.0; // 0 if nothing was held out
  double iterationsPerSecond = 0.0;
};

// Trains Gaussians on a project with images and camera poses in sparse/0, and
// writes a 3DGS .ply. The Gaussians start at the sparse points and neither
// split nor multiply yet, so detail is limited by how many points there are.
TrainReport run_train(vk::Context &ctx, const TrainOptions &options);

} // namespace splat
