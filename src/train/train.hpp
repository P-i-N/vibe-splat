#pragma once

#include "core/vk_context.hpp"

#include <cstdint>
#include <filesystem>

namespace splat {

struct TrainOptions {
  std::filesystem::path project;
  std::filesystem::path out;
  int iterations = 30000;
  float lambdaSsim = 0.2f;
  int shDegree = 3;
  int maxSize = 0;  // Shrink images to this; 0 keeps their size
  int maxViews = 0; // Use only the first n views; 0 uses all
  int holdout = 0;  // Keep every nth view out of training; 0 trains on all
  std::uint32_t seed = 1;
  int reportEvery = 500;
  int saveEvery = 5000; // 0 only saves at the end

  bool densify = true;
  float gradientThreshold = 2e-4f;
  double vramBudgetGiB = 0.0; // 0: 90% of the GPU's free memory
  std::size_t maxGaussians = 0; // 0: as many as the budget allows
  // Share of the Gaussians removed, those adding least to the training views,
  // at the end of densification and again halfway through the rest. 0: none.
  // 70% each time leaves a tenth, for about 0.1 dB of PSNR on held-out views.
  double pruneShare = 0.7;
};

struct TrainReport {
  float firstLoss = 0.f;
  float lastLoss = 0.f;
  double trainPsnr = 0.0;
  double holdoutPsnr = 0.0; // 0 if nothing was held out
  double iterationsPerSecond = 0.0;
  std::size_t gaussians = 0;
  std::uint64_t peakGpuBytes = 0;
  std::size_t heldBack = 0;
};

// Trains Gaussians on a project with images and camera poses in sparse/0, and
// writes a 3DGS .ply. The Gaussians start at the sparse points and multiply
// where the images ask for more detail, within the memory budget.
TrainReport run_train(vk::Context &ctx, const TrainOptions &options);

} // namespace splat
