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
  // Learn each view's exposure, so that the Gaussians need not take it up.
  // For photos whose exposure varies, as auto exposure in video does; on
  // evenly exposed photos it costs about 0.5 dB on held-out views.
  bool exposure = false;
  bool fullAdam = false;       // Update every Gaussian each step, not only those in view: slower, sharper
  bool fullResolution = false; // Train at full size throughout, without the coarse start
  float gradientThreshold = 2e-4f;
  double vramBudgetGiB = 0.0; // 0: 90% of the GPU's free memory
  std::size_t maxGaussians = 0; // 0: as many as the budget allows
  // Share of the Gaussians removed, those adding least to the training views,
  // at the end of densification and again halfway through the rest. 0: none.
  // 70% each time leaves a tenth, for about 0.1 dB of PSNR on held-out views.
  double pruneShare = 0.7;
  // The gradient threshold rises by this factor during densification, centred
  // on gradientThreshold, so that broad shapes come first (Improving ADC). 1: fixed.
  double thresholdRise = 1.0;
  float splitThreshold = 0.f;     // Above 0: see DensifyOptions::splitThreshold
  double denseFraction = 0.01;    // Of the scene's extent: larger Gaussians split, smaller clone
  int densifyInterval = 100;      // Iterations between densification rounds
  int scoreViews = 0;             // Above 0: views scored per round, see DensifyOptions::scoredViews
  float scoreMinPixels = 5.f;     // See DensifyOptions::minMarkedPixels
  int explore = 0;                // Random splits and random new Gaussians per round
  bool significancePrune = false; // Of the too transparent remove the half adding least, and none for size
  bool pointExtent = false;       // The scene's extent from the sparse points, not the cameras
};

struct TrainReport {
  float firstLoss = 0.f;
  float lastLoss = 0.f;
  double trainPsnr = 0.0;
  double holdoutPsnr = 0.0; // 0 if nothing was held out
  double holdoutFittedPsnr = 0.0; // With each view's exposure fitted, see cross_fitted_psnr()
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
