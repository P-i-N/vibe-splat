#pragma once

#include "core/vk_context.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace splat {

struct TrainOptions {
  std::filesystem::path project;
  std::filesystem::path out;
  int iterations = 30000;
  float lambdaSsim = 0.2f;
  int shDegree = 3;
  int maxSize = 0;  // Shrink images to this; 0 keeps their size
  int maxViews = 0; // Use only the first n views; 0 uses all
  int holdout = 0;  // Keep the first and every nth view out of training, as 3DGS does; 0 trains on all
  std::uint32_t seed = 1;
  int reportEvery = 500;
  int saveEvery = 5000; // 0 only saves at the end

  bool densify = true;
  // How Gaussians are added and removed: the paper's clone, split and prune
  // (Adc), or 3DGS-MCMC's relocation within the cap, with no opacity reset (Mcmc).
  enum class DensifyMode { Adc, Mcmc };
  DensifyMode densifyMode = DensifyMode::Adc;
  // Learn each view's exposure, so that the Gaussians need not take it up.
  // For photos whose exposure varies, as auto exposure in video does.
  bool exposure = false;
  bool fullAdam = false;       // Update every Gaussian each step, not only those in view: slower
  bool fullResolution = false; // Train at full size throughout, without the coarse start
  float gradientThreshold = 2e-4f;
  double vramBudgetGiB = 0.0; // 0: 90% of the GPU's free memory
  std::size_t maxGaussians = 0; // 0: as many as the budget allows
  // Share of the Gaussians removed, those adding least to the training views,
  // at the end of densification and again halfway through the rest. 0: none.
  // Densifying by score (scoreViews) already keeps the count low, and over 5000
  // iterations 30% each time then cost 0.3 dB and 70% cost 3 dB on held-out views.
  double pruneShare = 0.0;
  // The gradient threshold rises by this factor during densification, centred
  // on gradientThreshold, so that broad shapes come first (Improving ADC). 1: fixed.
  double thresholdRise = 1.0;
  float splitThreshold = 1.2e-3f; // Above 0: see DensifyOptions::splitThreshold
  double denseFraction = 0.001;   // Of the scene's extent: larger Gaussians split, smaller clone
  int densifyInterval = 100;      // Iterations between densification rounds
  int scoreViews = 10;            // Above 0: views scored per round, see DensifyOptions::scoredViews
  float scoreMinPixels = 5.f;     // See DensifyOptions::minMarkedPixels
  int explore = 0;                // Random splits and random new Gaussians per round
  bool significancePrune = false; // Of the too transparent remove the half adding least, and none for size
  // Image sizes from the photos' spectra, Gaussians growing with the size, and
  // the position rate held until full size (DashGaussian). Off: it cost 0.7 dB
  // on the bicycle scene at 7000 iterations against the fixed 15% and 35%.
  bool adaptiveSchedule = false;
  // The Gaussian count may grow to this multiple of the starting points (with
  // the adaptive schedule, raised by what the scene asks for); 0: no such cap.
  double maxGrowth = 4.0;
  // 3DGS-MCMC's penalties on the loss, reg * mean(sigmoid(opacity)) and
  // reg * mean(exp(scale)) over the Gaussians, so that unneeded ones fade and
  // shrink. 0 is off.
  float opacityReg = 0.f;
  float scaleReg = 0.f;
  // 3DGS-MCMC's exploration noise on positions, as a multiple of the position
  // rate (its paper and gsplat use 5e5). 0 is off.
  float positionNoise = 0.f;
  // Mip-Splatting's filters: the 2D Mip filter in place of the 0.3 pixel
  // dilation, and the 3D smoothing filter from the training cameras.
  bool mipFilter2d = false;
  bool mipFilter3d = false;
  // Edge-aware long-axis splitting, see DensifyOptions::edgeSplit.
  bool edgeSplit = false;
  // Refine each training view's camera along with the Gaussians, for poses that are a little off.
  bool refinePoses = false;
  std::size_t densePoints = 50000;  // Start from up to this many more points, by patch matching
  // After densification, steps from the curvature instead of Adam's second moment (3DGS^2-TR).
  bool secondOrder = false;
  bool pointExtent = false;       // The scene's extent from the sparse points, not the cameras
  // Turn and move the saved Gaussians so the floor is level at y = 0, -y up.
  // Training itself stays in the cameras' frame.
  bool alignFloor = true;
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
  // The progress reports' PSNR on the probe view (the first held-out view, or
  // the first training view), by iteration, as measured before that
  // iteration's pruning and opacity reset.
  std::vector<std::pair<int, double>> progressPsnr;
};

// Trains Gaussians on a project with images and camera poses in sparse/0, and
// writes a 3DGS .ply. The Gaussians start at the sparse points and multiply
// where the images ask for more detail, within the memory budget.
TrainReport run_train(vk::Context &ctx, const TrainOptions &options);

} // namespace splat
