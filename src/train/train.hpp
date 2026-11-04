#pragma once

#include "core/vk_context.hpp"
#include "render/camera.hpp"
#include "sfm/colmap_model.hpp"
#include "splat/cgc.hpp"
#include "train/schedule.hpp"
#include "train/trainer.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
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
  double vramBudgetGiB = 0.0; // 0: a share of the GPU's free memory, see vramShare
  double vramShare = 0.9;     // Of the free memory, when no budget is given
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
  // Below 0: 4 for every 7000 iterations (see growth_factor()).
  double maxGrowth = -1.0;
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
  // This share of Gaussians, the least opaque, is removed 300 iterations after
  // each opacity reset (recovery-aware pruning, Improving Densification in 3DGS). 0: none.
  double recoveryPrune = 0.0;
  // SH dropout: with this probability per iteration, a Gaussian keeps only the
  // degrees up to 0, 1 and 2 before 20%, 40% and the rest of training. 0: off.
  float shDropout = 0.f;
  // Refine each training view's camera along with the Gaussians, for poses that are a little off.
  bool refinePoses = false;
  std::size_t densePoints = 50000;  // Start from up to this many more points, by patch matching
  std::size_t sparsePoints = 0;     // Start from a random sample of at most this many sparse points; 0: all
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

// One posed photo, undistorted and ready to train on.
struct TrainView {
  Camera camera;
  std::vector<std::uint8_t> rgba;
  std::string name;
};

// Coarse to fine: training starts at a quarter of the image size and is at
// half size until coarseUntil[0] of the way through, full size after.
constexpr std::size_t kCoarseLevels = 2;

// The CPU side of training: the sparse model, its decoded images and the
// starting points. Loading takes a while for many photos, so it is separate
// and needs no GPU.
struct TrainData {
  ColmapModel model;
  std::vector<TrainView> trainViews, holdoutViews;
  // Coarse to fine: smallerViews[l][i] is trainViews[i] at 1 / 2^(l + 1) of its
  // size. Empty with TrainOptions::fullResolution.
  std::vector<std::vector<TrainView>> smallerViews;
  // Until which share of training each smaller size is used: the fixed 35% and
  // 15%, or from the photos' spectra with the adaptive schedule.
  std::array<double, kCoarseLevels> coarseUntil{0.35, 0.15};
  double extent = 1.0;       // Radius of the camera positions around their mean (or the points' mean depth)
  double typicalDepth = 1.0; // Median distance of the points, plus the extent
  // The sparse points, and up to TrainOptions::densePoints more from patch matching.
  std::vector<ColmapPoint> startingPoints;
};

// The pinhole camera of a posed image, with its lens distortion ignored, for
// images resized by `scale`.
[[nodiscard]] Camera posed_camera(const ColmapImage &image, const ColmapCamera &camera, double scale = 1.0);

// Loads project/sparse/0 and its images, on all cores, then finds the dense
// starting points. `progress` is called after each image with how many are
// done and how many there are, from the loading threads but never two at once.
// The multiple of the starting points the Gaussians may grow to: maxGrowth when
// set (0: no cap), else 4 for every 7000 iterations, at least 4. The cap keeps
// short runs fast, where more Gaussians add little; a long run held to it hit
// the cap within 2000 of 30000 iterations and then gained no detail.
[[nodiscard]] double growth_factor(const TrainOptions &options) noexcept;

[[nodiscard]] TrainData load_train_data(const TrainOptions &options,
                                        const std::function<void(std::size_t, std::size_t)> &progress = {});

// The training loop, one iteration at a time, so that a viewer can show it as it
// goes. run_train() drives it to the end.
class TrainSession {
public:
  // `log` receives the progress lines; by default they go to stdout.
  TrainSession(vk::Context &ctx, const TrainOptions &options, TrainData data,
               std::function<void(const std::string &)> log = {});
  ~TrainSession();
  TrainSession(const TrainSession &) = delete;
  TrainSession &operator=(const TrainSession &) = delete;

  [[nodiscard]] bool finished() const noexcept { return iteration_ >= options_.iterations; }
  // Runs the next iteration: a step, and densification, reports and checkpoints when due.
  StepStats step();
  // Measures PSNR on every view, saves the result and the previews.
  TrainReport finish();
  // Saves the current Gaussians, through a temporary file.
  void save(const std::filesystem::path &path);

  [[nodiscard]] Trainer &trainer() noexcept { return *trainer_; }
  [[nodiscard]] const Trainer &trainer() const noexcept { return *trainer_; }
  [[nodiscard]] const TrainData &data() const noexcept { return data_; }
  [[nodiscard]] const TrainOptions &options() const noexcept { return options_; }
  [[nodiscard]] int iteration() const noexcept { return iteration_; }
  // Index into data().trainViews of the view the last step trained on.
  [[nodiscard]] std::size_t last_view() const noexcept { return lastView_; }
  // PSNR of the probe view at the last report, 0 before the first.
  [[nodiscard]] double last_psnr() const noexcept { return lastPsnr_; }
  [[nodiscard]] const TrainReport &report() const noexcept { return report_; }
  // Time spent inside step(), so that pauses do not count.
  [[nodiscard]] double training_seconds() const noexcept { return trainingSeconds_; }
  [[nodiscard]] double budget_gib() const noexcept { return budgetGiB_; }

private:
  struct ViewExposure; // What each view's camera did to its photo, learned
  struct ViewPose;     // How far each view's camera has been refined
  template <class F> auto fitting(F &&work);
  // Which of the smaller sizes `iteration` trains at: kCoarseLevels at full size.
  [[nodiscard]] std::size_t coarse_level(int iteration) const noexcept;
  // The view to train on, at the size the schedule asks for at `iteration`.
  [[nodiscard]] const TrainView &view_at(std::size_t index, int iteration) const;
  void refill_order();
  // Mip-Splatting's 3D smoothing filter from the cameras of the current size.
  void refresh_filter(int iteration);
  // Saves, and returns how the floor was found, or nothing if it was not looked for.
  std::string save_output(const std::filesystem::path &path, const CgcOptions &cgcOptions);

  vk::Context &ctx_;
  TrainOptions options_;
  TrainData data_;
  std::function<void(const std::string &)> log_;
  TrainerOptions trainerOptions_;
  std::unique_ptr<Trainer> trainer_;
  double budgetGiB_ = 0.0;

  int densifyFrom_ = 0, densifyUntil_ = 0, pruneAgainAt_ = 0, shInterval_ = 1;
  std::vector<ViewExposure> exposures_;
  std::vector<ViewPose> poses_;
  std::mt19937 rng_, densifyRng_;
  std::vector<std::size_t> order_;
  LearningRates rates_;
  double positionStart_ = 0.0, positionEnd_ = 0.0;
  GrowthBudget growth_;
  double entriesPerVisible_ = 16.0;
  double lossSum_ = 0.0;
  int lossCount_ = 0;
  int lastOpacityReset_ = 0;
  bool checkpointDue_ = false;
  double densifySeconds_ = 0.0;
  int densifyRounds_ = 0;
  double pruneSeconds_ = 0.0;
  std::vector<Camera> allCameras_; // Training and held-out, for levelling the floor
  std::vector<float> filter3d_;
  std::size_t filterLevel_ = ~std::size_t{0}; // Stale, and also after any change of the count outside a round
  int iteration_ = 0;
  std::size_t lastView_ = 0;
  double lastPsnr_ = 0.0;
  double trainingSeconds_ = 0.0;
  TrainReport report_;
};

// Trains Gaussians on a project with images and camera poses in sparse/0, and
// writes a 3DGS .ply. The Gaussians start at the sparse points and multiply
// where the images ask for more detail, within the memory budget.
TrainReport run_train(vk::Context &ctx, const TrainOptions &options);

} // namespace splat
