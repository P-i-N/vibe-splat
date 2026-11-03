#pragma once

#include "core/vk_context.hpp"
#include "render/camera.hpp"
#include "sfm/colmap_model.hpp"
#include "train/trainer.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
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
  int holdout = 0;  // Keep every nth view out of training; 0 trains on all
  std::uint32_t seed = 1;
  int reportEvery = 500;
  int saveEvery = 5000; // 0 only saves at the end

  bool densify = true;
  float gradientThreshold = 2e-4f;
  double vramBudgetGiB = 0.0; // 0: a share of the GPU's free memory, see vramShare
  double vramShare = 0.9;     // Of the free memory, when no budget is given
  std::size_t maxGaussians = 0; // 0: as many as the budget allows
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

// One posed photo, undistorted and ready to train on.
struct TrainView {
  Camera camera;
  std::vector<std::uint8_t> rgba;
  std::string name;
};

// The CPU side of training: the sparse model and its decoded images. Loading
// takes a while for many photos, so it is separate and needs no GPU.
struct TrainData {
  ColmapModel model;
  std::vector<TrainView> trainViews, holdoutViews;
  double extent = 1.0;       // Radius of the camera positions around their mean
  double typicalDepth = 1.0; // Median distance of the points, plus the extent
};

// The pinhole camera of a posed image, with its lens distortion ignored, for
// images resized by `scale`.
[[nodiscard]] Camera posed_camera(const ColmapImage &image, const ColmapCamera &camera, double scale = 1.0);

// Loads project/sparse/0 and its images, on all cores. `progress` is called
// after each image with how many are done and how many there are, from the
// loading threads but never two at once.
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
  void save(const std::filesystem::path &path) const;

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
  template <class F> auto fitting(F &&work);

  vk::Context &ctx_;
  TrainOptions options_;
  TrainData data_;
  std::function<void(const std::string &)> log_;
  TrainerOptions trainerOptions_;
  std::unique_ptr<Trainer> trainer_;
  double budgetGiB_ = 0.0;

  int densifyFrom_ = 0, densifyUntil_ = 0, shInterval_ = 1;
  std::mt19937 rng_, densifyRng_;
  std::vector<std::size_t> order_;
  LearningRates rates_;
  double positionStart_ = 0.0, positionEnd_ = 0.0;
  double entriesPerVisible_ = 16.0;
  double lossSum_ = 0.0;
  int lossCount_ = 0;
  int lastOpacityReset_ = 0;
  bool checkpointDue_ = false;
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
