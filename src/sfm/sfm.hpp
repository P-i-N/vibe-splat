#pragma once

#include "ingest/ingest.hpp"
#include "sfm/colmap_model.hpp"

#include <filesystem>

namespace splat {

struct SfmImportReport {
  ColmapStats stats;
  std::size_t ingested = 0;     // Images in the project
  std::size_t unregistered = 0; // Project images the model has no pose for
};

// Imports a COLMAP sparse model into a project that has been ingested. Model
// images are matched to project images by file name. The model is stored as
// project/sparse/0/*.bin with names relative to the project (images/...). Model
// images that are not in the project are dropped. Throws std::runtime_error
// if no image matches.
SfmImportReport import_colmap_model(const std::filesystem::path &project, const std::filesystem::path &modelFolder);

enum class Matcher {
  Auto,       // Exhaustive for small photo sets, sequential for videos and large sets
  Exhaustive, // Every pair of images
  Sequential, // Neighbours in input order, plus a few at growing distances
};

struct SfmOptions {
  std::filesystem::path project;
  int maxFeatures = 4000; // Per image
  Matcher matcher = Matcher::Auto;
  int window = 8;  // Neighbours per image for the sequential matcher
  int threads = 0; // 0 uses all cores
};

// One camera's intrinsics before and after bundle adjustment.
struct SfmCameraReport {
  int id = 0;
  int width = 0;
  int height = 0;
  FocalSource focalSource = FocalSource::Default; // Where the initial focal came from
  double initialFocal = 0.0;                      // Pixels, from ingest
  double focal = 0.0;                             // Pixels, after refinement (equal when manual)
  double radial = 0.0;                            // SIMPLE_RADIAL k
};

struct SfmReport {
  std::vector<SfmCameraReport> cameras;
  ColmapStats stats;
  std::size_t ingested = 0;
  std::size_t unregistered = 0;
  std::size_t pairsTried = 0;
  std::size_t pairsVerified = 0;
};

// Estimates camera poses and a sparse point cloud from an ingested project, and
// writes them to project/sparse/0 in COLMAP's binary format, plus preview.ply
// for `splat view`. Throws std::runtime_error if no reconstruction can start.
//
// The steps, in order:
//   1. SIFT features in every image.
//   2. Matching, then a five-point RANSAC to keep geometrically sound pairs.
//   3. Tracks: chains of matches that follow one 3D point across images.
//   4. A first pair of images with enough baseline, triangulated.
//   5. Repeatedly: register the image seeing the most known points (P3P RANSAC),
//      triangulate what it adds, and bundle adjust now and then.
//   6. A final bundle adjustment that also refines focal length and distortion.
SfmReport run_sfm(const SfmOptions &options);

} // namespace splat
