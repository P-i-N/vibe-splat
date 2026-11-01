#include "sfm/sfm.hpp"

#include "ingest/ingest.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace splat {

namespace fs = std::filesystem;

SfmImportReport import_colmap_model(const fs::path &project, const fs::path &modelFolder) {
  const IngestManifest manifest = load_manifest(project / "ingest.json");
  ColmapModel model = load_colmap_model(modelFolder);

  std::map<std::string, std::string> projectFiles; // File name -> path relative to the project
  for (const IngestImage &image : manifest.images) {
    projectFiles[fs::path(image.file).filename().string()] = image.file;
  }

  std::set<std::string> registered;
  std::set<std::uint32_t> dropped;
  for (auto it = model.images.begin(); it != model.images.end();) {
    const auto match = projectFiles.find(fs::path(it->second.name).filename().string());
    if (match == projectFiles.end()) {
      dropped.insert(it->first);
      it = model.images.erase(it);
    } else {
      it->second.name = match->second;
      registered.insert(match->second);
      ++it;
    }
  }
  if (registered.empty()) {
    throw std::runtime_error(std::format("no image of the model in {} is part of project {}", modelFolder.string(), project.string()));
  }

  // Tracks must not refer to images that are gone, and points need two views to mean anything.
  if (!dropped.empty()) {
    std::set<std::uint64_t> removedPoints;
    for (ColmapPoint &point : model.points) {
      std::erase_if(point.track, [&](const ColmapTrackElement &t) { return dropped.contains(t.imageId); });
      if (point.track.size() < 2) removedPoints.insert(point.id);
    }
    std::erase_if(model.points, [&](const ColmapPoint &p) { return removedPoints.contains(p.id); });
    for (auto &[id, image] : model.images) {
      for (ColmapObservation &o : image.observations) {
        if (removedPoints.contains(o.pointId)) o.pointId = kNoPoint;
      }
    }
  }

  save_colmap_model(model, project / "sparse" / "0");

  SfmImportReport report;
  report.stats = compute_stats(model);
  report.ingested = manifest.images.size();
  report.unregistered = manifest.images.size() - registered.size();
  return report;
}

} // namespace splat
