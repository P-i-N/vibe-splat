#include "ingest/ingest.hpp"

#include "core/progress.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <span>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace splat {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

namespace {

constexpr int kManifestVersion = 1;
constexpr double kDefaultFocalFactor = 1.2; // Same guess COLMAP starts from
constexpr double kFullFrameDiagonalMm = 43.2666;

std::string lower_extension(const fs::path &path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

bool is_image_file(const fs::path &path) {
  const std::string ext = lower_extension(path);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tga";
}

bool is_video_file(const fs::path &path) {
  const std::string ext = lower_extension(path);
  return ext == ".mp4" || ext == ".mov" || ext == ".m4v" || ext == ".mkv" || ext == ".webm" || ext == ".avi";
}

bool is_unsupported_photo(const fs::path &path) {
  const std::string ext = lower_extension(path);
  return ext == ".heic" || ext == ".heif" || ext == ".avif" || ext == ".dng" || ext == ".cr2" || ext == ".nef" ||
         ext == ".arw" || ext == ".webp" || ext == ".tif" || ext == ".tiff";
}

// Compares runs of digits by value, the rest case-insensitively.
bool natural_less(const std::string &a, const std::string &b) {
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j]))) {
      while (i < a.size() && a[i] == '0') ++i;
      while (j < b.size() && b[j] == '0') ++j;
      std::size_t ie = i, je = j;
      while (ie < a.size() && std::isdigit(static_cast<unsigned char>(a[ie]))) ++ie;
      while (je < b.size() && std::isdigit(static_cast<unsigned char>(b[je]))) ++je;
      if (ie - i != je - j) return ie - i < je - j;
      if (const int c = a.compare(i, ie - i, b, j, je - j); c != 0) return c < 0;
      i = ie;
      j = je;
    } else {
      const int ca = std::tolower(static_cast<unsigned char>(a[i]));
      const int cb = std::tolower(static_cast<unsigned char>(b[j]));
      if (ca != cb) return ca < cb;
      ++i;
      ++j;
    }
  }
  return a.size() - i < b.size() - j;
}

std::vector<std::uint8_t> read_file(const fs::path &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open file");
  std::vector<std::uint8_t> data(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!in) throw std::runtime_error("cannot read file");
  return data;
}

// What identifies one physical camera setting; images that share it can share intrinsics.
using CameraKey = std::tuple<int, int, std::int64_t, int, std::string, std::string, std::int64_t, std::int64_t>;

class CameraSet {
public:
  // Scale goes from the size the focal estimate was made for to the stored size.
  int add(const RgbImage &stored, double scale, const FocalEstimate &focal, const ExifData &exif, const ManualIntrinsics &manual) {
    IngestCamera camera;
    camera.width = stored.width;
    camera.height = stored.height;
    camera.fx = camera.fy = focal.pixels * scale;
    camera.cx = manual.cx ? *manual.cx * scale : stored.width * 0.5;
    camera.cy = manual.cy ? *manual.cy * scale : stored.height * 0.5;
    camera.focalSource = focal.source;
    camera.make = exif.make;
    camera.model = exif.model;

    // Round the focal length so that metadata noise does not split a camera.
    const CameraKey key{camera.width,
                        camera.height,
                        std::llround(camera.fx * 100.0),
                        static_cast<int>(camera.focalSource),
                        camera.make,
                        camera.model,
                        std::llround(camera.cx * 100.0),
                        std::llround(camera.cy * 100.0)};
    const std::lock_guard lock(mutex_);
    const auto [it, inserted] = ids_.try_emplace(key, static_cast<int>(ids_.size()) + 1);
    if (inserted) {
      camera.id = it->second;
      cameras_.push_back(std::move(camera));
    }
    return it->second;
  }

  // Ids in the order of the images that use them, so results do not depend on threading.
  void finish(IngestManifest &manifest) const {
    std::map<int, int> renumber;
    for (IngestImage &image : manifest.images) {
      const auto [it, inserted] = renumber.try_emplace(image.cameraId, static_cast<int>(renumber.size()) + 1);
      if (inserted) {
        auto camera = *std::find_if(cameras_.begin(), cameras_.end(), [&](const IngestCamera &c) { return c.id == image.cameraId; });
        camera.id = it->second;
        manifest.cameras.push_back(std::move(camera));
      }
      image.cameraId = it->second;
    }
  }

private:
  std::mutex mutex_;
  std::map<CameraKey, int> ids_;
  std::vector<IngestCamera> cameras_;
};

struct Source {
  fs::path path;
  bool video = false;
};

// Stores one finished image and returns its manifest entry.
struct ImageStore {
  const IngestOptions &options;
  CameraSet &cameras;

  [[nodiscard]] IngestImage store(RgbImage image, const std::string &name, const ExifData &exif, const fs::path &source,
                                  std::optional<double> videoSeconds) {
    const FocalEstimate focal = estimate_focal(image.width, image.height, exif, options.manual);
    const int originalLongest = std::max(image.width, image.height);
    image = downscale_to_fit(std::move(image), options.maxSize);
    const double scale = static_cast<double>(std::max(image.width, image.height)) / originalLongest;

    const std::string file = "images/" + name + (options.png ? ".png" : ".jpg");
    write_image(image, options.project / file, options.jpegQuality);

    IngestImage entry;
    entry.file = file;
    entry.cameraId = cameras.add(image, scale, focal, exif, options.manual);
    entry.source = source.string();
    entry.videoSeconds = videoSeconds;
    entry.captureTime = exif.dateTimeOriginal;
    return entry;
  }
};

// Removes what an earlier ingest wrote, and nothing else.
void clear_previous(const fs::path &project) {
  const fs::path manifestPath = project / "ingest.json";
  if (!fs::exists(manifestPath)) return;
  for (const IngestImage &image : load_manifest(manifestPath).images) {
    fs::remove(project / image.file);
  }
  fs::remove(manifestPath);
}

} // namespace

const char *to_string(FocalSource source) noexcept {
  switch (source) {
  case FocalSource::Manual: return "manual";
  case FocalSource::Exif35mm: return "exif-35mm";
  case FocalSource::ExifPlane: return "exif-focal-plane";
  default: return "default";
  }
}

FocalSource focal_source_from_string(const std::string &text) {
  for (const FocalSource s : {FocalSource::Manual, FocalSource::Exif35mm, FocalSource::ExifPlane, FocalSource::Default}) {
    if (text == to_string(s)) return s;
  }
  throw std::runtime_error(std::format("unknown focal source '{}'", text));
}

FocalEstimate estimate_focal(int width, int height, const ExifData &exif, const ManualIntrinsics &manual) {
  const double longest = std::max(width, height);
  const double diagonal = std::hypot(static_cast<double>(width), static_cast<double>(height));

  if (manual.focalPx) return {*manual.focalPx, FocalSource::Manual};
  if (manual.focalMm) {
    if (!manual.sensorWidthMm) throw std::runtime_error("--focal-mm needs --sensor-width-mm");
    return {*manual.focalMm / *manual.sensorWidthMm * longest, FocalSource::Manual};
  }
  if (manual.focal35mm) return {*manual.focal35mm * diagonal / kFullFrameDiagonalMm, FocalSource::Manual};
  if (manual.fovDeg) {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    return {0.5 * width / std::tan(0.5 * *manual.fovDeg * kDegToRad), FocalSource::Manual};
  }

  // The 35 mm equivalent is defined by the diagonal, so it holds for any aspect ratio.
  if (exif.focalLength35mm) return {*exif.focalLength35mm * diagonal / kFullFrameDiagonalMm, FocalSource::Exif35mm};

  // Focal plane resolution counts pixels of the camera's own full-size image,
  // which may be larger than this file.
  if (exif.focalLengthMm && exif.focalPlaneXResolution) {
    const double recorded = exif.pixelXDimension && exif.pixelYDimension
                                ? std::max(*exif.pixelXDimension, *exif.pixelYDimension)
                                : longest;
    if (recorded > 0.0) {
      return {*exif.focalLengthMm * *exif.focalPlaneXResolution * longest / recorded, FocalSource::ExifPlane};
    }
  }
  return {kDefaultFocalFactor * longest, FocalSource::Default};
}

void save_manifest(const IngestManifest &manifest, const fs::path &path) {
  json root;
  root["version"] = kManifestVersion;
  json cameras = json::array();
  for (const IngestCamera &c : manifest.cameras) {
    cameras.push_back({{"id", c.id},
                       {"model", "PINHOLE"},
                       {"width", c.width},
                       {"height", c.height},
                       {"fx", c.fx},
                       {"fy", c.fy},
                       {"cx", c.cx},
                       {"cy", c.cy},
                       {"focal_source", to_string(c.focalSource)},
                       {"device_make", c.make},
                       {"device_model", c.model}});
  }
  root["cameras"] = std::move(cameras);
  json images = json::array();
  for (const IngestImage &image : manifest.images) {
    json entry = {{"file", image.file}, {"camera", image.cameraId}, {"source", image.source}};
    if (image.videoSeconds) entry["video_seconds"] = *image.videoSeconds;
    if (!image.captureTime.empty()) entry["capture_time"] = image.captureTime;
    images.push_back(std::move(entry));
  }
  root["images"] = std::move(images);

  std::ofstream out(path, std::ios::binary);
  out << root.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
  out.close();
  if (!out) throw std::runtime_error(std::format("cannot write {}", path.string()));
}

IngestManifest load_manifest(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
  try {
    const json root = json::parse(in);
    if (root.at("version").get<int>() != kManifestVersion) {
      throw std::runtime_error("unsupported version");
    }
    IngestManifest manifest;
    for (const json &c : root.at("cameras")) {
      IngestCamera camera;
      camera.id = c.at("id").get<int>();
      camera.width = c.at("width").get<int>();
      camera.height = c.at("height").get<int>();
      camera.fx = c.at("fx").get<double>();
      camera.fy = c.at("fy").get<double>();
      camera.cx = c.at("cx").get<double>();
      camera.cy = c.at("cy").get<double>();
      camera.focalSource = focal_source_from_string(c.at("focal_source").get<std::string>());
      camera.make = c.value("device_make", "");
      camera.model = c.value("device_model", "");
      manifest.cameras.push_back(std::move(camera));
    }
    for (const json &i : root.at("images")) {
      IngestImage image;
      image.file = i.at("file").get<std::string>();
      image.cameraId = i.at("camera").get<int>();
      image.source = i.at("source").get<std::string>();
      if (i.contains("video_seconds")) image.videoSeconds = i.at("video_seconds").get<double>();
      image.captureTime = i.value("capture_time", "");
      manifest.images.push_back(std::move(image));
    }
    return manifest;
  } catch (const std::exception &e) {
    throw std::runtime_error(std::format("{}: invalid manifest: {}", path.string(), e.what()));
  }
}

std::vector<fs::path> expand_inputs(const std::vector<fs::path> &inputs) {
  std::vector<fs::path> files;
  for (const fs::path &input : inputs) {
    if (fs::is_directory(input)) {
      std::vector<fs::path> found;
      for (const fs::directory_entry &entry : fs::directory_iterator(input)) {
        if (entry.is_regular_file() && (is_image_file(entry.path()) || is_video_file(entry.path()) || is_unsupported_photo(entry.path()))) {
          found.push_back(entry.path());
        }
      }
      std::sort(found.begin(), found.end(), [](const fs::path &a, const fs::path &b) {
        return natural_less(a.filename().string(), b.filename().string());
      });
      files.insert(files.end(), found.begin(), found.end());
    } else if (fs::exists(input)) {
      files.push_back(input);
    } else {
      throw std::runtime_error(std::format("{} does not exist", input.string()));
    }
  }
  return files;
}

IngestReport run_ingest(const IngestOptions &options) {
  if (options.manual.focalMm && !options.manual.sensorWidthMm) {
    throw std::runtime_error("--focal-mm needs --sensor-width-mm");
  }

  IngestReport report;
  std::vector<Source> sources;
  for (const fs::path &file : expand_inputs(options.inputs)) {
    if (is_image_file(file)) {
      sources.push_back({file, false});
    } else if (is_video_file(file)) {
      if (!video_supported()) throw std::runtime_error("cannot read videos: splat was built without FFmpeg and ffmpeg is not on PATH");
      sources.push_back({file, true});
    } else {
      const std::string hint = is_unsupported_photo(file) ? " (convert it to JPEG or PNG first)" : "";
      std::println(stderr, "{}", console::paint(console::Color::Yellow, std::format("Skipping {}: unsupported file type{}", file.string(), hint), stderr));
      ++report.failed;
    }
  }
  if (sources.empty()) throw std::runtime_error("no images or videos to ingest");

  const fs::path manifestPath = options.project / "ingest.json";
  if (fs::exists(manifestPath) && !options.force) {
    throw std::runtime_error(std::format("{} already has an ingest; pass --force to replace it", options.project.string()));
  }
  clear_previous(options.project);
  fs::create_directories(options.project / "images");

  CameraSet cameras;
  ImageStore store{options, cameras};

  // One result list per source, so the output order follows the input order
  // whatever order the threads finish in.
  std::vector<std::vector<IngestImage>> results(sources.size());
  std::mutex reportMutex;
  const auto report_failure = [&](Progress &bar, const fs::path &path, const std::exception &e) {
    const std::lock_guard lock(reportMutex);
    bar.message(console::paint(console::Color::Yellow, std::format("Skipping {}: {}", path.string(), e.what()), stderr), true);
    ++report.failed;
  };

  // Names are fixed up front: a photo's number is its position among the photos.
  std::vector<std::size_t> photos;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i].video) photos.push_back(i);
  }

  Progress photoBar("Photos", photos.size());
  std::atomic<std::size_t> next{0};
  const auto photo_worker = [&] {
    for (std::size_t job = next++; job < photos.size(); job = next++) {
      const std::size_t index = photos[job];
      const fs::path &path = sources[index].path;
      try {
        const std::vector<std::uint8_t> bytes = read_file(path);
        const ExifData exif = parse_jpeg_exif(bytes);
        RgbImage image = apply_orientation(decode_image(bytes), exif.orientation);
        results[index].push_back(store.store(std::move(image), std::format("photo_{:06}", job + 1), exif, path, std::nullopt));
      } catch (const std::exception &e) {
        report_failure(photoBar, path, e);
      }
      photoBar.advance(1, path.filename().string());
    }
  };
  const unsigned threadCount = options.threads > 0 ? static_cast<unsigned>(options.threads)
                                                   : std::max(1u, std::thread::hardware_concurrency());
  {
    std::vector<std::jthread> workers;
    for (unsigned t = 0; t < std::min<std::size_t>(threadCount, photos.size()); ++t) workers.emplace_back(photo_worker);
  }
  photoBar.finish();

  // Video decoding is already threaded inside FFmpeg.
  int videoNumber = 0;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i].video) continue;
    ++videoNumber;
    int frameNumber = 0;
    Progress videoBar(std::format("Video {}", videoNumber), 0);
    try {
      extract_video_frames(sources[i].path, options.video, [&](VideoFrame &&frame) {
        ++frameNumber;
        const std::string name = std::format("video{:02}_{:06}", videoNumber, frameNumber);
        results[i].push_back(store.store(std::move(frame.image), name, ExifData{}, sources[i].path, frame.seconds));
        videoBar.set(static_cast<std::size_t>(frameNumber), sources[i].path.filename().string());
      });
    } catch (const std::exception &e) {
      report_failure(videoBar, sources[i].path, e);
    }
  }

  IngestManifest manifest;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i].video) report.videoFrames += results[i].size();
    else report.images += results[i].size();
    manifest.images.insert(manifest.images.end(), results[i].begin(), results[i].end());
  }
  if (manifest.images.empty()) throw std::runtime_error("none of the inputs could be read");

  cameras.finish(manifest);
  report.cameras = manifest.cameras.size();
  save_manifest(manifest, manifestPath);
  return report;
}

} // namespace splat
