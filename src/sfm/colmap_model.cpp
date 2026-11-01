#include "sfm/colmap_model.hpp"

#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace splat {

namespace fs = std::filesystem;

namespace {

// Checks counts against the bytes left, so a corrupt count fails cleanly
// instead of asking for terabytes of memory.
class Reader {
public:
  explicit Reader(const fs::path &path) : path_(path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error(std::format("cannot open {}", path.string()));
    data_.resize(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(data_.data(), static_cast<std::streamsize>(data_.size()));
    if (!in) throw std::runtime_error(std::format("cannot read {}", path.string()));
  }

  template <class T> [[nodiscard]] T read() {
    static_assert(std::is_trivially_copyable_v<T>);
    require(sizeof(T));
    T value;
    std::memcpy(&value, data_.data() + at_, sizeof(T));
    at_ += sizeof(T);
    return value;
  }

  [[nodiscard]] std::string read_string() {
    const auto end = data_.find('\0', at_);
    if (end == std::string::npos) fail();
    std::string text = data_.substr(at_, end - at_);
    at_ = end + 1;
    return text;
  }

  [[nodiscard]] std::size_t checked_count(std::uint64_t count, std::size_t minBytesEach) {
    if (count > (data_.size() - at_) / minBytesEach) fail();
    return static_cast<std::size_t>(count);
  }

  [[nodiscard]] bool at_end() const { return at_ == data_.size(); }

  [[noreturn]] void fail() const { throw std::runtime_error(std::format("{} is truncated or corrupt", path_.string())); }

private:
  void require(std::size_t bytes) const {
    if (data_.size() - at_ < bytes) fail();
  }

  fs::path path_;
  std::string data_;
  std::size_t at_ = 0;
};

class Writer {
public:
  explicit Writer(const fs::path &path) : path_(path), out_(path, std::ios::binary) {}
  template <class T> void write(const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out_.write(reinterpret_cast<const char *>(&value), sizeof(T));
  }
  void write_string(const std::string &text) { out_.write(text.c_str(), static_cast<std::streamsize>(text.size() + 1)); }
  void finish() {
    out_.close();
    if (!out_) throw std::runtime_error(std::format("cannot write {}", path_.string()));
  }

private:
  fs::path path_;
  std::ofstream out_;
};

glm::dvec3 rotate(const std::array<double, 4> &q, const glm::dvec3 &v) {
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double w = q[0] / norm;
  const glm::dvec3 u(q[1] / norm, q[2] / norm, q[3] / norm);
  return v + 2.0 * glm::cross(u, glm::cross(u, v) + w * v);
}

} // namespace

glm::dvec3 ColmapImage::to_camera(const glm::dvec3 &world) const { return rotate(rotation, world) + translation; }

glm::dvec3 ColmapImage::center() const {
  const std::array<double, 4> inverse{rotation[0], -rotation[1], -rotation[2], -rotation[3]};
  return -rotate(inverse, translation);
}

const char *colmap_model_name(int model) noexcept {
  static constexpr const char *kNames[] = {"SIMPLE_PINHOLE", "PINHOLE",  "SIMPLE_RADIAL",        "RADIAL",
                                           "OPENCV",         "OPENCV_FISHEYE", "FULL_OPENCV",    "FOV",
                                           "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"};
  return model >= 0 && model < static_cast<int>(std::size(kNames)) ? kNames[model] : nullptr;
}

int colmap_model_param_count(int model) noexcept {
  static constexpr int kCounts[] = {3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12};
  return model >= 0 && model < static_cast<int>(std::size(kCounts)) ? kCounts[model] : -1;
}

ColmapModel load_colmap_model(const fs::path &folder) {
  ColmapModel model;

  {
    Reader in(folder / "cameras.bin");
    const std::size_t count = in.checked_count(in.read<std::uint64_t>(), 24);
    for (std::size_t i = 0; i < count; ++i) {
      ColmapCamera camera;
      camera.id = in.read<std::uint32_t>();
      camera.model = in.read<std::int32_t>();
      camera.width = in.read<std::uint64_t>();
      camera.height = in.read<std::uint64_t>();
      const int params = colmap_model_param_count(camera.model);
      if (params < 0) throw std::runtime_error(std::format("unknown camera model id {}", camera.model));
      camera.params.resize(static_cast<std::size_t>(params));
      for (double &p : camera.params) p = in.read<double>();
      model.cameras[camera.id] = std::move(camera);
    }
    if (!in.at_end()) in.fail();
  }

  {
    Reader in(folder / "images.bin");
    const std::size_t count = in.checked_count(in.read<std::uint64_t>(), 73);
    for (std::size_t i = 0; i < count; ++i) {
      ColmapImage image;
      image.id = in.read<std::uint32_t>();
      for (double &q : image.rotation) q = in.read<double>();
      image.translation.x = in.read<double>();
      image.translation.y = in.read<double>();
      image.translation.z = in.read<double>();
      image.cameraId = in.read<std::uint32_t>();
      image.name = in.read_string();
      image.observations.resize(in.checked_count(in.read<std::uint64_t>(), 24));
      for (ColmapObservation &o : image.observations) {
        o.xy.x = in.read<double>();
        o.xy.y = in.read<double>();
        o.pointId = in.read<std::uint64_t>();
      }
      if (!model.cameras.contains(image.cameraId)) {
        throw std::runtime_error(std::format("image {} uses unknown camera {}", image.name, image.cameraId));
      }
      if (image.rotation[0] == 0.0 && image.rotation[1] == 0.0 && image.rotation[2] == 0.0 && image.rotation[3] == 0.0) {
        throw std::runtime_error(std::format("image {} has a zero rotation", image.name));
      }
      model.images[image.id] = std::move(image);
    }
    if (!in.at_end()) in.fail();
  }

  {
    Reader in(folder / "points3D.bin");
    const std::size_t count = in.checked_count(in.read<std::uint64_t>(), 51);
    model.points.resize(count);
    for (ColmapPoint &point : model.points) {
      point.id = in.read<std::uint64_t>();
      point.position.x = in.read<double>();
      point.position.y = in.read<double>();
      point.position.z = in.read<double>();
      for (std::uint8_t &c : point.color) c = in.read<std::uint8_t>();
      point.error = in.read<double>();
      point.track.resize(in.checked_count(in.read<std::uint64_t>(), 8));
      for (ColmapTrackElement &t : point.track) {
        t.imageId = in.read<std::uint32_t>();
        t.pointIndex = in.read<std::uint32_t>();
      }
    }
    if (!in.at_end()) in.fail();
  }
  return model;
}

void save_colmap_model(const ColmapModel &model, const fs::path &folder) {
  fs::create_directories(folder);

  Writer cameras(folder / "cameras.bin");
  cameras.write<std::uint64_t>(model.cameras.size());
  for (const auto &[id, camera] : model.cameras) {
    cameras.write<std::uint32_t>(camera.id);
    cameras.write<std::int32_t>(camera.model);
    cameras.write<std::uint64_t>(camera.width);
    cameras.write<std::uint64_t>(camera.height);
    for (const double p : camera.params) cameras.write(p);
  }
  cameras.finish();

  Writer images(folder / "images.bin");
  images.write<std::uint64_t>(model.images.size());
  for (const auto &[id, image] : model.images) {
    images.write<std::uint32_t>(image.id);
    for (const double q : image.rotation) images.write(q);
    images.write(image.translation.x);
    images.write(image.translation.y);
    images.write(image.translation.z);
    images.write<std::uint32_t>(image.cameraId);
    images.write_string(image.name);
    images.write<std::uint64_t>(image.observations.size());
    for (const ColmapObservation &o : image.observations) {
      images.write(o.xy.x);
      images.write(o.xy.y);
      images.write<std::uint64_t>(o.pointId);
    }
  }
  images.finish();

  Writer points(folder / "points3D.bin");
  points.write<std::uint64_t>(model.points.size());
  for (const ColmapPoint &point : model.points) {
    points.write<std::uint64_t>(point.id);
    points.write(point.position.x);
    points.write(point.position.y);
    points.write(point.position.z);
    for (const std::uint8_t c : point.color) points.write(c);
    points.write(point.error);
    points.write<std::uint64_t>(point.track.size());
    for (const ColmapTrackElement &t : point.track) {
      points.write<std::uint32_t>(t.imageId);
      points.write<std::uint32_t>(t.pointIndex);
    }
  }
  points.finish();
}

bool colmap_project(const ColmapCamera &camera, const glm::dvec3 &p, glm::dvec2 &pixel) {
  if (p.z <= 0.0) return false;
  const double x = p.x / p.z, y = p.y / p.z;
  const double r2 = x * x + y * y;
  const auto &k = camera.params;
  switch (camera.model) {
  case 0: // SIMPLE_PINHOLE: f, cx, cy
    pixel = {k[0] * x + k[1], k[0] * y + k[2]};
    return true;
  case 1: // PINHOLE: fx, fy, cx, cy
    pixel = {k[0] * x + k[2], k[1] * y + k[3]};
    return true;
  case 2: { // SIMPLE_RADIAL: f, cx, cy, k
    const double d = 1.0 + k[3] * r2;
    pixel = {k[0] * d * x + k[1], k[0] * d * y + k[2]};
    return true;
  }
  case 3: { // RADIAL: f, cx, cy, k1, k2
    const double d = 1.0 + r2 * (k[3] + k[4] * r2);
    pixel = {k[0] * d * x + k[1], k[0] * d * y + k[2]};
    return true;
  }
  case 4: { // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    const double d = 1.0 + r2 * (k[4] + k[5] * r2);
    const double xd = d * x + 2.0 * k[6] * x * y + k[7] * (r2 + 2.0 * x * x);
    const double yd = d * y + k[6] * (r2 + 2.0 * y * y) + 2.0 * k[7] * x * y;
    pixel = {k[0] * xd + k[2], k[1] * yd + k[3]};
    return true;
  }
  default: return false;
  }
}

ColmapStats compute_stats(const ColmapModel &model) {
  ColmapStats stats;
  stats.cameras = model.cameras.size();
  stats.images = model.images.size();
  stats.points = model.points.size();

  std::size_t trackElements = 0;
  for (const ColmapPoint &point : model.points) trackElements += point.track.size();
  if (stats.points > 0) stats.meanTrackLength = static_cast<double>(trackElements) / static_cast<double>(stats.points);

  std::map<std::uint64_t, const ColmapPoint *> byId;
  for (const ColmapPoint &point : model.points) byId[point.id] = &point;

  double errorSum = 0.0;
  std::size_t errorCount = 0, observed = 0;
  for (const auto &[id, image] : model.images) {
    const ColmapCamera &camera = model.cameras.at(image.cameraId);
    for (const ColmapObservation &o : image.observations) {
      if (o.pointId == kNoPoint) continue;
      ++observed;
      const auto it = byId.find(o.pointId);
      glm::dvec2 pixel;
      if (it != byId.end() && colmap_project(camera, image.to_camera(it->second->position), pixel)) {
        errorSum += glm::length(pixel - o.xy);
        ++errorCount;
      }
    }
  }
  if (stats.images > 0) stats.meanObservationsPerImage = static_cast<double>(observed) / static_cast<double>(stats.images);
  if (errorCount > 0) {
    stats.meanReprojectionError = errorSum / static_cast<double>(errorCount);
  } else {
    stats.meanReprojectionError = std::numeric_limits<double>::quiet_NaN();
  }
  return stats;
}

} // namespace splat
