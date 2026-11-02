#include "splat/gaussian_cloud.hpp"

#include "splat/cgc.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace splat {
namespace {

constexpr float kShC0 = 0.28209479177387814f;

enum class PlyType { I8, U8, I16, U16, I32, U32, F32, F64 };

std::optional<PlyType> parse_type(std::string_view t) {
  if (t == "char" || t == "int8") return PlyType::I8;
  if (t == "uchar" || t == "uint8") return PlyType::U8;
  if (t == "short" || t == "int16") return PlyType::I16;
  if (t == "ushort" || t == "uint16") return PlyType::U16;
  if (t == "int" || t == "int32") return PlyType::I32;
  if (t == "uint" || t == "uint32") return PlyType::U32;
  if (t == "float" || t == "float32") return PlyType::F32;
  if (t == "double" || t == "float64") return PlyType::F64;
  return std::nullopt;
}

std::size_t type_size(PlyType t) {
  switch (t) {
  case PlyType::I8:
  case PlyType::U8: return 1;
  case PlyType::I16:
  case PlyType::U16: return 2;
  case PlyType::I32:
  case PlyType::U32:
  case PlyType::F32: return 4;
  case PlyType::F64: return 8;
  }
  return 0;
}

template <class T> T read_le(const std::byte *p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) {
    v = std::byteswap(v);
  }
  return v;
}

float read_binary(PlyType t, const std::byte *p) {
  switch (t) {
  case PlyType::I8: return static_cast<float>(read_le<std::int8_t>(p));
  case PlyType::U8: return static_cast<float>(read_le<std::uint8_t>(p));
  case PlyType::I16: return static_cast<float>(read_le<std::int16_t>(p));
  case PlyType::U16: return static_cast<float>(read_le<std::uint16_t>(p));
  case PlyType::I32: return static_cast<float>(read_le<std::int32_t>(p));
  case PlyType::U32: return static_cast<float>(read_le<std::uint32_t>(p));
  case PlyType::F32: return std::bit_cast<float>(read_le<std::uint32_t>(p));
  case PlyType::F64: return static_cast<float>(std::bit_cast<double>(read_le<std::uint64_t>(p)));
  }
  return 0.f;
}

struct Property {
  std::string name;
  PlyType type = PlyType::F32;
  std::size_t offset = 0; // Byte offset within a binary vertex record
};

struct Header {
  bool binary = true;
  std::size_t vertexCount = 0;
  std::vector<Property> props;
  std::size_t stride = 0;
};

Header parse_header(std::istream &in, const std::filesystem::path &path) {
  std::string line;
  if (!std::getline(in, line) || line.substr(0, 3) != "ply") {
    throw std::runtime_error(std::format("{}: not a PLY file", path.string()));
  }
  Header h;
  bool inVertex = false;
  bool sawVertex = false;
  bool sawFormat = false;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    std::istringstream ls(line);
    std::string word;
    ls >> word;
    if (word == "end_header") {
      break;
    }
    if (word == "comment" || word == "obj_info" || word.empty()) {
      continue;
    }
    if (word == "format") {
      std::string fmt;
      ls >> fmt;
      if (fmt == "binary_little_endian") {
        h.binary = true;
      } else if (fmt == "ascii") {
        h.binary = false;
      } else {
        throw std::runtime_error(std::format("{}: unsupported PLY format '{}'", path.string(), fmt));
      }
      sawFormat = true;
    } else if (word == "element") {
      std::string name;
      std::size_t count = 0;
      ls >> name >> count;
      if (name == "vertex") {
        if (sawVertex) {
          throw std::runtime_error(std::format("{}: more than one vertex element", path.string()));
        }
        inVertex = true;
        sawVertex = true;
        h.vertexCount = count;
      } else {
        if (!sawVertex && count > 0) {
          throw std::runtime_error(
              std::format("{}: element '{}' before the vertex element is not supported", path.string(), name));
        }
        inVertex = false;
      }
    } else if (word == "property") {
      if (!inVertex) {
        continue; // Properties of elements after "vertex" are never read.
      }
      std::string typeName;
      ls >> typeName;
      if (typeName == "list") {
        throw std::runtime_error(std::format("{}: list properties on vertices are not supported", path.string()));
      }
      std::string name;
      ls >> name;
      const auto type = parse_type(typeName);
      if (!type) {
        throw std::runtime_error(std::format("{}: unknown property type '{}'", path.string(), typeName));
      }
      h.props.push_back({name, *type, h.stride});
      h.stride += type_size(*type);
    }
  }
  if (!sawFormat || !sawVertex) {
    throw std::runtime_error(std::format("{}: PLY header has no format or vertex element", path.string()));
  }
  return h;
}

int degree_from_rest(std::size_t restCount) {
  switch (restCount) {
  case 0: return 0;
  case 9: return 1;
  case 24: return 2;
  case 45: return 3;
  default: return -1;
  }
}

} // namespace

void GaussianCloud::resize(std::size_t count, int degree) {
  shDegree = degree;
  positions.assign(count * 3, 0.f);
  scales.assign(count * 3, 0.f);
  rotations.assign(count * 4, 0.f);
  opacities.assign(count, 0.f);
  sh.assign(count * 3 * static_cast<std::size_t>(shCoeffs()), 0.f);
}

void GaussianCloud::validate() const {
  const std::size_t n = size();
  if (shDegree < 0 || shDegree > 3) {
    throw std::runtime_error("GaussianCloud: SH degree must be 0..3");
  }
  if (positions.size() != n * 3 || scales.size() != n * 3 || rotations.size() != n * 4 ||
      sh.size() != n * 3 * static_cast<std::size_t>(shCoeffs())) {
    throw std::runtime_error("GaussianCloud: inconsistent array sizes");
  }
}

Bounds compute_bounds(const GaussianCloud &cloud) {
  Bounds b;
  const std::size_t n = cloud.size();
  if (n == 0) {
    return b;
  }
  std::vector<float> axis(n);
  for (int a = 0; a < 3; ++a) {
    b.min[a] = std::numeric_limits<float>::max();
    b.max[a] = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i) {
      axis[i] = cloud.positions[i * 3 + a];
      b.min[a] = std::min(b.min[a], axis[i]);
      b.max[a] = std::max(b.max[a], axis[i]);
    }
    auto mid = axis.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(axis.begin(), mid, axis.end());
    b.center[a] = *mid;
  }
  std::vector<float> dist(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float dx = cloud.positions[i * 3 + 0] - b.center[0];
    const float dy = cloud.positions[i * 3 + 1] - b.center[1];
    const float dz = cloud.positions[i * 3 + 2] - b.center[2];
    dist[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  auto p90 = dist.begin() + static_cast<std::ptrdiff_t>(std::min(n - 1, n * 9 / 10));
  std::nth_element(dist.begin(), p90, dist.end());
  b.radius = std::max(*p90, 1e-3f);
  return b;
}

GaussianCloud load_ply(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::format("Cannot open '{}'", path.string()));
  }
  const Header h = parse_header(in, path);

  std::unordered_map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < h.props.size(); ++i) {
    index[h.props[i].name] = i;
  }
  auto require = [&](const std::string &name) {
    const auto it = index.find(name);
    if (it == index.end()) {
      throw std::runtime_error(std::format("{}: missing property '{}' (is this a Gaussian splat file?)",
                                           path.string(), name));
    }
    return it->second;
  };

  std::size_t restCount = 0;
  while (index.contains(std::format("f_rest_{}", restCount))) {
    ++restCount;
  }
  const int degree = degree_from_rest(restCount);
  if (degree < 0) {
    throw std::runtime_error(std::format("{}: unexpected number of f_rest properties ({})", path.string(), restCount));
  }

  const std::array<std::size_t, 3> pos{require("x"), require("y"), require("z")};
  const std::array<std::size_t, 3> scl{require("scale_0"), require("scale_1"), require("scale_2")};
  const std::array<std::size_t, 4> rot{require("rot_0"), require("rot_1"), require("rot_2"), require("rot_3")};
  const std::size_t opa = require("opacity");
  const std::array<std::size_t, 3> dc{require("f_dc_0"), require("f_dc_1"), require("f_dc_2")};
  std::vector<std::size_t> rest(restCount);
  for (std::size_t i = 0; i < restCount; ++i) {
    rest[i] = index.at(std::format("f_rest_{}", i));
  }

  GaussianCloud cloud;
  cloud.resize(h.vertexCount, degree);
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  std::vector<float> values(h.props.size());

  auto store = [&](std::size_t v) {
    for (int a = 0; a < 3; ++a) {
      cloud.positions[v * 3 + a] = values[pos[a]];
      cloud.scales[v * 3 + a] = values[scl[a]];
      cloud.sh[(v * coeffs) * 3 + a] = values[dc[a]];
    }
    for (int a = 0; a < 4; ++a) {
      cloud.rotations[v * 4 + a] = values[rot[a]];
    }
    cloud.opacities[v] = values[opa];
    // f_rest is channel-major in the file: all R coefficients, then G, then B.
    for (std::size_t c = 0; c < 3 && coeffs > 1; ++c) {
      for (std::size_t k = 1; k < coeffs; ++k) {
        cloud.sh[(v * coeffs + k) * 3 + c] = values[rest[c * (coeffs - 1) + (k - 1)]];
      }
    }
  };

  if (h.binary) {
    constexpr std::size_t kChunk = 65536;
    std::vector<std::byte> buffer(h.stride * kChunk);
    for (std::size_t first = 0; first < h.vertexCount; first += kChunk) {
      const std::size_t n = std::min(kChunk, h.vertexCount - first);
      in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(n * h.stride));
      if (static_cast<std::size_t>(in.gcount()) != n * h.stride) {
        throw std::runtime_error(std::format("{}: file is truncated", path.string()));
      }
      for (std::size_t i = 0; i < n; ++i) {
        const std::byte *rec = buffer.data() + i * h.stride;
        for (std::size_t p = 0; p < h.props.size(); ++p) {
          values[p] = read_binary(h.props[p].type, rec + h.props[p].offset);
        }
        store(first + i);
      }
    }
  } else {
    for (std::size_t v = 0; v < h.vertexCount; ++v) {
      for (std::size_t p = 0; p < h.props.size(); ++p) {
        double d = 0.0;
        if (!(in >> d)) {
          throw std::runtime_error(std::format("{}: file is truncated", path.string()));
        }
        values[p] = static_cast<float>(d);
      }
      store(v);
    }
  }
  return cloud;
}

void save_ply(const GaussianCloud &cloud, const std::filesystem::path &path) {
  cloud.validate();
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error(std::format("Cannot write '{}'", path.string()));
  }
  const std::size_t n = cloud.size();
  const std::size_t coeffs = static_cast<std::size_t>(cloud.shCoeffs());
  const std::size_t restCount = 3 * (coeffs - 1);

  std::string header = std::format("ply\nformat binary_little_endian 1.0\nelement vertex {}\n", n);
  for (const char *name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"}) {
    header += std::format("property float {}\n", name);
  }
  for (std::size_t i = 0; i < restCount; ++i) {
    header += std::format("property float f_rest_{}\n", i);
  }
  for (const char *name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
    header += std::format("property float {}\n", name);
  }
  header += "end_header\n";
  out.write(header.data(), static_cast<std::streamsize>(header.size()));

  const std::size_t floatsPerVertex = 9 + restCount + 8;
  std::vector<float> rec(floatsPerVertex);
  for (std::size_t v = 0; v < n; ++v) {
    std::size_t k = 0;
    for (int a = 0; a < 3; ++a) rec[k++] = cloud.positions[v * 3 + a];
    for (int a = 0; a < 3; ++a) rec[k++] = 0.f; // normals
    for (int a = 0; a < 3; ++a) rec[k++] = cloud.sh[(v * coeffs) * 3 + a];
    for (std::size_t c = 0; c < 3 && coeffs > 1; ++c) {
      for (std::size_t j = 1; j < coeffs; ++j) rec[k++] = cloud.sh[(v * coeffs + j) * 3 + c];
    }
    rec[k++] = cloud.opacities[v];
    for (int a = 0; a < 3; ++a) rec[k++] = cloud.scales[v * 3 + a];
    for (int a = 0; a < 4; ++a) rec[k++] = cloud.rotations[v * 4 + a];
    if constexpr (std::endian::native == std::endian::big) {
      for (float &f : rec) f = std::bit_cast<float>(std::byteswap(std::bit_cast<std::uint32_t>(f)));
    }
    out.write(reinterpret_cast<const char *>(rec.data()), static_cast<std::streamsize>(rec.size() * sizeof(float)));
  }
  if (!out) {
    throw std::runtime_error(std::format("Failed writing '{}'", path.string()));
  }
}

std::size_t remove_non_finite(GaussianCloud &cloud) {
  const std::size_t n = cloud.size();
  const std::size_t floats = static_cast<std::size_t>(3 * cloud.shCoeffs());
  const auto finite = [](std::span<const float> values) { return std::ranges::all_of(values, [](float v) { return std::isfinite(v); }); };
  std::size_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!finite(std::span(cloud.positions).subspan(i * 3, 3)) || !finite(std::span(cloud.scales).subspan(i * 3, 3)) ||
        !finite(std::span(cloud.rotations).subspan(i * 4, 4)) || !std::isfinite(cloud.opacities[i]) ||
        !finite(std::span(cloud.sh).subspan(i * floats, floats))) {
      continue;
    }
    if (kept != i) {
      std::copy_n(cloud.positions.begin() + static_cast<std::ptrdiff_t>(i * 3), 3, cloud.positions.begin() + static_cast<std::ptrdiff_t>(kept * 3));
      std::copy_n(cloud.scales.begin() + static_cast<std::ptrdiff_t>(i * 3), 3, cloud.scales.begin() + static_cast<std::ptrdiff_t>(kept * 3));
      std::copy_n(cloud.rotations.begin() + static_cast<std::ptrdiff_t>(i * 4), 4, cloud.rotations.begin() + static_cast<std::ptrdiff_t>(kept * 4));
      cloud.opacities[kept] = cloud.opacities[i];
      std::copy_n(cloud.sh.begin() + static_cast<std::ptrdiff_t>(i * floats), floats, cloud.sh.begin() + static_cast<std::ptrdiff_t>(kept * floats));
    }
    ++kept;
  }
  cloud.positions.resize(kept * 3);
  cloud.scales.resize(kept * 3);
  cloud.rotations.resize(kept * 4);
  cloud.opacities.resize(kept);
  cloud.sh.resize(kept * floats);
  return n - kept;
}

CloudFormat cloud_format(const std::filesystem::path &path) {
  std::string ext = path.extension().string();
  std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".cgc" ? CloudFormat::Cgc : CloudFormat::Ply;
}

void save_cloud(const GaussianCloud &cloud, const std::filesystem::path &path) { save_cloud(cloud, path, cloud_format(path)); }

void save_cloud(const GaussianCloud &cloud, const std::filesystem::path &path, CloudFormat format) {
  save_cloud(cloud, path, format, CgcOptions{});
}

void save_cloud(const GaussianCloud &cloud, const std::filesystem::path &path, CloudFormat format, const CgcOptions &options) {
  if (format == CloudFormat::Cgc) {
    save_cgc(cloud, path, options);
  } else {
    save_ply(cloud, path);
  }
}

GaussianCloud load_cloud(const std::filesystem::path &path) {
  return cloud_format(path) == CloudFormat::Cgc ? load_cgc(path) : load_ply(path);
}

GaussianCloud make_test_scene(std::size_t count, std::uint32_t seed) {
  // Convention: y points down (as in COLMAP), so the floor is at +y.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  std::normal_distribution<float> normal(0.f, 1.f);

  GaussianCloud cloud;
  cloud.resize(count, 1);
  const std::size_t coeffs = 4;
  const std::size_t sphereCount = count * 3 / 5;
  constexpr int kSpheres = 8;
  constexpr float kRing = 2.5f;
  constexpr float kSphereRadius = 0.6f;
  constexpr float kFloorY = kSphereRadius;
  constexpr float kFloorHalf = 4.5f;

  auto set_color = [&](std::size_t v, float r, float g, float b) {
    cloud.sh[(v * coeffs) * 3 + 0] = (r - 0.5f) / kShC0;
    cloud.sh[(v * coeffs) * 3 + 1] = (g - 0.5f) / kShC0;
    cloud.sh[(v * coeffs) * 3 + 2] = (b - 0.5f) / kShC0;
  };
  auto hue = [](float h, float out[3]) {
    for (int c = 0; c < 3; ++c) {
      const float k = std::fmod(h * 6.f + static_cast<float>(4 - 2 * c) + 6.f, 6.f);
      out[c] = 1.f - std::clamp(std::min(k, 4.f - k), 0.f, 1.f);
    }
  };

  for (std::size_t v = 0; v < count; ++v) {
    cloud.rotations[v * 4 + 0] = 1.f; // identity (w, x, y, z)
    cloud.opacities[v] = 4.f;         // sigmoid(4) ~= 0.98
    if (v < sphereCount) {
      const int s = static_cast<int>(v % kSpheres);
      const float angle = 2.f * std::numbers::pi_v<float> * static_cast<float>(s) / kSpheres;
      float dir[3] = {normal(rng), normal(rng), normal(rng)};
      const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) + 1e-9f;
      for (float &d : dir) d /= len;
      cloud.positions[v * 3 + 0] = kRing * std::cos(angle) + kSphereRadius * dir[0];
      cloud.positions[v * 3 + 1] = kSphereRadius * dir[1];
      cloud.positions[v * 3 + 2] = kRing * std::sin(angle) + kSphereRadius * dir[2];
      for (int a = 0; a < 3; ++a) cloud.scales[v * 3 + a] = std::log(0.03f);
      float rgb[3];
      hue(static_cast<float>(s) / kSpheres, rgb);
      // Simple shading baked into the colour: lighter towards the top (-y).
      const float shade = 0.65f + 0.35f * (-dir[1]);
      set_color(v, rgb[0] * shade, rgb[1] * shade, rgb[2] * shade);
      // A little view dependence (degree-1 SH) so the SH path is exercised.
      for (int c = 0; c < 3; ++c) {
        cloud.sh[(v * coeffs + 3) * 3 + c] = 0.15f * (c == 2 ? 1.f : -0.5f);
      }
    } else {
      const float x = (uni(rng) * 2.f - 1.f) * kFloorHalf;
      const float z = (uni(rng) * 2.f - 1.f) * kFloorHalf;
      cloud.positions[v * 3 + 0] = x;
      cloud.positions[v * 3 + 1] = kFloorY;
      cloud.positions[v * 3 + 2] = z;
      cloud.scales[v * 3 + 0] = std::log(0.05f);
      cloud.scales[v * 3 + 1] = std::log(0.002f);
      cloud.scales[v * 3 + 2] = std::log(0.05f);
      const bool dark = (static_cast<int>(std::floor(x)) + static_cast<int>(std::floor(z))) % 2 == 0;
      const float g = dark ? 0.25f : 0.8f;
      set_color(v, g, g, g);
    }
  }
  return cloud;
}

} // namespace splat
