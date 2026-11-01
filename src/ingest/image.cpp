#include "ingest/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include <stb_image.h>
// offscreen.cpp has its own static copy of the writer, for its PNG output.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace splat {

RgbImage decode_image(std::span<const std::uint8_t> encoded) {
  if (encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("image file is too large");
  }
  int width = 0, height = 0, channels = 0;
  stbi_uc *data = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels, 3);
  if (data == nullptr) {
    throw std::runtime_error(std::format("cannot decode image: {}", stbi_failure_reason()));
  }
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.assign(data, data + static_cast<std::size_t>(width) * height * 3);
  stbi_image_free(data);
  return image;
}

void write_image(const RgbImage &image, const std::filesystem::path &path, int jpegQuality) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const std::string name = path.string();
  int ok = 0;
  if (ext == ".png") {
    ok = stbi_write_png(name.c_str(), image.width, image.height, 3, image.pixels.data(), image.width * 3);
  } else if (ext == ".jpg" || ext == ".jpeg") {
    ok = stbi_write_jpg(name.c_str(), image.width, image.height, 3, image.pixels.data(), jpegQuality);
  } else {
    throw std::runtime_error(std::format("cannot write image format '{}'", ext));
  }
  if (ok == 0) {
    throw std::runtime_error(std::format("cannot write {}", name));
  }
}

RgbImage apply_orientation(RgbImage image, int orientation) {
  if (orientation <= 1 || orientation > 8) return image;

  const int w = image.width, h = image.height;
  const bool swap = orientation_swaps_axes(orientation);
  RgbImage out;
  out.width = swap ? h : w;
  out.height = swap ? w : h;
  out.pixels.resize(image.pixels.size());

  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      // Source pixel for each output pixel. The orientation says where the
      // stored image's first row and column belong when displayed.
      int sx = 0, sy = 0;
      switch (orientation) {
      case 2: sx = w - 1 - x; sy = y; break;         // Mirrored horizontally
      case 3: sx = w - 1 - x; sy = h - 1 - y; break; // Rotated 180
      case 4: sx = x; sy = h - 1 - y; break;         // Mirrored vertically
      case 5: sx = y; sy = x; break;                 // Transposed
      case 6: sx = y; sy = h - 1 - x; break;         // Needs 90 clockwise
      case 7: sx = w - 1 - y; sy = h - 1 - x; break; // Transversed
      default: sx = w - 1 - y; sy = x; break;        // 8: needs 90 counter-clockwise
      }
      const std::size_t from = (static_cast<std::size_t>(sy) * w + sx) * 3;
      const std::size_t to = (static_cast<std::size_t>(y) * out.width + x) * 3;
      std::copy_n(image.pixels.data() + from, 3, out.pixels.data() + to);
    }
  }
  return out;
}

RgbImage downscale_to_fit(RgbImage image, int maxSize) {
  const int longest = std::max(image.width, image.height);
  if (maxSize <= 0 || longest <= maxSize) return image;

  const double scale = static_cast<double>(maxSize) / longest;
  RgbImage out;
  out.width = std::max(1, static_cast<int>(std::lround(image.width * scale)));
  out.height = std::max(1, static_cast<int>(std::lround(image.height * scale)));
  out.pixels.resize(static_cast<std::size_t>(out.width) * out.height * 3);

  // Each output pixel averages the source rectangle it covers, with partial
  // weights at the edges, one pass per axis.
  struct Span {
    int first;
    int last;
    float firstWeight;
    float lastWeight;
    float total;
  };
  const auto make_spans = [](int from, int to) {
    std::vector<Span> spans(static_cast<std::size_t>(to));
    const double step = static_cast<double>(from) / to;
    for (int i = 0; i < to; ++i) {
      const double begin = i * step, end = (i + 1) * step;
      Span s{};
      s.first = static_cast<int>(begin);
      s.last = std::min(from - 1, static_cast<int>(std::ceil(end)) - 1);
      s.firstWeight = static_cast<float>(std::min(end, s.first + 1.0) - begin);
      s.lastWeight = s.last > s.first ? static_cast<float>(end - s.last) : 0.f;
      s.total = static_cast<float>(end - begin);
      spans[static_cast<std::size_t>(i)] = s;
    }
    return spans;
  };
  const auto xs = make_spans(image.width, out.width);
  const auto ys = make_spans(image.height, out.height);

  std::vector<float> row(static_cast<std::size_t>(image.width) * 3);
  std::vector<float> accum(static_cast<std::size_t>(out.width) * 3);
  for (int y = 0; y < out.height; ++y) {
    const Span &sy = ys[static_cast<std::size_t>(y)];
    std::fill(accum.begin(), accum.end(), 0.f);
    for (int sourceY = sy.first; sourceY <= sy.last; ++sourceY) {
      const float rowWeight = sourceY == sy.first ? sy.firstWeight : sourceY == sy.last ? sy.lastWeight : 1.f;
      const std::uint8_t *src = image.pixels.data() + static_cast<std::size_t>(sourceY) * image.width * 3;
      for (int x = 0; x < out.width; ++x) {
        const Span &sx = xs[static_cast<std::size_t>(x)];
        float sum[3]{};
        for (int sourceX = sx.first; sourceX <= sx.last; ++sourceX) {
          const float w = sourceX == sx.first ? sx.firstWeight : sourceX == sx.last ? sx.lastWeight : 1.f;
          for (int c = 0; c < 3; ++c) sum[c] += w * static_cast<float>(src[static_cast<std::size_t>(sourceX) * 3 + c]);
        }
        for (int c = 0; c < 3; ++c) accum[static_cast<std::size_t>(x) * 3 + c] += rowWeight * sum[c] / sx.total;
      }
    }
    std::uint8_t *dst = out.pixels.data() + static_cast<std::size_t>(y) * out.width * 3;
    for (std::size_t i = 0; i < accum.size(); ++i) {
      dst[i] = static_cast<std::uint8_t>(std::clamp(std::lround(accum[i] / sy.total), 0l, 255l));
    }
  }
  return out;
}

double sharpness(std::span<const std::uint8_t> grey, int width, int height, std::ptrdiff_t stride) {
  if (width < 3 || height < 3) return 0.0;
  double sum = 0.0, sumSquares = 0.0;
  for (int y = 1; y < height - 1; ++y) {
    const std::uint8_t *p = grey.data() + y * stride;
    for (int x = 1; x < width - 1; ++x) {
      const double laplacian = 4.0 * p[x] - p[x - 1] - p[x + 1] - p[x - stride] - p[x + stride];
      sum += laplacian;
      sumSquares += laplacian * laplacian;
    }
  }
  const double n = static_cast<double>(width - 2) * (height - 2);
  const double mean = sum / n;
  return sumSquares / n - mean * mean;
}

} // namespace splat
