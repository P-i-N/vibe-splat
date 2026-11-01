#include "ingest/exif.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string_view>

namespace splat {

namespace {

constexpr std::uint16_t kTagMake = 0x010F;
constexpr std::uint16_t kTagModel = 0x0110;
constexpr std::uint16_t kTagOrientation = 0x0112;
constexpr std::uint16_t kTagExifIfd = 0x8769;
constexpr std::uint16_t kTagDateTimeOriginal = 0x9003;
constexpr std::uint16_t kTagFocalLength = 0x920A;
constexpr std::uint16_t kTagPixelXDimension = 0xA002;
constexpr std::uint16_t kTagPixelYDimension = 0xA003;
constexpr std::uint16_t kTagFocalPlaneXRes = 0xA20E;
constexpr std::uint16_t kTagFocalPlaneYRes = 0xA20F;
constexpr std::uint16_t kTagFocalPlaneUnit = 0xA210;
constexpr std::uint16_t kTagFocalLength35mm = 0xA405;

constexpr std::uint16_t kTypeAscii = 2;
constexpr std::uint16_t kTypeShort = 3;
constexpr std::uint16_t kTypeLong = 4;
constexpr std::uint16_t kTypeRational = 5;

// Bounds-checked reads in the file's byte order. Out-of-range reads return nullopt.
class TiffReader {
public:
  TiffReader(std::span<const std::uint8_t> data, bool littleEndian) : data_(data), little_(littleEndian) {}

  [[nodiscard]] std::size_t size() const { return data_.size(); }

  [[nodiscard]] std::optional<std::uint16_t> u16(std::size_t offset) const {
    if (offset > data_.size() || data_.size() - offset < 2) return std::nullopt;
    const std::uint32_t a = data_[offset], b = data_[offset + 1];
    return static_cast<std::uint16_t>(little_ ? a | (b << 8) : (a << 8) | b);
  }

  [[nodiscard]] std::optional<std::uint32_t> u32(std::size_t offset) const {
    const auto a = u16(offset), b = u16(offset + 2);
    if (!a || !b) return std::nullopt;
    return little_ ? static_cast<std::uint32_t>(*a) | (static_cast<std::uint32_t>(*b) << 16)
                   : (static_cast<std::uint32_t>(*a) << 16) | static_cast<std::uint32_t>(*b);
  }

  [[nodiscard]] std::optional<double> rational(std::size_t offset) const {
    const auto num = u32(offset), den = u32(offset + 4);
    if (!num || !den || *den == 0) return std::nullopt;
    return static_cast<double>(*num) / static_cast<double>(*den);
  }

  [[nodiscard]] std::string ascii(std::size_t offset, std::size_t count) const {
    if (offset > data_.size() || data_.size() - offset < count) return {};
    std::string_view view(reinterpret_cast<const char *>(data_.data() + offset), count);
    view = view.substr(0, view.find('\0'));
    while (!view.empty() && view.back() == ' ') view.remove_suffix(1);
    return std::string(view);
  }

private:
  std::span<const std::uint8_t> data_;
  bool little_;
};

struct Entry {
  std::uint16_t tag;
  std::uint16_t type;
  std::uint32_t count;
  std::size_t valueOffset; // Where the value lives, inline or out of line
};

[[nodiscard]] std::size_t type_size(std::uint16_t type) {
  switch (type) {
  case kTypeShort: return 2;
  case kTypeLong: return 4;
  case kTypeRational: return 8;
  default: return 1;
  }
}

[[nodiscard]] std::optional<std::uint32_t> integer(const TiffReader &r, const Entry &e) {
  if (e.type == kTypeShort) {
    const auto v = r.u16(e.valueOffset);
    return v ? std::optional<std::uint32_t>(*v) : std::nullopt;
  }
  if (e.type == kTypeLong) return r.u32(e.valueOffset);
  return std::nullopt;
}

[[nodiscard]] std::optional<double> real(const TiffReader &r, const Entry &e) {
  if (e.type == kTypeRational) return r.rational(e.valueOffset);
  const auto v = integer(r, e);
  return v ? std::optional<double>(*v) : std::nullopt;
}

[[nodiscard]] std::string text(const TiffReader &r, const Entry &e) {
  return e.type == kTypeAscii ? r.ascii(e.valueOffset, e.count) : std::string{};
}

[[nodiscard]] std::optional<double> positive(std::optional<double> value) {
  return value && *value > 0.0 ? value : std::nullopt;
}

template <class Visit> void visit_ifd(const TiffReader &r, std::size_t offset, Visit &&visit) {
  const auto count = r.u16(offset);
  if (!count) return;
  for (std::size_t i = 0; i < *count; ++i) {
    const std::size_t at = offset + 2 + i * 12;
    const auto tag = r.u16(at), type = r.u16(at + 2);
    const auto n = r.u32(at + 4);
    if (!tag || !type || !n) return;
    Entry entry{*tag, *type, *n, at + 8};
    if (static_cast<std::uint64_t>(type_size(*type)) * *n > 4) {
      const auto pointer = r.u32(at + 8);
      if (!pointer) return;
      entry.valueOffset = *pointer;
    }
    visit(entry);
  }
}

} // namespace

ExifData parse_tiff_exif(std::span<const std::uint8_t> tiff) {
  ExifData exif;
  if (tiff.size() < 8) return exif;
  const bool little = tiff[0] == 'I' && tiff[1] == 'I';
  if (!little && !(tiff[0] == 'M' && tiff[1] == 'M')) return exif;
  const TiffReader r(tiff, little);
  const auto magic = r.u16(2);
  const auto ifd0 = r.u32(4);
  if (magic != 42 || !ifd0) return exif;

  std::optional<std::size_t> exifIfd;
  visit_ifd(r, *ifd0, [&](const Entry &e) {
    switch (e.tag) {
    case kTagMake: exif.make = text(r, e); break;
    case kTagModel: exif.model = text(r, e); break;
    case kTagOrientation:
      if (const auto v = integer(r, e); v && *v >= 1 && *v <= 8) exif.orientation = static_cast<int>(*v);
      break;
    case kTagExifIfd:
      if (const auto v = integer(r, e)) exifIfd = *v;
      break;
    default: break;
    }
  });
  if (!exifIfd) return exif;

  std::optional<double> xRes, yRes;
  std::uint32_t unit = 2; // EXIF default: inches
  visit_ifd(r, *exifIfd, [&](const Entry &e) {
    switch (e.tag) {
    case kTagDateTimeOriginal: exif.dateTimeOriginal = text(r, e); break;
    case kTagFocalLength: exif.focalLengthMm = positive(real(r, e)); break;
    case kTagFocalLength35mm: exif.focalLength35mm = positive(real(r, e)); break;
    case kTagPixelXDimension: exif.pixelXDimension = integer(r, e); break;
    case kTagPixelYDimension: exif.pixelYDimension = integer(r, e); break;
    case kTagFocalPlaneXRes: xRes = positive(real(r, e)); break;
    case kTagFocalPlaneYRes: yRes = positive(real(r, e)); break;
    case kTagFocalPlaneUnit: unit = integer(r, e).value_or(2); break;
    default: break;
    }
  });

  // Convert to pixels per millimetre. Units: 2 inch, 3 cm, 4 mm, 5 um, 6 nm.
  const double perMm = unit == 2   ? 1.0 / 25.4
                       : unit == 3 ? 0.1
                       : unit == 4 ? 1.0
                       : unit == 5 ? 1000.0
                       : unit == 6 ? 1000000.0
                                   : 0.0;
  if (perMm > 0.0) {
    if (xRes) exif.focalPlaneXResolution = *xRes * perMm;
    if (yRes) exif.focalPlaneYResolution = *yRes * perMm;
  }
  return exif;
}

ExifData parse_jpeg_exif(std::span<const std::uint8_t> jpeg) {
  static constexpr std::uint8_t kExifHeader[] = {'E', 'x', 'i', 'f', 0, 0};
  if (jpeg.size() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) return {};
  std::size_t at = 2;
  while (at + 4 <= jpeg.size()) {
    if (jpeg[at] != 0xFF) return {};
    const std::uint8_t marker = jpeg[at + 1];
    if (marker == 0xFF) { // Fill byte
      ++at;
      continue;
    }
    if (marker == 0xDA || marker == 0xD9) return {}; // Image data: EXIF always comes first
    const std::size_t length = (static_cast<std::size_t>(jpeg[at + 2]) << 8) | jpeg[at + 3];
    if (length < 2 || at + 2 + length > jpeg.size()) return {};
    if (marker == 0xE1 && length >= 2 + sizeof(kExifHeader) &&
        std::memcmp(jpeg.data() + at + 4, kExifHeader, sizeof(kExifHeader)) == 0) {
      return parse_tiff_exif(jpeg.subspan(at + 4 + sizeof(kExifHeader), length - 2 - sizeof(kExifHeader)));
    }
    at += 2 + length;
  }
  return {};
}

} // namespace splat
