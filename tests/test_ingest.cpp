#include "ingest/exif.hpp"
#include "ingest/image.hpp"
#include "ingest/ingest.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef SPLAT_WITH_VIDEO
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/imgutils.h>
}
#endif

using namespace splat;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
namespace fs = std::filesystem;

namespace {

fs::path temp_dir(const char *name) {
  const fs::path path = fs::temp_directory_path() / name;
  fs::remove_all(path);
  fs::create_directories(path);
  return path;
}

// Builds a TIFF block with an IFD0 and an EXIF IFD, in either byte order.
class TiffBuilder {
public:
  explicit TiffBuilder(bool little) : little_(little) {}

  struct Tag {
    std::uint16_t id;
    std::uint16_t type;
    std::uint32_t count;
    std::vector<std::uint8_t> value; // Encoded in file byte order
  };

  std::vector<std::uint8_t> u16(std::uint32_t v) const {
    const auto a = static_cast<std::uint8_t>(v & 0xFF), b = static_cast<std::uint8_t>((v >> 8) & 0xFF);
    return little_ ? std::vector<std::uint8_t>{a, b} : std::vector<std::uint8_t>{b, a};
  }
  std::vector<std::uint8_t> u32(std::uint32_t v) const {
    auto lo = u16(v & 0xFFFF), hi = u16(v >> 16);
    auto &first = little_ ? lo : hi;
    const auto &second = little_ ? hi : lo;
    first.insert(first.end(), second.begin(), second.end());
    return first;
  }
  Tag short_tag(std::uint16_t id, std::uint32_t v) const { return {id, 3, 1, u16(v)}; }
  Tag rational_tag(std::uint16_t id, std::uint32_t num, std::uint32_t den) const {
    auto value = u32(num);
    const auto d = u32(den);
    value.insert(value.end(), d.begin(), d.end());
    return {id, 5, 1, value};
  }
  Tag ascii_tag(std::uint16_t id, const std::string &text) const {
    std::vector<std::uint8_t> value(text.begin(), text.end());
    value.push_back(0);
    return {id, 2, static_cast<std::uint32_t>(value.size()), value};
  }

  std::vector<std::uint8_t> build(std::vector<Tag> ifd0, const std::vector<Tag> &exifIfd) const {
    constexpr std::uint32_t kHeader = 8;
    const std::uint32_t ifd0Size = 2 + static_cast<std::uint32_t>(ifd0.size() + 1) * 12 + 4;
    const std::uint32_t exifOffset = kHeader + ifd0Size;
    Tag pointer{0x8769, 4, 1, u32(exifOffset)};
    ifd0.push_back(pointer);
    const std::uint32_t exifSize = 2 + static_cast<std::uint32_t>(exifIfd.size()) * 12 + 4;

    std::vector<std::uint8_t> out = little_ ? std::vector<std::uint8_t>{'I', 'I'} : std::vector<std::uint8_t>{'M', 'M'};
    append(out, u16(42));
    append(out, u32(kHeader));
    std::vector<std::uint8_t> overflow;
    const std::uint32_t overflowStart = exifOffset + exifSize;
    write_ifd(out, ifd0, overflow, overflowStart);
    write_ifd(out, exifIfd, overflow, overflowStart);
    append(out, overflow);
    return out;
  }

private:
  static void append(std::vector<std::uint8_t> &to, const std::vector<std::uint8_t> &from) {
    to.insert(to.end(), from.begin(), from.end());
  }
  void write_ifd(std::vector<std::uint8_t> &out, const std::vector<Tag> &tags, std::vector<std::uint8_t> &overflow,
                 std::uint32_t overflowStart) const {
    append(out, u16(static_cast<std::uint32_t>(tags.size())));
    for (const Tag &tag : tags) {
      append(out, u16(tag.id));
      append(out, u16(tag.type));
      append(out, u32(tag.count));
      if (tag.value.size() <= 4) {
        auto inline_value = tag.value;
        inline_value.resize(4);
        append(out, inline_value);
      } else {
        append(out, u32(overflowStart + static_cast<std::uint32_t>(overflow.size())));
        append(overflow, tag.value);
      }
    }
    append(out, u32(0));
  }
  bool little_;
};

std::vector<std::uint8_t> sample_tiff(bool little) {
  const TiffBuilder b(little);
  return b.build({b.ascii_tag(0x010F, "Valve"), b.ascii_tag(0x0110, "Test Cam  "), b.short_tag(0x0112, 6)},
                 {b.ascii_tag(0x9003, "2026:11:01 12:00:00"), b.rational_tag(0x920A, 50, 10), b.short_tag(0xA405, 26),
                  b.short_tag(0xA002, 4000), b.short_tag(0xA003, 3000), b.rational_tag(0xA20E, 25400, 10),
                  b.short_tag(0xA210, 2)});
}

// Wraps a TIFF block as APP1 in front of a JPEG file. `image` is a complete JPEG.
std::vector<std::uint8_t> jpeg_with_exif(const std::vector<std::uint8_t> &tiff, const std::vector<std::uint8_t> &image) {
  std::vector<std::uint8_t> out = {0xFF, 0xD8, 0xFF, 0xE1};
  const std::size_t length = 2 + 6 + tiff.size();
  out.push_back(static_cast<std::uint8_t>(length >> 8));
  out.push_back(static_cast<std::uint8_t>(length & 0xFF));
  for (const char c : {'E', 'x', 'i', 'f', '\0', '\0'}) out.push_back(static_cast<std::uint8_t>(c));
  out.insert(out.end(), tiff.begin(), tiff.end());
  out.insert(out.end(), image.begin() + 2, image.end());
  return out;
}

RgbImage gradient(int width, int height) {
  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t at = (static_cast<std::size_t>(y) * width + x) * 3;
      image.pixels[at] = static_cast<std::uint8_t>(x * 255 / std::max(1, width - 1));
      image.pixels[at + 1] = static_cast<std::uint8_t>(y * 255 / std::max(1, height - 1));
      image.pixels[at + 2] = 128;
    }
  }
  return image;
}

std::vector<std::uint8_t> read_bytes(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_bytes(const fs::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::uint8_t red_at(const RgbImage &image, int x, int y) {
  return image.pixels[(static_cast<std::size_t>(y) * image.width + x) * 3];
}

} // namespace

TEST_CASE("EXIF tags are read in both byte orders", "[ingest][exif]") {
  for (const bool little : {true, false}) {
    INFO("little endian: " << little);
    const ExifData exif = parse_tiff_exif(sample_tiff(little));
    CHECK(exif.make == "Valve");
    CHECK(exif.model == "Test Cam");
    CHECK(exif.orientation == 6);
    CHECK(exif.dateTimeOriginal == "2026:11:01 12:00:00");
    REQUIRE(exif.focalLengthMm);
    CHECK_THAT(*exif.focalLengthMm, WithinAbs(5.0, 1e-9));
    REQUIRE(exif.focalLength35mm);
    CHECK_THAT(*exif.focalLength35mm, WithinAbs(26.0, 1e-9));
    CHECK(exif.pixelXDimension == 4000u);
    CHECK(exif.pixelYDimension == 3000u);
    REQUIRE(exif.focalPlaneXResolution);
    CHECK_THAT(*exif.focalPlaneXResolution, WithinAbs(100.0, 1e-9)); // 2540 per inch
  }
}

TEST_CASE("EXIF parser survives truncated and corrupt data", "[ingest][exif]") {
  const auto tiff = sample_tiff(true);
  for (std::size_t size = 0; size < tiff.size(); ++size) {
    (void)parse_tiff_exif(std::span(tiff).first(size));
  }
  std::mt19937 rng(1234);
  for (int round = 0; round < 2000; ++round) {
    auto damaged = tiff;
    for (int i = 0; i < 8; ++i) damaged[rng() % damaged.size()] = static_cast<std::uint8_t>(rng());
    const ExifData exif = parse_tiff_exif(damaged);
    CHECK((exif.orientation >= 1 && exif.orientation <= 8));
  }
  const std::vector<std::uint8_t> junk = {0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF, 'E', 'x'};
  CHECK(parse_jpeg_exif(junk).orientation == 1);
  CHECK(parse_jpeg_exif({}).orientation == 1);
}

TEST_CASE("EXIF orientation produces the displayed image", "[ingest]") {
  // A 3x2 image with distinct values. Each case is the upright image as it
  // should be displayed; the stored file is what a camera would write for it.
  RgbImage stored;
  stored.width = 3;
  stored.height = 2;
  const std::uint8_t values[6] = {1, 2, 3, 4, 5, 6};
  for (const std::uint8_t v : values) stored.pixels.insert(stored.pixels.end(), {v, v, v});

  const auto rows = [](const RgbImage &image) {
    std::vector<int> out;
    for (std::size_t i = 0; i < image.pixels.size(); i += 3) out.push_back(image.pixels[i]);
    return out;
  };
  using V = std::vector<int>;
  CHECK(rows(apply_orientation(stored, 1)) == V{1, 2, 3, 4, 5, 6});
  CHECK(rows(apply_orientation(stored, 2)) == V{3, 2, 1, 6, 5, 4});
  CHECK(rows(apply_orientation(stored, 3)) == V{6, 5, 4, 3, 2, 1});
  CHECK(rows(apply_orientation(stored, 4)) == V{4, 5, 6, 1, 2, 3});
  CHECK(rows(apply_orientation(stored, 5)) == V{1, 4, 2, 5, 3, 6});
  CHECK(rows(apply_orientation(stored, 6)) == V{4, 1, 5, 2, 6, 3}); // 90 clockwise
  CHECK(rows(apply_orientation(stored, 7)) == V{6, 3, 5, 2, 4, 1});
  CHECK(rows(apply_orientation(stored, 8)) == V{3, 6, 2, 5, 1, 4}); // 90 counter-clockwise

  const RgbImage turned = apply_orientation(stored, 6);
  CHECK(turned.width == 2);
  CHECK(turned.height == 3);
}

TEST_CASE("Downscaling keeps aspect ratio and average colour", "[ingest]") {
  const RgbImage unchanged = downscale_to_fit(gradient(100, 50), 100);
  CHECK(unchanged.width == 100);
  CHECK(downscale_to_fit(gradient(100, 50), 0).width == 100);

  const RgbImage small = downscale_to_fit(gradient(1000, 500), 333);
  CHECK(small.width == 333);
  CHECK(small.height == 167);
  REQUIRE(small.pixels.size() == 333u * 167u * 3u);

  // A uniform image must stay exactly uniform, also at fractional ratios.
  RgbImage flat;
  flat.width = 997;
  flat.height = 601;
  flat.pixels.assign(997u * 601u * 3u, 200);
  const RgbImage shrunk = downscale_to_fit(flat, 123);
  for (const std::uint8_t v : shrunk.pixels) REQUIRE(v == 200);

  // A gradient's corners stay near their extremes.
  CHECK(red_at(small, 0, 0) < 3);
  CHECK(red_at(small, small.width - 1, 0) > 252);
}

TEST_CASE("Sharpness ranks a sharp image above a blurred one", "[ingest]") {
  constexpr int kSize = 128;
  std::vector<std::uint8_t> sharp(kSize * kSize), blurred(kSize * kSize);
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) sharp[y * kSize + x] = ((x / 8 + y / 8) % 2) ? 220 : 30;
  }
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      int sum = 0;
      for (int dy = -4; dy <= 4; ++dy) {
        for (int dx = -4; dx <= 4; ++dx) {
          sum += sharp[std::clamp(y + dy, 0, kSize - 1) * kSize + std::clamp(x + dx, 0, kSize - 1)];
        }
      }
      blurred[y * kSize + x] = static_cast<std::uint8_t>(sum / 81);
    }
  }
  CHECK(sharpness(sharp, kSize, kSize, kSize) > 10.0 * sharpness(blurred, kSize, kSize, kSize));
  const std::vector<std::uint8_t> flat(kSize * kSize, 90);
  CHECK(sharpness(flat, kSize, kSize, kSize) == 0.0);
}

TEST_CASE("Focal length follows the documented priority", "[ingest]") {
  ExifData exif;
  ManualIntrinsics manual;

  // Nothing known: a labelled guess.
  auto f = estimate_focal(4000, 3000, exif, manual);
  CHECK(f.source == FocalSource::Default);
  CHECK_THAT(f.pixels, WithinAbs(4800.0, 1e-6));

  // Focal plane resolution, with the file smaller than the camera's image.
  exif.focalLengthMm = 5.0;
  exif.focalPlaneXResolution = 800.0; // px/mm at 4000 px wide
  exif.pixelXDimension = 4000;
  exif.pixelYDimension = 3000;
  f = estimate_focal(2000, 1500, exif, manual);
  CHECK(f.source == FocalSource::ExifPlane);
  CHECK_THAT(f.pixels, WithinAbs(2000.0, 1e-6));

  // A 35 mm equivalent beats it. 3:2 reduces to the textbook focal/36 * width.
  exif.focalLength35mm = 36.0;
  f = estimate_focal(6000, 4000, exif, manual);
  CHECK(f.source == FocalSource::Exif35mm);
  CHECK_THAT(f.pixels, WithinAbs(6000.0, 1.0));

  // Manual beats everything.
  manual.focalPx = 1234.0;
  f = estimate_focal(6000, 4000, exif, manual);
  CHECK(f.source == FocalSource::Manual);
  CHECK_THAT(f.pixels, WithinAbs(1234.0, 1e-9));

  ManualIntrinsics byFov;
  byFov.fovDeg = 90.0;
  CHECK_THAT(estimate_focal(2000, 1000, {}, byFov).pixels, WithinAbs(1000.0, 1e-6));

  ManualIntrinsics byMm;
  byMm.focalMm = 24.0;
  byMm.sensorWidthMm = 36.0;
  CHECK_THAT(estimate_focal(3000, 2000, {}, byMm).pixels, WithinAbs(2000.0, 1e-6));
  byMm.sensorWidthMm.reset();
  CHECK_THROWS(estimate_focal(3000, 2000, {}, byMm));
}

TEST_CASE("Manifest round-trips", "[ingest]") {
  IngestManifest m;
  IngestCamera c;
  c.id = 1;
  c.width = 1600;
  c.height = 1200;
  c.fx = c.fy = 1234.5;
  c.cx = 800;
  c.cy = 600;
  c.focalSource = FocalSource::Exif35mm;
  c.make = "Valve";
  c.model = "Test";
  m.cameras.push_back(c);
  m.images.push_back({"images/photo_000001.jpg", 1, "/a/b.jpg", std::nullopt, "2026:11:01 12:00:00"});
  m.images.push_back({"images/video01_000001.jpg", 1, "/a/v.mp4", 1.5, ""});

  const fs::path dir = temp_dir("splat_manifest");
  save_manifest(m, dir / "ingest.json");
  const IngestManifest loaded = load_manifest(dir / "ingest.json");
  REQUIRE(loaded.cameras.size() == 1);
  CHECK(loaded.cameras[0].fx == 1234.5);
  CHECK(loaded.cameras[0].focalSource == FocalSource::Exif35mm);
  CHECK(loaded.cameras[0].model == "Test");
  REQUIRE(loaded.images.size() == 2);
  CHECK(loaded.images[0].captureTime == "2026:11:01 12:00:00");
  CHECK_FALSE(loaded.images[0].videoSeconds);
  CHECK(loaded.images[1].videoSeconds == 1.5);

  write_bytes(dir / "bad.json", {'{', '}'});
  CHECK_THROWS(load_manifest(dir / "bad.json"));
  fs::remove_all(dir);
}

TEST_CASE("Folders expand in natural order", "[ingest]") {
  const fs::path dir = temp_dir("splat_expand");
  for (const char *name : {"img10.jpg", "img2.JPG", "img1.png", "notes.txt", "clip.mp4"}) {
    write_bytes(dir / name, {0});
  }
  const auto files = expand_inputs({dir});
  std::vector<std::string> names;
  for (const auto &f : files) names.push_back(f.filename().string());
  CHECK(names == std::vector<std::string>{"clip.mp4", "img1.png", "img2.JPG", "img10.jpg"});
  CHECK_THROWS(expand_inputs({dir / "missing"}));
  fs::remove_all(dir);
}

TEST_CASE("Ingesting photos applies EXIF, downscaling and manual intrinsics", "[ingest]") {
  const fs::path dir = temp_dir("splat_ingest_photos");
  const fs::path input = dir / "input";
  const fs::path project = dir / "project";
  fs::create_directories(input);

  // Encode through the same writer, then splice EXIF in as a camera would.
  write_image(gradient(400, 300), input / "plain.jpg", 95);
  const auto plain = read_bytes(input / "plain.jpg");

  const TiffBuilder b(true);
  // Orientation 6 (portrait shot) and a 26 mm equivalent.
  write_bytes(input / "b_rotated.jpg",
              jpeg_with_exif(b.build({b.ascii_tag(0x010F, "Valve"), b.ascii_tag(0x0110, "Cam"), b.short_tag(0x0112, 6)},
                                     {b.short_tag(0xA405, 26)}),
                             plain));
  fs::rename(input / "plain.jpg", input / "a_plain.jpg");
  write_image(gradient(40, 30), input / "c.png", 95);
  write_bytes(input / "d_broken.jpg", {0xFF, 0xD8, 0x00, 0x01});
  write_bytes(input / "e.heic", {0});

  IngestOptions options;
  options.inputs = {input};
  options.project = project;
  options.maxSize = 200;
  options.threads = 2;
  const IngestReport report = run_ingest(options);
  CHECK(report.images == 3);
  CHECK(report.failed == 2);
  CHECK(report.cameras == 3);

  const IngestManifest m = load_manifest(project / "ingest.json");
  REQUIRE(m.images.size() == 3);
  CHECK(m.images[0].file == "images/photo_000001.jpg");
  CHECK_THAT(m.images[0].source, ContainsSubstring("a_plain"));
  CHECK_THAT(m.images[1].source, ContainsSubstring("b_rotated"));

  const auto camera_of = [&](const IngestImage &image) { return m.cameras.at(static_cast<std::size_t>(image.cameraId - 1)); };
  const IngestCamera plainCamera = camera_of(m.images[0]);
  CHECK(plainCamera.width == 200);
  CHECK(plainCamera.height == 150);
  CHECK(plainCamera.focalSource == FocalSource::Default);
  CHECK_THAT(plainCamera.fx, WithinAbs(240.0, 1e-6)); // 1.2 * 200
  CHECK_THAT(plainCamera.cx, WithinAbs(100.0, 1e-9));

  const IngestCamera rotatedCamera = camera_of(m.images[1]);
  CHECK(rotatedCamera.width == 150); // Portrait after rotation
  CHECK(rotatedCamera.height == 200);
  CHECK(rotatedCamera.focalSource == FocalSource::Exif35mm);
  CHECK(rotatedCamera.make == "Valve");
  CHECK_THAT(rotatedCamera.fx, WithinAbs(26.0 * 250.0 / 43.2666, 1e-3));

  // The stored file really is what the manifest says, and really was rotated:
  // red rises left to right in the source, so top to bottom after 90 degrees.
  const RgbImage stored = decode_image(read_bytes(project / m.images[1].file));
  CHECK(stored.width == 150);
  CHECK(stored.height == 200);
  CHECK(red_at(stored, 75, 5) < 30);
  CHECK(red_at(stored, 75, 195) > 225);

  // A second run refuses, then replaces with --force and removes stale images.
  CHECK_THROWS(run_ingest(options));
  options.force = true;
  options.manual.focalPx = 1000.0;
  options.manual.cx = 100.0;
  options.inputs = {input / "a_plain.jpg"};
  const IngestReport second = run_ingest(options);
  CHECK(second.images == 1);
  CHECK_FALSE(fs::exists(project / m.images[1].file));

  // Manual values are given at input size and scale with the image.
  const IngestManifest forced = load_manifest(project / "ingest.json");
  REQUIRE(forced.cameras.size() == 1);
  CHECK(forced.cameras[0].focalSource == FocalSource::Manual);
  CHECK_THAT(forced.cameras[0].fx, WithinAbs(500.0, 1e-6));
  CHECK_THAT(forced.cameras[0].cx, WithinAbs(50.0, 1e-6));
  fs::remove_all(dir);
}

#ifdef SPLAT_WITH_VIDEO

namespace {

// Writes a short MPEG-4 video: a moving checkerboard, blurred except for one
// frame in each second, with an optional rotation flag like a phone would set.
void write_test_video(const fs::path &path, int seconds, int fps, int rotationCcw) {
  constexpr int kWidth = 320, kHeight = 240;
  AVFormatContext *format = nullptr;
  REQUIRE(avformat_alloc_output_context2(&format, nullptr, nullptr, path.string().c_str()) >= 0);
  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
  REQUIRE(codec != nullptr);
  AVStream *stream = avformat_new_stream(format, nullptr);
  AVCodecContext *encoder = avcodec_alloc_context3(codec);
  encoder->width = kWidth;
  encoder->height = kHeight;
  encoder->pix_fmt = AV_PIX_FMT_YUV420P;
  encoder->time_base = {1, fps};
  encoder->framerate = {fps, 1};
  encoder->bit_rate = 4'000'000;
  encoder->gop_size = 1;
  if (format->oformat->flags & AVFMT_GLOBALHEADER) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  REQUIRE(avcodec_open2(encoder, codec, nullptr) >= 0);
  REQUIRE(avcodec_parameters_from_context(stream->codecpar, encoder) >= 0);
  stream->time_base = encoder->time_base;
  if (rotationCcw != 0) {
    auto *side = av_packet_side_data_new(&stream->codecpar->coded_side_data, &stream->codecpar->nb_coded_side_data,
                                         AV_PKT_DATA_DISPLAYMATRIX, 9 * sizeof(std::int32_t), 0);
    REQUIRE(side != nullptr);
    av_display_rotation_set(reinterpret_cast<std::int32_t *>(side->data), rotationCcw);
  }
  REQUIRE(avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE) >= 0);
  REQUIRE(avformat_write_header(format, nullptr) >= 0);

  AVFrame *frame = av_frame_alloc();
  frame->format = AV_PIX_FMT_YUV420P;
  frame->width = kWidth;
  frame->height = kHeight;
  REQUIRE(av_frame_get_buffer(frame, 0) >= 0);
  AVPacket *packet = av_packet_alloc();

  const auto drain = [&] {
    while (avcodec_receive_packet(encoder, packet) >= 0) {
      packet->duration = 1; // Without it the MP4 edit list drops the last frame
      av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
      packet->stream_index = stream->index;
      av_interleaved_write_frame(format, packet);
    }
  };
  for (int n = 0; n < seconds * fps; ++n) {
    REQUIRE(av_frame_make_writable(frame) >= 0);
    // One frame per second is sharp: frame fps/2 of each second.
    const bool sharp = n % fps == fps / 2;
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        int value = ((x / 16 + y / 16) % 2) ? 220 : 30;
        if (!sharp) value = 125; // The "blurred" frames have no detail at all
        frame->data[0][y * frame->linesize[0] + x] = static_cast<std::uint8_t>(value);
      }
    }
    for (int y = 0; y < kHeight / 2; ++y) {
      std::fill_n(frame->data[1] + y * frame->linesize[1], kWidth / 2, std::uint8_t{128});
      std::fill_n(frame->data[2] + y * frame->linesize[2], kWidth / 2, std::uint8_t{128});
    }
    frame->pts = n;
    REQUIRE(avcodec_send_frame(encoder, frame) >= 0);
    drain();
  }
  avcodec_send_frame(encoder, nullptr);
  drain();
  av_write_trailer(format);
  avio_closep(&format->pb);
  av_packet_free(&packet);
  av_frame_free(&frame);
  avcodec_free_context(&encoder);
  avformat_free_context(format);
}

} // namespace

TEST_CASE("Video extraction keeps the sharpest frame of each interval", "[ingest][video]") {
  const fs::path dir = temp_dir("splat_ingest_video");
  write_test_video(dir / "clip.mp4", 4, 10, 0);

  std::vector<VideoFrame> frames;
  VideoOptions options;
  options.fps = 1.0;
  extract_video_frames(dir / "clip.mp4", options, [&](VideoFrame &&f) { frames.push_back(std::move(f)); });
  REQUIRE(frames.size() == 4);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    INFO("frame " << i);
    CHECK_THAT(frames[i].seconds, WithinAbs(static_cast<double>(i) + 0.5, 0.11));
    CHECK(frames[i].sharpness > 100.0);
    CHECK(frames[i].image.width == 320);
  }

  frames.clear();
  options.fps = 0.0;
  extract_video_frames(dir / "clip.mp4", options, [&](VideoFrame &&f) { frames.push_back(std::move(f)); });
  CHECK(frames.size() == 40);

  // A frame limit lowers the rate rather than dropping the end of the video.
  frames.clear();
  options.fps = 10.0;
  options.maxFrames = 2;
  extract_video_frames(dir / "clip.mp4", options, [&](VideoFrame &&f) { frames.push_back(std::move(f)); });
  REQUIRE(frames.size() == 2);
  CHECK(frames[1].seconds > 2.0);
  fs::remove_all(dir);
}

TEST_CASE("Video rotation metadata is applied", "[ingest][video]") {
  const fs::path dir = temp_dir("splat_ingest_rotated");
  // Phones shooting portrait store landscape frames with a -90 degree matrix.
  write_test_video(dir / "portrait.mp4", 1, 10, -90);
  std::vector<VideoFrame> frames;
  VideoOptions options;
  options.fps = 1.0;
  extract_video_frames(dir / "portrait.mp4", options, [&](VideoFrame &&f) { frames.push_back(std::move(f)); });
  REQUIRE(frames.size() == 1);
  CHECK(frames[0].image.width == 240);
  CHECK(frames[0].image.height == 320);
  fs::remove_all(dir);
}

TEST_CASE("Ingesting a video writes frames and a manifest", "[ingest][video]") {
  const fs::path dir = temp_dir("splat_ingest_full_video");
  write_test_video(dir / "clip.mp4", 3, 10, 0);
  IngestOptions options;
  options.inputs = {dir / "clip.mp4"};
  options.project = dir / "project";
  options.video.fps = 1.0;
  options.manual.focal35mm = 28.0;
  const IngestReport report = run_ingest(options);
  CHECK(report.videoFrames == 3);
  CHECK(report.cameras == 1);
  const IngestManifest m = load_manifest(dir / "project" / "ingest.json");
  REQUIRE(m.images.size() == 3);
  CHECK(m.images[2].file == "images/video01_000003.jpg");
  CHECK(fs::exists(dir / "project" / m.images[2].file));
  REQUIRE(m.images[2].videoSeconds);
  CHECK(m.cameras[0].focalSource == FocalSource::Manual);
  CHECK(m.cameras[0].width == 320);
  fs::remove_all(dir);
}

#endif
