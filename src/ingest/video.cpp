#include "ingest/video.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SPLAT_WITH_VIDEO

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace splat {

namespace {

// Long edge of the grey copy used to score sharpness. Full-size scoring would
// cost more than decoding, and the ranking barely changes.
constexpr int kScoreSize = 640;

std::string av_message(int code) {
  char text[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, text, sizeof(text));
  return text;
}

struct FormatDeleter {
  void operator()(AVFormatContext *p) const { avformat_close_input(&p); }
};
struct CodecDeleter {
  void operator()(AVCodecContext *p) const { avcodec_free_context(&p); }
};
struct FrameDeleter {
  void operator()(AVFrame *p) const { av_frame_free(&p); }
};
struct PacketDeleter {
  void operator()(AVPacket *p) const { av_packet_free(&p); }
};
struct SwsDeleter {
  void operator()(SwsContext *p) const { sws_freeContext(p); }
};
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;

// Clockwise rotation in degrees that displays the stream upright: 0, 90, 180 or 270.
int stream_rotation(const AVStream &stream) {
  const AVPacketSideData *side = av_packet_side_data_get(stream.codecpar->coded_side_data,
                                                         stream.codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
  if (side == nullptr || side->size < 9 * sizeof(std::int32_t)) return 0;
  const double counterClockwise = av_display_rotation_get(reinterpret_cast<const std::int32_t *>(side->data));
  if (std::isnan(counterClockwise)) return 0;
  const int quarter = static_cast<int>(std::lround(-counterClockwise / 90.0));
  return ((quarter % 4) + 4) % 4 * 90;
}

int rotation_to_orientation(int degrees) { return degrees == 90 ? 6 : degrees == 180 ? 3 : degrees == 270 ? 8 : 1; }

class FrameConverter {
public:
  FrameConverter(const AVFrame &frame) : width_(frame.width), height_(frame.height) {
    const auto format = static_cast<AVPixelFormat>(frame.format);
    const double scale = std::min(1.0, static_cast<double>(kScoreSize) / std::max(width_, height_));
    scoreWidth_ = std::max(3, static_cast<int>(std::lround(width_ * scale)));
    scoreHeight_ = std::max(3, static_cast<int>(std::lround(height_ * scale)));
    format_ = format;
    toRgb_.reset(sws_getContext(width_, height_, format, width_, height_, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr,
                                nullptr, nullptr));
    toGrey_.reset(sws_getContext(width_, height_, format, scoreWidth_, scoreHeight_, AV_PIX_FMT_GRAY8, SWS_BILINEAR,
                                 nullptr, nullptr, nullptr));
    if (!toRgb_ || !toGrey_) {
      throw std::runtime_error(std::format("unsupported pixel format {}", static_cast<int>(format)));
    }
    grey_.resize(static_cast<std::size_t>(scoreWidth_) * scoreHeight_);
  }

  [[nodiscard]] bool matches(const AVFrame &frame) const {
    return frame.width == width_ && frame.height == height_ && frame.format == format_;
  }

  [[nodiscard]] double score(const AVFrame &frame) {
    std::uint8_t *dst[4] = {grey_.data(), nullptr, nullptr, nullptr};
    const int dstStride[4] = {scoreWidth_, 0, 0, 0};
    sws_scale(toGrey_.get(), frame.data, frame.linesize, 0, height_, dst, dstStride);
    return sharpness(grey_, scoreWidth_, scoreHeight_, scoreWidth_);
  }

  [[nodiscard]] RgbImage to_rgb(const AVFrame &frame) {
    RgbImage image;
    image.width = width_;
    image.height = height_;
    image.pixels.resize(static_cast<std::size_t>(width_) * height_ * 3);
    std::uint8_t *dst[4] = {image.pixels.data(), nullptr, nullptr, nullptr};
    const int dstStride[4] = {width_ * 3, 0, 0, 0};
    sws_scale(toRgb_.get(), frame.data, frame.linesize, 0, height_, dst, dstStride);
    return image;
  }

private:
  int width_;
  int height_;
  int format_ = AV_PIX_FMT_NONE;
  int scoreWidth_ = 0;
  int scoreHeight_ = 0;
  SwsPtr toRgb_;
  SwsPtr toGrey_;
  std::vector<std::uint8_t> grey_;
};

} // namespace

bool video_supported() noexcept { return true; }

void extract_video_frames(const std::filesystem::path &path, const VideoOptions &options,
                          const std::function<void(VideoFrame &&)> &sink) {
  const auto fail = [&](const std::string &what, int code) {
    return std::runtime_error(std::format("{}: {}: {}", path.string(), what, av_message(code)));
  };

  AVFormatContext *rawFormat = nullptr;
  if (const int r = avformat_open_input(&rawFormat, path.string().c_str(), nullptr, nullptr); r < 0) {
    throw fail("cannot open", r);
  }
  const std::unique_ptr<AVFormatContext, FormatDeleter> format(rawFormat);
  if (const int r = avformat_find_stream_info(format.get(), nullptr); r < 0) {
    throw fail("cannot read stream info", r);
  }

  const AVCodec *codec = nullptr;
  const int streamIndex = av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (streamIndex < 0) throw fail("no decodable video stream", streamIndex);
  const AVStream &stream = *format->streams[streamIndex];

  const std::unique_ptr<AVCodecContext, CodecDeleter> decoder(avcodec_alloc_context3(codec));
  if (!decoder) throw std::runtime_error("out of memory");
  if (const int r = avcodec_parameters_to_context(decoder.get(), stream.codecpar); r < 0) {
    throw fail("bad codec parameters", r);
  }
  decoder->thread_count = 0; // Let FFmpeg pick
  if (const int r = avcodec_open2(decoder.get(), codec, nullptr); r < 0) throw fail("cannot open decoder", r);

  // Lower the rate if the limit would otherwise cut the video short.
  double fps = options.fps;
  double duration = 0.0;
  if (stream.duration != AV_NOPTS_VALUE) {
    duration = static_cast<double>(stream.duration) * av_q2d(stream.time_base);
  } else if (format->duration != AV_NOPTS_VALUE) {
    duration = static_cast<double>(format->duration) / AV_TIME_BASE;
  }
  if (options.maxFrames > 0 && duration > 0.0) {
    const double limited = options.maxFrames / duration;
    fps = fps > 0.0 ? std::min(fps, limited) : limited;
  }

  const double timeBase = av_q2d(stream.time_base);
  const int orientation = rotation_to_orientation(stream_rotation(stream));
  const double startSeconds = stream.start_time != AV_NOPTS_VALUE ? static_cast<double>(stream.start_time) * timeBase : 0.0;

  std::optional<FrameConverter> converter;
  FramePtr best;
  double bestScore = -1.0, bestSeconds = 0.0;
  std::int64_t bucket = -1, frameCounter = 0;
  int emitted = 0;

  const auto flush_best = [&] {
    if (!best) return;
    if (options.maxFrames > 0 && emitted >= options.maxFrames) {
      best.reset();
      return;
    }
    VideoFrame out;
    out.image = apply_orientation(converter->to_rgb(*best), orientation);
    out.seconds = bestSeconds;
    out.sharpness = bestScore;
    best.reset();
    bestScore = -1.0;
    ++emitted;
    sink(std::move(out));
  };

  const FramePtr frame(av_frame_alloc());
  const auto consume_frame = [&] {
    if (!converter || !converter->matches(*frame)) {
      flush_best(); // The stream changed resolution or format mid-way
      converter.emplace(*frame);
    }
    const std::int64_t index = frameCounter++;
    double seconds = 0.0;
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
      seconds = static_cast<double>(frame->best_effort_timestamp) * timeBase - startSeconds;
    } else {
      const double rate = av_q2d(stream.avg_frame_rate);
      seconds = rate > 0.0 ? static_cast<double>(index) / rate : 0.0;
    }
    const std::int64_t thisBucket = fps > 0.0 ? static_cast<std::int64_t>(std::floor(seconds * fps)) : index;
    if (thisBucket != bucket) {
      flush_best();
      bucket = thisBucket;
    }
    if (const double score = converter->score(*frame); score > bestScore) {
      best.reset(av_frame_clone(frame.get()));
      if (!best) throw std::runtime_error("out of memory");
      bestScore = score;
      bestSeconds = seconds;
    }
  };

  const auto drain = [&] {
    for (;;) {
      const int r = avcodec_receive_frame(decoder.get(), frame.get());
      if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return;
      if (r < 0) throw fail("decoding failed", r);
      consume_frame();
      av_frame_unref(frame.get());
    }
  };

  const std::unique_ptr<AVPacket, PacketDeleter> packet(av_packet_alloc());
  for (;;) {
    const int r = av_read_frame(format.get(), packet.get());
    if (r == AVERROR_EOF) break;
    if (r < 0) throw fail("reading failed", r);
    if (packet->stream_index == streamIndex) {
      // A damaged packet costs a frame, not the whole video.
      if (const int s = avcodec_send_packet(decoder.get(), packet.get()); s < 0 && s != AVERROR(EAGAIN) && s != AVERROR_INVALIDDATA) {
        av_packet_unref(packet.get());
        throw fail("decoding failed", s);
      }
      drain();
    }
    av_packet_unref(packet.get());
  }
  avcodec_send_packet(decoder.get(), nullptr);
  drain();
  flush_best();
  if (emitted == 0) throw std::runtime_error(std::format("{}: no frames could be decoded", path.string()));
}

} // namespace splat

#else

namespace splat {

bool video_supported() noexcept { return false; }

void extract_video_frames(const std::filesystem::path &, const VideoOptions &, const std::function<void(VideoFrame &&)> &) {
  throw std::runtime_error("this build of splat cannot read videos (built without FFmpeg)");
}

} // namespace splat

#endif
