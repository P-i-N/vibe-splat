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

// Without FFmpeg's libraries, run the ffmpeg program found on PATH. It decodes
// to PPM images on stdout, and its showinfo filter reports each frame's
// timestamp on stderr. ffmpeg applies rotation metadata itself.

#include <cctype>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string_view>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace splat {

namespace {

namespace fs = std::filesystem;

constexpr int kScoreSize = 640;

std::optional<fs::path> find_ffmpeg() {
#ifdef _WIN32
  constexpr wchar_t kSeparator = L';';
  constexpr const wchar_t *kName = L"ffmpeg.exe";
  std::wstring paths(32767, L'\0');
  paths.resize(GetEnvironmentVariableW(L"PATH", paths.data(), static_cast<DWORD>(paths.size())));
#else
  constexpr char kSeparator = ':';
  constexpr const char *kName = "ffmpeg";
  const char *env = std::getenv("PATH");
  const std::string paths = env != nullptr ? env : "";
#endif
  std::size_t begin = 0;
  while (begin <= paths.size()) {
    std::size_t end = paths.find(kSeparator, begin);
    if (end == paths.npos) end = paths.size();
    auto dir = paths.substr(begin, end - begin);
    begin = end + 1;
    if (dir.size() >= 2 && dir.front() == '"' && dir.back() == '"') dir = dir.substr(1, dir.size() - 2);
    if (dir.empty()) continue;
    std::error_code ec;
    const fs::path candidate = fs::path(dir) / kName;
    if (fs::is_regular_file(candidate, ec)) return candidate;
  }
  return std::nullopt;
}

// Looked up once: PATH does not change while splat runs.
const std::optional<fs::path> &ffmpeg_path() {
  static const std::optional<fs::path> path = find_ffmpeg();
  return path;
}

// A child process with its stdout and stderr connected to pipes. The
// destructor kills it if it is still running.
class ChildProcess {
public:
  ChildProcess(const fs::path &program, const std::vector<fs::path> &args) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE outWrite = nullptr, errWrite = nullptr;
    if (!CreatePipe(&out_, &outWrite, &inherit, 0) || !CreatePipe(&err_, &errWrite, &inherit, 0)) {
      throw std::runtime_error("cannot create a pipe for ffmpeg");
    }
    SetHandleInformation(out_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_, HANDLE_FLAG_INHERIT, 0);
    std::wstring commandLine = quote(program.native());
    for (const fs::path &arg : args) commandLine += L' ' + quote(arg.native());
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = outWrite;
    startup.hStdError = errWrite;
    PROCESS_INFORMATION info{};
    const BOOL started = CreateProcessW(program.c_str(), commandLine.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                        nullptr, nullptr, &startup, &info);
    CloseHandle(outWrite);
    CloseHandle(errWrite);
    if (!started) throw std::runtime_error(std::format("cannot run {} (error {})", program.string(), GetLastError()));
    CloseHandle(info.hThread);
    process_ = info.hProcess;
#else
    int outPipe[2], errPipe[2];
    if (pipe(outPipe) != 0) throw std::runtime_error("cannot create a pipe for ffmpeg");
    out_ = outPipe[0];
    if (pipe(errPipe) != 0) {
      close(outPipe[1]);
      throw std::runtime_error("cannot create a pipe for ffmpeg");
    }
    err_ = errPipe[0];
    fcntl(out_, F_SETFD, FD_CLOEXEC);
    fcntl(err_, F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, outPipe[1], 1);
    posix_spawn_file_actions_adddup2(&actions, errPipe[1], 2);
    posix_spawn_file_actions_addclose(&actions, outPipe[1]);
    posix_spawn_file_actions_addclose(&actions, errPipe[1]);
    std::vector<char *> argv{const_cast<char *>(program.c_str())};
    for (const fs::path &arg : args) argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);
    const int r = posix_spawn(&pid_, program.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(outPipe[1]);
    close(errPipe[1]);
    if (r != 0) {
      pid_ = -1;
      close_pipes();
      throw std::runtime_error(std::format("cannot run {}: {}", program.string(), std::strerror(r)));
    }
#endif
  }

  ChildProcess(const ChildProcess &) = delete;
  ChildProcess &operator=(const ChildProcess &) = delete;

  ~ChildProcess() {
    kill();
    close_pipes();
#ifdef _WIN32
    if (process_ != nullptr) CloseHandle(process_);
#endif
  }

  // Reads up to `size` bytes of stdout (or stderr); 0 means end of output.
  std::size_t read_out(std::uint8_t *data, std::size_t size) { return read(out_, data, size); }
  std::size_t read_err(std::uint8_t *data, std::size_t size) { return read(err_, data, size); }

  // Stops the process if it is still running, which ends its output.
  void kill() {
    if (!running()) return;
#ifdef _WIN32
    TerminateProcess(process_, 1);
#else
    ::kill(pid_, SIGKILL);
#endif
    static_cast<void>(wait());
  }

  // Waits for the process to exit and returns its exit code.
  int wait() {
#ifdef _WIN32
    WaitForSingleObject(process_, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process_, &code);
    exitCode_ = static_cast<int>(code);
#else
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    exitCode_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    pid_ = -1;
#endif
    return *exitCode_;
  }

private:
#ifdef _WIN32
  // Quotes an argument so CommandLineToArgvW gives it back unchanged.
  static std::wstring quote(const std::wstring &arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
    std::wstring out = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t c : arg) {
      if (c == L'\\') {
        ++backslashes;
        continue;
      }
      out.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
      backslashes = 0;
      out.push_back(c);
    }
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
  }

  static std::size_t read(HANDLE pipe, std::uint8_t *data, std::size_t size) {
    DWORD got = 0;
    if (!ReadFile(pipe, data, static_cast<DWORD>(std::min<std::size_t>(size, 1 << 20)), &got, nullptr)) return 0;
    return got;
  }

  [[nodiscard]] bool running() const { return process_ != nullptr && !exitCode_; }

  void close_pipes() {
    if (out_ != nullptr) CloseHandle(out_);
    if (err_ != nullptr) CloseHandle(err_);
    out_ = err_ = nullptr;
  }

  HANDLE process_ = nullptr;
  HANDLE out_ = nullptr;
  HANDLE err_ = nullptr;
#else
  static std::size_t read(int fd, std::uint8_t *data, std::size_t size) {
    for (;;) {
      const ssize_t got = ::read(fd, data, size);
      if (got >= 0) return static_cast<std::size_t>(got);
      if (errno != EINTR) return 0;
    }
  }

  [[nodiscard]] bool running() const { return pid_ > 0; }

  void close_pipes() {
    if (out_ >= 0) close(out_);
    if (err_ >= 0) close(err_);
    out_ = err_ = -1;
  }

  pid_t pid_ = -1;
  int out_ = -1;
  int err_ = -1;
#endif
  std::optional<int> exitCode_;
};

// What ffmpeg says on stderr: the input's duration, then one timestamp per
// frame from the showinfo filter. Filled by a reader thread.
class FfmpegLog {
public:
  void add_line(std::string_view line) {
    const std::lock_guard lock(mutex_);
    if (line.find("Parsed_showinfo") != line.npos) {
      const std::size_t at = line.find("pts_time:");
      if (at == line.npos || line.find(" n:") == line.npos) return;
      // "pts_time:NOPTS" parses as NaN, which the caller handles.
      times_.push_back(std::strtod(std::string(line.substr(at + 9, 32)).c_str(), nullptr));
      if (line.find("pts_time:NOPTS") != line.npos) times_.back() = std::nan("");
      changed_.notify_all();
      return;
    }
    if (const std::size_t at = line.find("Duration: "); at != line.npos && !duration_) {
      // "HH:MM:SS.ss", or "N/A" when unknown.
      const std::string_view text = line.substr(at + 10);
      const char *end = text.data() + text.size();
      int h = 0, m = 0;
      double s = 0.0;
      const auto field = [&](const char *from, auto &value, bool colonAfter) -> const char * {
        const auto r = std::from_chars(from, end, value);
        if (r.ec != std::errc{}) return nullptr;
        if (!colonAfter) return r.ptr;
        return r.ptr != end && *r.ptr == ':' ? r.ptr + 1 : nullptr;
      };
      const char *p = field(text.data(), h, true);
      if (p != nullptr) p = field(p, m, true);
      if (p != nullptr && field(p, s, false) != nullptr) duration_ = h * 3600.0 + m * 60.0 + s;
    }
    tail_.emplace_back(line);
    if (tail_.size() > 4) tail_.pop_front();
  }

  void finish() {
    const std::lock_guard lock(mutex_);
    done_ = true;
    changed_.notify_all();
  }

  // Waits for frame `index`'s timestamp in seconds. NaN if ffmpeg did not
  // report it. The duration is known by then if ffmpeg knows it.
  double frame_seconds(std::size_t index) {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return done_ || times_.size() > index; });
    return index < times_.size() ? times_[index] : std::nan("");
  }

  std::optional<double> duration() {
    const std::lock_guard lock(mutex_);
    return duration_;
  }

  // The last lines that were not frame reports, which hold any error.
  std::string tail() {
    const std::lock_guard lock(mutex_);
    std::string text;
    for (const std::string &line : tail_) text += (text.empty() ? "" : " / ") + line;
    return text;
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<double> times_;
  std::optional<double> duration_;
  std::deque<std::string> tail_;
  bool done_ = false;
};

// Buffered reading of ffmpeg's stdout.
class OutputReader {
public:
  explicit OutputReader(ChildProcess &process) : process_(process), buffer_(1 << 16) {}

  // -1 at end of output.
  int get() {
    if (pos_ == size_ && !fill()) return -1;
    return buffer_[pos_++];
  }

  bool read(std::uint8_t *data, std::size_t size) {
    while (size > 0) {
      if (pos_ == size_ && !fill()) return false;
      const std::size_t n = std::min(size, size_ - pos_);
      std::copy_n(buffer_.data() + pos_, n, data);
      pos_ += n;
      data += n;
      size -= n;
    }
    return true;
  }

private:
  bool fill() {
    size_ = process_.read_out(buffer_.data(), buffer_.size());
    pos_ = 0;
    return size_ > 0;
  }

  ChildProcess &process_;
  std::vector<std::uint8_t> buffer_;
  std::size_t pos_ = 0, size_ = 0;
};

// Reads one binary PPM image. Nothing at the end of the output.
std::optional<RgbImage> read_ppm(OutputReader &in) {
  int c = in.get();
  if (c < 0) return std::nullopt;
  const auto bad = [] { return std::runtime_error("ffmpeg sent a malformed image"); };
  if (c != 'P' || in.get() != '6') throw bad();
  const auto number = [&] {
    c = in.get();
    while (c == '#' || std::isspace(c)) {
      if (c == '#') {
        while (c >= 0 && c != '\n') c = in.get();
      }
      c = in.get();
    }
    long value = 0;
    if (c < '0' || c > '9') throw bad();
    while (c >= '0' && c <= '9') {
      value = value * 10 + (c - '0');
      if (value > 1 << 16) throw bad();
      c = in.get();
    }
    return static_cast<int>(value); // Leaves the single whitespace after it consumed
  };
  RgbImage image;
  image.width = number();
  image.height = number();
  if (number() != 255 || image.width == 0 || image.height == 0) throw bad();
  image.pixels.resize(static_cast<std::size_t>(image.width) * image.height * 3);
  if (!in.read(image.pixels.data(), image.pixels.size())) throw std::runtime_error("ffmpeg's output ended mid-frame");
  return image;
}

// Sharpness of a grey copy at most kScoreSize on the long edge, like the
// library path scores it.
double score(const RgbImage &image) {
  const double scale = std::min(1.0, static_cast<double>(kScoreSize) / std::max(image.width, image.height));
  const int w = std::max(3, static_cast<int>(std::lround(image.width * scale)));
  const int h = std::max(3, static_cast<int>(std::lround(image.height * scale)));
  std::vector<std::uint8_t> grey(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    const int y0 = y * image.height / h, y1 = std::max(y0 + 1, (y + 1) * image.height / h);
    for (int x = 0; x < w; ++x) {
      const int x0 = x * image.width / w, x1 = std::max(x0 + 1, (x + 1) * image.width / w);
      std::uint32_t sum = 0;
      for (int sy = y0; sy < y1; ++sy) {
        const std::uint8_t *p = image.pixels.data() + (static_cast<std::size_t>(sy) * image.width + x0) * 3;
        for (int sx = x0; sx < x1; ++sx, p += 3) sum += 77u * p[0] + 150u * p[1] + 29u * p[2];
      }
      grey[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(sum / (256u * (y1 - y0) * (x1 - x0)));
    }
  }
  return sharpness(grey, w, h, w);
}

} // namespace

bool video_supported() noexcept { return ffmpeg_path().has_value(); }

void extract_video_frames(const std::filesystem::path &path, const VideoOptions &options,
                          const std::function<void(VideoFrame &&)> &sink) {
  const std::optional<fs::path> &ffmpeg = ffmpeg_path();
  if (!ffmpeg) throw std::runtime_error("cannot read videos: splat was built without FFmpeg and ffmpeg is not on PATH");

  // "file:" stops ffmpeg from reading a name like "a:b.mp4" as a protocol.
  fs::path input = "file:";
  input += fs::absolute(path).native();
  ChildProcess process(*ffmpeg, {"-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-i", input, "-map",
                                 "0:v:0", "-vf", "showinfo", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-c:v",
                                 "ppm", "-f", "image2pipe", "-"});
  FfmpegLog log;
  std::jthread logReader([&] {
    std::string line;
    std::uint8_t chunk[4096];
    for (std::size_t n; (n = process.read_err(chunk, sizeof(chunk))) > 0;) {
      for (std::size_t i = 0; i < n; ++i) {
        if (chunk[i] == '\n' || chunk[i] == '\r') {
          if (!line.empty()) log.add_line(line);
          line.clear();
        } else {
          line.push_back(static_cast<char>(chunk[i]));
        }
      }
    }
    if (!line.empty()) log.add_line(line);
    log.finish();
  });
  // Runs before the reader thread is joined: if `sink` throws, ffmpeg would
  // otherwise wait on a full stdout pipe and never close stderr.
  struct KillOnExit {
    ChildProcess &process;
    ~KillOnExit() { process.kill(); }
  } killOnExit{process};
  const auto fail = [&](const std::string &what) {
    return std::runtime_error(std::format("{}: {}: {}", path.string(), what, log.tail()));
  };

  OutputReader out(process);
  double fps = options.fps;
  std::optional<RgbImage> best;
  double bestScore = -1.0, bestSeconds = 0.0, lastSeconds = 0.0;
  std::int64_t bucket = -1;
  int emitted = 0;

  const auto flush_best = [&] {
    if (!best) return;
    if (options.maxFrames > 0 && emitted >= options.maxFrames) {
      best.reset();
      return;
    }
    VideoFrame frame;
    frame.image = std::move(*best);
    frame.seconds = bestSeconds;
    frame.sharpness = bestScore;
    best.reset();
    bestScore = -1.0;
    ++emitted;
    sink(std::move(frame));
  };

  for (std::size_t index = 0;; ++index) {
    std::optional<RgbImage> image;
    try {
      image = read_ppm(out);
    } catch (const std::runtime_error &e) {
      throw fail(e.what());
    }
    if (!image) break;
    const double reported = log.frame_seconds(index);
    const double seconds = std::isnan(reported) ? lastSeconds : reported;
    lastSeconds = seconds;
    if (index == 0) {
      // Lower the rate if the limit would otherwise cut the video short.
      const std::optional<double> duration = log.duration();
      if (options.maxFrames > 0 && duration && *duration > 0.0) {
        const double limited = options.maxFrames / *duration;
        fps = fps > 0.0 ? std::min(fps, limited) : limited;
      }
    }
    const std::int64_t thisBucket =
        fps > 0.0 ? static_cast<std::int64_t>(std::floor(seconds * fps)) : static_cast<std::int64_t>(index);
    if (thisBucket != bucket) {
      flush_best();
      bucket = thisBucket;
    }
    if (const double s = score(*image); s > bestScore) {
      best = std::move(image);
      bestScore = s;
      bestSeconds = seconds;
    }
  }
  if (process.wait() != 0) {
    logReader.join();
    throw fail("ffmpeg failed");
  }
  flush_best();
  if (emitted == 0) throw std::runtime_error(std::format("{}: no frames could be decoded", path.string()));
}

} // namespace splat

#endif
