#include "core/progress.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace splat {

namespace console {

namespace {

bool env_set(const char *name) {
#ifdef _WIN32
  char *value = nullptr;
  std::size_t size = 0;
  const bool set = _dupenv_s(&value, &size, name) == 0 && value != nullptr && *value != '\0';
  std::free(value);
  return set;
#else
  const char *value = std::getenv(name);
  return value != nullptr && *value != '\0';
#endif
}

const char *code_of(Color color) {
  switch (color) {
  case Color::Red: return "\x1b[31m";
  case Color::Green: return "\x1b[32m";
  case Color::Yellow: return "\x1b[33m";
  case Color::Cyan: return "\x1b[36m";
  case Color::Gray: return "\x1b[90m";
  case Color::Bold: return "\x1b[1m";
  }
  return "";
}

} // namespace

bool is_terminal(std::FILE *stream) {
#ifdef _WIN32
  const int fd = _fileno(stream);
  if (fd < 0) return false;
  const HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  DWORD mode = 0;
  if (handle == INVALID_HANDLE_VALUE || !GetConsoleMode(handle, &mode)) return false;
  if ((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0 && !SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) return false;
  // The bar is drawn with block characters.
  static const bool utf8 = SetConsoleOutputCP(CP_UTF8) != 0;
  return utf8;
#else
  if (!isatty(fileno(stream))) return false;
  const char *term = std::getenv("TERM");
  return term == nullptr || std::string_view(term) != "dumb";
#endif
}

bool color_enabled(std::FILE *stream) { return is_terminal(stream) && !env_set("NO_COLOR"); }

int width(std::FILE *stream) {
#ifdef _WIN32
  const int fd = _fileno(stream);
  if (fd >= 0) {
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), &info)) {
      return std::max(info.srWindow.Right - info.srWindow.Left + 1, 1);
    }
  }
#else
  winsize size{};
  if (ioctl(fileno(stream), TIOCGWINSZ, &size) == 0 && size.ws_col > 0) return size.ws_col;
#endif
  return 80;
}

std::string paint(Color color, std::string_view text, std::FILE *stream) {
  if (!color_enabled(stream)) return std::string(text);
  return std::format("{}{}\x1b[0m", code_of(color), text);
}

} // namespace console

namespace {

constexpr const char *kFilled = "\xE2\x96\x88"; // Full block
constexpr const char *kEmpty = "\xE2\x96\x91";  // Light shade
constexpr std::size_t kLabelColumn = 9;
constexpr int kMinBarCells = 10;
constexpr int kMaxBarCells = 40;
constexpr auto kRedrawInterval = std::chrono::milliseconds(50);

std::string format_time(double seconds) {
  const long total = static_cast<long>(seconds + 0.5);
  if (total >= 3600) return std::format("{}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60);
  return std::format("{}:{:02}", total / 60, total % 60);
}

} // namespace

std::string Progress::render(const Line &line, int columns, bool color) {
  const int maxWidth = std::max(columns - 1, 19); // The last column is left alone so the line never wraps
  std::string out;
  int used = 0;
  const auto put = [&](std::string_view text, const char *code) {
    if (color && code != nullptr) {
      out += code;
      out += text;
      out += "\x1b[0m";
    } else {
      out += text;
    }
    used += static_cast<int>(text.size());
  };
  constexpr const char *kBoldCyan = "\x1b[1;36m";
  constexpr const char *kGray = "\x1b[90m";
  constexpr const char *kYellow = "\x1b[33m";
  constexpr const char *kGreen = "\x1b[32m";
  constexpr const char *kBold = "\x1b[1m";

  std::string label(line.label);
  if (maxWidth >= 60 && label.size() < kLabelColumn) label.resize(kLabelColumn, ' ');
  label += ' ';
  put(label, kBoldCyan);

  const auto info_piece = [&] {
    const int room = maxWidth - used - 1;
    if (line.info.empty() || room < 5) return;
    std::string text(line.info);
    if (static_cast<int>(text.size()) > room) {
      text.resize(static_cast<std::size_t>(room));
      text.back() = '~';
    }
    put(" ", nullptr);
    put(text, kGray);
  };

  if (line.total == 0) { // No total: a count and the time passed
    put(std::format("{}", line.done), kBold);
    const std::string elapsed = " " + format_time(line.elapsedSeconds);
    if (maxWidth - used >= static_cast<int>(elapsed.size())) put(elapsed, kYellow);
    info_piece();
    return out;
  }

  const std::size_t done = std::min(line.done, line.total);
  const std::string count = std::format("{:{}}/{}", done, std::to_string(line.total).size(), line.total);
  const std::string percent = std::format("{:3}%", done * 100 / line.total);

  // The bar gets about a third of the line. The count goes first when it will not fit.
  int cells = std::clamp(maxWidth * 35 / 100, kMinBarCells, kMaxBarCells);
  bool showCount = true;
  const auto room_for_bar = [&] { return maxWidth - used - 2 - 1 - static_cast<int>(percent.size()) - (showCount ? 1 + static_cast<int>(count.size()) : 0); };
  if (room_for_bar() < 6) showCount = false;
  cells = std::max(std::min(cells, room_for_bar()), 3);

  std::size_t filled = done * static_cast<std::size_t>(cells) / line.total;
  if (done < line.total) filled = std::min(filled, static_cast<std::size_t>(cells) - 1);
  put("[", kGray);
  std::string bar;
  for (std::size_t i = 0; i < filled; ++i) bar += kFilled;
  std::string rest;
  for (std::size_t i = filled; i < static_cast<std::size_t>(cells); ++i) rest += kEmpty;
  // The blocks are three bytes but one column wide.
  if (color) out += kGreen;
  out += bar;
  if (color) out += std::string("\x1b[0m") + kGray;
  out += rest;
  if (color) out += "\x1b[0m";
  used += cells;
  put("]", kGray);

  put(" " + percent, line.finished ? kGreen : kBold);
  if (showCount) put(" " + count, nullptr);

  std::string time;
  if (line.finished) {
    time = " in " + format_time(line.elapsedSeconds);
  } else if (done > 0 && line.elapsedSeconds >= 0.5) {
    time = " ETA " + format_time(line.elapsedSeconds * static_cast<double>(line.total - done) / static_cast<double>(done));
  }
  if (!time.empty() && maxWidth - used >= static_cast<int>(time.size())) put(time, kYellow);
  info_piece();
  return out;
}

Progress::Progress(std::string label, std::size_t total)
    : label_(std::move(label)), total_(total), enabled_(console::is_terminal(stdout)), start_(std::chrono::steady_clock::now()) {}

Progress::~Progress() { finish(); }

void Progress::start() {
  const std::lock_guard lock(mutex_);
  start_ = std::chrono::steady_clock::now();
}

void Progress::advance(std::size_t n, std::string_view info) {
  done_ += n;
  update(info, false);
}

void Progress::set(std::size_t done, std::string_view info) {
  done_ = done;
  update(info, false);
}

void Progress::update(std::string_view info, bool force) {
  if (!enabled_) return;
  // A thread that finds the bar busy leaves the drawing to the one holding it.
  std::unique_lock lock(mutex_, std::try_to_lock);
  if (!lock || finished_) return;
  if (!info.empty()) info_ = info;
  const auto now = std::chrono::steady_clock::now();
  if (!force && drawn_ && now - lastDraw_ < kRedrawInterval) return;
  lastDraw_ = now;
  draw();
}

void Progress::draw() {
  Line line;
  line.label = label_;
  line.done = done_;
  line.total = total_;
  line.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  line.info = info_;
  line.finished = finished_ || (total_ > 0 && line.done >= total_);
  const std::string text = render(line, console::width(stdout), console::color_enabled(stdout));
  std::fputs("\r", stdout);
  std::fputs(text.c_str(), stdout);
  std::fputs("\x1b[K", stdout);
  std::fflush(stdout);
  drawn_ = true;
}

void Progress::message(std::string_view text, bool toStderr) {
  const std::lock_guard lock(mutex_);
  const bool live = enabled_ && drawn_ && !finished_;
  if (live) {
    std::fputs("\r\x1b[K", stdout);
    std::fflush(stdout);
  }
  std::FILE *stream = toStderr ? stderr : stdout;
  std::fwrite(text.data(), 1, text.size(), stream);
  std::fputc('\n', stream);
  std::fflush(stream);
  if (live) draw();
}

void Progress::finish() {
  const std::lock_guard lock(mutex_);
  if (finished_) return;
  finished_ = true;
  if (!enabled_ || (!drawn_ && done_ == 0)) return;
  draw();
  std::fputs("\n", stdout);
  std::fflush(stdout);
}

} // namespace splat
