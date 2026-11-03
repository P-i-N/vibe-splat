#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>

namespace splat {

namespace console {

enum class Color { Red, Green, Yellow, Cyan, Gray, Bold };

// True if the stream is a terminal that understands ANSI escape sequences. On
// Windows this also switches the console to UTF-8 and turns escape processing on.
[[nodiscard]] bool is_terminal(std::FILE *stream);

// Colors are used on a terminal unless the NO_COLOR environment variable is set.
[[nodiscard]] bool color_enabled(std::FILE *stream);

// The terminal's width in columns, or 80 if it cannot be determined.
[[nodiscard]] int width(std::FILE *stream);

// The text in the color's escape codes, or unchanged where colors are off.
[[nodiscard]] std::string paint(Color color, std::string_view text, std::FILE *stream = stdout);

} // namespace console

// A progress bar on one terminal line, redrawn in place. It fits the terminal's
// current width: the bar takes about a third of the line and the details
// (count, time left, info text) are dropped from the right as the line gets
// narrower. When stdout is not a terminal nothing is drawn.
//
// advance() and set() may be called from several threads. Output of other
// kinds goes through message(), so that it does not break the bar's line.
class Progress {
public:
  // A total of 0 means unknown: no bar, just the count and the time passed.
  Progress(std::string label, std::size_t total);
  ~Progress();
  Progress(const Progress &) = delete;
  Progress &operator=(const Progress &) = delete;

  // Restarts the clock that time left is estimated from.
  void start();
  void advance(std::size_t n = 1, std::string_view info = {});
  void set(std::size_t done, std::string_view info = {});
  // Prints a line above the bar.
  void message(std::string_view text, bool toStderr = false);
  // Draws the bar one last time and moves to the next line. Called by the destructor.
  void finish();

  struct Line {
    std::string_view label;
    std::size_t done = 0;
    std::size_t total = 0;
    double elapsedSeconds = 0.0;
    std::string_view info;
    bool finished = false; // Shows the time taken instead of the time left
  };
  // The text of one line, at most `columns` characters wide.
  [[nodiscard]] static std::string render(const Line &line, int columns, bool color);

private:
  void draw();
  void update(std::string_view info, bool force);

  std::string label_;
  std::size_t total_;
  bool enabled_;
  std::atomic<std::size_t> done_{0};
  std::mutex mutex_;
  std::string info_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point lastDraw_{};
  bool drawn_ = false;
  bool finished_ = false;
};

} // namespace splat
