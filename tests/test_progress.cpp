#include "core/progress.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace splat;

namespace {

// The columns a line takes up: every UTF-8 lead byte, but no escape sequence.
int visible_columns(const std::string &text) {
  int columns = 0;
  bool escape = false;
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (escape) {
      if (c == 'm') escape = false;
    } else if (c == '\x1b') {
      escape = true;
    } else if ((byte & 0xC0) != 0x80) {
      ++columns;
    }
  }
  return columns;
}

Progress::Line sample_line() {
  Progress::Line line;
  line.label = "Matching";
  line.done = 42;
  line.total = 100;
  line.elapsedSeconds = 30.0;
  line.info = "17 verified";
  return line;
}

} // namespace

TEST_CASE("A progress line shows the percentage, count and time left", "[progress]") {
  const std::string text = Progress::render(sample_line(), 100, false);
  CHECK(text.find("Matching") == 0);
  CHECK(text.find(" 42%") != std::string::npos);
  CHECK(text.find("42/100") != std::string::npos);
  CHECK(text.find("ETA 0:41") != std::string::npos); // 30 s for 42, so 58 are left
  CHECK(text.find("17 verified") != std::string::npos);
}

TEST_CASE("A progress line fits any terminal width", "[progress]") {
  for (const bool color : {false, true}) {
    for (int columns = 20; columns <= 200; ++columns) {
      CAPTURE(columns, color);
      CHECK(visible_columns(Progress::render(sample_line(), columns, color)) <= columns - 1);
    }
  }
}

TEST_CASE("The bar grows with the terminal and the details drop out of narrow ones", "[progress]") {
  const std::string wide = Progress::render(sample_line(), 160, false);
  const std::string narrow = Progress::render(sample_line(), 30, false);
  CHECK(visible_columns(wide) > visible_columns(narrow));
  CHECK(wide.find("verified") != std::string::npos);
  CHECK(narrow.find("verified") == std::string::npos);
  CHECK(narrow.find("%") != std::string::npos);
}

TEST_CASE("A long info text is cut to the line", "[progress]") {
  Progress::Line line = sample_line();
  line.info = std::string(500, 'x');
  const std::string text = Progress::render(line, 80, false);
  CHECK(visible_columns(text) <= 79);
  CHECK(text.back() == '~');
}

TEST_CASE("A finished line shows the time taken", "[progress]") {
  Progress::Line line = sample_line();
  line.done = 100;
  line.finished = true;
  const std::string text = Progress::render(line, 100, false);
  CHECK(text.find("100%") != std::string::npos);
  CHECK(text.find("in 0:30") != std::string::npos);
  CHECK(text.find("ETA") == std::string::npos);
}

TEST_CASE("Without a total there is only a count", "[progress]") {
  Progress::Line line = sample_line();
  line.total = 0;
  line.done = 7;
  const std::string text = Progress::render(line, 100, false);
  CHECK(text.find("7 0:30") != std::string::npos);
  CHECK(text.find('/') == std::string::npos);
}

TEST_CASE("Colors are escape codes that plain output leaves out", "[progress]") {
  CHECK(Progress::render(sample_line(), 100, false).find('\x1b') == std::string::npos);
  CHECK(Progress::render(sample_line(), 100, true).find("\x1b[") != std::string::npos);
}
