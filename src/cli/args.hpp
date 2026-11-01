#pragma once

#include <algorithm>
#include <charconv>
#include <format>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splat::cli {

// Minimal argument parser: positionals plus "--name value" and "--flag".
class Args {
public:
  Args(const std::vector<std::string_view> &args, const std::vector<std::string_view> &flags) {
    for (std::size_t i = 0; i < args.size(); ++i) {
      const std::string_view a = args[i];
      if (a.starts_with("--")) {
        const std::string name(a.substr(2));
        if (std::find(flags.begin(), flags.end(), a) != flags.end()) {
          options_[name] = "1";
        } else if (i + 1 < args.size()) {
          options_[name] = std::string(args[++i]);
        } else {
          throw std::runtime_error(std::format("Option {} needs a value", a));
        }
      } else {
        positionals_.emplace_back(a);
      }
    }
  }

  [[nodiscard]] const std::vector<std::string> &positionals() const { return positionals_; }
  [[nodiscard]] bool has(const std::string &name) const { return options_.contains(name); }

  [[nodiscard]] std::string get(const std::string &name, std::string fallback) const {
    const auto it = options_.find(name);
    return it == options_.end() ? fallback : it->second;
  }

  template <class T> [[nodiscard]] T number(const std::string &name, T fallback) const {
    const auto it = options_.find(name);
    if (it == options_.end()) {
      return fallback;
    }
    T value{};
    const std::string &s = it->second;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
      throw std::runtime_error(std::format("--{} expects a number, got '{}'", name, s));
    }
    return value;
  }

  // Throws if an option was given that the command does not know.
  void allow_only(const std::vector<std::string_view> &known) const {
    for (const auto &[name, value] : options_) {
      if (std::find(known.begin(), known.end(), name) == known.end()) {
        throw std::runtime_error(std::format("Unknown option --{}", name));
      }
    }
  }

private:
  std::vector<std::string> positionals_;
  std::map<std::string, std::string> options_;
};

} // namespace splat::cli
