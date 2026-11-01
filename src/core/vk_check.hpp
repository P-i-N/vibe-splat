#pragma once

#include <volk.h>

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splat::vk {

// Thrown for any failed Vulkan call or unrecoverable setup problem.
class Error : public std::runtime_error {
public:
  Error(std::string message, VkResult result = VK_ERROR_UNKNOWN)
      : std::runtime_error(std::move(message)), result_(result) {}
  [[nodiscard]] VkResult result() const noexcept { return result_; }

private:
  VkResult result_;
};

[[nodiscard]] std::string_view to_string(VkResult result) noexcept;

void check(VkResult result, std::string_view what,
           std::source_location loc = std::source_location::current());

} // namespace splat::vk
