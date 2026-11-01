cmake_minimum_required(VERSION 3.28)

project(splat VERSION 0.1.0 LANGUAGES CXX)

# C++23 is the baseline. C++26 features are only used behind feature-test
# macros once GCC, Clang and MSVC all support them.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Self-contained binaries: static MSVC runtime on Windows, static libstdc++ on Linux.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(SPLAT_BUILD_TESTS "Build unit tests" ON)

find_package(volk CONFIG REQUIRED)
# vcpkg provides VulkanHeaders as a config package; system installs (used for
# quick local builds without vcpkg) provide the same Vulkan::Headers target
# through CMake's FindVulkan module.
find_package(VulkanHeaders CONFIG QUIET)
if(NOT TARGET Vulkan::Headers)
  find_package(Vulkan REQUIRED)
endif()
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
# stb is header-only; vcpkg installs it to include/, Debian/Ubuntu to include/stb/.
find_path(STB_INCLUDE_DIR stb_image_write.h PATH_SUFFIXES stb REQUIRED)
# The window viewer needs SDL3. Without it, `splat view` is not available but
# everything else (including headless `splat render`) still builds.
find_package(SDL3 CONFIG QUIET)
# Video input needs FFmpeg's libraries. Without them, `splat ingest` handles
# photos only.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
endif()

include(cmake/CompilerWarnings.cmake)
include(cmake/Shaders.cmake)

add_subdirectory(shaders)
add_subdirectory(src)

if(SPLAT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
