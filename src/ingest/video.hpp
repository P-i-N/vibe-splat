#pragma once

#include "ingest/image.hpp"

#include <filesystem>
#include <functional>

namespace splat {

struct VideoOptions {
  double fps = 2.0;  // Frames to keep per second of video; 0 keeps every frame
  int maxFrames = 0; // 0 means no limit. Lowers the rate to cover the whole video.
};

struct VideoFrame {
  RgbImage image;       // Already rotated according to the video's metadata
  double seconds = 0.0; // Position in the video
  double sharpness = 0.0;
};

// True if this build can read videos.
[[nodiscard]] bool video_supported() noexcept;

// Decodes a video and calls `sink` with the sharpest frame of each interval
// of 1/fps seconds, in order. Motion blur ruins training images, so this beats
// taking whichever frame lands on the interval. Throws std::runtime_error.
void extract_video_frames(const std::filesystem::path &path, const VideoOptions &options,
                          const std::function<void(VideoFrame &&)> &sink);

} // namespace splat
