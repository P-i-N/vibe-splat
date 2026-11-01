# vibe-splat

A Gaussian splat generator and viewer that runs on consumer GPUs using only
Vulkan. No CUDA or other vendor libraries, so it works on AMD and NVIDIA
alike. The main target is the Steam Machine (SteamOS); Windows is supported
too.

Input is a set of photos or a video, plus optional camera information read
from EXIF or given by hand. Output is a standard 3D Gaussian splatting `.ply`.

## Status

Milestones M1 (viewer), M2 (ingest) and M3 (camera poses) are done:

- `splat view scene.ply` opens a window and renders a 3DGS `.ply` in real time.
  Controls: left-drag orbits, right- or middle-drag pans, the wheel zooms,
  W/A/S/D/Q/E move, R resets the view and Esc quits. On a gamepad (Steam
  Machine controller, Xbox or PlayStation pads), the left stick moves, the
  right stick orbits, the triggers zoom and A resets the view.
- `splat render scene.ply -o out.png` renders one image without a window.
- `splat make-test-scene out.ply` writes a synthetic scene, so you can try
  everything without a trained splat.
- `splat info` lists Vulkan devices; `splat selftest` checks the GPU works.
- `splat ingest photos/ clip.mp4 -o project/` imports photos and videos into a
  project folder (see below).
- `splat sfm project/` estimates camera poses and a sparse point cloud (see
  below).
- `splat make-test-views dir/` renders images with known poses to try it on.
- `train` is a placeholder for a later milestone.

Any `.ply` in the standard 3DGS layout should work, with SH degree 0 to 3.
That is the layout written by the original implementation and most trainers.

| Milestone | Scope |
| --- | --- |
| M0 Foundation | CMake + vcpkg, Vulkan core, CI on Linux and Windows |
| M1 Viewer | Load `.ply` splats, GPU sort, render, window and controls |
| M2 Ingest | Photos, video frames, EXIF and manual intrinsics |
| M3 Camera poses | Our own structure from motion (CPU), COLMAP model import |
| M4 Trainer core | Vulkan forward, loss, backward, Adam |
| M5 Full training | Densification, SH degree 3, 8 GB VRAM budget |
| M6 Polish | GUI, live preview, gamepad, AppImage and Windows zip |

## How ingest works

`splat ingest` writes `images/` and `ingest.json` into the project folder, for
the later stages to read. Rerun it with `--force` to replace them.

- **Photos** can be JPEG, PNG, BMP or TGA. HEIC and raw files need converting
  first. EXIF orientation is applied, so stored images are always upright.
- **Videos** (needs a build with FFmpeg) keep the sharpest frame of each interval
  of 1/`--fps` seconds, since motion blur ruins training images. Rotation
  metadata is applied.
- **Size**: the longest side is reduced to `--max-size` (default 1600).
- **Cameras**: images with the same size and focal length share a camera in
  `ingest.json`. The focal length comes from the first of these available,
  and `focal_source` records which:
  1. `manual`: `--focal-35mm`, `--focal-px`, `--focal-mm` with
     `--sensor-width-mm`, or `--fov`
  2. `exif-35mm`: EXIF FocalLengthIn35mmFilm
  3. `exif-focal-plane`: EXIF FocalLength and FocalPlaneResolution
  4. `default`: a guess of 1.2 x the longest side, for camera-pose estimation
     to refine. Video always lands here unless given by hand.

## How pose estimation works

`splat sfm` is a small incremental structure-from-motion pipeline written from
scratch, needing only Eigen. It runs on the CPU:

1. **Features.** SIFT: extrema of a difference-of-Gaussians pyramid, described
   by gradient histograms (RootSIFT).
2. **Matching.** Brute-force mutual nearest neighbours with Lowe's ratio test.
   Every pair for up to 150 photos; for videos and larger sets, neighbours in
   input order plus a few at growing distances (`--matcher` overrides this).
3. **Verification.** RANSAC around the five-point essential matrix solver,
   using ingest's intrinsics. Pairs that do not fit a rigid camera motion go.
4. **Tracks.** Matches are chained into tracks that follow one 3D point.
5. **Start.** The best-matched pair with a wide enough baseline is triangulated.
6. **Growth.** The image seeing the most known points is registered with P3P
   RANSAC, what it newly sees is triangulated, and the model is bundle adjusted
   as it grows. Bundle adjustment is Levenberg-Marquardt with a Huber loss and
   the points eliminated through the Schur complement.
7. **Finish.** A final bundle adjustment refines focal length and radial
   distortion (a focal length given by hand stays fixed), and points missing by
   more than 1.5 pixels are dropped.

The result goes to `project/sparse/0` in COLMAP's binary format, with a
`SIMPLE_RADIAL` camera, so other tools can read it. `preview.ply` next to it
shows the points in `splat view`. `splat sfm project/ --import dir/` takes an
existing COLMAP model instead.

Try it with images whose true poses are known to the tests:

```
splat make-test-views views/
splat ingest views/ -o project/ --focal-px 500
splat sfm project/
splat view project/sparse/0/preview.ply
```

## How rendering works

Each frame runs three GPU stages:

1. **Preprocess** (`shaders/preprocess.comp`). Each Gaussian is culled
   against the view, its 3D covariance is projected to a 2D conic (EWA
   splatting), and its view-dependent colour is evaluated from spherical
   harmonics. It also writes a depth key.
2. **Sort** (`shaders/radix_*.comp`). A stable 8-bit-per-pass GPU radix sort
   orders the splats back to front.
3. **Draw** (`shaders/splat.vert`, `splat.frag`). Each visible splat becomes
   one instanced screen-space quad. The quads use an indirect draw whose
   count comes from step 1, with premultiplied alpha blending.

The device needs Vulkan 1.3, one graphics+compute queue, subgroup arithmetic
and `VK_KHR_push_descriptor`. Every current AMD, NVIDIA and Intel driver
supports all of these, and so does Mesa (RADV, ANV, lavapipe).

## Running

Users need nothing but a GPU driver with Vulkan 1.3. The program loads the
Vulkan loader at runtime and links everything else statically.

```
splat make-test-scene scene.ply
splat view scene.ply
splat render scene.ply -o scene.png --yaw 45 --pitch 30
splat info
```

3DGS data normally uses the COLMAP convention, where "up" is -y. If a scene
appears upside down, pass `--up +y` (or `+z`/`-z`).

## Building

Requirements: CMake 3.28+, Ninja, git, and a C++23 compiler (GCC 14+,
Clang 18+ or Visual Studio 2022 17.10+). Video input uses the system's FFmpeg
6.1 or newer, found through pkg-config (Ubuntu: `libavformat-dev libavcodec-dev
libswscale-dev`); without it, `splat ingest` handles photos only. Everything else, including the
shader compiler, is fetched and built by vcpkg, which is a git submodule.

```
git clone --recursive https://github.com/P-i-N/vibe-splat
cd vibe-splat
cmake --preset linux-release      # or windows-release from a VS developer prompt
cmake --build --preset linux-release
ctest --preset linux-release
```

The first configure builds the dependencies and takes a few minutes; later
builds reuse vcpkg's binary cache.

On Linux you can skip vcpkg for a quick local build if your distribution has
the packages (Ubuntu 24.04: `libvulkan-dev libvulkan-volk-dev
libvulkan-memory-allocator-dev glslang-tools catch2 libglm-dev libstb-dev
nlohmann-json3-dev libeigen3-dev`). Without SDL3 the window viewer is left out, and without
FFmpeg's development packages video input is, but everything else still builds:

```
cmake -G Ninja -B build/sys -DCMAKE_BUILD_TYPE=Debug
cmake --build build/sys
```

Tests need a Vulkan device. Without a GPU, Mesa's lavapipe
(`mesa-vulkan-drivers`) provides a software one.

## Layout

```
shaders/        GLSL shaders, compiled to SPIR-V and embedded at build time
src/core/       Vulkan context, buffers, images, compute pipelines, radix sort
src/splat/      Gaussian cloud data, .ply reading and writing, test scene
src/render/     Camera, splat renderer, offscreen rendering and PNG output
src/viewer/     SDL3 window, swapchain and input (built when SDL3 is found)
src/cli/        The `splat` command-line program
tests/          Catch2 unit tests
cmake/          Shader compilation and compiler settings
external/vcpkg  Package manager (submodule)
```

## Conventions

- C++23 baseline. C++26 features only behind feature-test macros once GCC,
  Clang and MSVC all ship them.
- Dependencies must be open source with permissive or LGPL licences.
- Every GPU kernel gets a CPU reference and a test.

## Licence

[0BSD](LICENSE): use it for anything, no conditions, no attribution needed.
Third-party libraries keep their own licences; release packages include
their notices (FFmpeg, for example, is LGPL and ships as shared libraries).
