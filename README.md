# vibe-splat

A Gaussian splat generator and viewer that runs on consumer GPUs using only
Vulkan. No CUDA or other vendor libraries, so it works on AMD and NVIDIA
alike. The main target is the Steam Machine (SteamOS); Windows is supported
too.

Input is a set of photos or a video, plus optional camera information read
from EXIF or given by hand. Output is a standard 3D Gaussian splatting `.ply`.

## Status

Milestone M1 (viewer) is done:

- `splat view scene.ply` opens a window and renders a 3DGS `.ply` in real time.
  Controls: left-drag orbits, right- or middle-drag pans, the wheel zooms,
  W/A/S/D/Q/E move, R resets the view and Esc quits. On a gamepad (Steam
  Machine controller, Xbox or PlayStation pads), the left stick moves, the
  right stick orbits, the triggers zoom and A resets the view.
- `splat render scene.ply -o out.png` renders one image without a window.
- `splat make-test-scene out.ply` writes a synthetic scene, so you can try
  everything without a trained splat.
- `splat info` lists Vulkan devices; `splat selftest` checks the GPU works.
- `ingest`, `sfm` and `train` are placeholders for later milestones.

Any `.ply` in the standard 3DGS layout should work, with SH degree 0 to 3.
That is the layout written by the original implementation and most trainers.

| Milestone | Scope |
| --- | --- |
| M0 Foundation | CMake + vcpkg, Vulkan core, CI on Linux and Windows |
| M1 Viewer | Load `.ply` splats, GPU sort, render, window and controls |
| M2 Ingest | Photos, video frames, EXIF and manual intrinsics |
| M3 Camera poses | COLMAP (CPU only) linked in, sparse model import |
| M4 Trainer core | Vulkan forward, loss, backward, Adam |
| M5 Full training | Densification, SH degree 3, 8 GB VRAM budget |
| M6 Polish | GUI, live preview, gamepad, AppImage and Windows zip |

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
Clang 18+ or Visual Studio 2022 17.10+). Everything else, including the
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
libvulkan-memory-allocator-dev glslang-tools catch2 libglm-dev libstb-dev`).
Without SDL3 the window viewer is left out, but everything else, including
`splat render`, still builds:

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
