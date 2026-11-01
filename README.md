# vibe-splat

A Gaussian splat generator and viewer that runs on consumer GPUs using only
Vulkan. No CUDA or other vendor libraries, so it works on AMD and NVIDIA
alike. The main target is the Steam Machine (SteamOS); Windows is supported
too.

Input is a set of photos or a video, plus optional camera information read
from EXIF or given by hand. Output is a standard 3D Gaussian splatting `.ply`.

## Status

Early skeleton (milestone M0). What works today:

- `splat info` lists Vulkan devices and what they support.
- `splat selftest` runs a compute kernel on the GPU and checks the result.
- `ingest`, `sfm`, `train` and `view` are placeholders for later milestones.

| Milestone | Scope |
| --- | --- |
| M0 Foundation | CMake + vcpkg, Vulkan core, CI on Linux and Windows |
| M1 Viewer | Load existing `.ply` splats, GPU sort, render |
| M2 Ingest | Photos, video frames, EXIF and manual intrinsics |
| M3 Camera poses | COLMAP (CPU only) linked in, sparse model import |
| M4 Trainer core | Vulkan forward, loss, backward, Adam |
| M5 Full training | Densification, SH degree 3, 8 GB VRAM budget |
| M6 Polish | GUI, live preview, gamepad, AppImage and Windows zip |

## Running

Users need nothing but a GPU driver with Vulkan 1.3. The program loads the
Vulkan loader at runtime and links everything else statically.

```
splat info
splat selftest
splat --help
```

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
libvulkan-memory-allocator-dev glslang-tools catch2`):

```
cmake -G Ninja -B build/sys -DCMAKE_BUILD_TYPE=Debug
cmake --build build/sys
```

Tests need a Vulkan device. Without a GPU, Mesa's lavapipe
(`mesa-vulkan-drivers`) provides a software one.

## Layout

```
shaders/        GLSL compute shaders, compiled to SPIR-V and embedded at build time
src/core/       Vulkan context, buffers, compute pipelines, self-test
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
