# vibe-splat

A Gaussian splat generator and viewer that runs on consumer GPUs using only
Vulkan. No CUDA or other vendor libraries, so it works on AMD and NVIDIA
alike. The main target is the Steam Machine (SteamOS); Windows is supported
too.

Input is a set of photos or a video, plus optional camera information read
from EXIF or given by hand. Output is a standard 3D Gaussian splatting `.ply`.

## Status

All milestones, M1 (viewer) to M6 (polish), are done:

- `splat view scene.ply` opens a window and renders a 3DGS `.ply` (or `.cgc`) in real time.
  `splat view project/` also shows the camera poses and sparse points, and can
  train the project while you watch (see below).
- `splat train project/ --view` trains in that window from the start.
- `splat render scene.ply -o out.png` renders one image without a window.
- `splat make-test-scene out.ply` writes a synthetic scene, so you can try
  everything without a trained splat.
- `splat info` lists Vulkan devices; `splat selftest` checks the GPU works.
- `splat ingest photos/ clip.mp4 -o project/` imports photos and videos into a
  project folder (see below).
- `splat sfm project/` estimates camera poses and a sparse point cloud (see
  below).
- `splat make-test-views dir/` renders images with known poses to try it on.
- `splat train project/` trains Gaussians from those poses and writes
  `project/splat.ply` (see below).

Any `.ply` in the standard 3DGS layout should work, with SH degree 0 to 3.
That is the layout written by the original implementation and most trainers.

| Milestone | Scope |
| --- | --- |
| M0 Foundation | CMake + vcpkg, Vulkan core, CI on Linux and Windows |
| M1 Viewer | Load `.ply` splats, GPU sort, render, window and controls |
| M2 Ingest | Photos, video frames, EXIF and manual intrinsics |
| M3 Camera poses | Our own structure from motion (GPU matching, CPU otherwise), COLMAP model import |
| M4 Trainer core | Vulkan forward, loss, backward, Adam |
| M5 Full training | Densification, SH degree 3, 8 GB VRAM budget |
| M6 Polish | GUI, live preview, gamepad, AppImage and Windows zip |

## The viewer

`splat view` takes a `.ply` or `.cgc`, a project folder, or nothing (then drop
files or folders on the window). A project folder opens its `splat.ply` or
`splat.cgc`, whichever is newer. With a project it draws each posed photo as a camera
frustum (blue for training, orange for held out, green for the one just trained
on) and can overlay the sparse points. Everything else sits in the corners:

- **Top left**: the scene, splat counts, frame rate and GPU. When looking
  through a photo, its name and a thumbnail.
- **Top right**: keys and mouse (H hides them).
- **Bottom left**: training. Set the iterations, image size, holdout, SH
  degree and whether to save `.ply` or compressed `.cgc`, then watch progress, loss, PSNR, Gaussians and VRAM, with pause,
  save and finish buttons. The Gaussians are drawn straight from the trainer's
  GPU buffers, so live preview costs no copies; a slider sets how much of each
  frame goes to training.
- **Bottom right**: view settings: splats or their centres, splat scale, SH
  degree, field of view, background, grid, axes, cameras and points.

| Input | Does |
| --- | --- |
| left drag / Ctrl+left drag | look around / orbit the target |
| right or middle drag / wheel | pan / zoom (Ctrl+wheel: field of view) |
| W A S D, Q E | move (Shift: faster) |
| F, R, T | frame the scene, reset the view, turntable |
| 1, 2 | splats, Gaussian centres as dots |
| C, P, G | cameras, sparse points, grid |
| [ ], double-click a camera | look through a photo; X (held) shows the photo itself |
| Space, Ctrl+S | pause training, save what it has so far |
| F12, Tab, H | screenshot (into the project or current folder), hide UI, help |
| Esc | leave the photo, or quit |

On a gamepad (Steam Machine controller, Xbox or PlayStation pads), the left
stick moves, the right stick orbits, the triggers zoom, A resets the view, B
leaves a photo, LB/RB step through photos, X switches to centres, Y hides the UI
and Start pauses training. Closing the window while training saves what it has.

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
scratch, needing only Eigen and, for matching, Vulkan:

1. **Features.** SIFT: extrema of a difference-of-Gaussians pyramid, described
   by gradient histograms (RootSIFT).
2. **Matching.** Brute-force mutual nearest neighbours with Lowe's ratio test.
   On the GPU, descriptors are quantised to 8 bits and compared with packed
   integer dot products in a compute shader (`--cpu` keeps it on the CPU,
   where Eigen matrix products do the same job more slowly). Every pair for up
   to 150 photos; for videos and larger sets, neighbours in input order plus a
   few at growing distances (`--matcher` overrides this).
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

## How training works

`splat train` is a differentiable renderer in Vulkan compute shaders. Nothing
does automatic differentiation: each pass has a hand-derived counterpart.

- **Forward.** Parameters become screen-space splats, which are sorted by depth.
  Each splat then adds one entry per 16x16 pixel tile its ellipse reaches (not
  just every tile of its bounding box), and a stable
  sort by tile leaves every tile with its own depth-sorted list. One workgroup
  per tile, one thread per pixel: the tile's splats are staged through shared
  memory and every thread blends its own pixel in registers, front to back.
  A pixel stops once its transmittance falls below 1e-4, and a tile once all
  its pixels have, so splats hidden behind opaque surfaces cost nothing. Tiles
  do not share pixels, so they composite in parallel with no blending hardware.
- **Loss.** 0.8 L1 + 0.2 D-SSIM, and its derivative at every pixel. The 11x11
  SSIM window is separable: a horizontal pass into shared memory and a
  vertical pass out of it, per tile.
- **Backward.** At a pixel `C = F + T (alpha c + (1 - alpha) B)`, with `B` what
  lies behind a splat and `T` the transmittance in front of it. So
  `dC/dalpha = T (c - B)` and `dC/dc = T alpha`, worked out back to front
  from where the forward pass stopped. Each pixel's share is reduced
  over its subgroup with one butterfly for all nine values, then added per
  splat with float atomics, or fixed-point integer atomics on GPUs without
  them. A last pass applies the chain rule down to position, scale, rotation,
  opacity and SH.
- **Overlap.** One submission per iteration (forward, loss, backward, Adam),
  and the next view's image is copied to staging memory on a worker thread
  while the GPU works on the current one.
- **Coarse to fine.** The first 15% of iterations train on images at a
  quarter of their size and the next 20% at half, made by averaging 2x2
  pixels, as the early iterations only need the broad shapes
  (`--full-resolution` turns this off).
- **Exposure.** With `--exposure`, each training view also learns a 3x4
  colour transform that the loss compares its photo through, so that
  changes in exposure and white balance between photos, as auto exposure
  makes in video, do not end up in the Gaussians. The output stays neutral.
- **Optimiser.** Adam, with the learning rates of the 3DGS paper. It only
  updates Gaussians the current view sees: the rest keep their values and
  moments, which also spares most of its memory traffic. `--full-adam`
  updates them all, which on the test set is a fifth slower and 0.5 dB
  better on held-out views.
- **Density.** Every 100 iterations in the first half of training, a Gaussian
  whose screen position keeps wanting to move is cloned if small and split in
  two if large, and nearly transparent ones are removed. Opacity is capped every
  3000 iterations, so that those no longer needed fade away.
- **Pruning.** At the end of densification and again halfway through the
  rest, every training view is rendered once without a
  step, adding up each Gaussian's blending weight over all their pixels. The
  70% adding least are removed (`--prune` sets the share, 0 turns it off),
  and the rest keep training to fill in. On the 194-photo test set at 10000
  iterations this ends with 193k Gaussians instead of 2.2M, for 0.1 dB less
  PSNR on held-out views, and the run takes 36% less time.
- **View dependence.** SH starts at degree 0 and gains a degree every 1000
  iterations, up to `--sh-degree`.
- **Memory.** A hard budget, by default 90% of the GPU's free memory
  (`--vram-budget` overrides it). At the limit, the Gaussians with the highest
  gradients densify first, and a view that still does not fit gives up the
  largest ones. At SH degree 3 a Gaussian takes about 1.1 KiB, plus 16 bytes per
  tile it covers.

Images are undistorted when loaded, since the renderer is a pinhole camera.
`src/train/reference.cpp` has every kernel again on the CPU: the tests check it
against finite differences, and the GPU against it.

`--holdout 8` keeps every 8th view out of training and reports PSNR on those,
which says more about quality than PSNR on images it has seen.

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

## Compressed output (.cgc)

Wherever `splat` writes or reads a scene, a `.cgc` extension selects CGC
(Compressed Gaussian Cloud) instead of `.ply`. It is meant for sending scenes
over the network, for example to a browser viewer. `splat train -o scene.cgc`
and `splat make-test-scene scene.cgc` write it directly, and `splat convert
scene.ply scene.cgc` converts an existing file. `view` and `render` open it
like a `.ply`.

CGC is lossy. Gaussians are split into three layers by importance, and fainter,
smaller ones are stored more coarsely. Within a layer they are sorted along a
Morton curve on one grid, axes are sorted so rotations and scales cost less, higher
SH bands share one codebook, and each byte column is its own LZMA stream. Training
itself always works at full precision. On public 3DGS scenes this is about 28x
smaller than the PLY, at 40 to 43 dB against it; `splat convert --small` gives about
31x at 39 to 41 dB, with weaker close-ups. Scenes with very few Gaussians and strongly
view-dependent colour lose more. Older `.cgc` files still load.

[web/cgc.js](web/cgc.js) is a dependency-free decoder for browsers, LZMA included.
The byte layout and the measurements behind it are in
`docs/superpowers/specs/2026-11-02-cgc-v2-format.md`.

### Pruning floaters

Trained scenes often contain a few "floaters": huge, nearly transparent
Gaussians far outside the scene that render as long streaks from any viewpoint
the training cameras did not cover. `splat convert` can drop them on the way
to either format, with both tests off unless asked for:

```
splat convert scene.ply scene.cgc --prune-sigma 1 --prune-radius 4
```

`--prune-sigma <units>` removes every Gaussian whose largest axis (one standard
deviation) is longer than that many world units. `--prune-radius <factor>`
removes every Gaussian farther from the scene centre than that many times the
radius holding 90% of all Gaussians; the centre is the per-axis median, so the
floaters themselves do not move it. Either option alone enables its test, and
`convert` prints how many Gaussians were removed. Start around `--prune-sigma
1` for a room-sized scene in metres and look at a render; opaque, well-placed
Gaussians are rarely that large.

## Running

Users need nothing but a GPU driver with Vulkan 1.3. The program loads the
Vulkan loader at runtime and links everything else statically.

CI builds the release packages: `splat-x86_64.AppImage` for Linux and SteamOS
(`packaging/make-appimage.sh`) and `splat-windows-x64.zip`. Both include the
licences of the libraries inside. Started without arguments from a desktop
(double-clicking `splat.exe`, or the AppImage), splat opens an empty viewer;
dropping a file or folder on `splat.exe` opens it, and `splat <path>` is short
for `splat view <path>`.

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
nlohmann-json3-dev libeigen3-dev`). Without SDL3 and Dear ImGui the window viewer is left out, and without
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
src/splat/      Gaussian cloud data, .ply and .cgc reading and writing, test scene
src/render/     Camera, splat renderer, offscreen rendering and PNG output
src/viewer/     SDL3 window, Dear ImGui interface, debug lines (built when SDL3 and ImGui are found)
src/cli/        The `splat` command-line program
packaging/      AppImage script, desktop entry, licence collection
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
