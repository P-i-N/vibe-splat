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
  `project/splat.ply` and `project/splat.cgc` (see below), plus
  `project/splat.html`: the scene in a single-page browser viewer, to open
  from disk or send to anyone.

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
- **Bottom left**: training. Set the iterations, image size, holdout and SH
  degree, then watch progress, loss, PSNR, Gaussians and VRAM, with pause,
  save and finish buttons. It saves both `splat.ply` and compressed `splat.cgc`;
  at the end the `.cgc` is compressed on another thread, as that takes about a
  minute per million Gaussians. The Gaussians are drawn straight from the trainer's
  GPU buffers, so live preview costs no copies; a slider sets how much of each
  frame goes to training.
- **Bottom right**: view settings: splats or their centres, splat scale, SH
  degree, field of view, move speed, background, grid, axes, cameras and
  points.

| Input | Does |
| --- | --- |
| left drag / Ctrl+left drag | look around / orbit the target |
| right or middle drag / wheel | pan / zoom (Ctrl+wheel: field of view) |
| W A S D, Q E | move (Shift: 4x faster; "move speed" slider scales both) |
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
- **Videos** (needs a build with FFmpeg, or `ffmpeg` 5.1+ on PATH) keep the sharpest frame of each interval
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
   by gradient histograms (RootSIFT), the 8192 with the most contrast per
   image.
2. **Matching.** Brute-force mutual nearest neighbours with Lowe's ratio test.
   On the GPU, descriptors are quantised to 8 bits and compared with packed
   integer dot products in a compute shader (`--cpu` keeps it on the CPU,
   where Eigen matrix products do the same job more slowly). Every pair for up
   to 150 photos; for videos and larger sets, neighbours in input order plus a
   few at growing distances (`--matcher` overrides this).
3. **Verification.** RANSAC around the five-point essential matrix solver,
   using ingest's intrinsics. Pairs that do not fit a rigid camera motion go.
4. **Tracks.** Matches are chained into tracks that follow one 3D point. A
   match that would give a track two features of one image, which a repeated
   pattern causes, is left out; the best-matched pairs go first.
5. **Start.** The best-matched pair with a wide enough baseline is triangulated.
6. **Growth.** The image seeing the most known points is registered with P3P
   RANSAC, what it newly sees is triangulated, and the model is bundle adjusted
   as it grows. Bundle adjustment is Levenberg-Marquardt with a Huber loss and
   the points eliminated through the Schur complement.
7. **Finish.** A final bundle adjustment refines focal length and radial
   distortion (a focal length given by hand stays fixed), and points missing by
   more than 1.5 pixels are dropped.

The pixel thresholds above are for images 1600 pixels across, ingest's default,
and grow in proportion for larger ones, whose features are located less
precisely in pixels. Held fixed, they had rejected most correct matches in
3840 pixel photos and left most of those photos without a pose.

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
- **Starting points.** Besides the sparse model, up to 50000 more points come
  from matching patches along epipolar lines between the photos, checked in a
  third one (`--dense-init` sets how many, 0 turns it off). The idea is from
  EDGS, without its neural matcher.
- **Coarse to fine.** Training starts on images at a quarter of their size
  until 15% of the way through, then half until 35%, made by averaging 2x2
  pixels, as the early iterations only need the broad shapes. The Gaussians may
  grow to `--max-growth` times the starting points, by default 4 for every 7000
  iterations: at 7000 that keeps the bicycle scene above 100 it/s on a laptop,
  where more adds little (24.1 dB held out at 400k, 24.2 at 600k). Held to 4
  over 30000 iterations, a guitar scene hit the cap after 2000 and gained no
  detail after that; with room to grow it reached 408k Gaussians and 2 dB more
  on its training views. `--adaptive-schedule`
  instead switches sizes where the photos' spectra fall off and lets the count
  grow with the size, as in DashGaussian: faster, but 0.7 dB worse on that
  scene at 7000 iterations (23.4 dB), and the default until now.
  `--full-resolution` stays at full size throughout.
- **Exposure.** With `--exposure`, each training view also learns a 3x4
  colour transform that the loss compares its photo through, so that
  changes in exposure and white balance between photos, as auto exposure
  makes in video, do not end up in the Gaussians. The output stays neutral.
  Held-out views are also scored with an exposure fitted on one half of each
  image and scored on the other, the only fair way to compare with and without.
- **Optimiser.** Adam, with the learning rates of the 3DGS paper. It only
  updates Gaussians the current view sees: the rest keep their values and
  moments, which also spares most of its memory traffic. `--full-adam`
  updates them all, about a tenth slower. It used to be 0.5 dB better on
  held-out views, but with scored densification it made no clear difference.
- **Density.** Every 100 iterations in the first half of training, 10 random
  views are scored, as in FastGS: for each Gaussian, how many pixels it reaches
  whose error is above a tenth of that view's worst. Only those reaching more
  than 5 per view may multiply (`--score-views 0` turns this off). Of those, a
  small one whose screen position keeps wanting to move is cloned. A large one
  is split in two on the same gradient without the cancelling between pixels
  pulling opposite ways, as in AbsGS. Nearly transparent ones are removed.
  Opacity is capped every 3000 iterations, so that those no longer needed fade away.
- **Pruning.** With `--prune <share>`, at the end of densification and again
  halfway through the rest, every training view is rendered once without a
  step, adding up each Gaussian's blending weight over all their pixels, and
  that share of those adding least is removed. Off by default: the scoring above
  already keeps the count down, and at 5000 iterations more costs real quality.
- **Options that did not help on the test scenes, and are off.** `--second-order`
  (3DGS^2-TR, after densification), `--explore` (random splits and new
  Gaussians, from the blur trap paper), `--threshold-rise` (Improving ADC),
  `--significance-prune` and `--point-extent`. From 3DGS-MCMC, measured on the
  bicycle scene at 7000 iterations against 24.1 dB held out at 107 it/s:
  `--opacity-reg 0.01 --scale-reg 0.01` (22.5-23.0 dB: with the opacity reset
  the penalties thin the cloud to 270k), `--densify mcmc` (relocation in place
  of clone, split and prune: 23.2 dB, 87 it/s as the relocated Gaussians are
  opaque and large), `--position-noise 5e5` on top (23.3 dB), and all of them
  together (23.7 dB, 98 it/s). Its gains are reported at 30000 iterations; at
  7000 the relocation has not converged. Mip-Splatting's filters,
  `--mip-filter` (both), `--mip-filter-2d` (a 0.1 pixel dilation with the
  opacity scaled to match) and `--mip-filter-3d` (each Gaussian widened to its
  smallest footprint in the photos, baked into saved scenes): 24.0-24.1 dB, the
  same as without, and the 3D filter costs a tenth of the speed (96 it/s).
  From Improving Densification in 3DGS: `--edge-split` (each candidate splits
  with probability score / the round's best, into two children along its
  longest axis, in place of cloning and splitting) grows only to 86k Gaussians
  and reaches 22.6 dB at 170-190 it/s, and `--recovery-prune 0.2` (the least
  opaque fifth removed 300 iterations after each opacity reset) 24.15-24.2 dB
  at 104-119 it/s, within noise of the default. `--sh-dropout 0.2` (Dropping
  Anchor and Spherical Harmonics: a Gaussian keeps only its low SH degrees on
  a fifth of the iterations): 24.1 dB, the same as without.
- **Poses.** `--refine-poses` refines the training cameras along with the
  Gaussians. It helps when poses are a little off and costs a little when they
  are good, so it is off by default.
- **View dependence.** SH starts at degree 0 and gains a degree every 1000
  iterations, up to `--sh-degree`, 2 by default. Degree 3, as in 3DGS, adds
  finer view-dependent colour; on bicycle at 15000 iterations it trained 8%
  slower and its final `.cgc` took 3x longer to compress.
- **Memory.** A hard budget, by default 90% of the GPU's free memory
  (`--vram-budget` overrides it). At the limit, the Gaussians with the highest
  gradients densify first, and a view that still does not fit gives up the
  largest ones. At SH degree 2 a Gaussian takes about 420 bytes (540 at
  degree 3), plus 16 bytes
  per tile it covers: values are floats, but SH above degree 0 half floats,
  Adam's moments are bfloat16, and gradients are never stored, as each step is
  taken where they are found.

Images are undistorted when loaded, since the renderer is a pinhole camera.
`src/train/reference.cpp` has every kernel again on the CPU: the tests check it
against finite differences, and the GPU against it.

`--holdout 8` keeps the first and every 8th view out of training, the split 3DGS
uses, and reports PSNR on those, which says more about quality than PSNR on
images it has seen.

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
over the network, for example to a browser viewer. `splat train` writes one
next to its `.ply`, `splat make-test-scene scene.cgc` writes it directly, and `splat convert
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

### Web viewer

[web/viewer.html](web/viewer.html) is a single-file WebGL 2 viewer for `.cgc`
and `.ply` scenes. Open it straight from disk, no web server needed, then use
**Open** or drop a file on the page. **Save as HTML** writes a copy of the
viewer with the scene embedded in it (base64), which opens anywhere as one
standalone file. Embed a `.cgc` rather than a `.ply`: it is about 28x smaller,
and browsers cannot hold a page of several hundred megabytes.

`splat train` writes such a page next to its output, `<out>.html`, with
the final `.cgc` inside; `--no-html` skips it. The program carries
`web/viewer.html` built in, so the page always matches the build.

Controls follow the native viewer: drag to look around, Ctrl+drag to orbit the
target, right drag (or Shift+drag) to pan, wheel to zoom, WASD/QE to move
(Shift: faster), R to reset the view, U to change the up axis (`-y` by
default). Two-finger touch pans and zooms. The page carries its own copy of
`cgc.js`, because pages opened from `file://` cannot import modules;
`tests/cgc_web_check.mjs` fails if the two copies drift apart.

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

`splat train` saves its output, checkpoints included, with the floor level at
y = 0 under the scene's centre, -y up. Training itself stays in the cameras'
frame; only the saved copy is turned (positions, orientations and the
view-dependent colour). Up comes from the cameras, whose x axes lie level when
held by people, and the floor is the plane holding the most sparse points,
by RANSAC, within 20 degrees of that, favouring level ones and with the
cameras above it. If no such plane stands out, the scene is only levelled by
the cameras' up. Pass `--no-align` to save in the cameras' frame.

The floor is found once, before training, from the camera poses and sparse
points alone, so every checkpoint is saved the same way and the viewer can
use it too: it shows a project's cameras, points and the Gaussians being
trained turned the same way, with the grid on the floor, so the view while
training matches the saved file.

## Building

Requirements: CMake 3.28+, Ninja, git, and a C++23 compiler (GCC 14+,
Clang 18+ or Visual Studio 2022 17.10+). Video input uses the system's FFmpeg
6.1 or newer, found through pkg-config (Ubuntu: `libavformat-dev libavcodec-dev
libswscale-dev`); without it, `splat ingest` runs the `ffmpeg` program from PATH
(5.1 or newer) for videos, or handles photos only. Everything else, including the
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
