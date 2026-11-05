# User guide

Everything `splat` can do, command by command. For what happens inside, see
[How it works](internals.md); for the short tour, the [README](../README.md).

## Commands

- `splat view scene.ply` opens a window and renders a 3DGS `.ply` (or `.cgc`) in real time.
  `splat view project/` also shows the camera poses and sparse points, and can
  train the project while you watch ([The viewer](#the-viewer)).
- `splat train project/ --view` trains in that window from the start.
- `splat render scene.ply -o out.png` renders one image without a window.
- `splat make-test-scene out.ply` writes a synthetic scene, so you can try
  everything without a trained splat.
- `splat info` lists Vulkan devices; `splat selftest` checks the GPU works.
- `splat ingest photos/ clip.mp4 -o project/` imports photos and videos into a
  project folder ([Importing photos and videos](#importing-photos-and-videos)).
- `splat sfm project/` estimates camera poses and a sparse point cloud
  ([how](internals.md#camera-poses-structure-from-motion)).
- `splat make-test-views dir/` renders images with known poses to try it on.
- `splat train project/` trains Gaussians from those poses and writes
  `project/splat.ply` and `project/splat.cgc` ([how](internals.md#training),
  [.cgc](#compressed-output-cgc)), plus
  `project/splat.html`: the scene in a single-page browser viewer, to open
  from disk or send to anyone.

Any `.ply` in the standard 3DGS layout should work, with SH degree 0 to 3.
That is the layout written by the original implementation and most trainers.

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

## Importing photos and videos

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

[web/cgc.js](../web/cgc.js) is a dependency-free decoder for browsers, LZMA included.
The byte layout and the measurements behind it are in
[the CGC v2 spec](superpowers/specs/2026-11-02-cgc-v2-format.md).

### Web viewer

[web/viewer.html](../web/viewer.html) is a single-file WebGL 2 viewer for `.cgc`
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
