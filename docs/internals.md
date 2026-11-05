# How it works

The algorithms behind each stage, for the curious and for contributors. For
using the program, see the [user guide](guide.md).

## Camera poses (structure from motion)

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

## Training

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
[`src/train/reference.cpp`](../src/train/reference.cpp) has every kernel again on the CPU: the tests check it
against finite differences, and the GPU against it.

`--holdout 8` keeps the first and every 8th view out of training, the split 3DGS
uses, and reports PSNR on those, which says more about quality than PSNR on
images it has seen.

## Rendering

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
