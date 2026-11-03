# Newer training techniques

Date: 2026-11-03
Status: done

## Intent

The user asked for recent 3DGS techniques that make training faster or more
accurate to be implemented, with no run longer than 5000 iterations.

## How it was measured

Every run: 5000 iterations, `--holdout 8`, on an AMD Steam Machine GPU (NAVI33).

- **truck**: 96 views at 1024x768, rendered from a public trained 3DGS scene,
  with poses and sparse points from `splat sfm`. Not real photographs.
- **a** (60 views at 1600x1200) and **c** (40 views at 640x480): synthetic scenes.
- **truck noisy**: truck with 0.3 degrees and 0.03 units of noise on the
  training poses only.

Repeated runs of one setup differ by about 0.1 dB, so smaller differences mean
nothing here. Nothing was measured at 30000 iterations, the default length.

## Results with the new defaults

| Scene | Before: held-out PSNR, time, Gaussians | After |
| --- | --- | --- |
| truck | 21.3 dB, 48 s, 81k | 23.4 dB, 27 s + 2.6 s matching, 196k |
| a | 30.0 dB, 56 s, 13k | 36.7 dB, 28 s + 0.9 s matching, 76k |
| c | 28.4 dB, 17 s, 13k | 32.7 dB, 16 s + 0.6 s matching, 62k |

Peak VRAM on truck went from 1.05 GiB to 0.30 GiB.

## What each one did on truck

| Technique | Against | Held-out PSNR | Time | Kept |
| --- | --- | --- | --- | --- |
| FastGS scoring with AbsGS splitting, no pruning | old defaults | 21.3 -> 22.9 dB | 48 -> 37 s | on |
| DashGaussian schedule and growth limit | that | 22.9 -> 23.0 dB | 37 -> 20 s | on |
| Dense starting points (EDGS, patch matching), 50000 | that | 22.9 -> 23.4 dB | 20 -> 27 + 2.6 s | on |
| `--refine-poses`, noisy poses | scoring | 16.9 -> 17.4 dB | same | flag |
| `--refine-poses`, good poses | 200000 dense points | 23.8 -> 23.5 dB | same | flag |
| `--second-order` | 200000 dense points | 23.8 -> 23.3 dB | same | flag |
| `--explore 20` | scoring | no clear change | same | flag |
| `--threshold-rise 4` | scoring | no clear change | same | flag |
| `--significance-prune` | new defaults | no clear change | same | flag |
| `--point-extent` | new defaults | no clear change | same | flag |
| `--exposure`, even photos | new defaults | no clear change | same | flag |

Papers report gains for the flags that did nothing here, on other scenes and at
30000 iterations, so this does not show they are wrong in general.

## How many dense starting points

More starting points make every iteration slower, as there are more Gaussians
to draw. At 5000 iterations on truck:

| `--dense-init` | Held-out PSNR | Training time | Gaussians |
| --- | --- | --- | --- |
| 0 | 22.9 dB | 20 s | 138k |
| 25000 | 23.4 dB | 24 s | 182k |
| 50000 | 23.5 dB | 27 s | 197k |
| 100000 | 23.4 dB | 33 s | 236k |
| 200000 | 23.7 dB | 42 s | 318k |

The default is 50000: 200000 bought about 0.2 dB more on truck for half as much
time again. Scene a still gained 1.2 dB from 200000 (36.7 to 37.9 dB).

Against the same total time, by giving the dense start fewer iterations, it is
even on truck and well ahead on the synthetic scenes:

| Scene | Without, 5000 iterations | With 50000, fewer iterations |
| --- | --- | --- |
| truck | 22.9 dB in 20 s | 22.9 dB in 18 s (3000) |
| a | 31.4 dB in 20 s | 35.8 dB in 19 s (3000) |
| c | 29.2 dB in 7 s | 31.8 dB in 7 s (2000) |

Matching stops once it has twice the points it needs, taking the reference
views in an order that spreads them over the scene, and searches each line a
pixel at a time before refining to half a pixel around the best match. That
took truck's matching from 7.7 s to 2.6 s with no change beyond noise.

## A bug this found

A prefetched image could leave a staging buffer too small for the next image
size, and the next larger image overflowed it into the memory the prefetch
thread writes to. Training then ran against a mix of two photos until the next
size change, costing up to 10 dB. It came with coarse-to-fine training, and
scoring made it likely. The prefetch itself was almost never used, as each
iteration replaced it before reading it. Both are fixed.

## Where the implementation differs from the papers

- FastGS: its pruning before the end of densification only applies with
  `--significance-prune`, and the rule it uses after that is left out. Pixels are counted where the backward
  pass visits them, so not behind a pixel's early stop. The minimum of its
  error range is taken as 0.
- DashGaussian: the image sizes stay at 1/4, 1/2 and full, as switch points only.
- EDGS: patch matching, not a neural matcher, and densification stays on, as
  training with `--no-densify` was worse (22.0 dB against 23.6 dB).
- 3DGS^2-TR: used only after densification, with its estimator in image space,
  and rotation stays with Adam, as the paper leaves a constant in its bound open.
- The blur trap paper does not give a seed's size, opacity or colour, or what
  "large" means. These are a typical Gaussian's size, 0.1, a random Gaussian's
  colour, and the larger half.
