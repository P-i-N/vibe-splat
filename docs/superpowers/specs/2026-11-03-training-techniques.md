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
| truck | 21.3 dB, 48 s, 81k | 23.8 dB, 42 s + 8 s matching, 317k |
| a | 30.0 dB, 56 s, 13k | 37.9 dB, 42 s, 165k |
| c | 28.4 dB, 17 s, 13k | 32.9 dB, 20 s, 81k |

Peak VRAM on truck went from 1.05 GiB to 0.42 GiB.

## What each one did on truck

| Technique | Against | Held-out PSNR | Time | Kept |
| --- | --- | --- | --- | --- |
| FastGS scoring with AbsGS splitting, no pruning | old defaults | 21.3 -> 22.9 dB | 48 -> 37 s | on |
| DashGaussian schedule and growth limit | that | 22.9 -> 23.0 dB | 37 -> 20 s | on |
| Dense starting points (EDGS, patch matching) | that | 22.8 -> 23.8 dB | 20 -> 42 s | on |
| `--refine-poses`, noisy poses | scoring | 16.9 -> 17.4 dB | same | flag |
| `--refine-poses`, good poses | new defaults | 23.8 -> 23.5 dB | same | flag |
| `--second-order` | new defaults | 23.8 -> 23.3 dB | same | flag |
| `--explore 20` | scoring | no clear change | same | flag |
| `--threshold-rise 4` | scoring | no clear change | same | flag |
| `--significance-prune` | new defaults | no clear change | same | flag |
| `--point-extent` | new defaults | no clear change | same | flag |
| `--exposure`, even photos | new defaults | no clear change | same | flag |

Papers report gains for the flags that did nothing here, on other scenes and at
30000 iterations, so this does not show they are wrong in general.

## A bug this found

A prefetched image could leave a staging buffer too small for the next image
size, and the next larger image overflowed it into the memory the prefetch
thread writes to. Training then ran against a mix of two photos until the next
size change, costing up to 10 dB. It came with coarse-to-fine training, and
scoring made it likely. The prefetch itself was almost never used, as each
iteration replaced it before reading it. Both are fixed.

## Where the implementation differs from the papers

- FastGS: only the pruning before the end of densification was considered, and
  the rule it uses after that is left out. Pixels are counted where the backward
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
