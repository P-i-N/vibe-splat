# Quality ladder for 7000-iteration training

Date: 2026-11-03
Status: done 2026-11-03; results at the end
Branch: claude/training-sota

## Intent

Raise the held-out PSNR that `splat train` reaches in 7000 iterations on a
laptop, without giving up its speed. The trainer already matches reference
3DGS: on the Mip-NeRF 360 bicycle scene at quarter resolution with the
official split it reaches 23.4 dB at 7000 iterations (gsplat 1.5.3 with the
paper's settings: 22.1 dB on the same machine, the published figure 23.7 dB),
with 365k Gaussians at about 140 iterations per second. Cross-rendering each
trainer's cloud with the other's rasteriser reproduces the PSNRs to 0.01 dB, so
the renderer is not where gains are. The gains are in what the literature added
since the paper: how Gaussians are added, moved and removed, how they are
filtered across resolutions, and how colour is regularised.

The owner wants both: more PSNR at 7000 iterations and training that stays
above 100 iterations per second.

## Goals

- Higher held-out PSNR on the measurement below, at 7000 iterations.
- Every technique is an option. A technique becomes a default only if it gains
  at least 0.2 dB over two runs each way and keeps the run at or above 100
  iterations per second.
- Techniques that lose stay in as options, off, with a line in the README's
  "Options that did not help" list, as the repository has done so far.

## Non-goals

- The 30000-iteration regime, millions of Gaussians, or quality that only
  shows there.
- Replacing the Gaussian kernel (Beta splatting) or the rasteriser.
- The CGC format, SfM or ingest.
- Backwards compatibility of saved scenes or of the viewer with other
  software: nobody else consumes them yet, and the owner has said they may
  change freely. The repository's own viewer is the only consumer and follows
  the trainer.

## Anti-goals

- A run below 100 iterations per second at 1237x822 on the owner's RTX 4070
  Laptop, whatever it gains. Lighter settings that bring it back above 100 are
  fine if they keep a gain.
- A change in what the existing defaults do that was not measured.
- Breaking the GPU-versus-CPU parity: every kernel has a CPU twin in
  `src/train/reference.cpp` and finite-difference tests in `tests/test_train.cpp`.

## Constraints

- Vulkan compute, C++23, hand-derived gradients. New per-Gaussian terms go
  into the chain-rule kernel `shaders/train_gradient.comp` and its CPU twin.
- Densification runs on the CPU every `--densify-interval` (100) iterations in
  `src/train/densify.cpp` over a downloaded cloud and `DensifyStats`, and goes
  back through `Trainer::apply_changes`, which keeps optimiser state for
  Gaussians with a `source`. Relocation and new split rules belong there.
- The VRAM budget turns into `DensifyOptions::maxGaussians`; with the adaptive
  schedule a growth budget (`GrowthBudget` in `src/train/schedule.hpp`) caps it
  further, and held back about 400k candidates per run on the measurement.
- Training is coarse to fine (quarter, half, full size); anything that depends
  on the pixel footprint must use the current level's camera.
- The GPU has no readily usable random numbers; per-element hash-based
  generators (for example PCG or xxhash of index and iteration) are the way to
  get per-iteration noise or dropout masks in a kernel.

## Measurement

The bicycle scene of Mip-NeRF 360, from the owner's `360_v2.zip`:

1. A folder with the 194 JPEGs of `bicycle/images_4` at its top level, named
   as in the archive, and `sparse/0` holding the archive's `images.bin` and
   `points3D.bin` unchanged and `cameras.bin` with the PINHOLE intrinsics
   scaled from 4946x3286 to the JPEGs' 1237x822 (fx, cx by 1237/4946, fy, cy by
   822/3286).
2. `splat train <folder> --iterations 7000 --holdout 8 --save-every 0 --no-align
   -o <scratch>.ply`, which trains on 169 views and reports PSNR on the 25
   held out (the first and every 8th, the split 3DGS uses).
3. Score: the "PSNR ... on held-out views" line. Speed: the "Finished in" line's
   iterations per second over the whole run. Baseline: 23.42 dB, 364887
   Gaussians, about 140 it/s. Identical runs differ by up to 0.3 dB, so a
   decision needs two runs each way, and gains under 0.2 dB are noise.
4. The hidden long test (`splat_tests "[.long]"`) and the regular `ctest` suite
   must still pass after each rung.

## Approaches considered

1. **A ladder of techniques, one rung at a time** (chosen). Each technique is
   an option, measured as above, defaulted on only if it wins. Breadth over
   depth; some rungs will be dropped after a minute of measurement.
2. **All in on 3DGS-MCMC.** The best-documented gain and a natural fit for the
   fixed budget, but a single bet that leaves the other levers untouched.
3. **Tuning only.** Sweep existing options; no new code; a few tenths at most,
   and earlier sessions already covered part of it.

The owner picked 1, with the cheapest of 3 folded into the first rung so that
the budget cap is settled before relocation is judged.

## The rungs, in order

Each rung ends with its measurement, its README lines and a commit. Each is
independent of the ones after it; a rung that loses is left as an option, off.

### Rung 0: settle the budget

Measure what decides how many Gaussians a run may have, since every later
rung works within that budget. `--fixed-schedule` removes the growth cap but
also moves the resolution switch points and the position-rate hold, so it
cannot say which part won. Add `--max-growth <factor>`: the growth cap as a
multiple of the starting count (the adaptive budget's final count is 5 times
it today; 0 means no cap), usable with either schedule. Then measure, twice
each: the default; the adaptive schedule with no cap and with factors 8 and
12; `--fixed-schedule`; and `--max-gaussians` at 400k and 600k. Pick the
default that wins, or keep the current one. The same cap semantics serve rung
1. (Builder's call, on the reviewer's point that the two effects were
confounded.)

### Rung 1: Markov chain Monte Carlo densification (3DGS-MCMC)

Reference: Kheradmand et al., NeurIPS 2024, and gsplat's implementation
(`gsplat/strategy/mcmc.py`, `gsplat/strategy/ops.py`, `gsplat/cuda/csrc/Relocation.cpp`
in gsplat 1.5.3), whose defaults are the starting values here.

Four parts, as separate options so each can be measured:

- **Opacity and scale penalties.** Add `reg * mean(sigmoid(opacity))` and
  `reg * mean(exp(scale))` to the loss, gsplat's defaults 0.01 each. They are
  gradients of one term per Gaussian: for the opacity logit x,
  `reg / N * sigmoid(x) * (1 - sigmoid(x))`; for a log scale s, `reg / N * exp(s)`.
  Added in the chain-rule kernel and its CPU twin, with N the current Gaussian
  count. Options `--opacity-reg`, `--scale-reg`, 0 off.
- **Relocation instead of pruning.** In `densify_cloud`, under a new mode,
  Gaussians at or below opacity 0.005 are not removed but moved onto live ones
  drawn with probability proportional to opacity. A live Gaussian that
  receives k dead ones becomes N = k + 1 copies at its position with the
  paper's distribution-preserving update, exactly as gsplat's
  `compute_relocation` does it: new opacity `o' = 1 - (1 - o)^(1/N)` and new
  scale `s' = s * o / D` with `D = sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k
  o'^(k+1) / sqrt(k+1)`. The copies and the source start with fresh optimiser
  moments (`source = -1`, as clones do today), which is also what gsplat does.
  (Builder's call: the formulas follow gsplat 1.5.3, not any paraphrase.)
- **Growth by sampling.** Under the same mode, each round grows the count by
  5% towards the cap by drawing live Gaussians with probability proportional
  to opacity and applying the same split-into-N update, instead of
  gradient-driven clone and split. The gradient thresholds and the FastGS
  scoring then play no part; the opacity reset is off in this mode (MCMC has
  none). The cap is the smallest of the VRAM-derived limit, `--max-gaussians`
  and `--max-growth` times the starting count; the `GrowthBudget`'s
  observation of asked-for growth is not used in this mode, since it would
  only ever see the 5%. MCMC is measured at a cap equal to the baseline's
  final count (about 365k) for a like-for-like comparison, and at rung 0's
  winning cap. (Builder's call.)
- **Position noise.** After each optimiser step, every Gaussian's position
  gets `noise_lr * lr_position * sigma(1 - opacity) * Sigma * n`, with n a
  standard normal 3-vector, Sigma the Gaussian's full covariance (not its
  square root: the paper and gsplat's `inject_noise_to_position` multiply by
  the covariance, and `noise_lr` is tuned for that), `sigma(x) = 1 / (1 +
  exp(-100 (x - 0.995)))` and gsplat's `noise_lr = 5e5`. A small kernel
  over the position group after Adam, with a hash-based generator seeded by
  Gaussian index and iteration. Option `--position-noise <noise_lr>`, 0 off.

Option `--densify mcmc` selects relocation and growth; the penalties and noise
are usable with either densification. Expected gain from the literature:
0.5 dB on Mip-NeRF 360 at 30k; unknown at 7k, which is why it is measured.
Cost: one extra kernel pass per iteration and the same CPU round as today.

### Rung 2: Mip-Splatting filters

Reference: Yu et al., CVPR 2024, and its code (`autonomousvision/mip-splatting`).

- **3D smoothing filter.** For each Gaussian, the smallest pixel footprint any
  training camera that sees it has at its depth, `f = depth / focal`, and the
  filter variance `(0.2 f)^2` added to the 3D covariance on all three axes;
  the opacity is scaled by `sqrt(det(Sigma) / det(Sigma + filter))` so the
  integral is kept. Recomputed at every densification round over the training
  cameras of the current coarse-to-fine level (the paper recomputes every 100
  iterations). Stored per Gaussian next to the parameters; it is not a trained
  parameter, but the scale gradient must go through the sum.
- **2D Mip filter.** Replace the fixed 0.3 pixel dilation of the 2D covariance
  with 0.1 and scale alpha by `sqrt(det(cov2d) / det(cov2d + 0.1 I))`, in the
  preprocess kernel and its CPU twin, with the matching derivative.

- **Saved scenes.** The 3D filter is baked into the saved cloud, as
  Mip-Splatting's export does: each scale becomes `sqrt(s^2 + filter)` and the
  opacity takes the same `sqrt(det / det)` factor, so files stay plain
  `GaussianCloud`s, checkpoints, CGC and floor alignment included. The 2D
  filter cannot be baked; if the option wins, the viewer's `preprocess.comp`
  and `splat.frag` and the CPU reference's `prepare()` adopt the same 0.1
  dilation and alpha factor, and the graphics-parity test in
  `tests/test_train.cpp` keeps tying the two together. The measurement is the
  training renderer's held-out PSNR, as for every other rung. (Owner's
  decision: the output may depend on this repository's viewer.)

Option `--mip-filter`, both parts together, since the paper found neither
sufficient alone. Expected: small at a single scale in the paper, but training
here starts at a quarter of the size, which is the multi-scale case the filters
are for. Cost: one small pass per densification round and a few operations per
splat.

### Rung 3: edge-aware splitting and recovery-aware pruning

Reference: Deng et al., "Improving Densification in 3D Gaussian Splatting for
High-Fidelity Rendering", CVPR 2026 (arXiv 2508.12313).

- **Edge-aware score.** A per-pixel edge weight from a Laplacian of each
  training image, computed once at load for each level. The FastGS scoring
  pass already renders sampled views and accumulates per-Gaussian pixel counts
  through an error mask; extend it to also accumulate `sum_p w_p * alpha_p` with
  the edge weight (the raw per-splat slots have a free 14th). Candidates remain
  those over the absolute-gradient threshold; each splits with probability
  `score / (largest score among this round's candidates)`, so the strongest
  edge candidate always splits and the rest less often than today. Fewer
  splits per round is part of the paper's point and helps speed. (Builder's
  call on the normalisation, which the paper leaves open.)
- **Long-axis split.** Replace clone and split for the chosen candidates: two
  children on the parent's longest axis at distance `0.45 L0` from its centre,
  longest semi-axis `L0 - d`, the other two `R0 * sqrt(1 - d^2 / L0^2)`,
  opacity 0.6 of the parent's.
- **Recovery-aware pruning.** 300 iterations after each opacity reset, remove
  the 20% of Gaussians with the lowest opacity. Only meaningful with the
  classic densification (MCMC has no reset).

Options `--edge-split` and `--recovery-prune`. Expected: the paper's +0.7 dB
over 3DGS at 30k with half the Gaussians; at 7k unknown. Cost: an extra
accumulation in the scoring pass already run.

### Rung 4: spherical-harmonic dropout

Reference: Fang et al., "Dropping Anchor and Spherical Harmonics for
Sparse-view Gaussian Splatting", CVPR 2026 (arXiv 2602.20933): per Gaussian,
with probability 0.2 per iteration, zero the SH coefficients above a retained
degree that rises through training (the paper: degree 0 until 20%, 1 until
40%, 2 until 60% of training; scaled to the iteration count here). In the
preprocess kernel, with a hash-based mask, and zero gradients for the dropped
coefficients. Option `--sh-dropout <probability>`, 0 off. The anchor dropout
of the same paper is for sparse views and is left out. Expected: small; it
targets the colour speckle seen on the sky-facing views. Cost: nil.

## Deliverables per rung

- The option(s), in `TrainOptions`, the CLI parser and `--help`.
- CPU twins and tests: densification rules are tested on the CPU in
  `tests/test_densify.cpp` (relocation formulas, the long-axis split, the
  recovery prune); kernel changes extend the GPU-versus-CPU and
  finite-difference tests in `tests/test_train.cpp`.
- The README's "How training works" list: one bullet for a winner, one line
  in the "did not help" bullet for a loser, with the measured numbers.
- A commit per rung with the measurement in its message, pushed to the
  branch.

## Left to the builder

- Kernel-level layout: whether noise and dropout get their own kernels or
  ride in Adam and preprocess; the hash generator.
- Exactly how the schedules scale to 7000 iterations (the paper values are
  for 30k or 10k), kept proportional as the rest of the trainer does.
- How relocation interacts with `--prune` and the significance pruning when
  both are on; the simplest consistent rule is fine.
- Test design within the existing Catch2 files.
- The order of the sub-measurements inside a rung, as long as each option is
  measured on its own and the rung ends with the winning combination.

## Results

Bicycle scene, `images_4`, official split, 7000 iterations, held-out PSNR on
the owner's RTX 4070 Laptop, two runs each unless one is listed. The baseline
before the ladder was the adaptive schedule with growth 5: 23.42 / 23.44 dB,
365k Gaussians, about 140 it/s.

| Rung | Configuration | Held-out dB | Gaussians | it/s | Kept |
|---|---|---|---|---|---|
| 0 | adaptive, no cap / growth 8 / 12 | 23.59 / 23.49-23.58 | 600k / 400-500k | 88-123 | no |
| 0 | fixed schedule, no cap | 24.29 / 24.19 | 617-702k | 88-90 | no |
| 0 | fixed, 400k / 450k / 500k / 600k | 24.08 / 24.14 / 24.19 / 24.23 | | 104 / 102 / 93 / 91 | |
| 0 | **fixed, growth 4 (new default)** | **24.11, 24.10** | **417k** | **107-117** | **yes** |
| 1 | `--opacity-reg 0.01 --scale-reg 0.01` | 22.98 / 22.52 | 270k | 104-110 | no |
| 1 | `--densify mcmc` | 23.19 / 23.18, 23.21 after the fix below | 417k | 76-83, 87 after | no |
| 1 | `--densify mcmc --position-noise 5e5` | 23.31 / 23.27 | 417k | 76-90 | no |
| 1 | all four | 23.72 / 23.70, 23.69 after | 417k | 84-91, 98 after | no |
| 2 | `--mip-filter` | 24.02 / 24.07 | 417k | 95-98 | no |
| 2 | `--mip-filter-2d` / `--mip-filter-3d` | 24.08 / 24.09 | 417k | 108 / 96 | no |
| 3 | `--edge-split` | 22.65 / 22.63 | 86k | 168-184 | no |
| 3 | `--recovery-prune 0.2` | 24.15 / 24.20 | 405k | 104-119 | no |
| 3 | both | 22.62 / 22.55 | 70k | 177-190 | no |
| 4 | `--sh-dropout 0.2` | 24.08 / 24.09 (mask on at evaluation), 24.08 / 24.10 after the fix | 417k | 104-116 | no |

A final review found three faults, fixed in the branch: SH dropout was still
applied in the reports and the final evaluation (it now applies to training
steps only; rung 4 re-measured, unchanged), the FastGS scoring pass still ran
under `--densify mcmc` (now gated; MCMC re-measured at 87 and 98 it/s, same
PSNR), and `--mip-filter-3d` lost its filter after `--prune` or a budget
recovery (now refreshed before the next step and before any save).

The ladder's gain is rung 0's 0.7 dB, from the fixed 15% / 35% coarse-to-fine
schedule in place of the adaptive one. Every technique from the literature
is in as an option, off: at 7000 iterations none of them beats the baseline
by the 0.2 dB this design asked for, and 3DGS-MCMC and the 3D filter also cost
speed. Two are worth another look under a different rule than this ladder's:
recovery pruning is as good and a little faster, and edge-aware splitting
reaches 22.6 dB with a fifth of the Gaussians at 1.7 times the speed, so a
split probability that starves growth less might do better than either.
