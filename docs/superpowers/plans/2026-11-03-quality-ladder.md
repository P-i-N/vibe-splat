# Quality Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `splat train`'s held-out PSNR at 7000 iterations on the official bicycle data, keeping the run at or above 100 iterations per second, by adding and measuring five rungs of techniques as options.

**Architecture:** Every technique is an option in `TrainOptions`, parsed in `cmd_train`, applied in `run_train`'s loop or in the CPU densification, with GPU kernels mirrored in `src/train/reference.cpp` and tied down by the GPU-versus-CPU tests. Each rung ends with a measurement on the official data, a README line and a commit; a technique becomes a default only if it wins.

**Tech Stack:** C++23, Vulkan compute (GLSL 460 in `shaders/`), Catch2 (`tests/`), CMake preset `windows-release` (build through `vcvars64.bat`), ctest.

**Spec:** `docs/superpowers/specs/2026-11-03-quality-ladder-design.md`

## Global Constraints

- Measurement: the folder built from `360_v2.zip` as the spec's "Measurement" section says (currently `C:\Users\janp\AppData\Local\Temp\claude\C--GitHub-vibe-splat\e0d0aeae-65bc-4d56-8c36-f4ad894160ff\scratchpad\official`), with `splat train <folder> --iterations 7000 --report-every 7000 --holdout 8 --save-every 0 --no-align -o <scratch>.ply`. Score: "PSNR ... on held-out views"; speed: the "Finished in" line. Two runs per decision; a gain under 0.2 dB is noise.
- Baseline at the start of this plan: 23.42 dB held-out, 364887 Gaussians, about 140 it/s with the defaults; 24.19-24.29 dB at 617k-702k Gaussians and 88-90 it/s with `--fixed-schedule`; 24.07-24.09 dB at 400k and 100-107 it/s with `--fixed-schedule --max-gaussians 400000`.
- A default changes only on a gain of at least 0.2 dB over two runs each way that keeps at least 100 it/s on the measurement.
- Every new GPU term has a CPU twin in `src/train/reference.cpp`; `ctest` (126 tests) and `splat_tests "[.long]"` pass after every task.
- Build: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build --preset windows-release'`; tests: `ctest --test-dir build/windows-release --output-on-failure`; one Catch2 case: `build/windows-release/tests/splat_tests.exe "<name>"`. No commas in Catch2 test names.
- Commits go on `claude/training-sota`, with the measured numbers in the message, and end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. `--max-growth 0` with the VRAM-derived cap: the VRAM limit must still bind (a run must never allocate past the budget). Pinned in Task 1's test of `GrowthBudget::limit`.
2. `--densify mcmc` when every Gaussian is dead or the cloud is at the cap: relocation must not divide by zero or empty the cloud. Pinned in Task 3's tests.
3. Position noise on a Gaussian with a degenerate scale (log scale below -20) or opacity exactly 1: positions must stay finite and unmoved respectively. Pinned in Task 4's test.
4. The 3D smoothing filter for a Gaussian no training camera sees: the filter is 0 and the opacity factor 1. Pinned in Task 7's test.
5. Edge-aware splitting in a round where every candidate's score is 0: fall back to today's rule, every candidate splits. Pinned in Task 9's test.

---

## Rung 0: settle the budget

### Task 1: `--max-growth` and the rung 0 measurement

**Files:**
- Modify: `src/train/schedule.hpp:25-36`, `src/train/schedule.cpp:76-81`
- Modify: `src/train/train.hpp` (TrainOptions), `src/train/train.cpp:453` (`GrowthBudget growth(...)`), `src/cli/main.cpp` (`--help`, `allow_only`, parsing)
- Test: `tests/test_densify.cpp` (or wherever `GrowthBudget` is tested today; `grep -rn GrowthBudget tests/` first)
- Modify: `README.md` "How training works", the Coarse to fine bullet.

**Interfaces:**
- Produces: `GrowthBudget(std::size_t initial, double factor)`; `factor <= 0` means no cap: `limit()` returns `std::numeric_limits<std::size_t>::max()` and `observe()` is a no-op. `TrainOptions::maxGrowth` (double, default 5.0). CLI `--max-growth <factor>`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST_CASE("A growth factor sets the final count", "[schedule]") {
  GrowthBudget budget(1000, 8.0);
  CHECK(budget.final_count() == 8000.0);
  CHECK(budget.limit(1.0, 1.0) == 8000);
}
TEST_CASE("A growth factor of zero never caps", "[schedule]") {
  GrowthBudget budget(1000, 0.0);
  CHECK(budget.limit(4.0, 0.0) == std::numeric_limits<std::size_t>::max());
  budget.observe(5);
  CHECK(budget.limit(1.0, 1.0) == std::numeric_limits<std::size_t>::max());
}
```

- [ ] **Step 2: Run them to verify they fail** (no two-argument constructor).

- [ ] **Step 3: Implement** the constructor and the no-cap behaviour in `schedule.hpp/.cpp`; thread `options.maxGrowth` into `GrowthBudget growth(startingPoints.size(), options.maxGrowth)` in `train.cpp`; parse `--max-growth` (default 5.0, clamp at 0) and list it in `--help` under the schedule options. In `train.cpp`, `std::min(densifyOptions.maxGaussians, growth.limit(...))` already keeps the VRAM cap when the limit is the maximum.

- [ ] **Step 4: Build; run `splat_tests "[schedule]"` and `ctest`; all pass.**

- [ ] **Step 5: Measure** (two runs each, record held-out dB, Gaussians, it/s): default; `--max-growth 0`; `--max-growth 8`; `--max-growth 12`; `--fixed-schedule --max-gaussians 450000`; `--fixed-schedule --max-gaussians 500000`. With the fixed-schedule rows already measured (see Global Constraints), decide: which of {adaptive schedule, fixed schedule} and which cap gives the most held-out dB at or above 100 it/s. Change `TrainOptions` defaults accordingly (`adaptiveSchedule`, `maxGrowth`, `maxGaussians`), and if the fixed schedule becomes the default, make `--fixed-schedule`'s opposite `--adaptive-schedule` in the CLI and `--help`.

- [ ] **Step 6: README**: rewrite the Coarse to fine bullet to say what the default now is and the measured numbers for both; `tests/test_densify.cpp` long test still passes with the new defaults.

- [ ] **Step 7: Commit** `Settle the growth cap and schedule for 7000 iterations` with the table of measurements.

---

## Rung 1: 3DGS-MCMC

### Task 2: opacity and scale penalties

**Files:**
- Modify: `shaders/train_gradient.comp` (add a push constant block; add the terms to `gradOpacities[i]` and `gradScales`), `src/train/trainer.hpp` (`TrainerOptions::opacityReg`, `scaleReg`, floats, default 0), `src/train/trainer.cpp` (the `gradient_.dispatch` at line ~430: pass `GradientPush{opacityReg / count_, scaleReg / count_}`; `gradient_` is constructed with push size 0 today, give it `sizeof(GradientPush)`)
- Modify: `src/train/reference.cpp` (the CPU gradient function used by the test "GPU loss and gradients match the CPU reference": add the same terms under the same options), `src/train/train.hpp` (`TrainOptions::opacityReg`, `scaleReg`), `src/train/train.cpp` (copy into `trainerOptions`), `src/cli/main.cpp` (`--opacity-reg <x>`, `--scale-reg <x>`)
- Test: `tests/test_train.cpp`

**Interfaces:**
- Produces: `TrainerOptions::opacityReg`, `TrainerOptions::scaleReg` (loss terms `reg * mean(sigmoid(opacity))` and `reg * mean(exp(scale))` over the N Gaussians); gradient per Gaussian `reg / N * sigmoid(x) * (1 - sigmoid(x))` on the opacity logit and `reg / N * exp(s)` on each log scale, added to whatever the pixels contribute, also for Gaussians no pixel touched (the kernel's early `touched` return must not skip them when a reg is on).

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("Opacity and scale penalties add their gradient to every Gaussian", "[train][vulkan]") {
  // A cloud entirely behind the camera: no pixel gradient at all.
  // With opacityReg = 0.5 and scaleReg = 0.25 and N Gaussians of logit opacity x and log scale s,
  // every gradient must equal 0.5 / N * sigmoid(x) * (1 - sigmoid(x)) and 0.25 / N * exp(s) within 1e-6,
  // and with both regs 0 every gradient must be exactly 0.
}
```
Use `Trainer::compute_gradients` then `trainer.gradients()` (see `Trainer::Gradients` in `trainer.hpp`).

- [ ] **Step 2: Run it; it fails** (gradients are 0).
- [ ] **Step 3: Implement** kernel, push constants, options, CLI, and the CPU twin; the existing "GPU loss and gradients match the CPU reference" test gets a section with both regs at 0.01 to pin the twin.
- [ ] **Step 4: Build; `splat_tests "[train]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add opacity and scale penalties (3DGS-MCMC)`.

### Task 3: relocation and growth by sampling

**Files:**
- Modify: `src/train/densify.hpp`, `src/train/densify.cpp` (new function), `src/train/trainer.hpp/.cpp` (`Trainer::relocate`), `src/train/train.hpp` (`enum class DensifyMode { Adc, Mcmc }; DensifyMode densifyMode = DensifyMode::Adc;`), `src/train/train.cpp` (densify block, lines ~573-620), `src/cli/main.cpp` (`--densify adc|mcmc`)
- Test: `tests/test_densify.cpp`

**Interfaces:**
- Produces:
```cpp
struct RelocateOptions {
  float minOpacity = 0.005f;   // At or below: dead, to be relocated
  double growth = 0.05;        // Share of the live count added per round, within maxGaussians
  std::size_t maxGaussians = ~std::size_t{0};
};
// 3DGS-MCMC: dead Gaussians move onto live ones drawn in proportion to opacity,
// then the count grows by `growth` the same way. A target that becomes N copies
// gets opacity 1 - (1 - o)^(1/N) and scale s * o / D, with
// D = sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1), as in gsplat.
[[nodiscard]] DensifyResult relocate_cloud(const GaussianCloud &cloud, const RelocateOptions &options, std::mt19937 &rng);
DensifyResult Trainer::relocate(const RelocateOptions &options, std::mt19937 &rng); // relocate_cloud on cloud(), then apply_changes
```
  `DensifyResult::source` is -1 for every relocated copy and for the targets they joined (fresh moments); other survivors keep their index. `result.cloned` counts relocated dead, `result.seeded` the growth copies, `result.pruned` 0.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST_CASE("A dead Gaussian is relocated onto a live one", "[densify]") {
  // line_of(2, 0.1f): opacities 0.9 and 0.001. After relocate_cloud the cloud still has 2 Gaussians,
  // both at the live one's position, opacity 1 - sqrt(1 - 0.9) each (N = 2), scale 0.1 * 0.9 / D(N=2, o') each,
  // D computed in the test from the formula; source is -1 for both.
}
TEST_CASE("Relocation targets are drawn in proportion to opacity", "[densify]") {
  // 1000 dead at opacity 0.001, two live at opacity 0.9 and 0.09, seed 1: the dead land on the first live
  // one between 850 and 950 times.
}
TEST_CASE("Growth adds five percent within the cap", "[densify]") {
  // 1000 live, no dead, maxGaussians 1030: result has 1030 Gaussians, seeded == 30;
  // maxGaussians 1000: unchanged count, seeded == 0.
}
TEST_CASE("Relocation with no live Gaussian leaves the cloud alone", "[densify]") {
  // all opacities 0.001: result equals the input, nothing moved.
}
```

- [ ] **Step 2: Run them; they fail** (no `relocate_cloud`).
- [ ] **Step 3: Implement** `relocate_cloud`: a dead mask; alive indices; `std::discrete_distribution` over live opacities; for each dead draw a target; growth draws `min(floor(growth * live), cap - count)` more targets; per target with k draws, N = k + 1, the update applied to the target and k copies (dead slots and new slots); fresh `source`. Then `Trainer::relocate`, `DensifyMode`, the CLI, and in `run_train`: under `Mcmc`, in place of `trainer.densify(...)`, call `trainer.relocate` with `maxGaussians` = the same cap the ADC path computes (VRAM, `--max-gaussians`, `--max-growth` times the starting count via `growth.final_count()`; do not call `growth.observe` in this mode); skip the opacity reset; `densifyFrom`, `densifyUntil` and the interval apply as for ADC.
- [ ] **Step 4: Build; `splat_tests "[densify]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add MCMC relocation and growth as a densification mode`.

### Task 4: position noise

**Files:**
- Create: `shaders/train_noise.comp` (add it to `shaders/CMakeLists.txt` like the others)
- Modify: `src/train/trainer.hpp/.cpp` (`TrainerOptions::positionNoise` float, default 0 = off; a `noise_` pipeline; `record_noise(cmd, scaler, iteration)` after `record_adam` inside `run` when `positionNoise > 0`), `src/train/train.hpp` (`TrainOptions::positionNoise`), `src/train/train.cpp` (copy into trainer options; pass the per-iteration scaler), `src/cli/main.cpp` (`--position-noise <x>`)
- Test: `tests/test_train.cpp`

**Interfaces:**
- Produces: kernel over Gaussians: `p += scaler * sigma(1 - opacity) * Sigma * n`, `Sigma = R diag(exp(2 s)) R^T`, `sigma(x) = 1 / (1 + exp(-100 (x - 0.995)))`, `n` three standard normals from a hash of (Gaussian index, iteration, component) (PCG-style integer hash to uniforms, Box-Muller to normals); `scaler = positionNoise * rates.position` for the current iteration. Push: `{uint count; float scaler; uint iteration;}`. Exposed for tests as `void Trainer::add_position_noise(float scaler, std::uint32_t iteration)`.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("Position noise moves transparent Gaussians by their covariance and leaves opaque ones", "[train][vulkan]") {
  // 4000 Gaussians at the origin, scale 1 (log 0), identity rotation:
  // half with opacity logit -20 (transparent), half with logit +20 (opaque).
  // add_position_noise(0.5f, 7): transparent ones' x has standard deviation within 10% of 0.5 and mean within 0.05 of 0;
  // opaque ones have not moved at all; all positions finite. A Gaussian with log scale -30 stays finite.
}
```
- [ ] **Step 2: Run it; it fails.**
- [ ] **Step 3: Implement** kernel, dispatch, options, CLI.
- [ ] **Step 4: Build; `splat_tests "[train]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add position noise (3DGS-MCMC)`.

### Task 5: rung 1 measurement

- [ ] **Step 1: Measure**, two runs each, on top of rung 0's defaults: (a) `--opacity-reg 0.01 --scale-reg 0.01`; (b) `--densify mcmc` at a cap equal to the baseline's final count (`--max-gaussians <rung 0 count>`); (c) `--densify mcmc` at rung 0's winning cap; (d) `--position-noise 5e5` with (c); (e) all four together; (f) if (e) is below 100 it/s, (e) with the cap lowered until it is above.
- [ ] **Step 2: Decide** defaults by the Global Constraints rule; set them in `TrainOptions`.
- [ ] **Step 3: README**: a Density bullet for MCMC if it wins (what it does, the numbers), otherwise one line in the "did not help" bullet per losing option, with numbers.
- [ ] **Step 4: `ctest` and the long test pass; commit** `Measure 3DGS-MCMC at 7000 iterations` with the table.

---

## Rung 2: Mip-Splatting filters

### Task 6: the 2D Mip filter

**Files:**
- Modify: `shaders/train_preprocess.comp:61-63,102` (dilation and opacity), `shaders/train_gradient.comp` (the derivative of the opacity factor through `G` and `raw[at]`), `shaders/preprocess.comp` (viewer, same dilation and factor, unconditional once the option becomes the default, else behind the same push flag), `src/train/reference.cpp:135,146` (`prepare()`), `src/train/trainer.hpp/.cpp` (`TrainerOptions::mipFilter2d` bool, pushed to both kernels), `src/train/train.hpp` (`TrainOptions::mipFilter`), `src/cli/main.cpp` (`--mip-filter`)
- Test: `tests/test_train.cpp`

**Interfaces:**
- Produces: with the flag, `a = cov[0][0] + 0.1`, `c = cov[1][1] + 0.1`, and the splat's opacity is `o * sqrt(det(cov) / det(cov + 0.1 I))`; the chain rule carries `dL/dopacity_used * o * d(factor)/dcov` into the covariance gradient and `factor` into the opacity gradient. The CPU `prepare()` takes the same flag (a new field in its options struct).

- [ ] **Step 1: Write the failing tests**: a section "with the 2D Mip filter" in "GPU forward matches the CPU reference and the graphics renderer" (forward parity with the flag on both sides; the graphics path compared only once the viewer has the filter) and in "GPU loss and gradients match the CPU reference" (gradient parity), plus a CPU finite-difference check of the opacity-factor derivative in "Analytic gradients match finite differences on the CPU" with the flag on.
- [ ] **Step 2: Run; the new sections fail** (no flag).
- [ ] **Step 3: Implement** kernels, reference, options, CLI.
- [ ] **Step 4: Build; `splat_tests "[train]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add the 2D Mip filter (Mip-Splatting)`.

### Task 7: the 3D smoothing filter and its baking

**Files:**
- Create: `src/train/mip.hpp`, `src/train/mip.cpp`
- Modify: `shaders/train_preprocess.comp` (add a `Filter3d` buffer: `sigma = R diag(exp(2 s) + f) R^T`, opacity times `prod_a sqrt(exp(2 s_a) / (exp(2 s_a) + f))`), `shaders/train_gradient.comp` (same `D + f I` in the covariance chain, the opacity factor's derivative on each log scale), `src/train/reference.cpp` (same, the cloud gains a parallel `filter3d` vector in the reference's options), `src/train/trainer.hpp/.cpp` (`filter3d_` buffer sized per Gaussian, zero by default, rebuilt in `allocate_per_gaussian` and gathered in `apply_changes` like `first`; `void set_filter3d(std::span<const float>)`), `src/train/train.cpp` (after every densification round and at start: compute and upload; at `save_output`: bake), `src/CMakeLists.txt` (new source)
- Test: `tests/test_train.cpp`, `tests/test_densify.cpp` (CPU filter computation)

**Interfaces:**
- Produces:
```cpp
// Mip-Splatting's 3D smoothing filter: for each Gaussian, (0.2 * depth / focal)^2 at the
// camera that sees it with the smallest footprint, 0 if none sees it (in front of the
// camera and inside the image). focal is max(fx, fy) of that camera.
[[nodiscard]] std::vector<float> smoothing_filter(const GaussianCloud &cloud, std::span<const Camera> cameras);
// Folds the filter into the cloud: scale_a = 0.5 log(exp(2 s_a) + f), opacity logit of
// sigmoid(o) * prod_a sqrt(exp(2 s_a) / (exp(2 s_a) + f)).
void bake_filter(GaussianCloud &cloud, std::span<const float> filter);
```

- [ ] **Step 1: Write the failing tests**

```cpp
TEST_CASE("The smoothing filter is the smallest footprint of the cameras that see a Gaussian", "[densify]") {
  // Two cameras of focal 100 looking at the origin from distances 2 and 4: filter == (0.2 * 2 / 100)^2 within 1e-9.
  // A Gaussian behind both cameras: filter 0. (Build the cameras with test::camera_of or test_camera.)
}
TEST_CASE("Baking the filter keeps the Gaussian's integral", "[densify]") {
  // scale log 0.1 on all axes, opacity 0.5, filter 0.03: baked scale == 0.5 log(0.01 + 0.03) and
  // baked opacity == 0.5 * (0.01 / 0.04)^(3/2) within 1e-6; filter 0 leaves the cloud unchanged.
}
```
and sections with a nonzero filter in the three parity tests of Task 6.
- [ ] **Step 2: Run; fail.**
- [ ] **Step 3: Implement.** In `run_train`, the cameras passed are the current level's training cameras (`view_at(i, iteration).camera` for every training view); recompute right after `trainer.densify`/`relocate` and once before the loop; `save_output` bakes into the downloaded cloud when `options.mipFilter`. Kernels read the buffer only when the flag is on (zero filter is the identity anyway).
- [ ] **Step 4: Build; `splat_tests "[train]" "[densify]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add the 3D smoothing filter (Mip-Splatting)`.

### Task 8: rung 2 measurement

- [ ] **Step 1: Measure** `--mip-filter` twice (on top of the defaults rungs 0 and 1 settled), and once each of the 2D-only and 3D-only variants (give `TrainOptions` the two booleans; the CLI flag sets both) to know which part carried any gain.
- [ ] **Step 2: Decide**; if it wins, make the viewer's `preprocess.comp` filter unconditional and extend the graphics-parity test accordingly.
- [ ] **Step 3: README** bullet or "did not help" line with numbers; `ctest`, long test; **commit** `Measure the Mip-Splatting filters at 7000 iterations`.

---

## Rung 3: edge-aware splitting and recovery-aware pruning

### Task 9: edge-aware score and long-axis split

**Files:**
- Modify: `shaders/train_types.glsl` (`RAW_PER_SPLAT` 13 -> 14, comment slot 13: edge score), `src/train/trainer.cpp:24` (`kRawGradientsPerSplat = 14`), `shaders/train_loss.comp:81-86` (when scoring, alpha holds `error + 2 * edge` with `edge` the 4-neighbour Laplacian magnitude of the target's luminance over 4, in [0, 1], from the tile already in shared memory; `train_error_mask.comp` keeps the error part and writes `marked + 2 * edge` back), `shaders/train_backward.glsl:87,128` (decode the packing `a = (marked ? 2 : 0) + edge` with `edge` clamped to [0, 0.999]: `marked = a >= 2 ? 1 : 0`, `edge = a - 2 * marked`; `g[13] = edge * weight` with `weight` the per-pixel `T * alpha` already used for `g[9]`), `shaders/train_score.comp` (`densifyStats[i].more.w += raw[13] * toPixels`), `src/train/trainer.cpp:652-670` (`DensifyStats::edgeScore` from `more.w`), `src/train/densify.hpp/.cpp` (`DensifyStats::edgeScore`; `DensifyOptions::edgeSplit` bool; under it, candidates split with probability `edgeScore / max edgeScore` and the long-axis rule replaces both clone and split), `src/train/train.hpp/.cpp` (`TrainOptions::edgeSplit`), `src/cli/main.cpp` (`--edge-split`)
- Test: `tests/test_densify.cpp`, `tests/test_train.cpp`

**Interfaces:**
- Produces: `DensifyStats::edgeScore` (per Gaussian, sum over scored views of edge weight times blending weight, in pixels); `DensifyOptions::edgeSplit`. Long-axis split: children at `+-0.45 L0` along the parent's longest axis (its largest scale, in the rotation's frame), child longest semi-axis `0.55 L0`, the other two `R * sqrt(1 - 0.45^2)`, opacity logit of `0.6 * sigmoid(parent)`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST_CASE("Long-axis split places two children along the longest axis", "[densify]") {
  // One Gaussian at the origin, scales (log 1, log 0.2, log 0.2), identity rotation, opacity 0.5,
  // gradient over the threshold, edgeScore 1, edgeSplit on: result has 2 Gaussians at x = +-0.45,
  // scales (log 0.55, log(0.2 * sqrt(1 - 0.2025)), same), opacity 0.3 each, source -1.
}
TEST_CASE("Edge scores set the split probability", "[densify]") {
  // 1000 candidates over the threshold, edgeScore 1 for the first 500 and 0.1 for the rest, seed 1:
  // all of the first 500 split; between 30 and 70 of the rest.
}
TEST_CASE("All edge scores zero splits every candidate", "[densify]") {
  // edgeScore all 0 with edgeSplit on: every candidate over the threshold is split (today's rule).
}
```
and in `test_train.cpp`: "Scoring a view accumulates the edge score" next to the existing "Scoring a view counts the high-error pixels": a target with one vertical edge; Gaussians covering the edge get an edgeScore above those on flat colour.
- [ ] **Step 2: Run; fail.**
- [ ] **Step 3: Implement** in the order of the Files list; keep the error-mask semantics unchanged for `markedPixels`.
- [ ] **Step 4: Build; `splat_tests "[densify]" "[train]"`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add edge-aware long-axis splitting`.

### Task 10: recovery-aware pruning

**Files:**
- Modify: `src/train/densify.hpp/.cpp` (expose `[[nodiscard]] DensifyResult remove_least_opaque(const GaussianCloud &cloud, double share);` built on the file's `remove_highest` with negated opacities), `src/train/trainer.hpp/.cpp` (`std::size_t Trainer::remove_least_opaque_gaussians(double share)`), `src/train/train.hpp/.cpp` (`TrainOptions::recoveryPrune` double, default 0 = off; in the loop: when `recoveryPrune > 0 && iteration == lastOpacityReset + 300 && lastOpacityReset > 0`, remove that share, print a line like the `--prune` one), `src/cli/main.cpp` (`--recovery-prune <share>`)
- Test: `tests/test_densify.cpp`

- [ ] **Step 1: Write the failing test**: `line_of(10, 0.1f)` with opacities 0.1 .. 1.0, share 0.2: the two least opaque are gone, 8 remain in order, `pruned == 2`.
- [ ] **Step 2: Run; fail. Step 3: Implement. Step 4: Build, tests pass.**
- [ ] **Step 5: Commit** `Add recovery-aware pruning`.

### Task 11: rung 3 measurement

- [ ] **Step 1: Measure** twice each: `--edge-split`; `--recovery-prune 0.2` (only meaningful with `--densify adc`); both.
- [ ] **Step 2: Decide; README; `ctest`, long test; commit** `Measure edge-aware splitting and recovery pruning at 7000 iterations`.

---

## Rung 4: spherical-harmonic dropout

### Task 12: SH dropout

**Files:**
- Modify: `shaders/train_preprocess.comp` (SH evaluation: when `dropout > 0` and `hash(i, iteration) < dropout`, evaluate only degrees `<= retained`), `shaders/train_gradient.comp` (zero `gradSh` above `retained` for the same Gaussians, same hash), `src/train/trainer.hpp/.cpp` (`void set_sh_dropout(float probability, int retainedDegree, std::uint32_t iteration)`; pushed to both kernels), `src/train/train.hpp/.cpp` (`TrainOptions::shDropout` float, default 0; in the loop before `step`: retained degree 0 until 20% of training, 1 until 40%, 2 from then on, scaled by `options.iterations`), `src/cli/main.cpp` (`--sh-dropout <probability>`)
- Test: `tests/test_train.cpp`

- [ ] **Step 1: Write the failing test**: "SH dropout renders with the retained degrees only": `random_cloud(300, 1, 7)` at degree 3; `set_sh_dropout(1.f, 0, 1)` makes `render` equal the CPU reference render of the same cloud with coefficients above degree 0 zeroed (within 1e-3); `set_sh_dropout(0.f, 0, 1)` restores the plain render; with probability 0.5 the GPU gradients of the dropped Gaussians' higher coefficients are exactly 0 and of the others not all 0.
- [ ] **Step 2: Run; fail. Step 3: Implement. Step 4: Build, `[train]`, `ctest` pass.**
- [ ] **Step 5: Commit** `Add spherical-harmonic dropout`.

### Task 13: rung 4 measurement

- [ ] **Step 1: Measure** `--sh-dropout 0.2` twice.
- [ ] **Step 2: Decide; README; `ctest`, long test; commit** `Measure SH dropout at 7000 iterations`.

---

## Task 14: close the ladder

- [ ] **Step 1:** Add a results table to the spec (`docs/superpowers/specs/2026-11-03-quality-ladder-design.md`): every measured configuration, held-out dB, Gaussians, it/s; set its status to done. Update the memory note `bicycle-quality-ceiling` with the final defaults and numbers.
- [ ] **Step 2:** Run the full `ctest` and the long test once more with the final defaults; a last measurement run with no options to confirm the README's numbers.
- [ ] **Step 3: Commit** `Record the quality ladder's results` and push.
