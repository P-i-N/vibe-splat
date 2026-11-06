# Faster feature matching in `splat sfm`

Date: 2026-11-02
Status: approved (user asked for "optimize, move it to GPU, whatever works")

## Intent

`splat sfm` spends most of its time in matching: for every image pair it
compares 4000 x 4000 descriptors of 128 floats with a scalar triple loop. On
194 photos (about 2100 sequential pairs) this dominates the run; exhaustive
matching (18 721 pairs) is out of reach. The rest of the pipeline (features,
RANSAC, registration, bundle adjustment) stays as it is unless profiling shows
it matters.

## Goals

- Same matches as today in spirit: mutual nearest neighbours under Lowe's
  ratio test (0.8) on unit-length RootSIFT descriptors.
- Matching runs on the GPU through the existing Vulkan context when one is
  available, and on the CPU otherwise (Windows CI has no GPU). Both paths give
  results close enough that the reconstruction does not change in quality.
- Descriptors uploaded once per run, not once per pair.
- Memory stays bounded for large sets: descriptors are stored as 8-bit
  integers on the GPU (100 MB for 194 images at 4000 features).

## Non-goals

- Approximate nearest neighbours (kd-trees, vocabulary trees). Brute force is
  exact and fast enough once vectorised.
- Learned features or matchers.
- Changing which pairs are tried (`--matcher` stays).

## Approaches considered

1. **CPU GEMM through Eigen.** Treat descriptors as an N x 128 matrix and
   compute A * B^T in row blocks. Eigen's product is SIMD and cache blocked.
   10-30x over the scalar loop with 40 lines of code. Still CPU-bound on
   exhaustive sets.
2. **Vulkan compute, integer dot products.** Descriptors quantised to uint8
   (as COLMAP stores them). A shader tiles B through shared memory, each
   thread keeps one A row and its best/second best, and records B's best A
   through `atomicMax` on a packed (score << 16 | index) word. A second small
   shader applies the ratio and mutual tests and appends matches to a list.
   Integer 4x8-bit dot products (`GL_EXT_integer_dot_product`, core in
   Vulkan 1.3) make the inner loop 32 instructions per descriptor pair.
3. **Vulkan compute, float descriptors.** Simpler numerically but 4x the
   memory and slower (no packed dot).

Chosen: 2 for the GPU, with 1 as the CPU fallback and the reference the tests
compare against. 3 was dropped for memory.

## Design

- `match_features` (CPU) keeps its signature and switches to blocked Eigen
  products. It remains the reference implementation.
- A new `FeatureMatcher` class in `sfm/gpu_matcher.{hpp,cpp}` takes a
  `vk::Context`, uploads all descriptors (uint8, 128 per keypoint, with an
  offset table), and matches batches of pairs: one dispatch per pair for the
  score kernel, a barrier, one per pair for the compaction kernel, then one
  readback of the match lists. Batch size is chosen so the scratch buffers stay
  around 64 MB.
- `run_sfm` gets an optional `vk::Context *`; the CLI creates one unless
  `--cpu` is passed, and falls back to the CPU path if context creation throws.
  Matching then goes pair-batch by pair-batch through the GPU while the CPU
  threads run the five-point RANSAC on the finished batches.
- Quantisation: `round(255 * d)`; scores are `dot >> 1` so they fit in 16 bits
  with the index in the other 16 (lowest index wins ties).
- Ratio test on quantised scores: squared distance is `2 - 2 dot / 255^2`.

## Tests

- GPU and CPU matchers agree on at least 95% of matches for the synthetic test
  views (SKIP without a device).
- Existing `[sfm]` tests and the `cli_sfm` pipeline test pass on both paths.

## Decisions made while building

- Workgroups of 64 threads, B tiles of 64 descriptors in shared memory with a
  padded stride, batches of 256 pairs per submission, and a second shader that
  applies the ratio and mutual tests on the GPU so only the match lists come
  back. The CPU verifies one batch while the GPU matches the next.
- With a GPU, the automatic matcher goes exhaustive up to 500 photos instead
  of 150: a pair costs well under a millisecond, and the 194-photo test set
  turned out not to be in shooting order (sequential matching posed 8 images,
  exhaustive all 194).
- Profiling after the matcher landed showed feature extraction and the
  five-point RANSAC as the next costs, so both got cheap fixes: the Gaussian
  blur was rewritten so both passes vectorise, descriptors use polynomial
  `atan2` and `exp` (2e-4 rad and 1e-5 relative), and RANSAC stops once a
  solution with the caller's minimum inlier count would have been found with
  99.9% probability, which cuts hopeless pairs from 500 iterations to a few.

## Results (194 photos, 1600 px, RTX 4070 Laptop, 16 CPU threads)

| Run | Before | After |
| --- | --- | --- |
| Sequential pairs (2052), matching + verification | ~195 s | 3.6 s |
| Sequential, whole `splat sfm` | 234 s, 8 of 194 posed | 14 s |
| Exhaustive pairs (18721), matching + verification | not tried | 11-12 s (GPU 7-8 s) |
| Exhaustive, whole `splat sfm` (now the default with a GPU) | not tried | 46-53 s, 194 of 194 posed |
| CPU-only matching (`--cpu`), sequential | ~195 s | 34 s |
| Feature extraction | 13.6 s | 8-9 s |

## Follow-up: registration and bundle adjustment (same day)

That stage took 27-31 s on the exhaustive run, almost all of it in
`bundle_adjust`: building the Schur-reduced system single-threaded (19 s),
dense LDLT solves (7 s), and 40-iteration final rounds that changed the RMS by
a thousandth. Changes:

- The reduced system is built in parallel over points. Each thread sums into
  its own 8x8 blocks, one per pair of views that share a point (only the
  a <= b half, S being symmetric), and the blocks are added and scattered into
  the dense S once. Per-point storage is flat, allocated once per call.
- Dense Cholesky (LLT) with the pivoted LDLT as fallback. A sparse Cholesky
  (Eigen SimplicialLLT with AMD) was measured 3.5x slower: the views see each
  other too much for the fill-in to stay small.
- Iterations stop once a step lowers the cost by less than a relative 1e-6
  (1e-5 during growth) instead of 1e-9.

Result: 28.7 s -> 5.5-6.2 s for the stage, with the same 194 poses, 31865
points and 0.467 px mean reprojection error. The whole exhaustive run is now
26 s (features 8 s, matching 11 s, mapping 6 s), from 234 s at the start of
the day. What is left in the stage is the dense factorisation (2 s) and the
per-point Jacobians (1.4 s).
