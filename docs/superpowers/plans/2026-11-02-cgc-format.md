# CGC Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save and load Gaussian clouds in a compact `.cgc` format (per-splat SH degree, quantization, Morton-ordered 256-splat chunks, byte planes, LZMA), picked by file extension.

**Architecture:** A small LZMA wrapper (`splat/lzma.*`) around liblzma's `.lzma` alone format; the format itself in `splat/cgc.*` with pure helpers in `splat::cgc_detail` for unit tests; `save_cloud`/`load_cloud` in `gaussian_cloud.*` dispatch by extension; the CLI and trainer call those.

**Tech Stack:** C++23, CMake + vcpkg, liblzma 5.8.x, Catch2 3.

**Spec:** `docs/superpowers/specs/2026-11-02-cgc-format-design.md` (read it; this plan does not repeat the byte layout).

## Global Constraints

- Magic `CGC\0` (`43 47 43 00`), version u32 = 1, then u64 uncompressed payload size, then the LZMA stream.
- Little-endian everywhere. No `maxShDegree` in the payload header: the decoder takes the highest value of the SH degree column (0 for an empty cloud).
- LZMA: `.lzma` alone format via `lzma_alone_encoder` / `lzma_alone_decoder` (unknown size + end marker). Final saves use preset 9 with `LZMA_PRESET_EXTREME`; training checkpoints use preset 1.
- liblzma linked PRIVATE to `splat_core` through `find_package(liblzma CONFIG REQUIRED)` and `liblzma::liblzma`. No other new dependencies.
- `.cgc` detection is case-insensitive on the extension; anything else is PLY, written and read exactly as today.
- Errors throw `std::runtime_error` with the path in the message, like `load_ply`.
- Match the style of `src/splat/gaussian_cloud.cpp`: `std::format`, anonymous namespaces, sparse comments.
- Work on a new branch `cgc-format` cut from `main`. If `main` fails to build without `65d6995` (Include `<array>` in offscreen.hpp), cherry-pick that commit first.

## Format details

The spec's "Payload" section now pins every quantization rule, bit order and array order; follow it exactly. Two details it doesn't spell out:
- **Morton quantization:** 21 bits per axis within the scene box: `q = min(2^21 - 1, floor((p - min) / (max - min) * 2^21))`, 0 when the axis is flat.
- **SH indexing:** in `GaussianCloud`, band ℓ coefficient k (0 .. 2ℓ) of splat i, channel c is `sh[(i * coeffs + ℓ² + k) * 3 + c]`.

## Review Focus

- A cloud straight out of training with a NaN or Inf in some attribute: the save fails with a clear message instead of undefined float-to-int casts. Test in Task 3.
- A zero-length or unnormalized quaternion: it decodes to a valid unit quaternion (zero becomes identity). Test in Task 2.
- Extreme opacity logits (±50) and extreme log scales (−30, +10): they decode to finite values. Test in Task 3.
- A `.ply` file renamed to `.cgc`, or a `.cgc` cut short by an interrupted download: `load_cloud` throws `std::runtime_error`, it doesn't crash. Tests in Tasks 1 and 3.
- Exporting a large trained scene (about 1M splats): preset 9e on ~60 MB of payload can take a minute and around 700 MiB of encoder memory. Measure in Task 5 and lower the preset if it's unreasonable, recording the numbers.

---

### Task 1: LZMA wrapper and dependency

**Files:**
- Modify: `vcpkg.json` (add `"liblzma"`), `CMakeLists.txt` (`find_package(liblzma CONFIG REQUIRED)` next to the other `find_package` calls), `src/CMakeLists.txt` (add `splat/lzma.cpp`; `liblzma::liblzma` in the PRIVATE link list)
- Create: `src/splat/lzma.hpp`, `src/splat/lzma.cpp`
- Test: `tests/test_cgc.cpp` (new; add to `tests/CMakeLists.txt`)

**Interfaces:**
- Produces:
  - `std::vector<std::byte> splat::lzma_compress(std::span<const std::byte> data, int level = 9, bool extreme = true)`: complete `.lzma` alone stream. `lzma.hpp` must not include `<lzma.h>`.
  - `std::vector<std::byte> splat::lzma_decompress(std::span<const std::byte> stream, std::size_t expectedSize)`: throws `std::runtime_error` on any liblzma error, on truncated input, or when the output size differs from `expectedSize`. The caller adds the path to the message.

- [ ] **Step 1: Write the failing tests** in `tests/test_cgc.cpp`, tag `[cgc]`:
  - `LZMA round-trips data`: 100 000 bytes of a repeating pattern plus some pseudo-random bytes; `lzma_decompress(lzma_compress(d), d.size()) == d`; compressed size is less than `d.size()`.
  - `LZMA round-trips empty input`, and round-trips the same data with `level = 1, extreme = false`.
  - `LZMA rejects bad input`: (a) the stream with the last 10 bytes removed throws; (b) the stream with byte 0 (properties) set to `0xFF` throws; (c) the correct stream with `expectedSize + 1` throws.
- [ ] **Step 2: Configure and build to see it fail** — `cmake --preset windows-release` then `cmake --build --preset windows-release`. Expected: compile error, `splat/lzma.hpp` not found. (The first configure installs liblzma through vcpkg.)
- [ ] **Step 3: Implement** with `lzma_lzma_preset(&opts, level | (extreme ? LZMA_PRESET_EXTREME : 0))` + `lzma_alone_encoder`, and `lzma_alone_decoder(&strm, UINT64_MAX)`, running `lzma_code` until `LZMA_STREAM_END`. Always `lzma_end` (RAII guard).
- [ ] **Step 4: Run the tests** — `build/windows-release/tests/splat_tests.exe "[cgc]"`. Expected: all pass.
- [ ] **Step 5: Commit** — "Add liblzma and an LZMA alone-format wrapper"

### Task 2: CGC building blocks

**Files:**
- Create: `src/splat/cgc.hpp`, `src/splat/cgc.cpp` (add to `src/CMakeLists.txt`)
- Test: `tests/test_cgc.cpp`

**Interfaces:**
- Produces, in `namespace splat::cgc_detail` (declared in `cgc.hpp`):
  - `std::uint64_t morton_code(std::uint32_t x, std::uint32_t y, std::uint32_t z)`: 21 low bits of each; x → bit 0, y → bit 1, z → bit 2, then repeating.
  - `std::uint32_t pack_rotation(std::array<float, 4> wxyz)` and `std::array<float, 4> unpack_rotation(std::uint32_t)`: smallest three, as in the spec. A zero-length input packs as the identity `(1, 0, 0, 0)`.
  - `int choose_sh_degree(std::span<const float> sh, int degree, float opacityLogit, float threshold)`: `sh` is one splat's `3 * (degree + 1)²` values in `GaussianCloud` order.
  - `void append_planes(std::vector<std::byte> &out, std::span<const std::uint32_t> values)`, and `std::vector<std::uint32_t> read_planes(std::span<const std::byte> in, std::size_t count)`: 4-byte byte planes.

- [ ] **Step 1: Write the failing tests:**
  - `morton_code`: `(1,0,0) == 1`, `(0,1,0) == 2`, `(0,0,1) == 4`, `(3,0,0) == 9`, `(0x1FFFFF, 0x1FFFFF, 0x1FFFFF) == (1ull << 63) - 1`.
  - Rotation round trip: 1000 random unit quaternions (fixed seed) plus `(0,1,0,0)`, `(0,0,0,-1)`, `(-0.5,0.5,-0.5,0.5)` and a quaternion scaled by 3: decoded is unit length within 1e-5 and `|dot(original_normalized, decoded)| > 0.999`. Zero quaternion decodes to `(1,0,0,0)`.
  - `choose_sh_degree` with threshold `1.5f / 255`: all higher bands zero → 0; band-3 coefficient 0.5, opacity logit 5 → 3; band-1 coefficients 0.5 with bands 2 and 3 at 1e-4 → 1; band-1 coefficient 0.05 with logit 5 → 1, but the same with logit −8 → 0; degree-0 input → 0.
  - Planes: `{0x04030201, 0x08070605}` gives bytes `01 05 02 06 03 07 04 08`, and `read_planes` gives the input back.
- [ ] **Step 2: Build and run** — `cmake --build --preset windows-release`. Expected: compile errors for the missing header and functions.
- [ ] **Step 3: Implement.** Band ℓ's value per channel is `sqrt(Σ_k c² / (4π)) * sigmoid(opacityLogit)`. Drop bands from the top while the largest channel value is below the threshold. Rotation: normalize, choose the largest |component|, flip the sign so it's positive, store its index in bits 30–31 and the other three in index order in bits 0–9, 10–19, 20–29, symmetric 10-bit over ±1/√2.
- [ ] **Step 4: Run tests** — `build/windows-release/tests/splat_tests.exe "[cgc]"`. Expected: all pass.
- [ ] **Step 5: Commit** — "Add CGC building blocks: Morton codes, rotation packing, SH degree choice"

### Task 3: `save_cgc` and `load_cgc`

**Files:**
- Modify: `src/splat/cgc.hpp`, `src/splat/cgc.cpp`
- Test: `tests/test_cgc.cpp`

**Interfaces:**
- Consumes: Task 1 and Task 2 functions.
- Produces:
  - `struct splat::CgcOptions { float shThreshold = 1.5f / 255.f; int lzmaLevel = 9; bool lzmaExtreme = true; };`
  - `struct splat::CgcStats { std::uint64_t bytes = 0; std::array<std::size_t, 4> degreeCounts{}; };`
  - `CgcStats splat::save_cgc(const GaussianCloud &cloud, const std::filesystem::path &path, const CgcOptions &options = {})`
  - `GaussianCloud splat::load_cgc(const std::filesystem::path &path)`

- [ ] **Step 1: Write the failing tests:**
  - Round trip, for degree 0..3, 1000 splats from a fixed seed (positions in a 10-unit box, log scales in [−6, −1], random unit quaternions, opacity logits in [−4, 4], DC in [−1.5, 1.5], higher SH in [−0.3, 0.3]). Then:
    - same count; `loaded.shDegree == degree`.
    - Splats come back in Morton order, so match each loaded splat to an original by nearest position (tolerances, not exact equality). Brute force is fine at 1000.
    - Position error per axis ≤ half a step of its chunk box; on this data that's ≤ 0.01.
    - Log-scale error ≤ 0.05; `|dot|` of rotations > 0.999; sigmoid opacity within 1/256.
    - DC colour `0.28209479 · sh0 + 0.5`: within 0.004 for every splat inside its channel's 0.1–99.9 percentile range. At most 0.2% of splats per channel may fall outside (clamped).
    - Higher SH within `r / 15` for band 1 and `r / 7` for bands 2–3, where r is the largest absolute value (≤ 0.3), or equal to 0 when that splat's band was dropped.
  - `degreeCounts` sums to the count; a cloud whose higher bands are all zero gives `degreeCounts[0] == count` and loads with `shDegree == 0`.
  - Colour outlier: the same 1000-splat cloud plus one splat with `sh0 = 100` on every channel. The other splats keep the 0.004 colour tolerance, and the outlier decodes to a finite value.
  - Edge cases: 0 splats (loads as empty), 1, 256, 257 splats; 300 splats all at `(1, 2, 3)` with the same scale (flat boxes: no NaN, positions decode exactly to `(1, 2, 3)`); opacity logits ±50 and log scales −30 / +10 decode to finite values.
  - Non-finite input: a NaN position, and separately an Inf in `sh`, make `save_cgc` throw, and leave no file behind if none existed.
  - Malformed files throw `std::runtime_error`: a PLY written by `save_ply` loaded with `load_cgc`; a valid `.cgc` cut to half its size; a valid `.cgc` with byte 16 (first byte of the LZMA properties) set to `0xFF`; wrong version.
- [ ] **Step 2: Build and run** — Expected: compile errors for `save_cgc` / `load_cgc`.
- [ ] **Step 3: Implement** following the spec's encoding and decoding sections and the conventions above. While parsing, check every section against the remaining payload and throw on overrun, on leftover bytes, and on SH degree values > 3. Before Morton sorting, compute the scene box from the positions (an empty cloud writes count 0 and no columns).
- [ ] **Step 4: Run tests** — `build/windows-release/tests/splat_tests.exe "[cgc]"`. Expected: all pass.
- [ ] **Step 5: Add the render check** in `tests/test_cgc.cpp`, tag `[vulkan][cgc]`, skipping without Vulkan as `test_render.cpp` does: `make_test_scene(50'000)` → `save_cgc` → `load_cgc`; render both with one `SplatRenderer` each at 320×180 from `default_orbit(original)`; PSNR over RGB > 35 dB. Small local PSNR helper in the test.
- [ ] **Step 6: Run** — `build/windows-release/tests/splat_tests.exe "[cgc]"`. Expected: pass. If PSNR comes out below 35 dB, report the value; don't loosen the threshold silently.
- [ ] **Step 7: Commit** — "Add CGC save and load"

### Task 4: Choose the format by extension, CLI and trainer

**Files:**
- Modify: `src/splat/gaussian_cloud.hpp`, `src/splat/gaussian_cloud.cpp`, `src/cli/main.cpp` (`load_scene` ~line 178, `cmd_make_test_scene` ~line 224, help text ~lines 40–42, command table ~line 433), `src/train/train.cpp` (~lines 393 and 412), `tests/CMakeLists.txt`, `README.md`
- Test: `tests/test_cgc.cpp`

**Interfaces:**
- Consumes: `save_cgc`, `load_cgc`, `CgcStats` from Task 3.
- Produces (in `gaussian_cloud.hpp`):
  - `enum class CloudFormat { Ply, Cgc };`
  - `CloudFormat cloud_format(const std::filesystem::path &path)`: `Cgc` when the extension equals `.cgc` ignoring case.
  - `void save_cloud(const GaussianCloud &cloud, const std::filesystem::path &path)`, and an overload `save_cloud(cloud, path, CloudFormat format, const CgcOptions &options = {})`. `gaussian_cloud.hpp` forward-declares `struct CgcOptions;`, and the overload's default argument lives in `cgc.hpp`, or else use two overloads. The implementer picks whichever compiles cleanly without a circular include.
  - `GaussianCloud load_cloud(const std::filesystem::path &path)`.

- [ ] **Step 1: Write the failing tests:** `cloud_format` gives `Cgc` for `a.cgc`, `a.CGC`, `dir.x/b.Cgc`, and `Ply` for `a.ply`, `a`, `a.cgc.tmp`. `save_cloud` / `load_cloud` round-trip a 10-splat cloud through both `x.ply` (exact) and `x.CGC` (by tolerance). The `CloudFormat` overload writes CGC to `x.cgc.tmp`, and `load_cgc` reads it back.
- [ ] **Step 2: Build and run** — Expected: compile errors.
- [ ] **Step 3: Implement the dispatch.** Point `load_scene` and `cmd_make_test_scene` at it. In `train.cpp`, the checkpoint saves to `temporary` in `cloud_format(options.out)`, with `CgcOptions{.lzmaLevel = 1, .lzmaExtreme = false}`, and the final save calls `save_cloud(trainer.cloud(), options.out)`. Leave `reconstruct.cpp` on `save_ply`.
- [ ] **Step 4: Add `splat convert <in> <out>`:** `load_cloud`, then `save_cgc` for CGC output (print splats per SH degree from `CgcStats`) or `save_ply` otherwise. Print input size, output size and the ratio. Usage message and exit code 2 on wrong arguments, as the other commands do. Add a line to the `--help` text, and mention `.cgc` next to `view`, `render` and `make-test-scene`.
- [ ] **Step 5: CLI tests** in `tests/CMakeLists.txt`: `cli_convert_cgc` (`splat convert cli_scene.ply cli_scene.cgc`, depends on `cli_make_test_scene`) and `cli_render_cgc` (`splat render cli_scene.cgc -o cli_render_cgc.png --width 320 --height 180`, depends on `cli_convert_cgc`).
- [ ] **Step 6: README:** a short "Compressed output (.cgc)" section. Cover what it is, that `-o scene.cgc` on `train` or `make-test-scene` writes it, `splat convert`, that it's lossy, and that the layout is in the spec.
- [ ] **Step 7: Run everything** — `cmake --build --preset windows-release` then `ctest --preset windows-release`. Expected: all tests pass, including the new CLI tests.
- [ ] **Step 8: Commit** — "Pick PLY or CGC by file extension; add splat convert"

### Task 5: Measure on a real scene

**Files:** none (results go in the final report; tune the presets in `lzma.cpp` only if the numbers call for it)

- [ ] **Step 1:** Find a trained `splat.ply` (ask the user for one if none is in `build/` or a project folder). Run `splat convert <scene>.ply <scene>.cgc` and record the sizes, ratio, SH degree histogram and wall time. Also record the encoder's peak memory from Task Manager or `/usr/bin/time`.
- [ ] **Step 2:** Run `splat render` on both files with the same arguments and compare the PNGs by eye. If wall time for about 1M splats is over ~60 s, try preset 9 without EXTREME and preset 6, record size and time for each, and keep the best trade-off.
- [ ] **Step 3:** Commit any preset change with the measurements in the message.
