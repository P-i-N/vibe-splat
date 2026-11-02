# CGC: Compressed Gaussian Cloud format

Date: 2026-11-02
Status: approved

## Intent

Trained scenes are meant to be viewed in a browser, where download size is the
main constraint. The standard 3DGS `.ply` stores 62 floats (248 bytes) per
Gaussian at SH degree 3, including 12 bytes of always-zero normals. CGC is a
compact export format built for that use. A browser viewer with a hand-written
LZMA decoder will follow later; this work covers only the C++ side.

## Goals

- `save` writes CGC when the output path ends in `.cgc` (case-insensitive);
  anything else writes `.ply` exactly as today.
- `load` reads CGC when the input path ends in `.cgc`, decodes it into an
  ordinary `GaussianCloud`, and everything downstream (`view`, `render`) runs
  unchanged.
- Everything internal (training, densification, SfM preview, test scenes) keeps
  working on full-precision data and `.ply`. CGC is only an export and import
  format.
- Each splat gets its own SH degree (0 to 3), chosen at save time.
- Attributes are quantized, splats are Morton-ordered and grouped into chunks of
  256, columns are split into byte planes, and the whole payload is compressed
  with LZMA through `liblzma`.

## Non-goals (for now)

- Delta coding of any column. Explicitly deferred.
- Browser decoder or web viewer.
- SH codebooks / vector quantization.
- Quantization-aware fine-tuning during training.
- Several independent LZMA streams (parallel or progressive decoding).
- Keeping the original splat order. A loaded CGC cloud is in Morton order.
- Bit-exact round trip. CGC is lossy.

## Anti-goals

- A CGC file that looks clearly different from the PLY it came from in
  `splat render`. The test suite checks render PSNR (see Testing).
- Format details that would make the later hand-written browser decoder hard:
  no xz container, no checksums or filters the decoder would have to
  implement, everything little-endian, fixed-size fields.

## Constraints

- C++23, matching the existing code style in `src/splat/`.
- The new dependency is `liblzma` from vcpkg (port 5.8.4, 0BSD licence, same as
  the project). Link it through `find_package(liblzma CONFIG REQUIRED)` and
  `liblzma::liblzma`, PRIVATE to `splat_core`.
- No other new dependencies.

## Approaches considered (positions)

- **A. One shared grid with delta coding.** Best compression, but delta coding
  is deferred for now.
- **B. Chunks of 256 splats, each with its own bounding box, 11/10/11-bit
  positions.** Chosen. Simple, a fixed 4 bytes per position before LZMA, and
  precision follows local density.
- **C. Shared-grid deltas plus stored chunk boxes.** Rejected: the boxes repeat
  information the viewer can compute from the positions.

## File layout

All integers and floats are little-endian.

### Outer file

| Field | Type | Notes |
|---|---|---|
| magic | 4 bytes | `CGC\0` (`43 47 43 00`) |
| version | u32 | 1 |
| payload size | u64 | uncompressed payload size, for checking |
| LZMA stream | rest of file | `.lzma` "alone" format: 13-byte header (properties byte, u32 dictionary size, u64 uncompressed size) followed by raw LZMA1 data. Written with `lzma_alone_encoder`, read with `lzma_alone_decoder`. That encoder writes the size as unknown (all ones) and ends the data with an end marker, so decoders must handle the end marker; the real size is the outer `payload size`. |

The `.lzma` alone format is what every small reference LZMA decoder reads, which
keeps the browser decoder simple. Encoder preset: 9 with `LZMA_PRESET_EXTREME`,
unless measurements show a better `lc/lp/pb` setting (builder's call; record
the result).

### Payload (inside the LZMA stream)

Header, then columns in this order. "Planes" means a column of N-byte values is
stored as N consecutive arrays: all byte 0s (least significant), then all
byte 1s, and so on. Floats are split as their 32-bit patterns.

Quantization rules, used by every field below:

- **Unsigned, N bits, over min..max:** `q = round((v − min) / (max − min) · (2^N − 1))`,
  clamped; decode `min + q / (2^N − 1) · (max − min)`. If `max = min`, q = 0
  and the value decodes to `min`.
- **Symmetric, N bits, over ±r:** with `h = 2^(N−1) − 1`,
  `q = clamp(round(v / r · h), −h, h) + h` (codes 0 .. 2h, so 0 is exact);
  decode `(q − h) / h · r`. If `r = 0`, q = h and the value decodes to 0.
- Codes narrower than their storage sit in the low bits; unused high bits
  are 0.

1. **Payload header**
   - `count` u32: number of splats
   - `colorMin[3]`, `colorMax[3]` f32: range of the DC colour per channel
   - `shRange[3]` f32: symmetric quantization range ±r for SH bands 1, 2, 3
     (0 for bands no splat uses)
2. **Chunk table**, `ceil(count / 256)` entries, each 12 f32: position min xyz,
   position max xyz, log-scale min xyz, log-scale max xyz. One chunk-major
   array of `12 · chunks` floats (chunk 0's 12 values, then chunk 1's), stored
   as planes of 4 bytes.
3. **Positions**, u32 per splat: x in bits 0 to 10 (11 bits), y in bits 11 to 20
   (10 bits), z in bits 21 to 31 (11 bits), unsigned within the splat's chunk
   box. Planes of 4 bytes.
4. **Rotations**, u32 per splat, "smallest three": normalize the quaternion (a
   zero-length one becomes the identity), flip its sign so the
   largest-magnitude component is positive, drop that component, store its
   index (0 to 3, in w, x, y, z order) in bits 30 to 31 and the other three,
   in w, x, y, z order, in bits 0–9, 10–19 and 20–29, each symmetric 10-bit
   over ±1/√2. Planes of 4 bytes.
5. **Scales**, 3 × u8 per splat: log scale, unsigned 8-bit within the chunk's
   log-scale range. Stored as three arrays (all x, all y, all z).
6. **Colour and opacity**, 4 × u8 per splat, stored as four arrays (R, G, B, A):
   - colour = `0.28209479 · sh0 + 0.5`, unsigned 8-bit within
     `colorMin..colorMax` for that channel. The encoder sets that range to the
     0.1st and 99.9th percentiles of the channel and clamps outliers, so a
     few extreme values don't coarsen the steps for everyone.
   - opacity = `sigmoid(opacity)`, `q = clamp(floor(α · 256), 0, 255)`
7. **SH degree**, u8 per splat (0 to 3).
8. **SH bands**, for ℓ = 1, 2, 3 in turn, only for splats whose degree is at
   least ℓ, in splat order. Band ℓ has 2ℓ + 1 coefficients × 3 colour channels.
   Stored column by column: coefficient is the outer loop, channel the inner
   one, and each (coefficient, channel) pair is one array holding that value
   for every splat that has the band. Each value is a u8, symmetric over ±r
   for that band: 5 bits for band 1 and 4 bits for bands 2 and 3.

## Encoding steps

1. **Choose the SH degree per splat.** Real SH are orthonormal, so the RMS
   colour change a band ℓ adds over all view directions is
   `sqrt(Σ c² / 4π)` per channel, where the sum runs over that band's
   coefficients. Multiply by `sigmoid(opacity)` so faint splats are judged by
   what they actually contribute. Starting from the top band, drop band ℓ while
   the largest of the three channel values is below a threshold, default
   `1.5 / 255`, and every higher band has already been dropped. The threshold is
   a field in a `CgcOptions` struct.
2. **Morton order.** Quantize positions to 21 bits per axis within the scene's
   bounding box, interleave into a 63-bit code, and sort splats by it (stable
   sort).
3. **Chunks.** Consecutive runs of 256 splats in Morton order. The last chunk
   may be shorter. Compute the position and log-scale min/max per chunk. A
   chunk whose box is flat on one axis (min = max) must decode without dividing
   by zero.
4. **Quantize** each column as described above. For the SH
   range r of band ℓ, take the 99.9th percentile of |c| over all coefficients of
   splats that keep band ℓ, and clamp values outside it.
5. **Write the payload** with byte-plane splitting, then compress it with LZMA.

## Decoding

Decompress, check the magic, version and sizes, then rebuild a `GaussianCloud`:

- `shDegree` = the highest value in the SH degree column (0 for an empty
  cloud). Splats with a lower degree get zeros for the bands they don't have.
- Positions and scales are dequantized with the unsigned rule inside their
  chunk's ranges.
- Rotations are rebuilt to a unit quaternion (w, x, y, z): the dropped
  component is `sqrt(max(0, 1 − sum of squares of the others))`.
- Colour goes back to `sh0 = (colour − 0.5) / 0.28209479`.
- Opacity goes back to a logit, using `(q + 0.5) / 256` so it stays finite.
- Any malformed file (bad magic or version, wrong sizes, truncated stream,
  LZMA error) throws `std::runtime_error` with the path in the message, as
  `load_ply` does.

## Code changes

- `src/splat/cgc.hpp`, `src/splat/cgc.cpp`: `CgcOptions`, `save_cgc`,
  `load_cgc`, plus the pure helpers (SH degree choice, Morton code, rotation
  packing) exposed in a `detail` namespace or similar so they can be unit
  tested.
- `src/splat/gaussian_cloud.hpp/.cpp`: `save_cloud(cloud, path)` and
  `load_cloud(path)`, which pick CGC or PLY by extension.
- `src/cli/main.cpp`: `load_scene` uses `load_cloud`; `make-test-scene` uses
  `save_cloud`. Help text mentions `.cgc`.
- New CLI command `splat convert <in> <out>`, which loads with `load_cloud` and
  saves with `save_cloud`, so existing `.ply` scenes can be turned into `.cgc`
  without retraining. It prints the input and output sizes and how many splats
  ended up at each SH degree.
- `src/train/train.cpp`: the final save and checkpoints use the format chosen
  by `options.out`'s extension. The checkpoint's temporary path ends in
  `.tmp`, so the format must come from `options.out`, not from the temporary
  path (for example an explicit `save_cgc` / `save_ply` choice, or a helper
  that takes the format separately).
- `src/sfm/reconstruct.cpp` keeps writing `preview.ply`.
- `vcpkg.json`, `CMakeLists.txt`, `src/CMakeLists.txt`: add `liblzma`.
- `README.md`: a short section on CGC and `convert`.

## Testing

- Unit tests (`tests/test_cgc.cpp`):
  - Morton code: bit interleaving on known values, and sort order.
  - Rotation packing: random unit quaternions survive with small error (dot
    product above 0.999 in absolute value), including ones where w is not the
    largest component, and sign-flipped inputs.
  - SH degree choice: zero higher bands give degree 0; a strong band 3 keeps
    degree 3; a strong band 1 with weak bands 2 and 3 gives degree 1; low
    opacity lowers the degree.
  - Round trip on a synthetic cloud for SH degrees 0 to 3: same count; each
    attribute within its quantization step; decoded quaternions are unit
    length.
  - Edge cases: empty cloud, 1 splat, exactly 256 and 257 splats, every splat
    at the same position (flat chunk boxes).
  - Malformed files throw: wrong magic, wrong version, truncated file, a
    corrupted LZMA properties byte. LZMA1 has no checksum, so a flipped byte
    in the middle of the data may decode silently into wrong values; the
    decoder only guarantees it won't read out of bounds.
  - Non-finite input (NaN or Inf in any attribute): `save_cgc` throws before
    writing anything.
  - Extension dispatch: `.CGC` and `.cgc` both produce CGC; `.ply` stays PLY.
- Render check: render `make_test_scene` from the PLY data and from its CGC
  round trip with the existing offscreen renderer, and require PSNR above
  35 dB (adjust after the first measurement if needed, and record why).
- CLI tests: `make-test-scene` → `convert` to `.cgc` → `render` the `.cgc`.

## Left to the builder

- Exact `lzma` presets and `lc/lp/pb`, after a quick measurement.
- Internal structure of `cgc.cpp`.
- Whether `convert` also prints a quality number.

## Calls made without explicit agreement

- Single LZMA stream in `.lzma` alone format.
- Per-chunk ranges for position and scale; scene-wide range for colour.
- 10-bit smallest-three rotations; 8-bit scale, colour and opacity; 5/4/4-bit
  SH; SH range at the 99.9th percentile.
- SH degree threshold of 1.5/255 RMS, weighted by opacity.
- The `splat convert` command.
- Checkpoints follow the output file's format, but are compressed with a fast
  LZMA preset (a `CgcOptions` field), so they don't stall training. Only the
  final save uses the slow, strongest preset. The file format is the same.
- Colour range at the 0.1/99.9 percentiles instead of min/max.
- Exact quantization rules, bit and array orders, and the unknown-size LZMA
  header, written down after the review flagged them as open.
