# CGC version 2

Date: 2026-11-02
Replaces the layout in `2026-11-02-cgc-format-design.md` (version 1, still readable).

## Results

Measured on public scenes trained with the original 3DGS code to 30 000
iterations. PSNR compares renders of the file with renders of the source PLY,
over eight orbit views at 960x540 on a black background; "min" is the worst view,
always a close-up.

| Scene | Splats | PLY | v1 | v2 | v2 `--small` |
|---|---|---|---|---|---|
| bonsai | 1.24 M | 308.7 MB | 28.20 MB, 45.3 dB (min 44.0) | 11.17 MB, 42.6 dB (min 38.1) | 9.95 MB, 41.3 dB (min 35.8) |
| truck | 2.06 M | 510.0 MB | 46.94 MB, 45.7 dB (min 41.9) | 18.43 MB, 43.0 dB (min 36.8) | 16.43 MB, 40.9 dB (min 33.3) |
| bicycle | 6.13 M | 1520.7 MB | about 136 MB, 42.9 dB (min 41.5) | 54.52 MB, 40.4 dB (min 38.3) | 48.97 MB, 38.6 dB (min 35.9) |

The bicycle v1 figure comes from the experiment bench, which reproduced bonsai's
v1 file to within 1.5 %.

`web/cgc.js`, with its own LZMA decoder, decodes bonsai in about 1.9 s and truck in
about 3.1 s on one thread under Node, and agrees with the C++ decoder on both.

**Weak spot:** a 13 373-splat scene trained by this repo for 5000 iterations on
synthetic views goes from 0.37 MB and 41.0 dB (min 39.2) in v1 to 0.22 MB and 35.4 dB
(min 30.3). Few splats with strongly view-dependent colour leave the codebook too
little to share. No scene trained by this repo on real photos has been measured.

These are PSNR figures against the source splats, not against photographs, and
orbit views are not the training cameras. v2 gives up 2.5 to 3 dB of mean PSNR and 3
to 6 dB in the worst view against v1, in exchange for files 2.5 times smaller.

## Layout

All values are little endian. Every header field is 4 bytes.

File header, 36 bytes: magic `CGC\0`, u32 version = 2, u32 count, u32 SH degree,
u32 layer count, f32 centre[3], f32 radius.

Then one 92-byte header per layer, most important layer first:

| Offset | Field |
|---|---|
| 0 | u32 count |
| 4 | u32 position bits |
| 8 | u32 rotation bits |
| 12 | u32 alpha bits |
| 16 | u32 colour bits[3] |
| 28 | u32 codebook size K |
| 32 | f32 position low[3] |
| 44 | f32 position step |
| 48 | f32 scale low |
| 52 | f32 scale step |
| 56 | f32 gap 1 step |
| 60 | f32 gap 2 step |
| 64 | f32 colour low[3] |
| 76 | f32 colour step[3] |
| 88 | f32 codebook step |

Then, for each layer with count > 0, its columns in the order below. Each column
is a u32 compressed size followed by a `.lzma` "alone" stream (lc = lp = pb = 0,
size written as unknown, with an end marker). A column's raw size follows from
the headers. There is nothing after the last column.

With n = layer count and D = 3 * ((degree + 1)^2 - 1):

1. **Position gaps**, ceil(3 * position bits / 8) columns of n bytes, least
   significant byte first. Together they hold the gap to the previous Morton
   code (the first gap is from 0). Summing them gives the code; bit 3i is x's
   bit i, 3i + 1 is y's, 3i + 2 is z's. Then, per axis,
   `y = low + q * step`, `t = sign(y) * expm1(|y|)`, `position = t * radius + centre`.
2. **Rotation**, 3 columns of n bytes: x, y, z of the quaternion, zigzag coded
   (`0, -1, 1, -2, ...` as `0, 1, 2, 3, ...`). Multiply by
   `0.70710678 / (2^(bits - 1) - 1)`. `w = sqrt(max(0, 1 - x² - y² - z²))`.
3. **Scale**, 3 columns of n bytes, q0, q1, q2. Log scales are
   `s0 = low + q0 * step`, `s1 = s0 - q1 * gap1`, `s2 = s1 - q2 * gap2`.
4. **Colour**, 3 columns of n bytes: Y, Co, Cg, each `low + q * step`.
   `t = Y - Cg / 2`, `R = t + Co / 2`, `G = Y + Cg / 2`, `B = t - Co / 2`, and
   `sh0 = (colour - 0.5) / 0.28209479`.
5. **Alpha**, 1 column of n bytes: `alpha = (q + 0.5) / 2^bits`, and the stored
   opacity is its logit.

If D > 0:

6. **SH labels**, 2 columns of n bytes: low and high byte of an index below K.
7. **SH codebook**, 2 columns of K * D bytes: low and high byte of a zigzag coded
   value. Dimension is the outer loop, so `value(a, t)` is at `a * K + t`. Multiply
   by the codebook step. Each triple of dimensions is one SH coefficient (from
   coefficient 1 up) in Y, Co, Cg, turned to RGB with the formula in step 4, with no
   offset.

## Why each part

Measured with an experiment bench on bonsai and truck, one change at a time.
Figures are bytes per splat, against v1's 22.7.

- **SH codebook (11.2 -> 1.6).** One codebook over all 45 coefficients beat scalar
  quantization at any rate, per-band codebooks and luma/chroma codebooks. At 40.8 dB
  scalar costs 3.2, while a 16 k codebook reaches 41.4 dB for 1.7.
- **Weighting centres by opacity times area (+2 dB for no bytes).** The distortion
  that shows comes from large, opaque splats.
- **Sorted positions on one global grid (3.76 -> 1.8).** Only gaps between Morton
  codes are stored, which removes the entropy of the splats' order. The symlog
  warp keeps floaters from coarsening the scene. Chunks tied to splat size were
  worse (1.85 at 43.8 dB against 1.76 at 45.0 dB).
- **Canonical axes (2 bits, plus cheaper scales).** Sorting axes by size lets
  scales be stored as gaps. Flipping two axes multiplies the quaternion by i, j
  or k, so w can always be the largest, with no index and no decoder cost.
- **YCoCg colour (2.29 -> 1.37).** RGB deltas between neighbours do not help;
  trained colours are too noisy.
- **Importance layers (-12 to -20 %, no loss in the worst view at 40/80).** The
  faintest, smallest splats tolerate far coarser values.
- **lc = lp = pb = 0, LZMA preset 6.** Separate columns make contexts almost
  useless, and preset 9 extreme gained nothing here. Streams being independent lets a
  viewer decode them in parallel.

Tried and left out:

- **Dropping the least important splats.** Bonsai lost 0.3 dB for 30 % of the
  splats, but truck lost 4 dB for 10 %, so it is not safe as a default.
- Hilbert order, colour and alpha deltas, octahedral normal plus angle rotations,
  and coarser rotations for round splats: no gain or a worse trade.

## Encoder

- Layers cut at the 40th and 80th percentile of `opacity * s0 * s1` (70th and
  95th with `--small`), with these parameters from most to least important:

  | | Position | Rotation | Scale, gap 1, gap 2 steps | Alpha | YCoCg | K |
  |---|---|---|---|---|---|---|
  | Top | 15 | 8 | 0.05, 0.05, 0.1 | 7 | 7, 7, 7 | n / 16 |
  | Middle | 14 | 6 | 0.1, 0.1, 0.2 | 6 | 6, 6, 6 | n / 120 |
  | Low | 13 | 5 | 0.15, 0.15, 0.3 | 5 | 6, 5, 5 | n / 500 |

  K is at least n / 4 up to 1024, and at most 65 536.
- Log scales are clamped to at least -9, as in v1. The scale step grows if the
  largest log scale would not fit in 8 bits.
- Colour ranges skip the most extreme 0.01 % at each end (at least one splat in
  layers of 64 or more), so one outlier cannot coarsen a channel.
- Training checkpoints use `CgcOptions::fast`: at most 1024 codes, a smaller
  k-means sample and LZMA preset 1.
