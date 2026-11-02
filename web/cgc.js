// A dependency-free CGC version 2 decoder for browsers and Node, including its
// own LZMA decoder. decodeCgc(arrayBuffer) returns flat typed arrays:
//   count, shDegree,
//   positions (3 per splat), scales (3, log), rotations (4, w x y z),
//   opacities (1, logit), sh (3 * (shDegree + 1)^2, [coefficient][rgb]).
// The byte layout is in docs/superpowers/specs/2026-11-02-cgc-v2-format.md.

const SH_C0 = 0.28209479177387814;
const ROTATION_RANGE = 0.70710678;

// Decodes a .lzma ("alone") stream with an end marker into `size` bytes.
export function lzmaDecode(input, size) {
  const props = input[0];
  if (props >= 225) throw new Error('bad LZMA properties');
  const pb = Math.floor(props / 45);
  const lp = Math.floor((props % 45) / 9);
  const lc = props % 9;
  const out = new Uint8Array(size);
  const init = 1024;
  const fill = (n) => new Uint16Array(n).fill(init);

  const isMatch = fill(192);
  const isRep = fill(12);
  const isRep0 = fill(12);
  const isRep1 = fill(12);
  const isRep2 = fill(12);
  const isRep0Long = fill(192);
  const literal = fill(0x300 << (lc + lp));
  const posSlot = fill(4 * 64);
  const posSpecial = fill(115);
  const align = fill(16);
  const newLength = () => ({ choice: fill(2), low: fill(16 * 8), mid: fill(16 * 8), high: fill(256) });
  const matchLength = newLength();
  const repLength = newLength();

  let pos = 18; // 13-byte header, then the range coder's zero byte and 4 code bytes
  if (input.length < 18) throw new Error('truncated LZMA stream');
  let range = 0xffffffff;
  let code = ((input[14] << 24) | (input[15] << 16) | (input[16] << 8) | input[17]) >>> 0;

  const normalize = () => {
    if (range < 0x1000000) {
      range = (range << 8) >>> 0;
      code = ((code << 8) | input[pos++]) >>> 0;
    }
  };
  const bit = (probs, i) => {
    normalize();
    const p = probs[i];
    const bound = (range >>> 11) * p;
    if (code < bound) {
      range = bound;
      probs[i] = p + ((2048 - p) >> 5);
      return 0;
    }
    range = (range - bound) >>> 0;
    code = (code - bound) >>> 0;
    probs[i] = p - (p >> 5);
    return 1;
  };
  const tree = (probs, offset, bits) => {
    let m = 1;
    for (let i = 0; i < bits; ++i) m = (m << 1) | bit(probs, offset + m);
    return m - (1 << bits);
  };
  const reverseTree = (probs, offset, bits) => {
    let m = 1;
    let sym = 0;
    for (let i = 0; i < bits; ++i) {
      const b = bit(probs, offset + m);
      m = (m << 1) | b;
      sym |= b << i;
    }
    return sym;
  };
  const length = (l, posState) => {
    if (bit(l.choice, 0) === 0) return tree(l.low, posState * 8, 3);
    if (bit(l.choice, 1) === 0) return 8 + tree(l.mid, posState * 8, 3);
    return 16 + tree(l.high, 0, 8);
  };

  const pbMask = (1 << pb) - 1;
  const lpMask = (1 << lp) - 1;
  let state = 0;
  let rep0 = 0;
  let rep1 = 0;
  let rep2 = 0;
  let rep3 = 0;
  let n = 0;
  for (;;) {
    const posState = n & pbMask;
    if (bit(isMatch, (state << 4) + posState) === 0) {
      if (n >= size) throw new Error('LZMA data is larger than expected');
      const prev = n > 0 ? out[n - 1] : 0;
      const base = 0x300 * (((n & lpMask) << lc) + (prev >> (8 - lc)));
      let sym = 1;
      if (state >= 7) {
        let match = out[n - rep0 - 1];
        do {
          const matchBit = (match >> 7) & 1;
          match <<= 1;
          const b = bit(literal, base + ((1 + matchBit) << 8) + sym);
          sym = (sym << 1) | b;
          if (matchBit !== b) break;
        } while (sym < 0x100);
      }
      while (sym < 0x100) sym = (sym << 1) | bit(literal, base + sym);
      out[n++] = sym & 0xff;
      state = state < 4 ? 0 : state < 10 ? state - 3 : state - 6;
      continue;
    }
    let len;
    if (bit(isRep, state) !== 0) {
      if (n === 0) throw new Error('corrupt LZMA data');
      if (bit(isRep0, state) === 0) {
        if (bit(isRep0Long, (state << 4) + posState) === 0) {
          state = state < 7 ? 9 : 11;
          if (n >= size) throw new Error('LZMA data is larger than expected');
          out[n] = out[n - rep0 - 1];
          ++n;
          continue;
        }
      } else {
        let dist;
        if (bit(isRep1, state) === 0) {
          dist = rep1;
        } else {
          if (bit(isRep2, state) === 0) {
            dist = rep2;
          } else {
            dist = rep3;
            rep3 = rep2;
          }
          rep2 = rep1;
        }
        rep1 = rep0;
        rep0 = dist;
      }
      len = length(repLength, posState);
      state = state < 7 ? 8 : 11;
    } else {
      rep3 = rep2;
      rep2 = rep1;
      rep1 = rep0;
      len = length(matchLength, posState);
      state = state < 7 ? 7 : 10;
      const slot = tree(posSlot, Math.min(len, 3) * 64, 6);
      if (slot < 4) {
        rep0 = slot;
      } else {
        const direct = (slot >> 1) - 1;
        let dist = ((2 | (slot & 1)) * 2 ** direct);
        if (slot < 14) {
          dist += reverseTree(posSpecial, dist - slot, direct);
        } else {
          let high = 0;
          for (let i = direct - 4; i > 0; --i) {
            normalize();
            range >>>= 1;
            const b = code >= range ? 1 : 0;
            if (b) code = (code - range) >>> 0;
            high = high * 2 + b;
          }
          dist += high * 16 + reverseTree(align, 0, 4);
        }
        rep0 = dist;
        if (rep0 === 0xffffffff) break; // end marker
      }
    }
    if (rep0 >= n) throw new Error('corrupt LZMA data');
    len += 2;
    if (len > size - n) throw new Error('LZMA data is larger than expected');
    const from = n - rep0 - 1;
    for (let i = 0; i < len; ++i) out[n + i] = out[from + i];
    n += len;
  }
  if (n !== size) throw new Error(`LZMA data has ${n} bytes, expected ${size}`);
  return out;
}

const unzigzag = (z) => (z & 1 ? -((z + 1) / 2) : z / 2);

export function decodeCgc(buffer) {
  const view = new DataView(buffer);
  const u32 = (at) => view.getUint32(at, true);
  const f32 = (at) => view.getFloat32(at, true);
  if (buffer.byteLength < 36 || u32(0) !== 0x00434743) throw new Error('not a CGC file');
  if (u32(4) !== 2) throw new Error(`unsupported CGC version ${u32(4)}`);
  const count = u32(8);
  const shDegree = u32(12);
  const layerCount = u32(16);
  const center = [f32(20), f32(24), f32(28)];
  const radius = f32(32);
  const coeffs = (shDegree + 1) * (shDegree + 1);
  const dims = (coeffs - 1) * 3;

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  const sh = new Float32Array(count * coeffs * 3);

  let at = 36 + layerCount * 92;
  const column = (size) => {
    const length = u32(at);
    const stream = new Uint8Array(buffer, at + 4, length);
    at += 4 + length;
    return lzmaDecode(stream, size);
  };

  let first = 0;
  for (let l = 0; l < layerCount; ++l) {
    const h = 36 + l * 92;
    const n = u32(h);
    if (n === 0) continue;
    const posBits = u32(h + 4);
    const rotBits = u32(h + 8);
    const alphaBits = u32(h + 12);
    const bookSize = u32(h + 28);
    const posLo = [f32(h + 32), f32(h + 36), f32(h + 40)];
    const posStep = f32(h + 44);
    const scaleLo = f32(h + 48);
    const scaleStep = f32(h + 52);
    const gap1Step = f32(h + 56);
    const gap2Step = f32(h + 60);
    const colorLo = [f32(h + 64), f32(h + 68), f32(h + 72)];
    const colorStep = [f32(h + 76), f32(h + 80), f32(h + 84)];
    const bookStep = f32(h + 88);

    // Positions: Morton gaps. Codes reach 63 bits, so they are summed and
    // split in 24-bit limbs that never leave a 32-bit integer.
    const planes = [];
    const planeCount = Math.ceil((posBits * 3) / 8);
    for (let p = 0; p < planeCount; ++p) planes.push(column(n));
    const limbs = [0, 0, 0];
    for (let j = 0; j < n; ++j) {
      let g0 = 0;
      let g1 = 0;
      let g2 = 0;
      for (let p = 0; p < planeCount; ++p) {
        const shifted = planes[p][j] << (8 * (p % 3));
        if (p < 3) g0 |= shifted;
        else if (p < 6) g1 |= shifted;
        else g2 |= shifted;
      }
      limbs[0] += g0;
      limbs[1] += g1 + (limbs[0] >>> 24);
      limbs[0] &= 0xffffff;
      limbs[2] += g2 + (limbs[1] >>> 24);
      limbs[1] &= 0xffffff;
      const q = [0, 0, 0];
      for (let b = 0; b < posBits; ++b) {
        for (let a = 0; a < 3; ++a) {
          const index = b * 3 + a;
          q[a] += ((limbs[Math.floor(index / 24)] >>> (index % 24)) & 1) * 2 ** b;
        }
      }
      for (let a = 0; a < 3; ++a) {
        const y = posLo[a] + q[a] * posStep;
        positions[(first + j) * 3 + a] = Math.sign(y) * Math.expm1(Math.abs(y)) * radius + center[a];
      }
    }

    const rot = [column(n), column(n), column(n)];
    const scale = [column(n), column(n), column(n)];
    const color = [column(n), column(n), column(n)];
    const alpha = column(n);
    const rotationStep = ROTATION_RANGE / ((1 << (rotBits - 1)) - 1);
    const levels = 2 ** alphaBits;
    for (let j = 0; j < n; ++j) {
      const out = first + j;
      const x = unzigzag(rot[0][j]) * rotationStep;
      const y = unzigzag(rot[1][j]) * rotationStep;
      const z = unzigzag(rot[2][j]) * rotationStep;
      rotations[out * 4] = Math.sqrt(Math.max(0, 1 - x * x - y * y - z * z));
      rotations[out * 4 + 1] = x;
      rotations[out * 4 + 2] = y;
      rotations[out * 4 + 3] = z;
      const s0 = scaleLo + scale[0][j] * scaleStep;
      const s1 = s0 - scale[1][j] * gap1Step;
      scales[out * 3] = s0;
      scales[out * 3 + 1] = s1;
      scales[out * 3 + 2] = s1 - scale[2][j] * gap2Step;
      const Y = colorLo[0] + color[0][j] * colorStep[0];
      const Co = colorLo[1] + color[1][j] * colorStep[1];
      const Cg = colorLo[2] + color[2][j] * colorStep[2];
      const t = Y - 0.5 * Cg;
      sh[out * coeffs * 3] = (t + 0.5 * Co - 0.5) / SH_C0;
      sh[out * coeffs * 3 + 1] = (Y + 0.5 * Cg - 0.5) / SH_C0;
      sh[out * coeffs * 3 + 2] = (t - 0.5 * Co - 0.5) / SH_C0;
      const a = (alpha[j] + 0.5) / levels;
      opacities[out] = Math.log(a / (1 - a));
    }

    if (dims > 0) {
      const labelLow = column(n);
      const labelHigh = column(n);
      const bookLow = column(bookSize * dims);
      const bookHigh = column(bookSize * dims);
      const book = new Float32Array(bookSize * dims);
      for (let t = 0; t < bookSize; ++t) {
        const value = (a) => unzigzag(bookLow[a * bookSize + t] | (bookHigh[a * bookSize + t] << 8)) * bookStep;
        for (let a = 0; a < dims; a += 3) {
          const Y = value(a);
          const Co = value(a + 1);
          const Cg = value(a + 2);
          const u = Y - 0.5 * Cg;
          book[t * dims + a] = u + 0.5 * Co;
          book[t * dims + a + 1] = Y + 0.5 * Cg;
          book[t * dims + a + 2] = u - 0.5 * Co;
        }
      }
      for (let j = 0; j < n; ++j) {
        const label = labelLow[j] | (labelHigh[j] << 8);
        if (label >= bookSize) throw new Error('SH label outside the codebook');
        sh.set(book.subarray(label * dims, (label + 1) * dims), ((first + j) * coeffs + 1) * 3);
      }
    }
    first += n;
  }
  if (first !== count) throw new Error('layer sizes do not add up');
  if (at !== buffer.byteLength) throw new Error('file has unexpected trailing data');
  return { count, shDegree, positions, scales, rotations, opacities, sh };
}
