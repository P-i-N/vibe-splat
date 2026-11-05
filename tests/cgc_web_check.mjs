// Checks web/cgc.js against the C++ decoder: argv[2] is a .cgc file and argv[3]
// the .ply that `splat convert` decoded it to.
import { readFileSync } from 'node:fs';
import { decodeCgc } from '../web/cgc.js';

const file = readFileSync(process.argv[2]);
const start = performance.now();
const cloud = decodeCgc(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const milliseconds = performance.now() - start;

const ply = readFileSync(process.argv[3]);
const headerEnd = ply.indexOf('end_header\n') + 'end_header\n'.length;
const header = ply.toString('latin1', 0, headerEnd);
const names = [...header.matchAll(/property float (\S+)/g)].map((m) => m[1]);
const count = Number(/element vertex (\d+)/.exec(header)[1]);
if (count !== cloud.count) throw new Error(`count ${cloud.count}, expected ${count}`);

const data = ply.subarray(headerEnd);
const value = (i, name) => data.readFloatLE((i * names.length + names.indexOf(name)) * 4);
const coeffs = (cloud.shDegree + 1) * (cloud.shDegree + 1);
let worst = 0;
const compare = (actual, expected) => {
  worst = Math.max(worst, Math.abs(actual - expected) / Math.max(1, Math.abs(expected)));
};
for (let i = 0; i < count; ++i) {
  for (let a = 0; a < 3; ++a) {
    compare(cloud.positions[i * 3 + a], value(i, 'xyz'[a]));
    compare(cloud.scales[i * 3 + a], value(i, `scale_${a}`));
    compare(cloud.sh[i * coeffs * 3 + a], value(i, `f_dc_${a}`));
  }
  for (let a = 0; a < 4; ++a) compare(cloud.rotations[i * 4 + a], value(i, `rot_${a}`));
  compare(cloud.opacities[i], value(i, 'opacity'));
  for (let r = 0; r < (coeffs - 1) * 3; ++r) {
    // PLY stores f_rest by channel, then by coefficient.
    const k = 1 + (r % (coeffs - 1));
    const c = Math.floor(r / (coeffs - 1));
    compare(cloud.sh[(i * coeffs + k) * 3 + c], value(i, `f_rest_${r}`));
  }
}
if (worst > 1e-3) throw new Error(`largest relative difference ${worst}`);

// web/viewer.html carries its own copy of the decoder, since pages opened from
// file:// cannot import modules. It must stay identical.
const unix = (path) => readFileSync(new URL(path, import.meta.url), 'utf8').replaceAll('\r\n', '\n');
if (!unix('../web/viewer.html').includes(unix('../web/cgc.js').trimEnd())) {
  throw new Error('web/viewer.html does not contain web/cgc.js verbatim; paste it in again');
}
console.log(`${count} splats decoded in ${milliseconds.toFixed(0)} ms, largest relative difference ${worst.toExponential(2)}`);
