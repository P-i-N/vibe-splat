// Checks a viewer page that `splat train` wrote: argv[2] is the .html, whose
// scene tag must hold a CGC file that web/cgc.js decodes.
import { readFileSync } from 'node:fs';
import { decodeCgc } from '../web/cgc.js';

const html = readFileSync(process.argv[2], 'latin1');
const match = /<script id="scene" type="application\/octet-stream" data-name="([^"]*)">([^<]*)<\/script>/.exec(html);
if (!match) throw new Error('no scene in the page');
const bytes = Buffer.from(match[2], 'base64');
const cloud = decodeCgc(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
if (cloud.count === 0) throw new Error('the page holds an empty scene');
if (!html.includes('function decodeCgc')) throw new Error('the page has no viewer');
console.log(`${match[1]}: ${cloud.count} splats, SH degree ${cloud.shDegree}, page ${(html.length / 1e6).toFixed(2)} MB`);
