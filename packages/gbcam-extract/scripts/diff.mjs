// Count differing pixels between two grayscale gbcam PNGs.
// Usage: node scripts/diff.mjs <a.png> <b.png>
import sharp from "sharp";
const [, , a, b] = process.argv;
const la = await sharp(a).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const lb = await sharp(b).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const A = la.data, B = lb.data;
const n = Math.min(A.length, B.length) / 4;
let diff = 0;
for (let i = 0; i < n; i++) if (A[i * 4] !== B[i * 4]) diff++;
console.log(`diff=${diff} / ${n} (${(100 * diff / n).toFixed(2)}%)  ${la.info.width}x${la.info.height} vs ${lb.info.width}x${lb.info.height}`);
