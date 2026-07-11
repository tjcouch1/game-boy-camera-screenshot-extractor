/**
 * analyze-sample.ts — print per-palette nearest-target pixel support and
 * channel stats for a 128x112 _sample.png intermediate.
 *
 * Usage: node dist-scripts/analyze-sample.js <sample.png> [...]
 */

import sharp from "sharp";

const TARGETS: [string, number, number][] = [
  ["BK", 0, 0],
  ["DG", 148, 148],
  ["LG", 255, 148],
  ["WH", 255, 255],
];

async function analyze(path: string) {
  const { data, info } = await sharp(path)
    .removeAlpha()
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const n = info.width * info.height;
  const counts = [0, 0, 0, 0];
  const margins: number[][] = [[], [], [], []];
  const bByClass: number[][] = [[], [], [], []];
  for (let i = 0; i < n; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    let best = 0,
      bestD = Infinity,
      second = Infinity;
    for (let t = 0; t < 4; t++) {
      const d = Math.hypot(r - TARGETS[t][1], g - TARGETS[t][2]);
      if (d < bestD) {
        second = bestD;
        bestD = d;
        best = t;
      } else if (d < second) second = d;
    }
    counts[best]++;
    margins[best].push(second - bestD);
    bByClass[best].push(b);
  }
  console.log(`\n=== ${path} (${n}px) ===`);
  for (let t = 0; t < 4; t++) {
    const m = margins[t].sort((a, b2) => a - b2);
    const bb = bByClass[t].sort((a, b2) => a - b2);
    const q = (arr: number[], f: number) =>
      arr.length ? arr[Math.floor(f * (arr.length - 1))].toFixed(0) : "-";
    console.log(
      `${TARGETS[t][0]}: n=${counts[t]} (${((100 * counts[t]) / n).toFixed(2)}%)  margin p10/p50/p90=${q(m, 0.1)}/${q(m, 0.5)}/${q(m, 0.9)}  B p10/p50/p90=${q(bb, 0.1)}/${q(bb, 0.5)}/${q(bb, 0.9)}`,
    );
  }
}

async function main() {
  for (const p of process.argv.slice(2)) await analyze(p);
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
