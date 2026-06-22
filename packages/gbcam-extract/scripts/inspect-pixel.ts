/**
 * Deep diagnostic: for a specific (test, x, y), print the pixel's RGB,
 * the empirical cluster centers (3D RGB), distances to each, and the
 * RGB of every neighbour within a radius. Used to understand what's
 * actually happening at the difficult pixels.
 */
import * as path from "node:path";
import sharp from "sharp";

const PALETTE_LABEL: Record<number, string> = {
  0: "BK",
  82: "DG",
  165: "LG",
  255: "WH",
};
const REF_MAP: Record<string, string> = {
  "thing-1": "thing-output-corrected.png",
  "thing-2": "thing-output-corrected.png",
  "thing-3": "thing-output-corrected.png",
  "zelda-poster-1": "zelda-poster-output-corrected.png",
  "zelda-poster-2": "zelda-poster-output-corrected.png",
  "zelda-poster-3": "zelda-poster-output-corrected.png",
  "park-1": "park-output-corrected.png",
  "bathhouse-1": "bathhouse-output-corrected.png",
};

async function loadPNG(p: string) {
  const { data, info } = await sharp(p)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

interface Probe {
  name: string;
  x: number;
  y: number;
}

const PROBES: Probe[] = [
  { name: "thing-1", x: 83, y: 70 },
  { name: "thing-1", x: 75, y: 72 },
  { name: "thing-1", x: 77, y: 72 },
  { name: "thing-1", x: 89, y: 77 },
  { name: "thing-2", x: 28, y: 21 },
  { name: "thing-2", x: 24, y: 23 },
  { name: "thing-2", x: 58, y: 27 },
  { name: "thing-2", x: 61, y: 28 },
  { name: "thing-2", x: 10, y: 44 },
  { name: "thing-2", x: 24, y: 44 },
  { name: "thing-2", x: 42, y: 47 },
  { name: "thing-2", x: 40, y: 51 },
  { name: "thing-2", x: 36, y: 61 },
  { name: "zelda-poster-1", x: 78, y: 86 },
  { name: "zelda-poster-1", x: 62, y: 92 },
  { name: "zelda-poster-1", x: 7, y: 80 },
  { name: "zelda-poster-2", x: 18, y: 9 },
  { name: "zelda-poster-2", x: 127, y: 72 },
  { name: "zelda-poster-2", x: 126, y: 77 },
  { name: "zelda-poster-2", x: 0, y: 85 },
];

const root = path.resolve(import.meta.dirname, "..", "..", "..");

async function inspect(p: Probe) {
  const outPath = path.join(root, "test-output", p.name, `${p.name}_gbcam.png`);
  const refPath = path.join(root, "test-input", REF_MAP[p.name]);
  const samplePath = path.join(
    root,
    "test-output",
    p.name,
    "debug",
    `${p.name}_sample.png`,
  );
  const out = await loadPNG(outPath);
  const ref = await loadPNG(refPath);
  const sample = await loadPNG(samplePath);

  const W = out.width;
  const H = out.height;

  // Empirical 3D cluster centers from reference (the "ideal" centers — what
  // each colour ACTUALLY looks like in this camera image's pixels).
  const cR = [0, 0, 0, 0];
  const cG = [0, 0, 0, 0];
  const cB = [0, 0, 0, 0];
  const cN = [0, 0, 0, 0];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const rv = ref.data[i];
      let pi: number;
      if (rv === 0) pi = 0;
      else if (rv === 82) pi = 1;
      else if (rv === 165) pi = 2;
      else pi = 3;
      cR[pi] += sample.data[i];
      cG[pi] += sample.data[i + 1];
      cB[pi] += sample.data[i + 2];
      cN[pi]++;
    }
  }
  for (let p2 = 0; p2 < 4; p2++) {
    cR[p2] /= cN[p2];
    cG[p2] /= cN[p2];
    cB[p2] /= cN[p2];
  }

  const pi = (p.y * W + p.x) * 4;
  const R = sample.data[pi];
  const G = sample.data[pi + 1];
  const B = sample.data[pi + 2];
  const result = PALETTE_LABEL[out.data[pi]];
  const refL = PALETTE_LABEL[ref.data[pi]];

  console.log(`\n=== ${p.name} (${p.x},${p.y}) result=${result} ref=${refL} ===`);
  console.log(
    `pixel RGB=(${R},${G},${B})`,
  );
  console.log(
    `ref-empirical cluster centers (3D, from ref labels in this image):`,
  );
  for (let k = 0; k < 4; k++) {
    console.log(
      `  ${["BK", "DG", "LG", "WH"][k]} = (R${cR[k].toFixed(1)}, G${cG[k].toFixed(1)}, B${cB[k].toFixed(1)})`,
    );
  }
  // 3D distance to each center
  for (let k = 0; k < 4; k++) {
    const d = (R - cR[k]) ** 2 + (G - cG[k]) ** 2 + (B - cB[k]) ** 2;
    console.log(
      `  d(${["BK", "DG", "LG", "WH"][k]}) = ${Math.sqrt(d).toFixed(1)}`,
    );
  }

  // Neighbour grid (5x5)
  console.log(`5x5 neighbourhood (label@RGB):`);
  for (let dy = -2; dy <= 2; dy++) {
    const row: string[] = [];
    for (let dx = -2; dx <= 2; dx++) {
      const nx = p.x + dx;
      const ny = p.y + dy;
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
        row.push("        --        ");
        continue;
      }
      const ni = (ny * W + nx) * 4;
      const lbl = PALETTE_LABEL[ref.data[ni]];
      const r = sample.data[ni];
      const g = sample.data[ni + 1];
      const b = sample.data[ni + 2];
      row.push(`${lbl}(${r.toString().padStart(3, " ")},${g.toString().padStart(3, " ")},${b.toString().padStart(3, " ")})`);
    }
    console.log("  " + row.join(" "));
  }
}

for (const p of PROBES) {
  await inspect(p);
}
