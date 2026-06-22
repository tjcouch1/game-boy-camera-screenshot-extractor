// Find exact error pixel locations and their sample RGB values.
import * as path from "node:path";
import sharp from "sharp";

const TESTS = [
  "thing-1",
  "thing-2",
  "zelda-poster-1",
  "zelda-poster-2",
  "park-1",
  "zelda-poster-3",
  "thing-3",
  "bathhouse-1",
];
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

const PALETTE_LABEL: Record<number, string> = {
  0: "BK",
  82: "DG",
  165: "LG",
  255: "WH",
};

async function loadPNG(p: string) {
  const { data, info } = await sharp(p).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

const root = path.resolve(import.meta.dirname, "..", "..", "..");

for (const name of TESTS) {
  const outPath = path.join(root, "test-output", name, `${name}_gbcam.png`);
  const refPath = path.join(root, "test-input", REF_MAP[name]);
  const samplePath = path.join(
    root,
    "test-output",
    name,
    "debug",
    `${name}_sample.png`,
  );

  const out = await loadPNG(outPath);
  const ref = await loadPNG(refPath);
  const sample = await loadPNG(samplePath);

  console.log(`\n=== ${name} ===`);

  let count = 0;
  for (let y = 0; y < out.height; y++) {
    for (let x = 0; x < out.width; x++) {
      const o = (y * out.width + x) * 4;
      const ov = out.data[o];
      const rv = ref.data[o];
      if (ov !== rv) {
        const sR = sample.data[o];
        const sG = sample.data[o + 1];
        const sB = sample.data[o + 2];
        const neighbors: string[] = [];
        for (const [dx, dy] of [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ]) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < out.width && ny >= 0 && ny < out.height) {
            const ni = (ny * out.width + nx) * 4;
            const nLabel = PALETTE_LABEL[ref.data[ni]];
            const nR = sample.data[ni];
            const nG = sample.data[ni + 1];
            const nB = sample.data[ni + 2];
            neighbors.push(`${nLabel}(${nR},${nG},${nB})`);
          }
        }
        console.log(
          `  (${x},${y}) result=${PALETTE_LABEL[ov]} ref=${PALETTE_LABEL[rv]} ` +
            `RGB=(${sR},${sG},${sB})  neigh=${neighbors.join(" ")}`,
        );
        count++;
      }
    }
  }
  console.log(`  total errors: ${count}`);
}
