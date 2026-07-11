/**
 * debug-one.ts — run a single image through the full pipeline with debug
 * output, dumping the final image, intermediates, debug images, and
 * structured metrics JSON.
 *
 * Usage: node dist-scripts/debug-one.js <input-image> <output-dir> [--no-locate]
 */

import { resolve, join, basename, extname, dirname } from "path";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import sharp from "sharp";
import { initOpenCV } from "../src/init-opencv.js";
import { processPicture } from "../src/index.js";
import type { GBImageData } from "../src/common.js";

async function loadImage(filePath: string): Promise<GBImageData> {
  const img = sharp(filePath).rotate().removeAlpha().ensureAlpha();
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.length);
  return {
    data: new Uint8ClampedArray(rgba),
    width: info.width,
    height: info.height,
  };
}

async function saveImage(img: GBImageData, outPath: string): Promise<void> {
  const dir = dirname(outPath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  await sharp(Buffer.from(img.data.buffer), {
    raw: { width: img.width, height: img.height, channels: 4 },
  })
    .png()
    .toFile(outPath);
}

async function main() {
  const args = process.argv.slice(2).filter((a) => a !== "--");
  const noLocate = args.includes("--no-locate");
  const positional = args.filter((a) => !a.startsWith("--"));
  const inputPath = resolve(positional[0]);
  const outDir = resolve(positional[1]);
  const stem = basename(inputPath, extname(inputPath));

  await initOpenCV();
  const input = await loadImage(inputPath);
  const result = await processPicture(input, {
    scale: 8,
    debug: true,
    locate: !noLocate,
  });

  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
  await saveImage(result.grayscale, join(outDir, `${stem}_gbcam.png`));

  const debugDir = join(outDir, "debug");
  if (result.intermediates) {
    for (const [name, img] of Object.entries(result.intermediates)) {
      if (img) await saveImage(img as GBImageData, join(debugDir, `${stem}_${name}.png`));
    }
  }
  if (result.debug?.images) {
    for (const [name, img] of Object.entries(result.debug.images)) {
      await saveImage(img, join(debugDir, `${stem}_${name}.png`));
    }
  }
  if (result.debug) {
    writeFileSync(
      join(debugDir, `${stem}_debug.json`),
      JSON.stringify(
        { metrics: result.debug.metrics, log: result.debug.log },
        null,
        2,
      ),
      "utf-8",
    );
    for (const line of result.debug.log) console.log(line);
  }
  console.log(`\nWrote ${join(outDir, `${stem}_gbcam.png`)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
