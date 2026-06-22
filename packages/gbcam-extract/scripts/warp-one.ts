/**
 * warp-one.ts — run the warp step alone on a single image (e.g. a locate
 * output) with a debug collector, and print the warp metrics. Diagnostic.
 *
 * Usage: pnpm warp-one -- <path-to-locate-or-cropped.png> [--scale 8]
 */
import sharp from "sharp";
import { initOpenCV } from "../src/init-opencv.js";
import { warp } from "../src/warp.js";
import { createDebugCollector } from "../src/debug.js";
import type { GBImageData } from "../src/common.js";

async function loadImage(filePath: string): Promise<GBImageData> {
  const { data, info } = await sharp(filePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const rgba = new Uint8ClampedArray(info.width * info.height * 4);
  rgba.set(data);
  return { data: rgba, width: info.width, height: info.height };
}

async function main() {
  const argv = process.argv.slice(2);
  const path = argv.find((a) => !a.startsWith("--"));
  const scaleArg = argv.indexOf("--scale");
  const scale = scaleArg >= 0 ? parseInt(argv[scaleArg + 1], 10) : 8;
  if (!path) {
    console.error("Usage: pnpm warp-one -- <path.png> [--scale 8]");
    process.exit(1);
  }
  const outArg = argv.indexOf("--out");
  const outPath = outArg >= 0 ? argv[outArg + 1] : undefined;
  await initOpenCV();
  const img = await loadImage(path);
  const dbg = createDebugCollector();
  const result = warp(img, { scale, debug: dbg });
  console.log(JSON.stringify(dbg.data.metrics.warp.subPixel, null, 1));
  if (outPath) {
    await sharp(Buffer.from(result.data.buffer), {
      raw: { width: result.width, height: result.height, channels: 4 },
    })
      .png()
      .toFile(outPath);
    console.log("wrote", outPath);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
