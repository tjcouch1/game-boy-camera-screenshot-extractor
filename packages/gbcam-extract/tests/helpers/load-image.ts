import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { GBImageData } from "../../src/common.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export async function loadImage(filePath: string): Promise<GBImageData> {
  const sharp = (await import("sharp")).default;
  // Auto-orient: applies any EXIF rotation so loaded pixels match visual
  // orientation. Phone photos (test-input-full/) often store landscape
  // images with a 180° EXIF rotation; without this, our detection runs in
  // storage-order coords while corners.json is in visual-order coords.
  const { data, info } = await sharp(filePath).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return {
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
    width: info.width,
    height: info.height,
  };
}

export function repoRoot(...segments: string[]): string {
  // From packages/gbcam-extract/tests/helpers/ -> repo root is 4 levels up
  return resolve(__dirname, "..", "..", "..", "..", ...segments);
}
