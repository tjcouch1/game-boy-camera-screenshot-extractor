import {
  type GBImageData,
  SCREEN_W,
  SCREEN_H,
  FRAME_THICK,
  CAM_W,
  CAM_H,
} from "./common.js";
import { type DebugCollector, cloneImage, strokeRect } from "./debug.js";

export interface CropOptions {
  scale?: number;
  debug?: DebugCollector;
  /**
   * Optional out-param: crop fills in the inner-border-to-white-frame
   * brightness ratio the orchestrator uses for processing-issue reporting
   * (works without `debug`). Healthy photos measure 0.78–0.96; a ratio near
   * or above 1 means the border reads as bright as the frame — a warp/
   * correct anomaly.
   */
  stats?: { borderToFrameRatio?: number };
}

/**
 * Crop step: remove the filmstrip frame, keeping only the 128x112 camera area.
 *
 * The input is a (SCREEN_W*scale) x (SCREEN_H*scale) perspective-corrected image.
 * The camera image starts at GB pixel (FRAME_THICK, FRAME_THICK) and is CAM_W x CAM_H
 * GB pixels, so in image-pixel coordinates:
 *   x: [FRAME_THICK*scale .. (FRAME_THICK+CAM_W)*scale)
 *   y: [FRAME_THICK*scale .. (FRAME_THICK+CAM_H)*scale)
 */
export function crop(input: GBImageData, options?: CropOptions): GBImageData {
  const scale = options?.scale ?? 8;
  const dbg = options?.debug;

  const expectedW = SCREEN_W * scale;
  const expectedH = SCREEN_H * scale;
  if (input.width !== expectedW || input.height !== expectedH) {
    throw new Error(
      `Unexpected input size ${input.width}x${input.height}; ` +
        `expected ${expectedW}x${expectedH} (scale=${scale})`,
    );
  }

  const x1 = FRAME_THICK * scale;
  const y1 = FRAME_THICK * scale;
  const outW = CAM_W * scale;
  const outH = CAM_H * scale;
  const x2 = x1 + outW;
  const y2 = y1 + outH;

  const data = new Uint8ClampedArray(outW * outH * 4);

  for (let y = 0; y < outH; y++) {
    const srcOffset = ((y1 + y) * input.width + x1) * 4;
    const dstOffset = y * outW * 4;
    data.set(
      input.data.subarray(srcOffset, srcOffset + outW * 4),
      dstOffset,
    );
  }

  // Brightness validation: inner border band should be darker than the white
  // frame. Computed unconditionally — the ratio also feeds processing-issue
  // reporting via options.stats.
  const borderMean = bandMean(input, x1 - scale, y1, scale, outH); // left border band
  const whiteMean = bandMean(input, 20 * scale, 1 * scale, 120 * scale, 3 * scale);
  if (options?.stats) {
    options.stats.borderToFrameRatio = borderMean / Math.max(whiteMean, 1);
  }

  if (dbg) {
    const ok = borderMean < whiteMean * 0.85;
    dbg.log(
      `[crop] inner-border mean=${borderMean.toFixed(1)} ` +
        `white-frame mean=${whiteMean.toFixed(1)} ` +
        `(ratio ${(borderMean / Math.max(whiteMean, 1)).toFixed(3)}, ${ok ? "OK" : "WARN"})`,
    );
    dbg.setMetrics("crop", {
      cameraRegion: { x: x1, y: y1, w: outW, h: outH },
      borderMean: Number(borderMean.toFixed(2)),
      whiteFrameMean: Number(whiteMean.toFixed(2)),
      borderToFrameRatio: Number((borderMean / Math.max(whiteMean, 1)).toFixed(4)),
      validation: ok ? "ok" : "warn",
    });

    // Overlay debug image: original with crop rectangle highlighted
    const overlay = cloneImage(input);
    const green: [number, number, number] = [0, 220, 0];
    const orange: [number, number, number] = [255, 140, 0];
    // Outer band marker (the inner-border region just outside the crop)
    strokeRect(
      overlay,
      x1 - scale,
      y1 - scale,
      outW + 2 * scale,
      outH + 2 * scale,
      orange,
      Math.max(2, Math.round(scale / 4)),
    );
    // Crop rectangle
    strokeRect(overlay, x1, y1, outW, outH, green, Math.max(3, Math.round(scale / 3)));
    // Suppress unused warnings for x2/y2 — they're conceptual and used for clarity above
    void x2;
    void y2;
    dbg.addImage("crop_a_region", overlay);
  }

  return { data, width: outW, height: outH };
}

/** Mean luminance (R+G+B / 3) over an axis-aligned band, clipped to image bounds. */
function bandMean(img: GBImageData, x: number, y: number, w: number, h: number): number {
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(img.width, Math.floor(x + w));
  const y1 = Math.min(img.height, Math.floor(y + h));
  let sum = 0;
  let count = 0;
  for (let py = y0; py < y1; py++) {
    for (let px = x0; px < x1; px++) {
      const i = (py * img.width + px) * 4;
      sum += img.data[i] + img.data[i + 1] + img.data[i + 2];
      count += 3;
    }
  }
  return count > 0 ? sum / count : 0;
}
