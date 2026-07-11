export type {
  GBImageData,
  PipelineResult,
  PipelineOptions,
  GBColorValue,
  StepName,
} from "./common.js";
export {
  GB_COLORS,
  STEP_ORDER,
  CAM_W,
  CAM_H,
  SCREEN_W,
  SCREEN_H,
  createGBImageData,
} from "./common.js";
export { initOpenCV } from "./init-opencv.js";
export { applyPalette } from "./palette.js";
export { locate } from "./locate.js";
export { warp } from "./warp.js";
export { correct } from "./correct.js";
export { crop } from "./crop.js";
export { sample } from "./sample.js";
export { quantize } from "./quantize.js";
export type { PaletteEntry } from "./data/palettes-generated.js";
export {
  MAIN_PALETTES,
  ADDITIONAL_PALETTES,
  FUN_PALETTES,
} from "./data/palettes-generated.js";
export type { Frame } from "./frames/types.js";
export {
  splitSheet,
  loadIndividualFrame,
  dedupeFrames,
  appendDeduped,
  frameFingerprint,
  composeFrame,
} from "./frames/index.js";

import type { GBImageData, PipelineResult, PipelineOptions } from "./common.js";
import { locate } from "./locate.js";
import { warp } from "./warp.js";
import { correct } from "./correct.js";
import { crop } from "./crop.js";
import { sample } from "./sample.js";
import { quantize } from "./quantize.js";
import { createDebugCollector } from "./debug.js";

export async function processPicture(
  input: GBImageData,
  options?: PipelineOptions,
): Promise<PipelineResult> {
  const scale = options?.scale ?? 8;
  const debug = options?.debug ?? false;
  const runLocate = options?.locate ?? true;
  const onProgress = options?.onProgress;

  const collector = debug ? createDebugCollector() : undefined;

  // Awaiting onProgress lets a caller return a Promise (e.g. a setTimeout(0)
  // yield) so the browser can repaint between synchronous pipeline steps.
  // Sync callers that return void produce a no-op `await undefined`.
  await onProgress?.("locate", 0);
  const located = runLocate ? locate(input, { debug: collector }) : input;
  await onProgress?.("locate", 100);

  await onProgress?.("warp", 0);
  const warped = warp(located, { scale, debug: collector });
  await onProgress?.("warp", 100);

  await onProgress?.("correct", 0);
  const corrected = correct(warped, { scale, debug: collector });
  await onProgress?.("correct", 100);

  await onProgress?.("crop", 0);
  const cropped = crop(corrected, { scale, debug: collector });
  await onProgress?.("crop", 100);

  await onProgress?.("sample", 0);
  let sampled = sample(cropped, { scale, debug: collector });
  await onProgress?.("sample", 100);

  await onProgress?.("quantize", 0);
  const stats: { valleyClamped?: boolean } = {};
  let quantized = quantize(sampled, {
    corrected,
    warped,
    scale,
    debug: collector,
    stats,
  });
  // Blur retry: when quantize reports the blur-filled-gap signature (the
  // LG/WH G-valley landed implausibly high), the image's content rows are
  // likely also vertically misphased — re-run sample with content-driven
  // row-phase correction and quantize again. Sharp images never take this
  // path, so it cannot regress them; on the blurred d-1 it cuts the diff
  // count by another ~4x.
  if (stats.valleyClamped) {
    collector?.log(
      "[pipeline] blur signature detected (clamped G-valley) — re-running sample with row-phase correction",
    );
    sampled = sample(cropped, { scale, rowPhase: true, debug: collector });
    quantized = quantize(sampled, {
      corrected,
      warped,
      scale,
      debug: collector,
    });
  }
  await onProgress?.("quantize", 100);

  const result: PipelineResult = { grayscale: quantized };
  if (debug) {
    result.intermediates = {
      locate: located,
      warp: warped,
      correct: corrected,
      crop: cropped,
      sample: sampled,
    };
    if (collector) {
      result.debug = collector.data;
    }
  }
  return result;
}
