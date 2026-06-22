// ─── Game Boy Camera palette (grayscale values) ───
export const GB_COLORS = [0, 82, 165, 255] as const;
export type GBColorValue = (typeof GB_COLORS)[number];

// ─── Screen geometry (in GB pixels) ───
export const SCREEN_W = 160;
export const SCREEN_H = 144;
export const FRAME_THICK = 16;
export const CAM_W = 128;
export const CAM_H = 112;

// Inner border outer-edge positions in GB pixel coords
export const INNER_TOP = FRAME_THICK - 1; // 15
export const INNER_BOT = FRAME_THICK + CAM_H; // 128
export const INNER_LEFT = FRAME_THICK - 1; // 15
export const INNER_RIGHT = FRAME_THICK + CAM_W; // 144

// ─── Pipeline step registry ───
export const STEP_ORDER = [
  "locate",
  "warp",
  "correct",
  "crop",
  "sample",
  "quantize",
] as const;
export type StepName = (typeof STEP_ORDER)[number];

// ─── Framework-agnostic image type ───
export interface GBImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

// ─── Pipeline API types ───
export interface PipelineResult {
  grayscale: GBImageData;
  intermediates?: {
    locate: GBImageData;
    warp: GBImageData;
    correct: GBImageData;
    crop: GBImageData;
    sample: GBImageData;
  };
  /**
   * Populated when `options.debug` is true. Contains diagnostic images,
   * structured per-step metrics, and a chronological log.
   */
  debug?: {
    images: Record<string, GBImageData>;
    log: string[];
    metrics: Record<string, Record<string, unknown>>;
  };
}

export interface PipelineOptions {
  scale?: number;
  debug?: boolean;
  onProgress?: (step: string, pct: number) => void | Promise<void>;
  /**
   * Run the {@link locate} step before {@link warp} to find the Game Boy
   * Screen within a full phone photo and produce an upright crop.
   *
   * Defaults to `true`. Set to `false` for inputs that are already cropped
   * and roughly upright (e.g. the existing `test-input/` and
   * `sample-pictures/` corpora) to skip the work.
   *
   * @default true
   */
  locate?: boolean;
}

// ─── Helpers ───

export function createGBImageData(width: number, height: number): GBImageData {
  return {
    data: new Uint8ClampedArray(width * height * 4),
    width,
    height,
  };
}

export function pixelIndex(img: GBImageData, x: number, y: number): number {
  return (y * img.width + x) * 4;
}

export function grayscaleToRGBA(
  gray: Uint8Array | Float32Array,
  width: number,
  height: number,
): GBImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < gray.length; i++) {
    const v = Math.round(gray[i]);
    const j = i * 4;
    data[j] = v;
    data[j + 1] = v;
    data[j + 2] = v;
    data[j + 3] = 255;
  }
  return { data, width, height };
}

export function rgbaToGrayscale(img: GBImageData): Float32Array {
  const gray = new Float32Array(img.width * img.height);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = img.data[i * 4];
  }
  return gray;
}
