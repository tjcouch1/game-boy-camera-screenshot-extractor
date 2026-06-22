/**
 * correct.ts — Front-light brightness gradient compensation
 *
 * Ported from gbcam_correct.py (972 lines).
 *
 * The GBA SP uses a side-mounted front-light that creates a smooth 2D brightness
 * gradient across the screen. The effect is affine per pixel: both the black floor
 * and white ceiling shift together.
 *
 * Algorithm:
 *   1. White surface estimation — sample the 4 filmstrip frame strips (top/bottom/
 *      left/right), compute 85th-percentile brightness per GB-pixel block, drop
 *      blocks below 75% of median, fit a degree-2 bivariate polynomial.
 *   2. Dark surface estimation (Coons patch) — sample the 4 inner border bands,
 *      smooth with uniform_filter1d, build Coons bilinear patch.
 *   3. Per-pixel affine correction — gain = (white - dark) / (255 - 82),
 *      offset = dark - gain * 82, corrected = clip(round((observed - offset) / gain), 0, 255)
 *   4. Iterative refinement — classify corrected pixels in [60, 110] as dark-gray,
 *      use their uncorrected values as interior calibration points, refit degree-4
 *      polynomial for dark surface, re-correct.
 */

import {
  type GBImageData,
  SCREEN_W,
  SCREEN_H,
  FRAME_THICK,
  CAM_W,
  CAM_H,
  INNER_TOP,
  INNER_BOT,
  INNER_LEFT,
  INNER_RIGHT,
  createGBImageData,
} from "./common.js";
import {
  type DebugCollector,
  cropImage,
  hstack,
  renderHeatmap,
} from "./debug.js";
import { FRAME_MASK, FRAME_MASK_W, FRAME_MASK_H } from "./frame-mask.js";
import { fitBivariateSurface } from "./poly-surface.js";

// ─── Constants ───

const TRUE_DARK = 82;
const TRUE_WHITE = 255;

// ─── Public interface ───

export interface CorrectOptions {
  scale?: number;
  polyDegree?: number;
  darkSmooth?: number;
  refinePasses?: number;
  debug?: DebugCollector;
}

/**
 * Correct step: compensate for front-light brightness gradient.
 *
 * Input: warped image at (SCREEN_W * scale) x (SCREEN_H * scale) px.
 * Output: same dimensions, brightness-normalised grayscale.
 */
export function correct(
  input: GBImageData,
  options?: CorrectOptions,
): GBImageData {
  const scale = options?.scale ?? 8;
  const polyDegree = options?.polyDegree ?? 2;
  const darkSmooth = options?.darkSmooth ?? 13;
  const refinePasses = options?.refinePasses ?? 1;
  const dbg = options?.debug;

  const expectedW = SCREEN_W * scale;
  const expectedH = SCREEN_H * scale;
  if (input.width !== expectedW || input.height !== expectedH) {
    throw new Error(
      `Unexpected input size ${input.width}x${input.height}; ` +
        `expected ${expectedW}x${expectedH} (scale=${scale})`,
    );
  }

  const W = input.width;
  const H = input.height;

  // Extract RGB channels from RGBA input
  // The warp output is color RGBA from the input photo
  const chR = new Float32Array(H * W);
  const chG = new Float32Array(H * W);
  const chB = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) {
    const idx = i * 4;
    chR[i] = input.data[idx];
    chG[i] = input.data[idx + 1];
    chB[i] = input.data[idx + 2];
  }

  // ── Perform per-channel correction ──

  // R channel: white=255, dark=148 (DG.R)
  const {
    ys: whiteYsR,
    xs: whiteXsR,
    vs: whiteVsR,
  } = collectWhiteSamples(chR, W, H, scale);
  const whiteSurfaceR = fitSurface(whiteYsR, whiteXsR, whiteVsR, H, W, polyDegree);
  const { left: leftR, right: rightR, top: topR, bot: botR } = collectDarkSamples(
    chR,
    W,
    H,
    scale,
  );
  let darkSurfaceR = buildDarkSurface(leftR, rightR, topR, botR, H, W, scale, darkSmooth);
  let correctedR = applyCorrectionChannel(chR, whiteSurfaceR, darkSurfaceR, W, H, 255, 148);

  // G channel: white=255, dark=148 (DG.G)
  const {
    ys: whiteYsG,
    xs: whiteXsG,
    vs: whiteVsG,
  } = collectWhiteSamples(chG, W, H, scale);
  const whiteSurfaceG = fitSurface(whiteYsG, whiteXsG, whiteVsG, H, W, polyDegree);
  const { left: leftG, right: rightG, top: topG, bot: botG } = collectDarkSamples(
    chG,
    W,
    H,
    scale,
  );
  let darkSurfaceG = buildDarkSurface(leftG, rightG, topG, botG, H, W, scale, darkSmooth);
  let correctedG = applyCorrectionChannel(chG, whiteSurfaceG, darkSurfaceG, W, H, 255, 148);

  // B channel: kept raw (downstream B reclassify uses empirical means
  // computed from current labels). Earlier 3-anchor piecewise-linear
  // attempts collapsed the camera-area B discriminator at saturation; the
  // raw range gives the existing reclassify what it expects.
  const correctedB = new Float32Array(chB);

  if (dbg) {
    dbg.log(
      `[correct] R: white samples kept=${whiteVsR.length} ` +
        `(median ${median(whiteVsR).toFixed(1)}, range ` +
        `${Math.min(...whiteVsR).toFixed(0)}–${Math.max(...whiteVsR).toFixed(0)})`,
    );
    dbg.log(
      `[correct] G: white samples kept=${whiteVsG.length} ` +
        `(median ${median(whiteVsG).toFixed(1)}, range ` +
        `${Math.min(...whiteVsG).toFixed(0)}–${Math.max(...whiteVsG).toFixed(0)})`,
    );
    dbg.log(
      `[correct] white surface R: ${surfRange(whiteSurfaceR)}; ` +
        `G: ${surfRange(whiteSurfaceG)}`,
    );
    dbg.log(
      `[correct] dark surface  R: ${surfRange(darkSurfaceR)}; ` +
        `G: ${surfRange(darkSurfaceG)}`,
    );
  }

  // ── Iterative refinement (optional: can apply per channel) ──
  let calCountR = 0;
  let calCountG = 0;
  for (let pass = 0; pass < refinePasses; pass++) {
    const refinedR = refinePassChannel(
      correctedR,
      chR,
      whiteSurfaceR,
      darkSurfaceR,
      W,
      H,
      scale,
      darkSmooth,
      255,
      148,
    );
    if (refinedR !== null) {
      darkSurfaceR = refinedR.darkSurface;
      correctedR = refinedR.corrected;
      calCountR = refinedR.calCount;
    }

    const refinedG = refinePassChannel(
      correctedG,
      chG,
      whiteSurfaceG,
      darkSurfaceG,
      W,
      H,
      scale,
      darkSmooth,
      255,
      148,
    );
    if (refinedG !== null) {
      darkSurfaceG = refinedG.darkSurface;
      correctedG = refinedG.corrected;
      calCountG = refinedG.calCount;
    }
  }

  if (dbg && refinePasses > 0) {
    dbg.log(
      `[correct] interior DG calibration: R=${calCountR}px G=${calCountG}px`,
    );
  }

  // ── Build output RGBA ──
  const output = createGBImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    const j = i * 4;
    output.data[j] = Math.max(0, Math.min(255, Math.round(correctedR[i])));
    output.data[j + 1] = Math.max(0, Math.min(255, Math.round(correctedG[i])));
    output.data[j + 2] = Math.max(0, Math.min(255, Math.round(correctedB[i])));
    output.data[j + 3] = 255;
  }

  if (dbg) {
    // a — Side-by-side camera region: input | output
    const camX = FRAME_THICK * scale;
    const camY = FRAME_THICK * scale;
    const camW = CAM_W * scale;
    const camH = CAM_H * scale;
    const beforeCam = cropImage(input, camX, camY, camW, camH);
    const afterCam = cropImage(output, camX, camY, camW, camH);
    dbg.addImage("correct_a_before_after", hstack(beforeCam, afterCam));

    // b — White surface heatmap (avg of R and G channels)
    const whiteAvg = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      whiteAvg[i] = (whiteSurfaceR[i] + whiteSurfaceG[i]) / 2;
    }
    dbg.addImage("correct_b_white_surface", renderHeatmap(whiteAvg, W, H));

    // c — Dark surface heatmap (avg of R and G channels)
    const darkAvg = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      darkAvg[i] = (darkSurfaceR[i] + darkSurfaceG[i]) / 2;
    }
    dbg.addImage("correct_c_dark_surface", renderHeatmap(darkAvg, W, H));

    // Frame post-correction p85 — verifies the correction landed where expected
    const framePost = framePost85(output);
    dbg.log(
      `[correct] frame post-correction p85: ` +
        `R=${framePost.R.toFixed(0)} G=${framePost.G.toFixed(0)} B=${framePost.B.toFixed(0)} ` +
        `(target #FFFFA5 = R255 G255 B165)`,
    );
    dbg.log(
      `[correct] camera region mean: ` +
        cameraRegionMean(output, scale)
          .map((v, i) => `${"RGB"[i]}=${v.toFixed(1)}`)
          .join(" "),
    );

    dbg.setMetrics("correct", {
      whiteSamples: { R: whiteVsR.length, G: whiteVsG.length },
      whiteSurfaceRange: {
        R: [Number(min(whiteSurfaceR).toFixed(2)), Number(max(whiteSurfaceR).toFixed(2))],
        G: [Number(min(whiteSurfaceG).toFixed(2)), Number(max(whiteSurfaceG).toFixed(2))],
      },
      darkSurfaceRange: {
        R: [Number(min(darkSurfaceR).toFixed(2)), Number(max(darkSurfaceR).toFixed(2))],
        G: [Number(min(darkSurfaceG).toFixed(2)), Number(max(darkSurfaceG).toFixed(2))],
      },
      dgCalibrationPixels: { R: calCountR, G: calCountG },
      framePostCorrectionP85: {
        R: Math.round(framePost.R),
        G: Math.round(framePost.G),
        B: Math.round(framePost.B),
      },
    });
  }

  return output;
}

// ─── Debug helpers (correct step) ───

function median(values: number[] | Float64Array | Float32Array): number {
  const arr = Array.from(values).sort((a, b) => a - b);
  const n = arr.length;
  if (n === 0) return 0;
  if (n % 2 === 0) return (arr[n / 2 - 1] + arr[n / 2]) / 2;
  return arr[(n - 1) / 2];
}

function min(arr: Float32Array): number {
  let m = Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
  return m;
}

function max(arr: Float32Array): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

function surfRange(arr: Float32Array): string {
  return `${min(arr).toFixed(0)}–${max(arr).toFixed(0)}`;
}

/** Compute p85 of each RGB channel across the top filmstrip strip blocks. */
function framePost85(img: GBImageData): { R: number; G: number; B: number } {
  // Reuse logic from Python: top strip GB rows 0..INNER_TOP-1, cols 10..SCREEN_W-10
  // We don't have scale here directly, derive from image width:
  const scale = img.width / SCREEN_W;
  const result = { R: 0, G: 0, B: 0 };
  for (const ch of [0, 1, 2] as const) {
    const vals: number[] = [];
    for (let gy = 0; gy < INNER_TOP; gy++) {
      for (let gx = 10; gx < SCREEN_W - 10; gx++) {
        const block: number[] = [];
        const y1 = gy * scale;
        const y2 = (gy + 1) * scale;
        const x1 = gx * scale;
        const x2 = (gx + 1) * scale;
        for (let y = y1; y < y2; y++) {
          for (let x = x1; x < x2; x++) {
            block.push(img.data[(y * img.width + x) * 4 + ch]);
          }
        }
        if (block.length > 0) {
          block.sort((a, b) => a - b);
          const idx = Math.floor(0.85 * (block.length - 1));
          vals.push(block[idx]);
        }
      }
    }
    if (vals.length > 0) {
      const med = median(vals);
      const filtered = vals.filter((v) => v > 0.5 * med);
      const v = filtered.length > 0 ? median(filtered) : med;
      if (ch === 0) result.R = v;
      else if (ch === 1) result.G = v;
      else result.B = v;
    }
  }
  return result;
}

function cameraRegionMean(img: GBImageData, scale: number): [number, number, number] {
  const x0 = FRAME_THICK * scale;
  const y0 = FRAME_THICK * scale;
  const x1 = (FRAME_THICK + CAM_W) * scale;
  const y1 = (FRAME_THICK + CAM_H) * scale;
  let sR = 0, sG = 0, sB = 0;
  let n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * img.width + x) * 4;
      sR += img.data[i];
      sG += img.data[i + 1];
      sB += img.data[i + 2];
      n++;
    }
  }
  return n > 0 ? [sR / n, sG / n, sB / n] : [0, 0, 0];
}

// ─── Frame-mask per-colour sample collection ───

/**
 * Sample one channel of the warped image at every frame-mask position of
 * the requested colour. Returns scattered (ys, xs, vs) suitable for
 * fitBivariateSurface. Sampling uses the inner block (excluding 1-pixel
 * borders) at GB-pixel resolution.
 */
function collectFrameColorSamples(
  channel: Float32Array,
  scale: number,
  color: number,
): { ys: number[]; xs: number[]; vs: number[] } {
  const ys: number[] = [];
  const xs: number[] = [];
  const vs: number[] = [];
  const half = Math.floor(scale / 2);
  const margin = Math.max(1, Math.floor(scale / 4));
  const W = SCREEN_W * scale;
  for (let fy = 0; fy < FRAME_MASK_H; fy++) {
    for (let fx = 0; fx < FRAME_MASK_W; fx++) {
      if (FRAME_MASK[fy * FRAME_MASK_W + fx] !== color) continue;
      // Average over the inner block (avoid border bleed)
      const y1 = fy * scale + margin;
      const y2 = (fy + 1) * scale - margin;
      const x1 = fx * scale + margin;
      const x2 = (fx + 1) * scale - margin;
      let sum = 0;
      let count = 0;
      for (let y = y1; y < y2; y++) {
        const rowBase = y * W;
        for (let x = x1; x < x2; x++) {
          sum += channel[rowBase + x];
          count++;
        }
      }
      if (count > 0) {
        ys.push(fy * scale + half);
        xs.push(fx * scale + half);
        vs.push(sum / count);
      }
    }
  }
  return { ys, xs, vs };
}

/**
 * Piecewise-linear B-channel correction with 3 anchors per location.
 *
 *   obs ≤ bkB        →  0
 *   bkB < obs ≤ whB  →  linear interp 0 → 165
 *   whB < obs ≤ dgB  →  linear interp 165 → 220  (DG target lowered from
 *                                                  255 to leave headroom)
 *   obs > dgB        →  linear extrap from 220 with same slope (no clamp)
 *
 * The DG target is 220 rather than the palette 255 to leave 35 units of
 * headroom for camera-area DG pixels — their observed B sits above the
 * polynomial's DG-anchor surface estimate (the frame anchors live on 1-px
 * dash edges with PSF bleed; camera-area DG has less bleed and higher B).
 * Without headroom, camera DG saturates to 255 alongside any PSF-bleed
 * warm pixel, destroying the B discriminator.
 */
function applyThreeAnchorB(
  channel: Float32Array,
  bkSurface: Float32Array,
  whSurface: Float32Array,
  dgSurface: Float32Array,
  W: number,
  H: number,
): Float32Array {
  const out = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) {
    const v = channel[i];
    const bk = bkSurface[i];
    const wh = whSurface[i];
    const dg = dgSurface[i];
    let mapped: number;
    if (wh - bk < 5 || dg - wh < 5) {
      mapped = v;
    } else if (v <= bk) {
      const slope = 165 / (wh - bk);
      mapped = 0 - slope * (bk - v);
    } else if (v <= wh) {
      mapped = ((v - bk) / (wh - bk)) * 165;
    } else if (v <= dg) {
      mapped = 165 + ((v - wh) / (dg - wh)) * (220 - 165);
    } else {
      const slope = (220 - 165) / (dg - wh);
      mapped = 220 + (v - dg) * slope;
    }
    out[i] = Math.max(0, Math.min(255, mapped));
  }
  return out;
}

// ─── Helper: GB block sample ───

function gbBlockSample(
  gray: Float32Array,
  W: number,
  gy: number,
  gx: number,
  scale: number,
  percentile: number,
): number {
  const y1 = gy * scale;
  const y2 = (gy + 1) * scale;
  const x1 = gx * scale;
  const x2 = (gx + 1) * scale;

  const values: number[] = [];
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      values.push(gray[y * W + x]);
    }
  }

  if (values.length === 0) return 0;
  return computePercentile(values, percentile);
}

function computePercentile(values: number[], percentile: number): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = (percentile / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

// ─── White surface estimation ───

function collectWhiteSamples(
  gray: Float32Array,
  W: number,
  H: number,
  scale: number,
): { ys: number[]; xs: number[]; vs: number[] } {
  const raw: Array<[number, number, number]> = [];
  const half = Math.floor(scale / 2);

  // Top strip: GB rows 0 .. INNER_TOP-1, safe cols 10 .. SCREEN_W-10
  for (let gy = 0; gy < INNER_TOP; gy++) {
    for (let gx = 10; gx < SCREEN_W - 10; gx++) {
      const v = gbBlockSample(gray, W, gy, gx, scale, 85);
      raw.push([gy * scale + half, gx * scale + half, v]);
    }
  }

  // Bottom strip: GB rows INNER_BOT+1 .. SCREEN_H-1
  for (let gy = INNER_BOT + 1; gy < SCREEN_H; gy++) {
    for (let gx = 10; gx < SCREEN_W - 10; gx++) {
      const v = gbBlockSample(gray, W, gy, gx, scale, 85);
      raw.push([gy * scale + half, gx * scale + half, v]);
    }
  }

  // Left strip: GB cols 0 .. INNER_LEFT-1, safe rows 10 .. SCREEN_H-10
  for (let gy = 10; gy < SCREEN_H - 10; gy++) {
    for (let gx = 0; gx < INNER_LEFT; gx++) {
      const v = gbBlockSample(gray, W, gy, gx, scale, 85);
      raw.push([gy * scale + half, gx * scale + half, v]);
    }
  }

  // Right strip: GB cols INNER_RIGHT+1 .. SCREEN_W-1
  for (let gy = 10; gy < SCREEN_H - 10; gy++) {
    for (let gx = INNER_RIGHT + 1; gx < SCREEN_W; gx++) {
      const v = gbBlockSample(gray, W, gy, gx, scale, 85);
      raw.push([gy * scale + half, gx * scale + half, v]);
    }
  }

  if (raw.length === 0) return { ys: [], xs: [], vs: [] };

  // Filter: drop blocks below 75% of median
  const allVals = raw.map((r) => r[2]);
  const med = computePercentile(allVals, 50);
  const threshold = 0.75 * med;
  const kept = raw.filter((r) => r[2] > threshold);

  return {
    ys: kept.map((r) => r[0]),
    xs: kept.map((r) => r[1]),
    vs: kept.map((r) => r[2]),
  };
}

// ─── Dark surface estimation ───

function collectDarkSamples(
  gray: Float32Array,
  W: number,
  _H: number,
  scale: number,
): {
  left: Float64Array;
  right: Float64Array;
  top: Float64Array;
  bot: Float64Array;
} {
  const gyRange = INNER_BOT - INNER_TOP + 1; // INNER_TOP to INNER_BOT inclusive
  const gxRange = INNER_RIGHT - INNER_LEFT + 1;

  const left = new Float64Array(gyRange);
  const right = new Float64Array(gyRange);
  for (let i = 0; i < gyRange; i++) {
    const gy = INNER_TOP + i;
    left[i] = gbBlockSample(gray, W, gy, INNER_LEFT, scale, 50);
    right[i] = gbBlockSample(gray, W, gy, INNER_RIGHT, scale, 50);
  }

  const top = new Float64Array(gxRange);
  const bot = new Float64Array(gxRange);
  for (let j = 0; j < gxRange; j++) {
    const gx = INNER_LEFT + j;
    top[j] = gbBlockSample(gray, W, INNER_TOP, gx, scale, 50);
    bot[j] = gbBlockSample(gray, W, INNER_BOT, gx, scale, 50);
  }

  return { left, right, top, bot };
}

// ─── Uniform filter 1D (simple moving average with nearest boundary) ───

export function uniformFilter1d(
  input: Float64Array,
  size: number,
): Float64Array {
  const half = Math.floor(size / 2);
  const n = input.length;
  const output = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = i - half; j <= i + half; j++) {
      let idx = j;
      // Nearest boundary: clamp to edge value (matches scipy mode='nearest')
      if (idx < 0) idx = 0;
      if (idx >= n) idx = n - 1;
      sum += input[idx];
    }
    output[i] = sum / size;
  }

  return output;
}

// ─── Linear interpolation (like numpy.interp) ───

function linearInterp(
  xp: Float64Array,
  yp: Float64Array,
  xNew: Float64Array,
): Float64Array {
  const result = new Float64Array(xNew.length);
  const n = xp.length;

  for (let i = 0; i < xNew.length; i++) {
    const x = xNew[i];
    if (x <= xp[0]) {
      result[i] = yp[0];
    } else if (x >= xp[n - 1]) {
      result[i] = yp[n - 1];
    } else {
      // Binary search for interval
      let lo = 0;
      let hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xp[mid] <= x) lo = mid;
        else hi = mid;
      }
      const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
      result[i] = yp[lo] * (1 - t) + yp[hi] * t;
    }
  }

  return result;
}

// ─── Coons bilinear patch ───

function buildDarkSurface(
  left: Float64Array,
  right: Float64Array,
  top: Float64Array,
  bot: Float64Array,
  H: number,
  W: number,
  scale: number,
  smoothK: number,
): Float32Array {
  // Smooth each border curve
  const ld = uniformFilter1d(left, smoothK);
  const rd = uniformFilter1d(right, smoothK);
  const td = uniformFilter1d(top, smoothK);
  const bd = uniformFilter1d(bot, smoothK);

  // Image-pixel centre positions of each border sample
  const gyCount = INNER_BOT - INNER_TOP + 1;
  const gxCount = INNER_RIGHT - INNER_LEFT + 1;
  const yRows = new Float64Array(gyCount);
  const xCols = new Float64Array(gxCount);
  const half = Math.floor(scale / 2);

  for (let i = 0; i < gyCount; i++) {
    yRows[i] = (INNER_TOP + i) * scale + half;
  }
  for (let j = 0; j < gxCount; j++) {
    xCols[j] = (INNER_LEFT + j) * scale + half;
  }

  const yStart = yRows[0];
  const yEnd = yRows[gyCount - 1];
  const xStart = xCols[0];
  const xEnd = xCols[gxCount - 1];

  // Build full-image coordinate arrays
  const yPx = new Float64Array(H);
  const xPx = new Float64Array(W);
  for (let i = 0; i < H; i++) yPx[i] = i;
  for (let i = 0; i < W; i++) xPx[i] = i;

  // Interpolate each border curve onto the full pixel axis
  const L = linearInterp(yRows, ld, yPx); // (H,) left boundary value at each row
  const R = linearInterp(yRows, rd, yPx); // (H,)
  const T = linearInterp(xCols, td, xPx); // (W,) top boundary value at each col
  const B = linearInterp(xCols, bd, xPx); // (W,)

  // Corner values (average of two meeting boundary curves)
  const TL = (ld[0] + td[0]) / 2;
  const TR = (rd[0] + td[gxCount - 1]) / 2;
  const BL = (ld[gyCount - 1] + bd[0]) / 2;
  const BR = (rd[gyCount - 1] + bd[gxCount - 1]) / 2;

  // Build Coons bilinear patch surface
  const ySpan = yEnd - yStart;
  const xSpan = xEnd - xStart;
  const surface = new Float32Array(H * W);

  for (let y = 0; y < H; y++) {
    const yn = Math.max(0, Math.min(1, (yPx[y] - yStart) / ySpan));
    const oneMinusYn = 1 - yn;
    const Ly = L[y];
    const Ry = R[y];

    for (let x = 0; x < W; x++) {
      const xn = Math.max(0, Math.min(1, (xPx[x] - xStart) / xSpan));
      const oneMinusXn = 1 - xn;

      // Coons bilinear patch:
      //   (1-xn)*L(y) + xn*R(y) + (1-yn)*T(x) + yn*B(x)
      //   - (1-xn)*(1-yn)*TL - xn*(1-yn)*TR - (1-xn)*yn*BL - xn*yn*BR
      surface[y * W + x] =
        oneMinusXn * Ly +
        xn * Ry +
        oneMinusYn * T[x] +
        yn * B[x] -
        oneMinusXn * oneMinusYn * TL -
        xn * oneMinusYn * TR -
        oneMinusXn * yn * BL -
        xn * yn * BR;
    }
  }

  return surface;
}

// ─── Bivariate polynomial surface fitting ───

/**
 * Build the Vandermonde design matrix for a bivariate polynomial.
 * Terms: x^dx * y^dy for all dx + dy <= degree.
 */
function buildDesignMatrix(
  yn: Float64Array,
  xn: Float64Array,
  degree: number,
): Float64Array {
  const n = yn.length;
  // Count terms
  let numTerms = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let dx = 0; dx <= degree - dy; dx++) {
      numTerms++;
    }
  }

  const A = new Float64Array(n * numTerms);
  let col = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let dx = 0; dx <= degree - dy; dx++) {
      for (let i = 0; i < n; i++) {
        A[i * numTerms + col] = Math.pow(yn[i], dy) * Math.pow(xn[i], dx);
      }
      col++;
    }
  }

  return A;
}

/**
 * Solve least-squares: A * coeffs = b
 * Using normal equations: (A^T A) coeffs = A^T b
 * With Cholesky-like approach via simple Gaussian elimination.
 */
function solveLeastSquares(
  A: Float64Array,
  b: Float64Array,
  rows: number,
  cols: number,
): Float64Array {
  // Compute A^T A (cols x cols)
  const AtA = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < rows; k++) {
        sum += A[k * cols + i] * A[k * cols + j];
      }
      AtA[i * cols + j] = sum;
      AtA[j * cols + i] = sum;
    }
  }

  // Compute A^T b (cols)
  const Atb = new Float64Array(cols);
  for (let i = 0; i < cols; i++) {
    let sum = 0;
    for (let k = 0; k < rows; k++) {
      sum += A[k * cols + i] * b[k];
    }
    Atb[i] = sum;
  }

  // Solve AtA * x = Atb via Gaussian elimination with partial pivoting
  // Augmented matrix [AtA | Atb]
  const aug = new Float64Array(cols * (cols + 1));
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < cols; j++) {
      aug[i * (cols + 1) + j] = AtA[i * cols + j];
    }
    aug[i * (cols + 1) + cols] = Atb[i];
  }

  const stride = cols + 1;
  for (let k = 0; k < cols; k++) {
    // Partial pivoting
    let maxVal = Math.abs(aug[k * stride + k]);
    let maxRow = k;
    for (let i = k + 1; i < cols; i++) {
      const val = Math.abs(aug[i * stride + k]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }
    if (maxRow !== k) {
      for (let j = k; j < stride; j++) {
        const tmp = aug[k * stride + j];
        aug[k * stride + j] = aug[maxRow * stride + j];
        aug[maxRow * stride + j] = tmp;
      }
    }

    const pivot = aug[k * stride + k];
    if (Math.abs(pivot) < 1e-12) continue; // singular

    for (let i = k + 1; i < cols; i++) {
      const factor = aug[i * stride + k] / pivot;
      for (let j = k; j < stride; j++) {
        aug[i * stride + j] -= factor * aug[k * stride + j];
      }
    }
  }

  // Back substitution
  const x = new Float64Array(cols);
  for (let i = cols - 1; i >= 0; i--) {
    let sum = aug[i * stride + cols];
    for (let j = i + 1; j < cols; j++) {
      sum -= aug[i * stride + j] * x[j];
    }
    const diag = aug[i * stride + i];
    x[i] = Math.abs(diag) > 1e-12 ? sum / diag : 0;
  }

  return x;
}

/**
 * Fit a bivariate polynomial surface and evaluate on full (H, W) grid.
 * Coordinates are normalised to [-1, 1] before fitting.
 */
function fitSurface(
  ys: number[],
  xs: number[],
  vals: number[],
  H: number,
  W: number,
  degree: number,
): Float32Array {
  const n = ys.length;
  if (n === 0) {
    // Return a flat surface at the mean of TRUE_WHITE
    return new Float32Array(H * W).fill(TRUE_WHITE);
  }

  // Normalise coordinates to [-1, 1]
  const ynS = new Float64Array(n);
  const xnS = new Float64Array(n);
  const vS = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    ynS[i] = (ys[i] / H) * 2 - 1;
    xnS[i] = (xs[i] / W) * 2 - 1;
    vS[i] = vals[i];
  }

  // Count terms
  let numTerms = 0;
  for (let dy = 0; dy <= degree; dy++) {
    for (let _dx = 0; _dx <= degree - dy; _dx++) {
      numTerms++;
    }
  }

  const A = buildDesignMatrix(ynS, xnS, degree);
  const coeffs = solveLeastSquares(A, vS, n, numTerms);

  // Evaluate on the full pixel grid
  const surface = new Float32Array(H * W);
  for (let y = 0; y < H; y++) {
    const ynVal = (y / H) * 2 - 1;
    for (let x = 0; x < W; x++) {
      const xnVal = (x / W) * 2 - 1;
      let val = 0;
      let col = 0;
      for (let dy = 0; dy <= degree; dy++) {
        for (let dx = 0; dx <= degree - dy; dx++) {
          val += coeffs[col] * Math.pow(ynVal, dy) * Math.pow(xnVal, dx);
          col++;
        }
      }
      surface[y * W + x] = val;
    }
  }

  return surface;
}

// ─── Per-pixel affine correction (per-channel) ───

function applyCorrectionChannel(
  channel: Float32Array,
  whiteSurface: Float32Array,
  darkSurface: Float32Array,
  W: number,
  H: number,
  whiteTarget: number,
  darkTarget: number,
): Float32Array {
  const corrected = new Float32Array(H * W);
  const span = whiteTarget - darkTarget;

  for (let i = 0; i < H * W; i++) {
    const ws = whiteSurface[i];
    const ds = darkSurface[i];
    const gain = Math.max(ws - ds, 5) / span;
    const offset = ds - gain * darkTarget;
    const val = (channel[i] - offset) / gain;
    corrected[i] = Math.max(0, Math.min(255, Math.round(val)));
  }

  return corrected;
}

// ─── Per-pixel affine correction (original grayscale version - deprecated) ───

// ─── Quick sample (inline sampling of camera area) ───

function quickSample(
  corrected: Float32Array,
  W: number,
  scale: number,
  hMargin: number = 2,
  vMargin: number = 1,
): Uint8Array {
  const out = new Uint8Array(CAM_H * CAM_W);

  for (let gy = 0; gy < CAM_H; gy++) {
    for (let gx = 0; gx < CAM_W; gx++) {
      const y1 = (FRAME_THICK + gy) * scale + vMargin;
      const y2 = (FRAME_THICK + gy + 1) * scale - vMargin;
      const x1 = (FRAME_THICK + gx) * scale + hMargin;
      const x2 = (FRAME_THICK + gx + 1) * scale - hMargin;

      const values: number[] = [];
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          values.push(corrected[y * W + x]);
        }
      }

      if (values.length === 0) {
        out[gy * CAM_W + gx] = 0;
      } else {
        // Median
        const med = computePercentile(values, 50);
        out[gy * CAM_W + gx] = Math.max(0, Math.min(255, Math.round(med)));
      }
    }
  }

  return out;
}

// ─── Collect border dark samples for polynomial fitting ───

function collectBorderDarkForPoly(
  channel: Float32Array,
  W: number,
  _H: number,
  scale: number,
  darkSmooth: number,
): { borderY: number[]; borderX: number[]; borderV: number[] } {
  const { left, right, top, bot } = collectDarkSamples(channel, W, _H, scale);

  const ld = uniformFilter1d(new Float64Array(left), darkSmooth);
  const rd = uniformFilter1d(new Float64Array(right), darkSmooth);
  const td = uniformFilter1d(new Float64Array(top), darkSmooth);
  const bd = uniformFilter1d(new Float64Array(bot), darkSmooth);

  const gyCount = INNER_BOT - INNER_TOP + 1;
  const gxCount = INNER_RIGHT - INNER_LEFT + 1;
  const half = Math.floor(scale / 2);

  const yRows: number[] = [];
  for (let i = 0; i < gyCount; i++) {
    yRows.push((INNER_TOP + i) * scale + half);
  }
  const xCols: number[] = [];
  for (let j = 0; j < gxCount; j++) {
    xCols.push((INNER_LEFT + j) * scale + half);
  }

  const borderY: number[] = [];
  const borderX: number[] = [];
  const borderV: number[] = [];

  // Left and right borders
  for (let i = 0; i < gyCount; i++) {
    borderY.push(yRows[i], yRows[i]);
    borderX.push(xCols[0], xCols[gxCount - 1]);
    borderV.push(ld[i], rd[i]);
  }

  // Top and bottom borders
  for (let j = 0; j < gxCount; j++) {
    borderY.push(yRows[0], yRows[gyCount - 1]);
    borderX.push(xCols[j], xCols[j]);
    borderV.push(td[j], bd[j]);
  }

  return { borderY, borderX, borderV };
}

// ─── Iterative refinement (per-channel) ───

function refinePassChannel(
  corrected: Float32Array,
  channel: Float32Array,
  whiteSurface: Float32Array,
  darkSurface: Float32Array,
  W: number,
  H: number,
  scale: number,
  darkSmooth: number,
  whiteTarget: number,
  darkTarget: number,
): { darkSurface: Float32Array; corrected: Float32Array; calCount: number } | null {
  // Quick-sample the corrected image to get per-GB-pixel brightness
  const sampled = quickSample(corrected, W, scale);

  // Classify pixels in [60, 110] as dark-gray (#525252)
  const edgeMargin = 3;
  const calY: number[] = [];
  const calX: number[] = [];
  const calV: number[] = [];
  const half = Math.floor(scale / 2);

  for (let gy = edgeMargin; gy < CAM_H - edgeMargin; gy++) {
    for (let gx = edgeMargin; gx < CAM_W - edgeMargin; gx++) {
      const val = sampled[gy * CAM_W + gx];
      // For R and G channels, dark target is 148 (DG.R = DG.G)
      // For B channel, dark target is different
      // Classify as dark if corrected value is close to darkTarget ± 30
      const darkMin = Math.max(0, darkTarget - 30);
      const darkMax = Math.min(255, darkTarget + 30);
      if (val >= darkMin && val <= darkMax) {
        // Use the uncorrected value at the block centre as interior calibration
        const py = (FRAME_THICK + gy) * scale + half;
        const px = (FRAME_THICK + gx) * scale + half;
        calY.push(py);
        calX.push(px);
        // Use median of the uncorrected block (same as _quick_sample uses median)
        const values: number[] = [];
        const y1 = (FRAME_THICK + gy) * scale + 1;
        const y2 = (FRAME_THICK + gy + 1) * scale - 1;
        const x1 = (FRAME_THICK + gx) * scale + 2;
        const x2 = (FRAME_THICK + gx + 1) * scale - 2;
        for (let y = y1; y < y2; y++) {
          for (let x = x1; x < x2; x++) {
            values.push(channel[y * W + x]);
          }
        }
        calV.push(
          values.length > 0 ? computePercentile(values, 50) : channel[py * W + px],
        );
      }
    }
  }

  if (calY.length < 50) return null;

  // Collect border dark samples
  const { borderY, borderX, borderV } = collectBorderDarkForPoly(channel, W, H, scale, darkSmooth);

  // Combine border + interior calibration points
  const allY = borderY.concat(calY);
  const allX = borderX.concat(calX);
  const allV = borderV.concat(calV);

  // Fit degree-4 polynomial
  const newDarkSurface = fitSurface(allY, allX, allV, H, W, 4);

  // Clamp: refined surface must not go below initial Coons estimate
  for (let i = 0; i < H * W; i++) {
    newDarkSurface[i] = Math.max(newDarkSurface[i], darkSurface[i]);
  }

  // Blend near edges: near camera area edges, blend toward Coons estimate
  const blendMargin = 4 * scale;
  const yCamStart = FRAME_THICK * scale;
  const yCamEnd = (FRAME_THICK + CAM_H) * scale;
  const xCamStart = FRAME_THICK * scale;
  const xCamEnd = (FRAME_THICK + CAM_W) * scale;
  const camH = yCamEnd - yCamStart;
  const camW = xCamEnd - xCamStart;

  for (let y = yCamStart; y < yCamEnd; y++) {
    const yRel = y - yCamStart;
    const yDist = Math.min(yRel, camH - 1 - yRel);
    for (let x = xCamStart; x < xCamEnd; x++) {
      const xRel = x - xCamStart;
      const distToEdge = Math.min(yDist, xRel, camW - 1 - xRel);
      if (distToEdge < blendMargin) {
        const blendFactor = distToEdge / blendMargin;
        const idx = y * W + x;
        newDarkSurface[idx] =
          (1 - blendFactor) * darkSurface[idx] +
          blendFactor * newDarkSurface[idx];
      }
    }
  }

  // Re-correct with refined dark surface
  const newCorrected = applyCorrectionChannel(
    channel,
    whiteSurface,
    newDarkSurface,
    W,
    H,
    whiteTarget,
    darkTarget,
  );

  return { darkSurface: newDarkSurface, corrected: newCorrected, calCount: calY.length };
}
