/**
 * Frame-aware classification using palette anchors extracted from the frame.
 *
 * The Game Boy filmstrip frame in the GBA SP image contains all four
 * palette colours (BK in dash bodies, DG in dash edges + inner border,
 * LG in dash anti-aliasing edges, WH in the frame background). The
 * `FRAME_MASK` constant marks which palette colour each frame pixel
 * should be. After warping, the observed (R, G, B) values at frame-mask
 * positions give us scattered samples for each colour, spread across the
 * image — enough to fit per-colour, per-channel bivariate surfaces that
 * predict the expected observed value for any colour at any location.
 *
 * Classification then becomes: for each pixel, find the colour C whose
 * predicted RGB at the pixel's location is closest to the actual observed
 * RGB. This naturally absorbs the front-light brightness gradient and
 * per-colour sub-pixel response curves — far more anchor data than the
 * 2-anchor (WH frame, DG border) affine in the existing correct step.
 */

import { type GBImageData, SCREEN_W, SCREEN_H, FRAME_THICK } from "./common.js";
import { FRAME_MASK, FRAME_MASK_W, FRAME_MASK_H } from "./frame-mask.js";
import { fitBivariateSurface } from "./poly-surface.js";

export interface FrameClassifier {
  /** 4 surfaces, indexed by palette colour 0..3 (BK, DG, LG, WH). */
  R: Float32Array[];
  G: Float32Array[];
  B: Float32Array[];
  W: number;
  H: number;
  /** Per-colour sample counts (BK, DG, LG, WH). */
  sampleCounts: [number, number, number, number];
  /** Per-colour global mean R/G/B from frame anchors (BK, DG, LG, WH). */
  meanR: [number, number, number, number];
  meanG: [number, number, number, number];
  meanB: [number, number, number, number];
}

/**
 * Build a Gaussian-weighted local-mean surface from scattered anchor samples.
 * For each (y, x) in the HxW output, the value is the σ-Gaussian-weighted
 * mean of the input samples. Unlike polynomial fitting, this can never
 * extrapolate outside the sample value range — predictions are always
 * convex combinations of observed samples.
 */
function gaussianAverage(
  ys: number[],
  xs: number[],
  vs: number[],
  H: number,
  W: number,
  sigma: number,
): Float32Array {
  const out = new Float32Array(H * W);
  if (ys.length === 0) return out;
  const twoSigma2 = 2 * sigma * sigma;
  // Fallback global mean for points where weight sum is zero.
  let globalMean = 0;
  for (const v of vs) globalMean += v;
  globalMean /= vs.length;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let wsum = 0;
      let vsum = 0;
      for (let i = 0; i < ys.length; i++) {
        const dy = y - ys[i];
        const dx = x - xs[i];
        const w = Math.exp(-(dy * dy + dx * dx) / twoSigma2);
        wsum += w;
        vsum += w * vs[i];
      }
      out[y * W + x] = wsum > 1e-12 ? vsum / wsum : globalMean;
    }
  }
  return out;
}

/** Clamp every entry of `surface` into [min(vals), max(vals)] in place. */
function clampToRange(surface: Float32Array, vals: number[]): void {
  if (vals.length === 0) return;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  for (let i = 0; i < surface.length; i++) {
    if (surface[i] < lo) surface[i] = lo;
    else if (surface[i] > hi) surface[i] = hi;
  }
}

/** Sample an 8×8-pixel-block-equivalent area of a per-channel sub-pixel image. */
function samplePixelBlock(
  image: GBImageData,
  scale: number,
  fy: number,
  fx: number,
): { R: number; G: number; B: number } {
  const innerStart = 1;
  const innerEnd = scale - 1;
  const innerW = innerEnd - innerStart;
  const bLo = innerStart;
  const bHi = innerStart + Math.floor(innerW / 3);
  const gLo = bHi;
  const gHi = innerStart + 2 * Math.floor(innerW / 3);
  const rLo = gHi + (scale >= 8 ? 1 : 0);
  const rHi = scale;
  const vMargin = Math.max(1, Math.floor(scale / 5));
  const y1 = fy * scale + vMargin;
  const y2 = (fy + 1) * scale - vMargin;
  const x0 = fx * scale;

  let rSum = 0, gSum = 0, bSum = 0;
  let rCount = 0, gCount = 0, bCount = 0;
  for (let y = y1; y < y2; y++) {
    const rowBase = y * image.width;
    for (let dx = bLo; dx < bHi; dx++) {
      bSum += image.data[(rowBase + x0 + dx) * 4 + 2];
      bCount++;
    }
    for (let dx = gLo; dx < gHi; dx++) {
      gSum += image.data[(rowBase + x0 + dx) * 4 + 1];
      gCount++;
    }
    for (let dx = rLo; dx < rHi; dx++) {
      rSum += image.data[(rowBase + x0 + dx) * 4];
      rCount++;
    }
  }
  return {
    R: rCount > 0 ? rSum / rCount : 0,
    G: gCount > 0 ? gSum / gCount : 0,
    B: bCount > 0 ? bSum / bCount : 0,
  };
}

export interface FrameAnchor {
  y: number;
  x: number;
  c: number;
  R: number;
  G: number;
  B: number;
}

/**
 * Collect raw per-color anchor samples (no surface fit). Used by k-NN.
 */
export function collectFrameAnchors(
  warped: GBImageData,
  scale: number = 8,
): FrameAnchor[] {
  if (
    warped.width !== SCREEN_W * scale ||
    warped.height !== SCREEN_H * scale
  ) {
    throw new Error(
      `Unexpected warped size ${warped.width}x${warped.height}; ` +
        `expected ${SCREEN_W * scale}x${SCREEN_H * scale} (scale=${scale})`,
    );
  }
  const anchors: FrameAnchor[] = [];
  for (let fy = 0; fy < FRAME_MASK_H; fy++) {
    for (let fx = 0; fx < FRAME_MASK_W; fx++) {
      const c = FRAME_MASK[fy * FRAME_MASK_W + fx];
      if (c >= 4) continue;
      const { R, G, B } = samplePixelBlock(warped, scale, fy, fx);
      anchors.push({ y: fy, x: fx, c, R, G, B });
    }
  }
  return anchors;
}

/**
 * Build a frame-aware classifier by sampling the corrected image at every
 * frame-mask pixel and fitting per-colour, per-channel polynomial surfaces.
 *
 * @param warped (SCREEN_W*scale, SCREEN_H*scale) RGBA image (warp or correct
 *   output is fine — both have the frame plus camera area).
 * @param scale image scale (default 8).
 * @param degree polynomial degree for surface fit (default 2).
 */
export function buildFrameClassifier(
  warped: GBImageData,
  scale: number = 8,
  degree: number = 2,
): FrameClassifier {
  if (
    warped.width !== SCREEN_W * scale ||
    warped.height !== SCREEN_H * scale
  ) {
    throw new Error(
      `Unexpected warped size ${warped.width}x${warped.height}; ` +
        `expected ${SCREEN_W * scale}x${SCREEN_H * scale} (scale=${scale})`,
    );
  }

  // Collect per-colour samples
  const ys: number[][] = [[], [], [], []];
  const xs: number[][] = [[], [], [], []];
  const rVals: number[][] = [[], [], [], []];
  const gVals: number[][] = [[], [], [], []];
  const bVals: number[][] = [[], [], [], []];

  for (let fy = 0; fy < FRAME_MASK_H; fy++) {
    for (let fx = 0; fx < FRAME_MASK_W; fx++) {
      const c = FRAME_MASK[fy * FRAME_MASK_W + fx];
      if (c >= 4) continue;
      const { R, G, B } = samplePixelBlock(warped, scale, fy, fx);
      ys[c].push(fy);
      xs[c].push(fx);
      rVals[c].push(R);
      gVals[c].push(G);
      bVals[c].push(B);
    }
  }

  const R: Float32Array[] = new Array(4);
  const G: Float32Array[] = new Array(4);
  const B: Float32Array[] = new Array(4);
  if (degree < 0) {
    // Negative degree = Gaussian-weighted local mean (no polynomial fit).
    // The kernel σ in pixels is |degree| (e.g. degree=-50 → σ=50).
    const sigma = -degree;
    for (let c = 0; c < 4; c++) {
      R[c] = gaussianAverage(ys[c], xs[c], rVals[c], FRAME_MASK_H, FRAME_MASK_W, sigma);
      G[c] = gaussianAverage(ys[c], xs[c], gVals[c], FRAME_MASK_H, FRAME_MASK_W, sigma);
      B[c] = gaussianAverage(ys[c], xs[c], bVals[c], FRAME_MASK_H, FRAME_MASK_W, sigma);
    }
  } else {
    for (let c = 0; c < 4; c++) {
      R[c] = fitBivariateSurface(ys[c], xs[c], rVals[c], FRAME_MASK_H, FRAME_MASK_W, degree);
      G[c] = fitBivariateSurface(ys[c], xs[c], gVals[c], FRAME_MASK_H, FRAME_MASK_W, degree);
      B[c] = fitBivariateSurface(ys[c], xs[c], bVals[c], FRAME_MASK_H, FRAME_MASK_W, degree);
      clampToRange(R[c], rVals[c]);
      clampToRange(G[c], gVals[c]);
      clampToRange(B[c], bVals[c]);
    }
  }

  const meanR: [number, number, number, number] = [0, 0, 0, 0];
  const meanG: [number, number, number, number] = [0, 0, 0, 0];
  const meanB: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    const n = ys[c].length;
    if (n === 0) continue;
    let sR = 0, sG = 0, sB = 0;
    for (let i = 0; i < n; i++) {
      sR += rVals[c][i];
      sG += gVals[c][i];
      sB += bVals[c][i];
    }
    meanR[c] = sR / n;
    meanG[c] = sG / n;
    meanB[c] = sB / n;
  }

  return {
    R,
    G,
    B,
    W: FRAME_MASK_W,
    H: FRAME_MASK_H,
    sampleCounts: [ys[0].length, ys[1].length, ys[2].length, ys[3].length],
    meanR,
    meanG,
    meanB,
  };
}

/**
 * Classify a single pixel given its position in frame coords and its observed
 * (R, G, B). Returns the palette index (0=BK, 1=DG, 2=LG, 3=WH).
 */
export function classifyByFrame(
  cls: FrameClassifier,
  frameY: number,
  frameX: number,
  R: number,
  G: number,
  B: number,
): number {
  const idx = frameY * cls.W + frameX;
  let bestC = 0;
  let bestD = Infinity;
  for (let c = 0; c < 4; c++) {
    const dR = R - cls.R[c][idx];
    const dG = G - cls.G[c][idx];
    const dB = B - cls.B[c][idx];
    const d = dR * dR + dG * dG + dB * dB;
    if (d < bestD) {
      bestD = d;
      bestC = c;
    }
  }
  return bestC;
}

/**
 * Map a camera-area pixel index back to its position in the frame grid.
 * Camera origin sits at (FRAME_THICK, FRAME_THICK) in screen coords.
 */
export function cameraToFrameCoords(
  camY: number,
  camX: number,
): { frameY: number; frameX: number } {
  return { frameY: camY + FRAME_THICK, frameX: camX + FRAME_THICK };
}
