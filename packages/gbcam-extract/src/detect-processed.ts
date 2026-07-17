/**
 * detect-processed.ts — Detect inputs that are already-processed Game Boy
 * Camera images (pipeline outputs fed back in) rather than phone photos of a
 * GBA SP screen, and recover the 128 × 112 four-color image directly instead
 * of running the photo pipeline on them.
 *
 * Detection runs signals from cheapest to strongest:
 *   1. Dimension gate — the image must be an exact integer multiple of one of
 *      the known output layouts: bare camera image (128 × 112), normal-framed
 *      screen (160 × 144), or wild-framed print (160 × 224).
 *   2. Integer downsample — each k × k block is reduced to its mean color.
 *   3. Four-color gate — the downsampled pixels must collapse into at most
 *      four tight color clusters. This holds even after lossy (JPEG/WebP)
 *      re-compression, where the literal color count balloons but the colors
 *      still cluster tightly around the four palette colors. Real photos have
 *      thousands of spread-out colors and never pass this gate.
 *   4. Palette mapping — clusters are mapped to the GB grayscale values
 *      {0, 82, 165, 255}: exact-grayscale images snap to the nearest value,
 *      colored images are matched against the known GB Camera palettes, and
 *      anything else (e.g. user-defined palettes) falls back to luminance
 *      rank order.
 */

import {
  type GBImageData,
  GB_COLORS,
  CAM_W,
  CAM_H,
  SCREEN_W,
  SCREEN_H,
  FRAME_THICK,
  grayscaleToRGBA,
} from "./common.js";
import type { Frame } from "./frames/types.js";
import { cropImage, type DebugCollector } from "./debug.js";
import { parseHex } from "./palette.js";
import {
  MAIN_PALETTES,
  ADDITIONAL_PALETTES,
  FUN_PALETTES,
} from "./data/palettes-generated.js";

/** Wild-frame print dimensions (GB Printer layout: full-width, extra-tall). */
export const WILD_W = 160;
export const WILD_H = 224;

/** Largest integer upscale factor we accept in the dimension gate. */
const MAX_SCALE = 32;

/** Colors within this Euclidean RGB distance join an existing cluster. */
const CLUSTER_ABSORB_DIST = 40;

/** More clusters than this and the image is certainly a photo — bail early. */
const MAX_CLUSTERS = 32;

/** The four biggest clusters must cover at least this fraction of pixels. */
const MIN_TOP4_COVERAGE = 0.97;

/** Any cluster beyond the top four must stay below this pixel fraction. */
const MAX_MINOR_CLUSTER_FRACTION = 0.005;

/** Max per-channel spread for a cluster to count as grayscale. */
const GRAYSCALE_CHANNEL_SPREAD = 14;

/** Max distance between a cluster and a known palette color to match it. */
const PALETTE_MATCH_DIST = 50;

/** Min fraction of matching non-hole pixels to accept a wild-frame match. */
const WILD_FRAME_MATCH_RATIO = 0.9;

export type ProcessedLayout = "bare" | "normal-frame" | "wild-frame";

export interface DetectProcessedOptions {
  /**
   * Frames whose geometry/artwork may appear around an already-processed
   * image. Used to locate the camera-image hole inside wild-framed inputs
   * (normal frames always have the hole at (16, 16), so they need no help).
   */
  knownFrames?: Frame[];
  debug?: DebugCollector;
}

export interface ProcessedDetection {
  /** Recovered 128 × 112 grayscale image (values in {0, 82, 165, 255}). */
  grayscale: GBImageData;
  layout: ProcessedLayout;
  /** Integer upscale factor the input was found at. */
  scale: number;
  /** Set when a wild-framed input matched a known frame's artwork. */
  matchedFrameId?: string;
}

interface Cluster {
  r: number;
  g: number;
  b: number;
  count: number;
}

const LAYOUTS: Array<{ layout: ProcessedLayout; w: number; h: number }> = [
  { layout: "bare", w: CAM_W, h: CAM_H },
  { layout: "normal-frame", w: SCREEN_W, h: SCREEN_H },
  { layout: "wild-frame", w: WILD_W, h: WILD_H },
];

function luminance(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function nearestGBValue(v: number): number {
  let nearest: number = GB_COLORS[0];
  for (const gv of GB_COLORS) {
    if (Math.abs(v - gv) < Math.abs(v - nearest)) nearest = gv;
  }
  return nearest;
}

function dist2(
  r0: number,
  g0: number,
  b0: number,
  r1: number,
  g1: number,
  b1: number,
): number {
  const dr = r0 - r1;
  const dg = g0 - g1;
  const db = b0 - b1;
  return dr * dr + dg * dg + db * db;
}

/** Mean-downsample an RGBA image by integer factor k into flat RGB arrays. */
function blockMeanDownsample(
  input: GBImageData,
  k: number,
  baseW: number,
  baseH: number,
): { r: Float32Array; g: Float32Array; b: Float32Array } {
  const r = new Float32Array(baseW * baseH);
  const g = new Float32Array(baseW * baseH);
  const b = new Float32Array(baseW * baseH);
  const norm = 1 / (k * k);
  for (let by = 0; by < baseH; by++) {
    for (let bx = 0; bx < baseW; bx++) {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      for (let dy = 0; dy < k; dy++) {
        let idx = ((by * k + dy) * input.width + bx * k) * 4;
        for (let dx = 0; dx < k; dx++) {
          sr += input.data[idx];
          sg += input.data[idx + 1];
          sb += input.data[idx + 2];
          idx += 4;
        }
      }
      const o = by * baseW + bx;
      r[o] = sr * norm;
      g[o] = sg * norm;
      b[o] = sb * norm;
    }
  }
  return { r, g, b };
}

/**
 * Greedy frequency-ordered clustering: unique colors are visited most-common
 * first; each either joins the nearest existing cluster (within
 * {@link CLUSTER_ABSORB_DIST}) or seeds a new one. Returns null when the
 * color structure cannot be a four-color image.
 */
function clusterColors(
  r: Float32Array,
  g: Float32Array,
  b: Float32Array,
): Cluster[] | null {
  const counts = new Map<number, number>();
  for (let i = 0; i < r.length; i++) {
    const key =
      (Math.round(r[i]) << 16) | (Math.round(g[i]) << 8) | Math.round(b[i]);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  const entries = [...counts.entries()].sort((a, b2) => b2[1] - a[1]);
  const clusters: Cluster[] = [];
  const maxD2 = CLUSTER_ABSORB_DIST * CLUSTER_ABSORB_DIST;

  for (const [key, count] of entries) {
    const cr = (key >> 16) & 0xff;
    const cg = (key >> 8) & 0xff;
    const cb = key & 0xff;
    let best = -1;
    let bestD2 = Infinity;
    for (let ci = 0; ci < clusters.length; ci++) {
      const c = clusters[ci];
      const d2 = dist2(cr, cg, cb, c.r, c.g, c.b);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = ci;
      }
    }
    if (best >= 0 && bestD2 <= maxD2) {
      const c = clusters[best];
      const total = c.count + count;
      c.r = (c.r * c.count + cr * count) / total;
      c.g = (c.g * c.count + cg * count) / total;
      c.b = (c.b * c.count + cb * count) / total;
      c.count = total;
    } else {
      if (clusters.length >= MAX_CLUSTERS) return null;
      clusters.push({ r: cr, g: cg, b: cb, count });
    }
  }

  clusters.sort((a, b2) => b2.count - a.count);
  const total = r.length;
  const top4 = clusters.slice(0, 4).reduce((s, c) => s + c.count, 0);
  if (top4 / total < MIN_TOP4_COVERAGE) return null;
  for (const c of clusters.slice(4)) {
    if (c.count / total >= MAX_MINOR_CLUSTER_FRACTION) return null;
  }
  return clusters.slice(0, 4);
}

/**
 * Map each cluster to one of the GB grayscale values. Tries, in order:
 * grayscale snap, known GB Camera palettes, luminance rank. Returns null
 * only when the grayscale snap is ambiguous (two clusters on the same GB
 * value), which indicates a non-GB grayscale image.
 */
function mapClustersToGB(
  clusters: Cluster[],
  debug?: DebugCollector,
): number[] | null {
  const isGrayscale = clusters.every((c) => {
    const spread =
      Math.max(c.r, c.g, c.b) - Math.min(c.r, c.g, c.b);
    return spread <= GRAYSCALE_CHANNEL_SPREAD;
  });

  if (isGrayscale) {
    const mapped = clusters.map((c) =>
      nearestGBValue(luminance(c.r, c.g, c.b)),
    );
    if (new Set(mapped).size === mapped.length) {
      debug?.log("[detect] palette mapping: grayscale snap");
      return mapped;
    }
    // Two distinct gray clusters landing on the same GB value means this is
    // grayscale but not GB-quantized (e.g. a plain grayscale photo).
    return null;
  }

  // Known GB Camera palettes. Palette color order is [255, 165, 82, 0].
  const paletteGBValues = [255, 165, 82, 0] as const;
  let bestMapping: number[] | null = null;
  let bestTotal = Infinity;
  let bestName = "";
  for (const entry of [
    ...MAIN_PALETTES,
    ...ADDITIONAL_PALETTES,
    ...FUN_PALETTES,
  ]) {
    const cols = entry.colors.map(parseHex);
    const mapping: number[] = [];
    const used = new Set<number>();
    let totalD = 0;
    let ok = true;
    for (const c of clusters) {
      let bi = -1;
      let bd2 = Infinity;
      for (let i = 0; i < 4; i++) {
        const d2 = dist2(c.r, c.g, c.b, cols[i][0], cols[i][1], cols[i][2]);
        if (d2 < bd2) {
          bd2 = d2;
          bi = i;
        }
      }
      if (bd2 > PALETTE_MATCH_DIST * PALETTE_MATCH_DIST || used.has(bi)) {
        ok = false;
        break;
      }
      used.add(bi);
      mapping.push(paletteGBValues[bi]);
      totalD += Math.sqrt(bd2);
    }
    if (ok && totalD < bestTotal) {
      bestTotal = totalD;
      bestMapping = mapping;
      bestName = entry.name;
    }
  }
  if (bestMapping) {
    debug?.log(
      `[detect] palette mapping: matched known palette "${bestName}" (total dist ${bestTotal.toFixed(1)})`,
    );
    return bestMapping;
  }

  // Luminance rank fallback for unknown (e.g. user-defined) palettes.
  debug?.log("[detect] palette mapping: luminance rank fallback");
  const order = clusters
    .map((c, i) => ({ i, l: luminance(c.r, c.g, c.b) }))
    .sort((a, b2) => a.l - b2.l);
  const mapped = new Array<number>(clusters.length);
  if (clusters.length === 4) {
    order.forEach((o, rank) => {
      mapped[o.i] = GB_COLORS[rank];
    });
  } else if (clusters.length === 1) {
    mapped[order[0].i] = nearestGBValue(order[0].l);
  } else {
    // 2 or 3 clusters: anchor extremes to 0/255, place middles by relative
    // position between them.
    const lMin = order[0].l;
    const lMax = order[order.length - 1].l;
    mapped[order[0].i] = 0;
    mapped[order[order.length - 1].i] = 255;
    for (let i = 1; i < order.length - 1; i++) {
      const t = (order[i].l - lMin) / Math.max(lMax - lMin, 1e-6);
      mapped[order[i].i] = t < 0.5 ? 82 : 165;
    }
  }
  return mapped;
}

/** Crop a 128 × 112 window out of a base-resolution GB-value buffer. */
function cropGBValues(
  values: Uint8Array,
  srcW: number,
  srcH: number,
  x0: number,
  y0: number,
): GBImageData {
  return cropImage(grayscaleToRGBA(values, srcW, srcH), x0, y0, CAM_W, CAM_H);
}

/**
 * Detect whether `input` is an already-processed Game Boy Camera image and,
 * if so, recover the 128 × 112 four-color image. Returns null when the input
 * does not look like a processed image (i.e. the photo pipeline should run).
 */
export function detectProcessedImage(
  input: GBImageData,
  opts?: DetectProcessedOptions,
): ProcessedDetection | null {
  const debug = opts?.debug;

  // 1. Dimension gate.
  let layout: ProcessedLayout | null = null;
  let baseW = 0;
  let baseH = 0;
  let scale = 0;
  for (const cand of LAYOUTS) {
    if (input.width % cand.w !== 0 || input.height % cand.h !== 0) continue;
    const kx = input.width / cand.w;
    const ky = input.height / cand.h;
    if (kx !== ky || kx < 1 || kx > MAX_SCALE) continue;
    layout = cand.layout;
    baseW = cand.w;
    baseH = cand.h;
    scale = kx;
    break;
  }
  if (!layout) return null;

  // 2. Downsample to base resolution.
  const { r, g, b } = blockMeanDownsample(input, scale, baseW, baseH);

  // 3. Four-color gate.
  const clusters = clusterColors(r, g, b);
  if (!clusters) {
    debug?.log(
      `[detect] dimensions matched ${layout} at ${scale}x but colors don't collapse to ≤4 clusters — treating as photo`,
    );
    return null;
  }

  // 4. Cluster → GB value mapping.
  const gbValues = mapClustersToGB(clusters, debug);
  if (!gbValues) {
    debug?.log(
      "[detect] color clusters are grayscale but not GB-quantized — treating as photo",
    );
    return null;
  }

  // Assign every pixel to its nearest cluster's GB value.
  const values = new Uint8Array(baseW * baseH);
  for (let i = 0; i < values.length; i++) {
    let bi = 0;
    let bd2 = Infinity;
    for (let ci = 0; ci < clusters.length; ci++) {
      const c = clusters[ci];
      const d2 = dist2(r[i], g[i], b[i], c.r, c.g, c.b);
      if (d2 < bd2) {
        bd2 = d2;
        bi = ci;
      }
    }
    values[i] = gbValues[bi];
  }

  debug?.setMetrics("detect", {
    layout,
    scale,
    clusterCenters: clusters.map((c, i) => ({
      r: Math.round(c.r),
      g: Math.round(c.g),
      b: Math.round(c.b),
      count: c.count,
      gbValue: gbValues[i],
    })),
  });

  // 5. Extract the camera region.
  if (layout === "bare") {
    debug?.log(`[detect] already-processed bare image at ${scale}x`);
    return {
      grayscale: cropGBValues(values, baseW, baseH, 0, 0),
      layout,
      scale,
    };
  }

  if (layout === "normal-frame") {
    debug?.log(
      `[detect] already-processed normal-framed image at ${scale}x — cropping hole at (${FRAME_THICK}, ${FRAME_THICK})`,
    );
    return {
      grayscale: cropGBValues(values, baseW, baseH, FRAME_THICK, FRAME_THICK),
      layout,
      scale,
    };
  }

  // Wild frame: the hole position varies per frame, so match the non-hole
  // artwork against known wild frames.
  let bestFrame: Frame | null = null;
  let bestRatio = 0;
  for (const frame of opts?.knownFrames ?? []) {
    if (frame.width !== WILD_W || frame.height !== WILD_H) continue;
    let match = 0;
    let outside = 0;
    for (let y = 0; y < WILD_H; y++) {
      for (let x = 0; x < WILD_W; x++) {
        const inHole =
          x >= frame.holeX &&
          x < frame.holeX + CAM_W &&
          y >= frame.holeY &&
          y < frame.holeY + CAM_H;
        if (inHole) continue;
        outside++;
        if (frame.pixels[y * WILD_W + x] === values[y * WILD_W + x]) match++;
      }
    }
    const ratio = outside > 0 ? match / outside : 0;
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestFrame = frame;
    }
  }

  if (bestFrame && bestRatio >= WILD_FRAME_MATCH_RATIO) {
    debug?.log(
      `[detect] already-processed wild-framed image at ${scale}x — matched frame "${bestFrame.id}" (${(bestRatio * 100).toFixed(1)}% artwork match), hole at (${bestFrame.holeX}, ${bestFrame.holeY})`,
    );
    return {
      grayscale: cropGBValues(
        values,
        baseW,
        baseH,
        bestFrame.holeX,
        bestFrame.holeY,
      ),
      layout,
      scale,
      matchedFrameId: bestFrame.id,
    };
  }

  // No frame matched — fall back to (16, 40), the hole position every wild
  // frame in the original USA/JPN sheets uses.
  const fallbackX = 16;
  const fallbackY = 40;
  debug?.log(
    `[detect] already-processed wild-framed image at ${scale}x — no known frame matched (best ${(bestRatio * 100).toFixed(1)}%), assuming hole at (${fallbackX}, ${fallbackY})`,
  );
  return {
    grayscale: cropGBValues(values, baseW, baseH, fallbackX, fallbackY),
    layout,
    scale,
  };
}
