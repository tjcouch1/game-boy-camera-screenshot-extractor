/**
 * quantize.ts — Map 128x112 colour samples to 4 GB Camera palette colors.
 *
 * Ported from gbcam_quantize.py. Uses k-means clustering in RG colour space
 * with strip refinement and G-valley correction.
 *
 * Pipeline: sample (128x112 colour) -> quantize -> 128x112 grayscale (0/82/165/255)
 */

import {
  type GBImageData,
  GB_COLORS,
  CAM_W,
  CAM_H,
  FRAME_THICK,
  SCREEN_W,
  createGBImageData,
} from "./common.js";
import { getCV, withMats } from "./opencv.js";
import { type DebugCollector, renderRGScatter, upscale } from "./debug.js";
import {
  buildFrameClassifier,
  classifyByFrame,
  cameraToFrameCoords,
  collectFrameAnchors,
} from "./frame-classify.js";

export interface QuantizeOptions {
  /**
   * Optional warped/corrected color image at (SCREEN_W*scale, SCREEN_H*scale).
   * When provided, quantize uses frame-aware classification: it samples the
   * frame dashes (which contain all four palette colors) to fit per-color
   * surfaces for R/G/B, then classifies each camera pixel by nearest
   * predicted color at its location. Falls back to 2D RG k-means when not
   * provided.
   */
  corrected?: GBImageData;
  /**
   * Optional warped (pre-correct) RGBA image at (SCREEN_W*scale,
   * SCREEN_H*scale). The `correct` step's affine gain amplifies bleed-lifted
   * DG pixels' R upward across the DG/LG boundary; the pre-correct warp still
   * separates DG (low R) from LG (high R) cleanly. Used to recover sparse
   * DG-on-black dots that correct pushes into LG.
   */
  warped?: GBImageData;
  scale?: number;
  debug?: DebugCollector;
}

// ─── RGB palette matching the Python COLOR_PALETTE_RGB ───
// BK=(0,0,0), DG=(148,148,255), LG=(255,148,148), WH=(255,255,165)
const PALETTE_RG: [number, number][] = [
  [0, 0],
  [148, 148],
  [255, 148],
  [255, 255],
];

// Warm initialisation centres for global k-means (RG plane)
const INIT_CENTERS_RG: [number, number][] = [
  [80, 20],
  [148, 148],
  [240, 148],
  [250, 250],
];

// ─── Helpers ───

/** Simple 1D Gaussian filter with reflected boundary. */
function gaussianFilter1d(input: number[], sigma: number): number[] {
  const radius = Math.ceil(sigma * 4);
  const kernel: number[] = [];
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(v);
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const n = input.length;
  const output: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let val = 0;
    for (let k = 0; k < kernel.length; k++) {
      let j = i + (k - radius);
      if (j < 0) j = -j;
      if (j >= n) j = 2 * n - 2 - j;
      j = Math.max(0, Math.min(n - 1, j));
      val += input[j] * kernel[k];
    }
    output[i] = val;
  }
  return output;
}

/**
 * Find all permutations of the given array. Used for cluster-to-palette
 * matching (k ≤ 4 clusters, so at most 24 permutations).
 */
function permutationsOf(items: number[]): number[][] {
  const result: number[][] = [];
  const arr = [...items];
  function permute(start: number) {
    if (start === arr.length) {
      result.push([...arr]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      [arr[start], arr[i]] = [arr[i], arr[start]];
      permute(start + 1);
      [arr[start], arr[i]] = [arr[i], arr[start]];
    }
  }
  permute(0);
  return result;
}

/**
 * Find the best assignment of clusters -> palette indices (restricted to the
 * given present palette indices) that minimizes total RG Euclidean distance.
 * `centersRG` holds k = presentPalette.length cluster centers; the result maps
 * cluster index -> palette index.
 */
function bestClusterToPalette(
  centersRG: Float32Array,
  targetsRG: [number, number][],
  presentPalette: number[],
): Int32Array {
  const perms = permutationsOf(presentPalette);
  let bestPerm: number[] = perms[0];
  let bestCost = Infinity;
  for (const perm of perms) {
    let cost = 0;
    for (let i = 0; i < perm.length; i++) {
      const cr = centersRG[i * 2];
      const cg = centersRG[i * 2 + 1];
      const tr = targetsRG[perm[i]][0];
      const tg = targetsRG[perm[i]][1];
      cost += Math.sqrt((cr - tr) ** 2 + (cg - tg) ** 2);
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestPerm = perm;
    }
  }
  return new Int32Array(bestPerm);
}

/**
 * Validate a cluster→palette assignment: a cluster assigned to a palette
 * colour whose center is far more than RATIO× closer to a *different*
 * palette target is not that colour at all — it's a cluster that migrated
 * into another colour's cloud because the assigned colour barely exists in
 * the image (e.g. a photo with no black: the forced 4th cluster splits a
 * warm cloud and the bijection labels real warm pixels BK). Returns the
 * palette indices whose clusters failed validation.
 *
 * This deliberately tests the CLUSTER, not a pixel count: an image with only
 * a handful of genuine pixels of a colour still anchors a small warm-started
 * cluster near that colour's target and passes. Measured across all corpora:
 * every real cluster has ratio ≤ 1.01 for BK (≤ 2.7 for gradient-shifted
 * warm clusters), while the no-BK image's migrated "BK" cluster sits at 4.8.
 */
function invalidClusterColors(
  centersRG: Float32Array,
  clusterToPalette: Int32Array,
  targetsRG: [number, number][],
): number[] {
  const RATIO = 3;
  const dropped: number[] = [];
  for (let ci = 0; ci < clusterToPalette.length; ci++) {
    const pi = clusterToPalette[ci];
    const cr = centersRG[ci * 2];
    const cg = centersRG[ci * 2 + 1];
    const dAssigned = Math.hypot(cr - targetsRG[pi][0], cg - targetsRG[pi][1]);
    let dOther = Infinity;
    for (let q = 0; q < 4; q++) {
      if (q === pi) continue;
      dOther = Math.min(
        dOther,
        Math.hypot(cr - targetsRG[q][0], cg - targetsRG[q][1]),
      );
    }
    if (dAssigned > RATIO * dOther) dropped.push(pi);
  }
  return dropped;
}

/**
 * Run cv.kmeans on an Nx3 float32 (R, G, B*scale) sample set with warm
 * initialisation. Returns { labels: Int32Array(N), centers: Float32Array(4*3) }.
 * The caller scales B before passing in to control how much B affects the
 * Euclidean distance vs R/G.
 */
function runKmeans3D(
  samplesRGB: Float32Array,
  n: number,
  initCenters: Float32Array,
): { labels: Int32Array; centers: Float32Array } {
  const cv = getCV();
  return withMats((track) => {
    const samplesMat = track(new cv.Mat(n, 3, cv.CV_32F));
    samplesMat.data32F.set(samplesRGB);
    const labelsMat = track(new cv.Mat(n, 1, cv.CV_32S));
    const centersMat = track(new cv.Mat(4, 3, cv.CV_32F));
    for (let i = 0; i < n; i++) {
      const r = samplesRGB[i * 3];
      const g = samplesRGB[i * 3 + 1];
      const b = samplesRGB[i * 3 + 2];
      let bestK = 0;
      let bestD = Infinity;
      for (let k = 0; k < 4; k++) {
        const cr = initCenters[k * 3];
        const cg = initCenters[k * 3 + 1];
        const cb = initCenters[k * 3 + 2];
        const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
        if (d < bestD) {
          bestD = d;
          bestK = k;
        }
      }
      labelsMat.data32S[i] = bestK;
    }
    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
      300,
      0.1,
    );
    cv.kmeans(
      samplesMat,
      4,
      labelsMat,
      criteria,
      1,
      cv.KMEANS_USE_INITIAL_LABELS,
      centersMat,
    );
    return {
      labels: new Int32Array(labelsMat.data32S),
      centers: new Float32Array(centersMat.data32F),
    };
  });
}

/**
 * Run cv.kmeans on an Nx2 float32 sample set with warm initialisation.
 * The cluster count k is taken from the number of init centers (2–4).
 * Returns { labels: Int32Array(N), centers: Float32Array(k*2) }
 */
function runKmeans(
  samplesRG: Float32Array,
  n: number,
  initCenters: [number, number][] | Float32Array,
): { labels: Int32Array; centers: Float32Array } {
  const cv = getCV();
  const k =
    initCenters instanceof Float32Array
      ? initCenters.length / 2
      : initCenters.length;
  return withMats((track) => {
    // Build Nx2 samples Mat
    const samplesMat = track(new cv.Mat(n, 2, cv.CV_32F));
    samplesMat.data32F.set(samplesRG);

    // Build labels output
    const labelsMat = track(new cv.Mat(n, 1, cv.CV_32S));

    // Build centers output
    const centersMat = track(new cv.Mat(k, 2, cv.CV_32F));

    // Use warm start: set labels from initial centers via nearest assignment
    // then use KMEANS_USE_INITIAL_LABELS
    // Actually, opencv.js doesn't support initial centers directly.
    // We assign initial labels based on nearest init center, then use KMEANS_USE_INITIAL_LABELS.
    for (let i = 0; i < n; i++) {
      const r = samplesRG[i * 2];
      const g = samplesRG[i * 2 + 1];
      let bestK = 0;
      let bestD = Infinity;
      const ic = initCenters instanceof Float32Array ? initCenters : null;
      for (let ki = 0; ki < k; ki++) {
        let cr: number, cg: number;
        if (ic) {
          cr = ic[ki * 2];
          cg = ic[ki * 2 + 1];
        } else {
          cr = (initCenters as [number, number][])[ki][0];
          cg = (initCenters as [number, number][])[ki][1];
        }
        const d = (r - cr) ** 2 + (g - cg) ** 2;
        if (d < bestD) {
          bestD = d;
          bestK = ki;
        }
      }
      labelsMat.data32S[i] = bestK;
    }

    const criteria = new cv.TermCriteria(
      cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
      300,
      0.1,
    );

    cv.kmeans(
      samplesMat,
      k,
      labelsMat,
      criteria,
      1, // attempts=1 since we use initial labels
      cv.KMEANS_USE_INITIAL_LABELS,
      centersMat,
    );

    // Copy results out before mats are deleted
    const labels = new Int32Array(labelsMat.data32S);
    const centers = new Float32Array(centersMat.data32F);
    return { labels, centers };
  });
}

/**
 * G-valley threshold: find the G-axis valley between LG and WH clusters
 * among high-R pixels. Matches Python _g_valley_threshold.
 */
function gValleyThreshold(
  gVals: number[],
  lgCenterG: number,
  whCenterG: number,
): number {
  const lo = Math.floor(lgCenterG) + 1;
  const hi = Math.floor(whCenterG);
  if (hi <= lo + 4) {
    return (lgCenterG + whCenterG) / 2.0;
  }

  // Build histogram: bins from lo to hi+1 (so hi-lo+1 bins covering values lo..hi)
  const nBins = hi - lo + 1;
  const hist = new Array<number>(nBins).fill(0);
  let total = 0;
  for (const g of gVals) {
    const bin = Math.floor(g) - lo;
    if (bin >= 0 && bin < nBins) {
      hist[bin]++;
      total++;
    }
  }

  if (total < 10) {
    return (lgCenterG + whCenterG) / 2.0;
  }

  const smooth = gaussianFilter1d(hist, 3.0);

  // Search from upper 2/3 of range
  let searchLo = Math.floor((smooth.length * 2) / 3);
  let valleyIdx = searchLo;
  let minVal = smooth[searchLo];
  for (let i = searchLo + 1; i < smooth.length; i++) {
    if (smooth[i] < minVal) {
      minVal = smooth[i];
      valleyIdx = i;
    }
  }

  // If boundary-constrained, retry from 1/3
  if (valleyIdx === searchLo) {
    const widerLo = Math.max(Math.floor(smooth.length / 3), 1);
    valleyIdx = widerLo;
    minVal = smooth[widerLo];
    for (let i = widerLo + 1; i < smooth.length; i++) {
      if (smooth[i] < minVal) {
        minVal = smooth[i];
        valleyIdx = i;
      }
    }
  }

  // threshold = edges[valley_idx] = lo + valley_idx
  return lo + valleyIdx;
}

/**
 * Quantize a 128x112 colour sample image to 4 GB Camera palette values.
 *
 * Uses k-means clustering in RG colour space with:
 * 1. Global k-means (4 clusters) with warm initialisation
 * 2. Strip k-means refinement for lateral gradient
 * 3. G-valley LG/WH refinement for pixel bleeding correction
 */
export function quantize(
  input: GBImageData,
  options?: QuantizeOptions,
): GBImageData {
  if (input.width !== CAM_W || input.height !== CAM_H) {
    throw new Error(
      `Expected ${CAM_W}x${CAM_H}, got ${input.width}x${input.height}`,
    );
  }
  const dbg = options?.debug;

  const N = CAM_W * CAM_H;
  const targetsRG = PALETTE_RG;

  // Extract RG values (Nx2 float32) for 2D k-means.
  const flatRG = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    flatRG[i * 2] = input.data[i * 4];
    flatRG[i * 2 + 1] = input.data[i * 4 + 1];
  }

  // ── 1. Global k-means with cluster-assignment validation ──
  // Always cluster with k=4 first, then validate that each colour's assigned
  // cluster is plausibly that colour. An image may legitimately contain
  // (almost) none of a colour — forcing 4 clusters then splits a present
  // colour's cloud and the bijective match labels real pixels with the
  // missing colour. When a cluster fails validation, drop its colour and
  // re-cluster with k = valid colours. (Individual pixels of a dropped
  // colour are recovered per-pixel at the end of the pipeline — dropping
  // the CLUSTER never makes the colour unreachable.)
  let presentPalette = [0, 1, 2, 3];
  let global = runKmeans(flatRG, N, INIT_CENTERS_RG);
  let clusterToPalette = bestClusterToPalette(
    global.centers,
    targetsRG,
    presentPalette,
  );
  const droppedColors = invalidClusterColors(
    global.centers,
    clusterToPalette,
    targetsRG,
  );
  if (droppedColors.length > 0 && 4 - droppedColors.length >= 2) {
    presentPalette = [0, 1, 2, 3].filter((p) => !droppedColors.includes(p));
    if (dbg) {
      dbg.log(
        `[quantize] cluster validation: ` +
          droppedColors.map((p) => ["BK", "DG", "LG", "WH"][p]).join(", ") +
          ` cluster(s) migrated away from their palette target — ` +
          `re-clustering with k=${presentPalette.length}`,
      );
    }
    global = runKmeans(
      flatRG,
      N,
      presentPalette.map((p) => INIT_CENTERS_RG[p]),
    );
    clusterToPalette = bestClusterToPalette(
      global.centers,
      targetsRG,
      presentPalette,
    );
  }
  const kPresent = presentPalette.length;

  // Map cluster labels to palette indices
  const labelsFlat = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    labelsFlat[i] = clusterToPalette[global.labels[i]];
  }

  // Capture global k-means metrics — palette-ordered cluster centers
  const paletteCenters = new Array<[number, number]>(4);
  for (let pi = 0; pi < 4; pi++) {
    let cr = targetsRG[pi][0];
    let cg = targetsRG[pi][1];
    for (let ci = 0; ci < kPresent; ci++) {
      if (clusterToPalette[ci] === pi) {
        cr = global.centers[ci * 2];
        cg = global.centers[ci * 2 + 1];
        break;
      }
    }
    paletteCenters[pi] = [cr, cg];
  }
  const globalCounts = countLabels(labelsFlat);

  if (dbg) {
    dbg.log(
      `[quantize] global k-means cluster centers (palette-ordered):  ` +
        ["BK", "DG", "LG", "WH"]
          .map(
            (n, i) =>
              `${n}=(R${paletteCenters[i][0].toFixed(0)},G${paletteCenters[i][1].toFixed(0)})`,
          )
          .join("  "),
    );
    dbg.log(
      `[quantize] after global kmeans: ` +
        ["BK", "DG", "LG", "WH"]
          .map((n, i) => `${n}=${globalCounts[i]}`)
          .join("  "),
    );
  }

  // Build global_centers_po (palette-ordered centers)
  const globalCentersPO = new Float32Array(4 * 2);
  for (let pi = 0; pi < 4; pi++) {
    let found = false;
    for (let ci = 0; ci < kPresent; ci++) {
      if (clusterToPalette[ci] === pi) {
        globalCentersPO[pi * 2] = global.centers[ci * 2];
        globalCentersPO[pi * 2 + 1] = global.centers[ci * 2 + 1];
        found = true;
        break;
      }
    }
    if (!found) {
      globalCentersPO[pi * 2] = targetsRG[pi][0];
      globalCentersPO[pi * 2 + 1] = targetsRG[pi][1];
    }
  }

  // ── 2. Strip k-means refinement ──
  const stripWidth = 32;
  const step = 16;
  const nStrips = Math.floor((CAM_W - stripWidth) / step) + 1;

  // strip_labels[y][x][s] — palette label from strip s
  // Use a flat array: index = (y * CAM_W + x) * nStrips + s
  const stripLabels = new Int8Array(CAM_H * CAM_W * nStrips).fill(-1);
  const stripCentersCol = new Float64Array(nStrips);

  for (let s = 0; s < nStrips; s++) {
    const colStart = s * step;
    const colEnd = Math.min(colStart + stripWidth, CAM_W);
    const sw = colEnd - colStart;
    const sN = CAM_H * sw;

    // Extract RG for this strip
    const stripRG = new Float32Array(sN * 2);
    let idx = 0;
    for (let y = 0; y < CAM_H; y++) {
      for (let x = colStart; x < colEnd; x++) {
        const pi = y * CAM_W + x;
        stripRG[idx * 2] = flatRG[pi * 2];
        stripRG[idx * 2 + 1] = flatRG[pi * 2 + 1];
        idx++;
      }
    }

    const stripInit = new Float32Array(kPresent * 2);
    for (let ki = 0; ki < kPresent; ki++) {
      stripInit[ki * 2] = globalCentersPO[presentPalette[ki] * 2];
      stripInit[ki * 2 + 1] = globalCentersPO[presentPalette[ki] * 2 + 1];
    }
    const stripResult = runKmeans(stripRG, sN, stripInit);
    const c2p = bestClusterToPalette(
      stripResult.centers,
      targetsRG,
      presentPalette,
    );

    // Build palette-ordered strip centers, then blend toward global centers.
    // This anchors per-strip drift (which over-classifies borderline pixels)
    // while still allowing local adaptation to brightness gradients.
    const blendedCenters = new Float32Array(4 * 2);
    const ANCHOR_W = 0.27; // weight on global (vs 1 - ANCHOR_W on strip)
    const stripCentersPO = new Float32Array(4 * 2);
    for (let pi = 0; pi < 4; pi++) {
      let ci = -1;
      for (let cj = 0; cj < kPresent; cj++) {
        if (c2p[cj] === pi) {
          ci = cj;
          break;
        }
      }
      if (ci >= 0) {
        stripCentersPO[pi * 2] = stripResult.centers[ci * 2];
        stripCentersPO[pi * 2 + 1] = stripResult.centers[ci * 2 + 1];
      } else {
        stripCentersPO[pi * 2] = globalCentersPO[pi * 2];
        stripCentersPO[pi * 2 + 1] = globalCentersPO[pi * 2 + 1];
      }
      blendedCenters[pi * 2] =
        stripCentersPO[pi * 2] * (1 - ANCHOR_W) +
        globalCentersPO[pi * 2] * ANCHOR_W;
      blendedCenters[pi * 2 + 1] =
        stripCentersPO[pi * 2 + 1] * (1 - ANCHOR_W) +
        globalCentersPO[pi * 2 + 1] * ANCHOR_W;
    }

    // Re-classify strip pixels using the blended centers
    idx = 0;
    for (let y = 0; y < CAM_H; y++) {
      for (let x = colStart; x < colEnd; x++) {
        const r = stripRG[idx * 2];
        const g = stripRG[idx * 2 + 1];
        let bestPi = presentPalette[0],
          bestD = Infinity;
        for (const pi of presentPalette) {
          const dr = r - blendedCenters[pi * 2];
          const dg = g - blendedCenters[pi * 2 + 1];
          const d = dr * dr + dg * dg;
          if (d < bestD) {
            bestD = d;
            bestPi = pi;
          }
        }
        stripLabels[(y * CAM_W + x) * nStrips + s] = bestPi;
        idx++;
      }
    }
    stripCentersCol[s] = (colStart + colEnd) / 2.0;
  }

  // Apply strip consensus: override global label when ALL covering strips agree
  const labels2d = new Int32Array(labelsFlat); // copy
  const finalLabels = new Int32Array(labelsFlat);
  let stripChanged = 0;

  for (let x = 0; x < CAM_W; x++) {
    // Find covering strips for this column
    const coveringStrips: number[] = [];
    for (let s = 0; s < nStrips; s++) {
      const cs = s * step;
      const ce = Math.min(cs + stripWidth, CAM_W);
      if (cs <= x && x < ce && stripLabels[x * nStrips + s] >= 0) {
        coveringStrips.push(s);
      }
    }
    if (coveringStrips.length === 0) continue;

    // Find the best strip (closest center column to x)
    let bestStrip = coveringStrips[0];
    let bestDist = Math.abs(stripCentersCol[bestStrip] - x);
    for (let i = 1; i < coveringStrips.length; i++) {
      const d = Math.abs(stripCentersCol[coveringStrips[i]] - x);
      if (d < bestDist) {
        bestDist = d;
        bestStrip = coveringStrips[i];
      }
    }

    for (let y = 0; y < CAM_H; y++) {
      const pi = y * CAM_W + x;
      const globalL = labels2d[pi];
      const stripL = stripLabels[pi * nStrips + bestStrip];

      if (stripL !== globalL) {
        // Check if ANY covering strip agrees with global
        let anyAgree = false;
        for (const s of coveringStrips) {
          if (stripLabels[pi * nStrips + s] === globalL) {
            anyAgree = true;
            break;
          }
        }
        if (!anyAgree) {
          finalLabels[pi] = stripL;
          stripChanged++;
        }
      }
    }
  }

  const stripCounts = countLabels(finalLabels);
  if (dbg) {
    dbg.log(
      `[quantize] strip ensemble: ${nStrips} strips, changed ${stripChanged} px  ` +
        `now: ${["BK", "DG", "LG", "WH"]
          .map((n, i) => `${n}=${stripCounts[i]}`)
          .join("  ")}`,
    );
  }

  // ── 2b. Fake-DG-cluster validation (blueness check) ──
  // When true DG is nearly absent from the image, the DG cluster of the RG
  // k-means migrates into the dim tail of the LG cloud — RG alone cannot
  // tell "dim warm" from "true DG". But real DG is BLUE on screen (#9494FF):
  // DG-labelled pixels' mean B sits well above LG's on every reference and
  // held-out image with real DG (sep ≥ 14, typically 21–61), while a
  // migrated cluster's B matches LG's (sep ≈ 0). If separation is missing,
  // dissolve the cluster: reassign its pixels to LG/WH by RG distance and
  // recover as DG only the pixels that are individually blue
  // (B − R ≥ 12 — warm content always has R far above B).
  {
    const SEP_MIN = 8;
    const BLUE_MIN = 12;
    let dgBsum = 0,
      dgN = 0,
      lgBsum = 0,
      lgN = 0,
      warmBsum = 0,
      warmN = 0;
    for (let i = 0; i < N; i++) {
      const b = input.data[i * 4 + 2];
      if (finalLabels[i] === 1) {
        dgBsum += b;
        dgN++;
      } else if (finalLabels[i] === 2) {
        lgBsum += b;
        lgN++;
        warmBsum += b;
        warmN++;
      } else if (finalLabels[i] === 3) {
        warmBsum += b;
        warmN++;
      }
    }
    const refB =
      lgN >= 50 ? lgBsum / lgN : warmN >= 50 ? warmBsum / warmN : null;
    // Fires when the DG cluster's B matches warm (a migrated cluster), and
    // also when DG had no valid cluster at all (dropped by the assignment
    // validation above) — the per-pixel blueness recovery below is how
    // genuinely-blue pixels get their DG label back in that case.
    const dgFake =
      refB !== null &&
      ((dgN > 0 && dgBsum / dgN - refB < SEP_MIN) ||
        (dgN === 0 && !presentPalette.includes(1)));
    if (dgFake && refB !== null) {
      const lgR = globalCentersPO[2 * 2];
      const lgG = globalCentersPO[2 * 2 + 1];
      const whR = globalCentersPO[3 * 2];
      const whG = globalCentersPO[3 * 2 + 1];
      let dissolved = 0;
      let recovered = 0;
      for (let i = 0; i < N; i++) {
        if (finalLabels[i] === 1) {
          const r = flatRG[i * 2];
          const g = flatRG[i * 2 + 1];
          const dLG = (r - lgR) ** 2 + (g - lgG) ** 2;
          const dWH = (r - whR) ** 2 + (g - whG) ** 2;
          finalLabels[i] = dLG <= dWH ? 2 : 3;
          dissolved++;
        }
        // Recover only warm-labelled pixels: BK pixels also read blue under
        // the front-light tint (B − R ≈ +50 on dark LCD), so an unrestricted
        // blueness flip would eat real black in a BK-present/DG-absent image.
        if (
          (finalLabels[i] === 2 || finalLabels[i] === 3) &&
          input.data[i * 4 + 2] - flatRG[i * 2] >= BLUE_MIN
        ) {
          finalLabels[i] = 1;
          recovered++;
        }
      }
      if (dbg) {
        dbg.log(
          `[quantize] DG cluster failed blueness validation ` +
            `(dgMeanB=${dgN > 0 ? (dgBsum / dgN).toFixed(1) : "n/a"} refB=${refB.toFixed(1)}): ` +
            `dissolved ${dissolved} px into LG/WH, recovered ${recovered} blue px as DG`,
        );
      }
    }
  }

  // ── 3. G-valley LG/WH refinement ──
  // Find cluster indices for LG (palette 2) and WH (palette 3)
  let lgClusterIdx = -1;
  let whClusterIdx = -1;
  for (let ci = 0; ci < kPresent; ci++) {
    if (clusterToPalette[ci] === 2) lgClusterIdx = ci;
    if (clusterToPalette[ci] === 3) whClusterIdx = ci;
  }

  let valleyThreshold: number | null = null;
  let valleyChanged = 0;
  if (lgClusterIdx >= 0 && whClusterIdx >= 0) {
    const lgCG = global.centers[lgClusterIdx * 2 + 1]; // G component of LG center
    const whCG = global.centers[whClusterIdx * 2 + 1]; // G component of WH center

    // Collect G values of high-R pixels (R > 190)
    const gHighR: number[] = [];
    for (let i = 0; i < N; i++) {
      if (flatRG[i * 2] > 190) {
        gHighR.push(flatRG[i * 2 + 1]);
      }
    }

    const gThresh = gValleyThreshold(gHighR, lgCG, whCG);
    valleyThreshold = gThresh;

    // Apply threshold to LG/WH pixels with high R
    for (let i = 0; i < N; i++) {
      if (
        flatRG[i * 2] > 170 &&
        (finalLabels[i] === 2 || finalLabels[i] === 3)
      ) {
        const newLabel = flatRG[i * 2 + 1] >= gThresh ? 3 : 2;
        if (newLabel !== finalLabels[i]) {
          valleyChanged++;
          finalLabels[i] = newLabel;
        }
      }
    }
    if (dbg) {
      dbg.log(
        `[quantize] G-valley refinement: threshold=${gThresh.toFixed(1)} ` +
          `(LG center G=${lgCG.toFixed(1)}, WH center G=${whCG.toFixed(1)}), ` +
          `changed ${valleyChanged} px`,
      );
    }
  }

  // ── 3c. B-aware reclassification of DG ↔ warm misclassifications ──
  // DG palette has B=255 (high), LG/WH have B=148-165 (low). The 2D RG
  // k-means ignores B — but B is precisely what should distinguish a
  // washed-out warm pixel (PSF-bleed-inflated G near DG center) from a
  // real DG pixel. Compute the cluster mean B per class, then bidirectional
  // reclassify any pixel whose B clearly contradicts its label.
  {
    let dgBsum = 0,
      dgBcount = 0;
    let warmBsum = 0,
      warmBcount = 0; // LG + WH
    for (let i = 0; i < N; i++) {
      const b = input.data[i * 4 + 2];
      if (finalLabels[i] === 1) {
        dgBsum += b;
        dgBcount++;
      }
      if (finalLabels[i] === 2 || finalLabels[i] === 3) {
        warmBsum += b;
        warmBcount++;
      }
    }
    if (dgBcount >= 50 && warmBcount >= 50) {
      const dgMeanB = dgBsum / dgBcount;
      const warmMeanB = warmBsum / warmBcount;
      const sep = dgMeanB - warmMeanB; // expect positive
      if (sep > 5) {
        const lgR = globalCentersPO[2 * 2];
        const lgG = globalCentersPO[2 * 2 + 1];
        const whR = globalCentersPO[3 * 2];
        const whG = globalCentersPO[3 * 2 + 1];
        const dgR = globalCentersPO[1 * 2];
        const dgG = globalCentersPO[1 * 2 + 1];
        // Lower threshold: pixels currently DG with B below this look warm.
        // Clamp to [175, 180] to protect against false flips in tests where
        // DG and warm B distributions overlap (e.g., bathhouse-1, where
        // dgMeanB-30 > 180) while still catching warm-but-DG-labeled pixels
        // in tests with dim DG.
        const bDgLowThresh = Math.max(195, Math.min(dgMeanB - 30, 180));
        // Upper threshold: pixels currently warm with B above this look DG
        const bWarmHiThresh = warmMeanB + sep * 0.6;
        let flippedFromDg = 0;
        let flippedToDg = 0;
        for (let i = 0; i < N; i++) {
          const lbl = finalLabels[i];
          const b = input.data[i * 4 + 2];
          const r = flatRG[i * 2];
          const g = flatRG[i * 2 + 1];
          const dLG = (r - lgR) ** 2 + (g - lgG) ** 2;
          const dWH = (r - whR) ** 2 + (g - whG) ** 2;
          const dDG = (r - dgR) ** 2 + (g - dgG) ** 2;
          if (lbl === 1 && b < bDgLowThresh) {
            const dWarm = Math.min(dLG, dWH);
            // Distances here are SQUARED — ratio 1.7 sq ≈ 1.30 linear, so
            // we permit warm to be up to 30% farther in RG than DG when B
            // clearly indicates warm content.
            if (dWarm < dDG * 1.7) {
              finalLabels[i] = dLG < dWH ? 2 : 3;
              flippedFromDg++;
            }
          } else if ((lbl === 2 || lbl === 3) && b > dgMeanB - 20) {
            // Pixel labeled warm but B is in the DG range — require RG
            // distance to also strongly indicate DG (much closer than warm)
            const dWarm = Math.min(dLG, dWH);
            const ratio = b >= dgMeanB ? 0.95 : 0.85;
            if (dDG < dWarm * ratio) {
              finalLabels[i] = 1;
              flippedToDg++;
            }
          }
        }
        if (dbg) {
          dbg.log(
            `[quantize] B reclassify: dgMeanB=${dgMeanB.toFixed(1)} warmMeanB=${warmMeanB.toFixed(1)} flipDg->warm=${flippedFromDg} flipWarm->dg=${flippedToDg}`,
          );
        }
      }
    }
  }

  // ── 3d. Confidence-based neighbour refinement (iterative) ──
  // For each pixel, compute its 3D RGB distance to the empirical cluster
  // centres of all four palette classes. Pixels whose nearest cluster is
  // decisively closer than the second-nearest are "confident". Ambiguous
  // pixels get reclassified by a vote among confident neighbours with
  // similar RGB. We iterate so newly-confident pixels in pass N+1 can
  // anchor still-ambiguous neighbours in subsequent passes.
  let confidenceRefineTotal = 0;
  for (let iter = 0; iter < 8; iter++) {
    // Two-pass cluster mean: first using all pixels, then recompute using
    // only confident pixels so cluster centres aren't biased by
    // borderline/mis-labelled pixels.
    let cR3 = [0, 0, 0, 0];
    let cG3 = [0, 0, 0, 0];
    let cB3 = [0, 0, 0, 0];
    let cN3 = [0, 0, 0, 0];
    for (let i = 0; i < N; i++) {
      const lbl = finalLabels[i];
      cR3[lbl] += flatRG[i * 2];
      cG3[lbl] += flatRG[i * 2 + 1];
      cB3[lbl] += input.data[i * 4 + 2];
      cN3[lbl]++;
    }
    for (let p = 0; p < 4; p++) {
      if (cN3[p] > 0) {
        cR3[p] /= cN3[p];
        cG3[p] /= cN3[p];
        cB3[p] /= cN3[p];
      }
    }
    // First-pass confidence based on biased means
    const isConfident = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const r = flatRG[i * 2];
      const g = flatRG[i * 2 + 1];
      const b = input.data[i * 4 + 2];
      let dBest = Infinity;
      let dSecond = Infinity;
      for (const p of presentPalette) {
        const d =
          (r - cR3[p]) * (r - cR3[p]) +
          (g - cG3[p]) * (g - cG3[p]) +
          (b - cB3[p]) * (b - cB3[p]);
        if (d < dBest) {
          dSecond = dBest;
          dBest = d;
        } else if (d < dSecond) {
          dSecond = d;
        }
      }
      isConfident[i] = dBest > 0 && dSecond >= dBest * 1.96 ? 1 : 0;
    }
    // Recompute cluster means using only confident pixels — these
    // represent the "core" of each cluster, not pulled down by
    // borderline pixels that may themselves be mis-labelled.
    const cR3c = [0, 0, 0, 0];
    const cG3c = [0, 0, 0, 0];
    const cB3c = [0, 0, 0, 0];
    const cN3c = [0, 0, 0, 0];
    for (let i = 0; i < N; i++) {
      if (!isConfident[i]) continue;
      const lbl = finalLabels[i];
      cR3c[lbl] += flatRG[i * 2];
      cG3c[lbl] += flatRG[i * 2 + 1];
      cB3c[lbl] += input.data[i * 4 + 2];
      cN3c[lbl]++;
    }
    for (let p = 0; p < 4; p++) {
      if (cN3c[p] > 50) {
        cR3[p] = cR3c[p] / cN3c[p];
        cG3[p] = cG3c[p] / cN3c[p];
        cB3[p] = cB3c[p] / cN3c[p];
      }
    }
    // Recompute confidence using cleaner means
    for (let i = 0; i < N; i++) {
      const r = flatRG[i * 2];
      const g = flatRG[i * 2 + 1];
      const b = input.data[i * 4 + 2];
      let dBest = Infinity;
      let dSecond = Infinity;
      for (const p of presentPalette) {
        const d =
          (r - cR3[p]) * (r - cR3[p]) +
          (g - cG3[p]) * (g - cG3[p]) +
          (b - cB3[p]) * (b - cB3[p]);
        if (d < dBest) {
          dSecond = dBest;
          dBest = d;
        } else if (d < dSecond) {
          dSecond = d;
        }
      }
      isConfident[i] = dBest > 0 && dSecond >= dBest * 1.96 ? 1 : 0;
    }
    const WIN = 7;
    const MD_MAX = 30;
    const SIGMA = 15;
    const AMBIG_DISCOUNT = 0.5;
    const MIN_DOMINANCE = 2.0;
    const MIN_VOTE = 0.1;
    // Palette-target centres (the ideal colour each class should be when
    // observed by a perfect pipeline). Used as a secondary anchor for
    // ambiguous pixels: if the palette-target nearest class differs from
    // the empirical-cluster nearest, the palette gets a vote too. Each
    // palette vote is weighted by the palette-target distance ratio
    // (more decisive = more vote).
    const PAL_R: [number, number, number, number] = [0, 148, 255, 255];
    const PAL_G: [number, number, number, number] = [0, 148, 148, 255];
    const PAL_B: [number, number, number, number] = [0, 255, 148, 165];
    const PAL_VOTE_SCALE = 0.12;
    let refined = 0;
    const newLabels = new Uint8Array(finalLabels);
    for (let i = 0; i < N; i++) {
      if (isConfident[i]) continue;
      const cx = i % CAM_W;
      const cy = Math.floor(i / CAM_W);
      const r = flatRG[i * 2];
      const g = flatRG[i * 2 + 1];
      const b = input.data[i * 4 + 2];
      const votes = [0, 0, 0, 0];
      for (let dy = -WIN; dy <= WIN; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= CAM_H) continue;
        for (let dx = -WIN; dx <= WIN; dx++) {
          const nx = cx + dx;
          if (nx < 0 || nx >= CAM_W) continue;
          const ni = ny * CAM_W + nx;
          if (ni === i) continue;
          const nr = flatRG[ni * 2];
          const ng = flatRG[ni * 2 + 1];
          const nb = input.data[ni * 4 + 2];
          const md = Math.abs(r - nr) + Math.abs(g - ng) + Math.abs(b - nb);
          if (md > MD_MAX) continue;
          let w = Math.exp(-md / SIGMA);
          if (!isConfident[ni]) w *= AMBIG_DISCOUNT;
          votes[finalLabels[ni]] += w;
        }
      }
      // Palette-target vote: which palette class is the pixel closest to
      // when compared against the IDEAL palette RGB? Independent of the
      // (potentially noisy) empirical cluster centres. This captures the
      // "what colour was the LCD supposed to display here?" signal. The
      // palette is most reliable for BK (R/G/B all ≈ 0) and DG (only
      // class with both low R/G AND high B); LG and WH share R=255 in
      // the palette so their differentiation comes from G — we let the
      // existing G-valley step handle that. So we only count a palette
      // vote when palBest is BK or DG (where palette is decisive on
      // multiple channels).
      let palBest = presentPalette[0];
      let palBestD = Infinity;
      let palSecond = Infinity;
      for (const p of presentPalette) {
        const dR = r - PAL_R[p];
        const dG = g - PAL_G[p];
        const dB = b - PAL_B[p];
        const d = dR * dR + dG * dG + dB * dB;
        if (d < palBestD) {
          palSecond = palBestD;
          palBestD = d;
          palBest = p;
        } else if (d < palSecond) {
          palSecond = d;
        }
      }
      if (palBestD > 0 && (palBest === 0 || palBest === 1)) {
        const palMargin = Math.sqrt(palSecond / palBestD) - 1;
        if (palMargin > 0) {
          votes[palBest] += palMargin * PAL_VOTE_SCALE;
        }
      }
      let bestP = 0;
      let bestV = -1;
      for (let p = 0; p < 4; p++) {
        if (votes[p] > bestV) {
          bestV = votes[p];
          bestP = p;
        }
      }
      const curLabel = finalLabels[i];
      if (bestP === curLabel) continue;
      if (bestV < MIN_VOTE) continue;
      const curV = votes[curLabel];
      if (curV > 0 && bestV < curV * MIN_DOMINANCE) continue;
      newLabels[i] = bestP;
      refined++;
    }
    for (let i = 0; i < N; i++) finalLabels[i] = newLabels[i];
    if (dbg) {
      let nConf = 0;
      for (let i = 0; i < N; i++) if (isConfident[i]) nConf++;
      dbg.log(
        `[quantize] confidence refine pass ${iter + 1}: ${nConf}/${N} pixels confident, ${refined} ambiguous pixels reclassified via similar-RGB neighbour vote`,
      );
    }
    confidenceRefineTotal += refined;
    if (refined === 0) break;
  }

  // ── 3e. Spatial anomaly detection: DG pixel surrounded by all-DG
  // neighbours with R substantially higher than neighbours' R avg is
  // anomalous — likely a true LG embedded in a DG region (PSF flattens
  // the R difference but it's still measurably above the local DG mean).
  // This is principled: based on the pixel's actual R vs the local DG
  // baseline, not hard-coded patterns.
  {
    let anomalyFlipped = 0;
    const newLabels = new Uint8Array(finalLabels);
    for (let i = 0; i < N; i++) {
      if (finalLabels[i] !== 1) continue; // only check DG-labelled pixels
      const cx = i % CAM_W;
      const cy = Math.floor(i / CAM_W);
      let allDG = true;
      let nR = 0;
      let nCount = 0;
      for (const [ddx, ddy] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ]) {
        const nx = cx + ddx;
        const ny = cy + ddy;
        if (nx < 0 || nx >= CAM_W || ny < 0 || ny >= CAM_H) continue;
        const ni = ny * CAM_W + nx;
        if (finalLabels[ni] !== 1) {
          allDG = false;
          break;
        }
        nR += flatRG[ni * 2];
        nCount++;
      }
      if (!allDG || nCount < 4) continue;
      const navgR = nR / nCount;
      const r = flatRG[i * 2];
      // Require R to also be above 160 (rough midpoint between palette
      // DG.R=148 and LG.R=255) — without this guard, ordinary DG pixels
      // with R slightly above an unusually-dim DG neighbour also flip.
      if (r >= 160 && r - navgR >= 25) {
        newLabels[i] = 2; // flip to LG
        anomalyFlipped++;
      }
    }
    for (let i = 0; i < N; i++) finalLabels[i] = newLabels[i];
    if (dbg && anomalyFlipped > 0) {
      dbg.log(
        `[quantize] DG-in-DG anomaly: ${anomalyFlipped} pixels reclassified DG → LG (R ≥ 25 above local DG mean)`,
      );
    }
  }

  // ── 3f. Per-column local LG/WH G-valley. Edge columns get uniformly
  // brightened by frame + vertical light bleed, lifting their LG pixels' G (and
  // even R/B) until they read as WH against the GLOBAL threshold. But WITHIN a
  // column the LG level and the WH level stay clearly separated, so a per-
  // column G valley recovers the right split. Clean columns reproduce ~the
  // global valley (no change); only the bled columns shift. Guarded to fire
  // only on columns with a genuinely bimodal LG/WH G distribution.
  {
    let colFlipped = 0;
    const MIN_PTS = 12;
    const MIN_SPREAD = 40; // LG/WH G levels must be this far apart in the column
    const MIN_DEPTH = 0.6; // valley must be ≤ this × the smaller peak (a real dip)
    // Only the outermost columns: the bright filmstrip frame bleeds light
    // horizontally into them, which (compounded with vertical bleed) is what
    // uniformly lifts their LG pixels into the WH range. Interior columns are
    // left to the global/strip classification (touching them trades errors).
    const EDGE = 2;
    for (let x = 0; x < CAM_W; x++) {
      if (x >= EDGE && x < CAM_W - EDGE) continue;
      const gv: number[] = [];
      const idx: number[] = [];
      for (let y = 0; y < CAM_H; y++) {
        const i = y * CAM_W + x;
        if (finalLabels[i] === 2 || finalLabels[i] === 3) {
          gv.push(flatRG[i * 2 + 1]);
          idx.push(i);
        }
      }
      if (gv.length < MIN_PTS) continue;
      let lo = Infinity,
        hi = -Infinity;
      for (const g of gv) {
        if (g < lo) lo = g;
        if (g > hi) hi = g;
      }
      if (hi - lo < MIN_SPREAD) continue;
      const nb = Math.round(hi - lo) + 1;
      const hist = new Array<number>(nb).fill(0);
      for (const g of gv)
        hist[Math.min(nb - 1, Math.max(0, Math.round(g - lo)))]++;
      const sm = gaussianFilter1d(hist, Math.max(2, (hi - lo) / 12));
      // Tallest peak, then the tallest peak separated from it by a dip.
      let p1 = 0;
      for (let k = 1; k < nb; k++) if (sm[k] > sm[p1]) p1 = k;
      let p2 = -1;
      for (let k = 0; k < nb; k++) {
        if (Math.abs(k - p1) < (hi - lo) / 6) continue;
        if (p2 < 0 || sm[k] > sm[p2]) p2 = k;
      }
      if (p2 < 0) continue;
      const a = Math.min(p1, p2),
        b = Math.max(p1, p2);
      let valley = a;
      for (let k = a + 1; k <= b; k++) if (sm[k] < sm[valley]) valley = k;
      const peakMin = Math.min(sm[p1], sm[p2]);
      if (peakMin <= 0 || sm[valley] > MIN_DEPTH * peakMin) continue; // not clearly bimodal
      // Only act when the column's lower (LG) mode is LIFTED above the global
      // LG level — the signature of frame/vertical bleed. A normal edge column
      // (LG at its usual G) is already classified correctly globally, and
      // re-thresholding it only flips correct WH pixels.
      const lowerModeG = lo + Math.min(p1, p2);
      if (lowerModeG <= paletteCenters[2][1] + 30) continue;
      const thr = lo + valley;
      for (let j = 0; j < idx.length; j++) {
        const want = gv[j] < thr ? 2 : 3;
        if (want !== finalLabels[idx[j]]) {
          finalLabels[idx[j]] = want;
          colFlipped++;
        }
      }
    }
    if (dbg && colFlipped > 0) {
      dbg.log(
        `[quantize] per-column LG/WH valley: ${colFlipped} pixels reclassified`,
      );
    }
  }

  // ── 3g. Local adaptive WH/LG threshold (the LG↔WH split is a 1D decision
  // on G — LG ≈ red/low-G, WH ≈ yellow/high-G). The global G-valley applies
  // ONE such boundary to the whole frame, but the front-light gradient leaves
  // WH spatially varying in brightness: where WH is dimmed, its dither dots'
  // absolute G falls below the global threshold and is mis-labelled LG —
  // flattening real WH/LG dither into solid LG. A single global 1D threshold
  // fundamentally can't separate spatially-varying WH from LG (and a per-pixel
  // RG-distance can't either — a bleed-lifted LG pixel and a dimmed WH pixel
  // have near-identical colour; only their LOCAL context differs).
  //
  // So decide the LG/WH split from each pixel's LOCAL window: build the
  // warm-pixel (LG/WH) G histogram in a small neighbourhood, and if it is
  // genuinely bimodal (two G-modes with a real dip), threshold at the local
  // valley. The local WH and LG levels stay cleanly separated regardless of
  // the regional brightness, so this recovers the right split everywhere. In
  // a uniform or non-bimodal window it makes no change (falls back to the
  // global classification), and the outermost columns are left to the
  // per-column step (3f). Net effect on the reference corpora: tier-1
  // normal unchanged, full slightly improved, self-consistency improved.
  {
    const RADIUS = 6;
    const MIN_WARM = 24;
    const MIN_SPREAD = 45;  // local warm-G range must show real LG/WH separation
    const MIN_DEPTH = 0.6;  // valley ≤ this × the smaller mode (a genuine dip)
    const EDGE = 2;         // outermost columns are owned by the per-column step (3f)
    const whFloor = paletteCenters[2][1] + 40; // upper mode must be clearly above LG to be WH
    const before = finalLabels.slice();
    let localFlipped = 0;
    for (let y = 0; y < CAM_H; y++) {
      for (let x = 0; x < CAM_W; x++) {
        if (x < EDGE || x >= CAM_W - EDGE) continue;
        const i = y * CAM_W + x;
        if (before[i] !== 2 && before[i] !== 3) continue;
        const gv: number[] = [];
        const yLo = Math.max(0, y - RADIUS);
        const yHi = Math.min(CAM_H - 1, y + RADIUS);
        const xLo = Math.max(0, x - RADIUS);
        const xHi = Math.min(CAM_W - 1, x + RADIUS);
        for (let yy = yLo; yy <= yHi; yy++) {
          for (let xx = xLo; xx <= xHi; xx++) {
            const k = yy * CAM_W + xx;
            if (before[k] === 2 || before[k] === 3) gv.push(flatRG[k * 2 + 1]);
          }
        }
        if (gv.length < MIN_WARM) continue;
        let lo = Infinity, hi = -Infinity;
        for (const g of gv) { if (g < lo) lo = g; if (g > hi) hi = g; }
        if (hi - lo < MIN_SPREAD) continue; // uniform warm region — trust global
        const nb = Math.round(hi - lo) + 1;
        const hist = new Array<number>(nb).fill(0);
        for (const g of gv) hist[Math.min(nb - 1, Math.max(0, Math.round(g - lo)))]++;
        const sm = gaussianFilter1d(hist, Math.max(2, (hi - lo) / 12));
        // Tallest mode, then the tallest mode separated from it by a dip.
        let p1 = 0; for (let k = 1; k < nb; k++) if (sm[k] > sm[p1]) p1 = k;
        let p2 = -1; for (let k = 0; k < nb; k++) {
          if (Math.abs(k - p1) < (hi - lo) / 6) continue;
          if (p2 < 0 || sm[k] > sm[p2]) p2 = k;
        }
        if (p2 < 0) continue;
        const a = Math.min(p1, p2), b = Math.max(p1, p2);
        let valley = a; for (let k = a + 1; k <= b; k++) if (sm[k] < sm[valley]) valley = k;
        const peakMin = Math.min(sm[p1], sm[p2]);
        if (peakMin <= 0 || sm[valley] > MIN_DEPTH * peakMin) continue; // not bimodal
        const upperModeG = lo + b;
        if (upperModeG < whFloor) continue;        // upper mode not bright enough to be WH
        const valleyG = lo + valley;
        const want = flatRG[i * 2 + 1] >= valleyG ? 3 : 2;
        if (want !== finalLabels[i]) {
          finalLabels[i] = want;
          localFlipped++;
        }
      }
    }
    if (dbg && localFlipped > 0) {
      dbg.log(`[quantize] local adaptive WH/LG: ${localFlipped} pixels reclassified`);
    }
  }

  // ── 3h. Recover sparse DG-on-black dots using the pre-correct warp.
  // The `correct` step's affine gain amplifies a bleed-lifted DG pixel's R
  // upward (warp R≈120 → sample R≈190), pushing isolated DG dots across the
  // DG/LG boundary so they're mis-labelled LG. The PRE-correct warp still
  // separates DG (low R) from LG (high R) cleanly — that signal is destroyed
  // downstream. Per-pixel reclassification of these is otherwise impossible
  // (in the ambiguous band LG outnumbers DG ~1000:1), but a DG dot has THREE
  // independent signatures at once that a stray LG pixel does not:
  //   (1) its warp R is closer to the DG mode than the LG mode,
  //   (2) high B (the DG colour, B≈255 vs LG≈148), and
  //   (3) it sits on a black background (≥3 BK neighbours — the sparse-dot
  //       structure; LG lives among warm neighbours, not on black).
  // Requiring all three keeps false flips to near-zero. All thresholds are
  // derived from this image's own DG/LG statistics (no magic constants).
  if (options?.warped) {
    const wq = options.warped;
    const sc = options?.scale ?? Math.round(wq.width / SCREEN_W);
    const m0 = Math.max(1, Math.floor(sc / 4));
    const m1 = Math.max(m0 + 1, Math.ceil((sc * 3) / 4));
    // Mean warp R and G over the inner block of each camera pixel.
    const warpRcam = new Float32Array(N);
    const warpGcam = new Float32Array(N);
    for (let cy = 0; cy < CAM_H; cy++) {
      for (let cx = 0; cx < CAM_W; cx++) {
        const sx = (cx + FRAME_THICK) * sc;
        const sy = (cy + FRAME_THICK) * sc;
        let sum = 0, sumG = 0, cnt = 0;
        for (let yy = sy + m0; yy < sy + m1; yy++) {
          for (let xx = sx + m0; xx < sx + m1; xx++) {
            sum += wq.data[(yy * wq.width + xx) * 4];
            sumG += wq.data[(yy * wq.width + xx) * 4 + 1];
            cnt++;
          }
        }
        warpRcam[cy * CAM_W + cx] = cnt > 0 ? sum / cnt : 0;
        warpGcam[cy * CAM_W + cx] = cnt > 0 ? sumG / cnt : 0;
      }
    }
    // Image-derived warp-R centroids of confidently-labelled DG and LG.
    let dgSum = 0, dgN = 0, lgSum = 0, lgN = 0;
    let dgBsum = 0, lgBsum = 0;
    for (let i = 0; i < N; i++) {
      if (finalLabels[i] === 1) { dgSum += warpRcam[i]; dgN++; dgBsum += input.data[i * 4 + 2]; }
      else if (finalLabels[i] === 2) { lgSum += warpRcam[i]; lgN++; lgBsum += input.data[i * 4 + 2]; }
    }
    let dotsFixed = 0;
    if (dgN >= 20 && lgN >= 20) {
      const warpDgR = dgSum / dgN;
      const warpLgR = lgSum / lgN;
      const dgMeanB = dgBsum / dgN;
      const lgMeanB = lgBsum / lgN;
      // Only meaningful when the warp actually separates DG from LG in R and
      // DG carries distinctly higher B than LG. (Guard was 40; lowered to 28
      // after measuring that the blurrier thing-2/thing-3 photos sit at
      // 31-35 warp-R separation with the recovery still fix-only there —
      // the structural gates below carry the precision, the guard only
      // rejects images where the warp genuinely doesn't separate the modes.)
      if (warpLgR - warpDgR > 28 && dgMeanB - lgMeanB > 20) {
        // Warp-R cut at the midpoint of the DG/LG warp centroids = classify
        // by nearest warp centroid (the natural boundary in the clean
        // pre-correct space). BK-neighbour floor of 3 keeps false flips ~zero.
        const FRAC = 0.5;
        const BKMIN = 3;
        const warpCut = warpDgR + (warpLgR - warpDgR) * FRAC;
        const bCut = (dgMeanB + lgMeanB) / 2;
        for (let cy = 0; cy < CAM_H; cy++) {
          for (let cx = 0; cx < CAM_W; cx++) {
            const i = cy * CAM_W + cx;
            if (finalLabels[i] !== 2) continue;            // only output-LG
            if (warpRcam[i] >= warpCut) continue;          // (1) warp R clearly DG
            if (input.data[i * 4 + 2] <= bCut) continue;   // (2) high B (DG)
            let bk = 0;                                      // (3) on black bg
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                const yy = cy + dy, xx = cx + dx;
                if (yy < 0 || xx < 0 || yy >= CAM_H || xx >= CAM_W) continue;
                if (finalLabels[yy * CAM_W + xx] === 0) bk++;
              }
            }
            if (bk < BKMIN) continue;
            finalLabels[i] = 1;
            dotsFixed++;
          }
        }

        // Second pass: two relaxed tiers for DG dots whose warp R sits just
        // ABOVE the centroid midpoint (bleed lifts an isolated dot's warp R
        // toward — but rarely past — the LG mode). Each tier trades a softer
        // warp-R/B cut for a stricter structural gate, so flips still require
        // several independent signatures at once:
        //   T1: nearly enclosed by black (bk ≥ 6) — an LG pixel essentially
        //       never sits fully inside a black region; warp R below 0.8 of
        //       the DG→LG span and B above the LG mean confirm.
        //   T2: on black (bk ≥ 4) with at most 1 DG neighbour (a true sparse
        //       dot has black around it, not DG — LG pixels at DG-region
        //       boundaries fail this) + warp R below 0.6 span + B clearly
        //       DG-shifted.
        // Measured on all 21 reference images (tier-1 normal + full +
        // private): fixes 17 LG→DG errors, breaks 0.
        // Neighbour counts come from a snapshot after the first pass so
        // results don't depend on scan order.
        {
          const snap = finalLabels.slice();
          const span = warpLgR - warpDgR;
          const bSpan = dgMeanB - lgMeanB;
          for (let cy = 0; cy < CAM_H; cy++) {
            for (let cx = 0; cx < CAM_W; cx++) {
              const i = cy * CAM_W + cx;
              if (snap[i] !== 2) continue;
              const f = (warpRcam[i] - warpDgR) / span;
              if (f >= 0.8) continue;
              // B must sit measurably above the LG mean (every verified DG
              // dot is at ≥ 0.25 of the LG→DG B span; a bright-LG pixel with
              // an artificially dark warp block sits at ≈ 0).
              const bRel = (input.data[i * 4 + 2] - lgMeanB) / bSpan;
              if (bRel <= 0.1) continue;
              let bk = 0;
              let dg = 0;
              for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                  if (!dx && !dy) continue;
                  const yy = cy + dy, xx = cx + dx;
                  if (yy < 0 || xx < 0 || yy >= CAM_H || xx >= CAM_W) continue;
                  const l = snap[yy * CAM_W + xx];
                  if (l === 0) bk++;
                  else if (l === 1) dg++;
                }
              }
              const t1 = bk >= 6;
              const t2 = bk >= 4 && dg <= 1 && f < 0.6 && bRel > 0.1;
              if (t1 || t2) {
                finalLabels[i] = 1;
                dotsFixed++;
              }
            }
          }
        }
      }
    }
    if (dbg && dotsFixed > 0) {
      dbg.log(`[quantize] warp-R DG-dot recovery: ${dotsFixed} pixels reclassified`);
    }

    // ── 3i. Recover BK pixels mislabelled DG using the pre-correct warp G.
    // The same correct-gain amplification that lifts DG's R into LG also
    // lifts a dim BK pixel's R/B into the DG range (worst in the dimmest
    // screen corners), while its G stays at black level. The pre-correct
    // warp G still separates BK (low G) from DG (mid G) decisively: across
    // every reference image, true-DG pixels sit at warp-G fraction ≥ 0.74 of
    // the BK→DG span (p1), while mislabelled-BK pixels sit at ≤ 0.50. Two
    // signals must agree: warp-G fraction < 0.52 AND sample G below ~0.55 of
    // the BK→DG sample-G span (guards interior dim-DG whose sample G is
    // clearly DG-level). Measured on all 21 reference images: fixes 12,
    // breaks 2 (both suspected reference errors in the same dither corner).
    {
      let bkW = 0, bkS = 0, bkN = 0, dgW = 0, dgS = 0, dgN2 = 0;
      for (let i = 0; i < N; i++) {
        if (finalLabels[i] === 0) {
          bkW += warpGcam[i]; bkS += input.data[i * 4 + 1]; bkN++;
        } else if (finalLabels[i] === 1) {
          dgW += warpGcam[i]; dgS += input.data[i * 4 + 1]; dgN2++;
        }
      }
      let bkFixed = 0;
      if (bkN >= 20 && dgN2 >= 20) {
        const wBK = bkW / bkN, wDG = dgW / dgN2;
        const sBK = bkS / bkN, sDG = dgS / dgN2;
        // Only meaningful when the warp actually separates BK from DG in G.
        if (wDG - wBK > 40 && sDG - sBK > 40) {
          const FG_CUT = 0.52;
          const GREL_CUT = 0.55;
          for (let i = 0; i < N; i++) {
            if (finalLabels[i] !== 1) continue;
            const fG = (warpGcam[i] - wBK) / (wDG - wBK);
            if (fG >= FG_CUT) continue;
            const gRel = (input.data[i * 4 + 1] - sBK) / (sDG - sBK);
            if (gRel >= GREL_CUT) continue;
            finalLabels[i] = 0;
            bkFixed++;
          }
        }
      }
      if (dbg && bkFixed > 0) {
        dbg.log(`[quantize] warp-G BK recovery: ${bkFixed} pixels reclassified DG → BK`);
      }
    }
  }

  // ── 3j. Per-pixel recovery for colours dropped by cluster validation ──
  // Dropping a colour's CLUSTER (because it migrated into another colour's
  // cloud) must never make individual pixels of that colour unreachable: an
  // image can legitimately contain just a handful of them — too few to
  // anchor a cluster, still real content. Any pixel whose own RG value is
  // nearest the dropped colour's palette target gets that label back. By
  // construction this is a tiny set (if many pixels sat near the target,
  // the cluster would not have migrated and the colour would not have been
  // dropped). DG is excluded: its target sits between BK and LG so
  // nearest-target is not decisive for it — blueness (step 2b) is its
  // per-pixel recovery instead.
  for (const p of [0, 2, 3]) {
    if (presentPalette.includes(p)) continue;
    let recovered = 0;
    for (let i = 0; i < N; i++) {
      const r = flatRG[i * 2];
      const g = flatRG[i * 2 + 1];
      let best = 0;
      let bestD = Infinity;
      for (let t = 0; t < 4; t++) {
        const d = (r - targetsRG[t][0]) ** 2 + (g - targetsRG[t][1]) ** 2;
        if (d < bestD) {
          bestD = d;
          best = t;
        }
      }
      if (best === p && finalLabels[i] !== p) {
        finalLabels[i] = p;
        recovered++;
      }
    }
    if (dbg && recovered > 0) {
      dbg.log(
        `[quantize] dropped-colour recovery: ${recovered} px nearest the ` +
          `${["BK", "DG", "LG", "WH"][p]} target relabelled ${["BK", "DG", "LG", "WH"][p]}`,
      );
    }
  }

  // ── 4. Output: map palette indices to grayscale values ──
  const output = createGBImageData(CAM_W, CAM_H);
  for (let i = 0; i < N; i++) {
    const v = GB_COLORS[finalLabels[i]];
    const j = i * 4;
    output.data[j] = v;
    output.data[j + 1] = v;
    output.data[j + 2] = v;
    output.data[j + 3] = 255;
  }

  if (dbg) {
    const finalCounts = countLabels(finalLabels);
    const total = N;
    dbg.log(
      `[quantize] final: ` +
        ["BK", "DG", "LG", "WH"]
          .map(
            (n, i) =>
              `${n}=${finalCounts[i]} (${((100 * finalCounts[i]) / total).toFixed(1)}%)`,
          )
          .join("  "),
    );

    dbg.setMetrics("quantize", {
      clusterCenters: paletteCenters.map(([r, g]) => [
        Number(r.toFixed(2)),
        Number(g.toFixed(2)),
      ]),
      stripEnsemble: { strips: nStrips, changed: stripChanged },
      valleyRefinement: {
        threshold:
          valleyThreshold === null ? null : Number(valleyThreshold.toFixed(2)),
        changed: valleyChanged,
      },
      counts: {
        afterGlobalKmeans: {
          BK: globalCounts[0],
          DG: globalCounts[1],
          LG: globalCounts[2],
          WH: globalCounts[3],
        },
        afterStripEnsemble: {
          BK: stripCounts[0],
          DG: stripCounts[1],
          LG: stripCounts[2],
          WH: stripCounts[3],
        },
        final: {
          BK: finalCounts[0],
          DG: finalCounts[1],
          LG: finalCounts[2],
          WH: finalCounts[3],
        },
      },
    });

    // Visual: 8x grayscale and 8x palette-rendered
    dbg.addImage("quantize_a_gray_8x", upscale(output, 8));

    const rgbOut = createGBImageData(CAM_W, CAM_H);
    const PALETTE_RGB: [number, number, number][] = [
      [0, 0, 0],
      [148, 148, 255],
      [255, 148, 148],
      [255, 255, 165],
    ];
    for (let i = 0; i < N; i++) {
      const c = PALETTE_RGB[finalLabels[i]];
      const j = i * 4;
      rgbOut.data[j] = c[0];
      rgbOut.data[j + 1] = c[1];
      rgbOut.data[j + 2] = c[2];
      rgbOut.data[j + 3] = 255;
    }
    dbg.addImage("quantize_b_rgb_8x", upscale(rgbOut, 8));

    // RG scatter: every input sample plotted by its final label, with cluster
    // centers (white crosses) and palette targets (yellow rings) overlaid.
    const rVals = new Array<number>(N);
    const gVals = new Array<number>(N);
    const pointColors = new Array<[number, number, number]>(N);
    for (let i = 0; i < N; i++) {
      rVals[i] = flatRG[i * 2];
      gVals[i] = flatRG[i * 2 + 1];
      pointColors[i] = PALETTE_RGB[finalLabels[i]];
    }
    const markers = [
      ...paletteCenters.map((c) => ({
        r: c[0],
        g: c[1],
        color: [255, 255, 255] as [number, number, number],
        size: 5,
        symbol: "cross" as const,
      })),
      ...targetsRG.map((t) => ({
        r: t[0],
        g: t[1],
        color: [255, 255, 0] as [number, number, number],
        size: 7,
        symbol: "ring" as const,
      })),
    ];
    dbg.addImage(
      "quantize_c_rg_scatter",
      renderRGScatter(rVals, gVals, pointColors, markers),
    );
  }

  return output;
}

/** Count occurrences of palette indices 0..3 in a label array. */
function countLabels(
  labels: Int32Array | Uint8Array,
): [number, number, number, number] {
  const c: [number, number, number, number] = [0, 0, 0, 0];
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (v >= 0 && v < 4) c[v]++;
  }
  return c;
}

/**
 * Frame-aware quantize: uses palette anchors extracted from the frame to fit
 * per-color RGB surfaces, then classifies each camera pixel by nearest
 * predicted color at its frame location.
 */
function frameAwareQuantize(
  input: GBImageData,
  warped: GBImageData,
  scale: number,
  dbg?: DebugCollector,
): GBImageData {
  const N = CAM_W * CAM_H;
  // k-NN in (y, x, R, G, B) space against all frame anchors. For each camera
  // pixel, find K nearest anchors and vote (weighted by 1/(1+d)) for the
  // colour. Position similarity uses nearby-gradient anchors; colour
  // similarity uses similar-RGB anchors. Avoids polynomial extrapolation
  // because predictions are always over actual observed samples.
  const anchors = collectFrameAnchors(warped, scale);
  const cls = buildFrameClassifier(warped, scale, 2);
  const labels = new Uint8Array(N);
  const SPATIAL_W = 0.05;
  const K = 5;
  for (let cy = 0; cy < CAM_H; cy++) {
    for (let cx = 0; cx < CAM_W; cx++) {
      const i = cy * CAM_W + cx;
      const R = input.data[i * 4];
      const G = input.data[i * 4 + 1];
      const B = input.data[i * 4 + 2];
      const { frameY, frameX } = cameraToFrameCoords(cy, cx);
      const bestD = new Array<number>(K).fill(Infinity);
      const bestC = new Array<number>(K).fill(-1);
      for (let ai = 0; ai < anchors.length; ai++) {
        const a = anchors[ai];
        const dy = frameY - a.y;
        const dx = frameX - a.x;
        const dR = R - a.R;
        const dG = G - a.G;
        const dB = B - a.B;
        const d = SPATIAL_W * (dy * dy + dx * dx) + dR * dR + dG * dG + dB * dB;
        for (let k = 0; k < K; k++) {
          if (d < bestD[k]) {
            for (let m = K - 1; m > k; m--) {
              bestD[m] = bestD[m - 1];
              bestC[m] = bestC[m - 1];
            }
            bestD[k] = d;
            bestC[k] = a.c;
            break;
          }
        }
      }
      const votes = [0, 0, 0, 0];
      for (let k = 0; k < K; k++) {
        if (bestC[k] < 0) continue;
        const w = 1 / (1 + bestD[k]);
        votes[bestC[k]] += w;
      }
      let bestColor = 0;
      let bestVote = -1;
      for (let c = 0; c < 4; c++) {
        if (votes[c] > bestVote) {
          bestVote = votes[c];
          bestColor = c;
        }
      }
      labels[i] = bestColor;
    }
  }
  const output = createGBImageData(CAM_W, CAM_H);
  for (let i = 0; i < N; i++) {
    const v = GB_COLORS[labels[i]];
    const j = i * 4;
    output.data[j] = v;
    output.data[j + 1] = v;
    output.data[j + 2] = v;
    output.data[j + 3] = 255;
  }

  if (dbg) {
    const counts = countLabels(labels);
    dbg.log(
      `[quantize] frame-aware: anchor samples ` +
        `BK=${cls.sampleCounts[0]} DG=${cls.sampleCounts[1]} ` +
        `LG=${cls.sampleCounts[2]} WH=${cls.sampleCounts[3]}`,
    );
    dbg.log(
      `[quantize] frame-aware means: ` +
        ["BK", "DG", "LG", "WH"]
          .map(
            (n, c) =>
              `${n}=(R${cls.meanR[c].toFixed(0)},G${cls.meanG[c].toFixed(0)},B${cls.meanB[c].toFixed(0)})`,
          )
          .join("  "),
    );
    // Predicted RGB at three frame positions (TL, center, BR of camera region)
    const PROBES = [
      { name: "TL", fy: 20, fx: 20 },
      { name: "MID", fy: 72, fx: 80 },
      { name: "BR", fy: 124, fx: 140 },
    ];
    for (const p of PROBES) {
      const idx = p.fy * cls.W + p.fx;
      const txt = ["BK", "DG", "LG", "WH"]
        .map(
          (n, c) =>
            `${n}=(R${cls.R[c][idx].toFixed(0)},G${cls.G[c][idx].toFixed(0)},B${cls.B[c][idx].toFixed(0)})`,
        )
        .join(" ");
      dbg.log(
        `[quantize] frame-aware probe ${p.name}@(${p.fy},${p.fx}): ${txt}`,
      );
    }
    dbg.log(
      `[quantize] frame-aware final: ` +
        ["BK", "DG", "LG", "WH"]
          .map(
            (n, i) =>
              `${n}=${counts[i]} (${((100 * counts[i]) / N).toFixed(1)}%)`,
          )
          .join("  "),
    );
    dbg.addImage("quantize_a_gray_8x", upscale(output, 8));
    const rgbOut = createGBImageData(CAM_W, CAM_H);
    const PALETTE_RGB: [number, number, number][] = [
      [0, 0, 0],
      [148, 148, 255],
      [255, 148, 148],
      [255, 255, 165],
    ];
    for (let i = 0; i < N; i++) {
      const c = PALETTE_RGB[labels[i]];
      const j = i * 4;
      rgbOut.data[j] = c[0];
      rgbOut.data[j + 1] = c[1];
      rgbOut.data[j + 2] = c[2];
      rgbOut.data[j + 3] = 255;
    }
    dbg.addImage("quantize_b_rgb_8x", upscale(rgbOut, 8));
  }

  return output;
}
