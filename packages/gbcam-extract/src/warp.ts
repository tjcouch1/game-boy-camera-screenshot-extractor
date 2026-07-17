/**
 * warp.ts — Perspective correction with inner-border refinement.
 *
 * Processing:
 *   1. Detect the four corners of the white filmstrip frame using brightness
 *      thresholding and contour analysis.
 *   2. Apply an initial perspective warp to (SCREEN_W*scale) x (SCREEN_H*scale).
 *   3. Two passes of perspective refinement: detect inner-border points,
 *      fit quadratic polynomials per edge, back-project corrected corners
 *      and re-warp.
 *   4. One non-linear remap pass: residual S-curve / lens-style curvature
 *      that a perspective transform cannot fix is removed by remapping each
 *      row's x-range and each column's y-range using the polynomial fits.
 */

import { type GBImageData, SCREEN_W, SCREEN_H, INNER_TOP, INNER_BOT, INNER_LEFT, INNER_RIGHT } from "./common.js";
import { getCV, withMats, imageDataToMat, matToImageData } from "./opencv.js";
import {
  type DebugCollector,
  cloneImage,
  drawLine,
  drawPolyline,
  fillCircle,
} from "./debug.js";

// ─── Public interface ───

export interface WarpOptions {
  scale?: number;
  threshold?: number;
  debug?: DebugCollector;
  /**
   * Optional out-param: warp fills in quality stats the orchestrator uses
   * for processing-issue reporting (works without `debug`).
   */
  stats?: { quadScore?: number };
}

export function warp(input: GBImageData, options?: WarpOptions): GBImageData {
  const scale = options?.scale ?? 8;
  // Starting brightness threshold for finding the white filmstrip frame
  // (#FFFFA5 reads ≈200+ even when dimmed; the surrounding LCD-black ring
  // reads far below). This is only the first attempt — corner detection
  // steps the threshold down internally when no acceptable quad is found.
  const threshVal = options?.threshold ?? 180;
  const dbg = options?.debug;

  const cv = getCV();

  const src = imageDataToMat(input);
  const bgr = new cv.Mat();
  cv.cvtColor(src, bgr, cv.COLOR_RGBA2BGR);
  src.delete();

  // a — Detect screen corners
  const detection = findScreenCornersWithMetrics(bgr, threshVal);
  const corners = detection.ordered;
  if (options?.stats) options.stats.quadScore = detection.score;

  if (dbg) {
    dbg.log(
      `[warp] corner detection: area=${detection.area.toFixed(0)} ` +
        `aspect=${detection.aspect.toFixed(3)} ` +
        `(target ${(SCREEN_W / SCREEN_H).toFixed(3)}) ` +
        `thresh=${detection.thresh} score=${detection.score.toFixed(4)}`,
    );
    dbg.log(
      `[warp] detected source corners (TL TR BR BL): ` +
        corners.map((c) => `(${Math.round(c[0])},${Math.round(c[1])})`).join(" "),
    );
    if (detection.score > 0.15) {
      dbg.log(`[warp] WARNING: quad quality is low — detection may be unreliable`);
    }

    const overlay = cloneImage(input);
    const green: [number, number, number] = [0, 255, 0];
    const cornerRadius = Math.max(6, Math.round(Math.min(input.width, input.height) / 200));
    for (const [x, y] of corners) fillCircle(overlay, x, y, cornerRadius, green);
    const polyThick = Math.max(2, Math.round(cornerRadius / 4));
    drawPolyline(overlay, corners.map(([x, y]) => [x, y] as [number, number]), green, polyThick, true);
    dbg.addImage("warp_a_corners", overlay);

    dbg.setMetrics("warp", {
      threshold: detection.thresh,
      contourArea: Math.round(detection.area),
      aspect: Number(detection.aspect.toFixed(4)),
      quadScore: Number(detection.score.toFixed(4)),
      sourceCorners: corners.map(([x, y]) => [Math.round(x), Math.round(y)]),
    });
  }

  // b — Initial perspective warp
  const init = initialWarp(bgr, corners, scale);

  // c — Perspective refinement passes (2 iterations is enough; a 3rd doesn't
  // measurably converge further given polynomial-fit noise).
  //
  // Each pass re-detects the frame border in the current warp and rebuilds
  // the transform by back-projecting onto the original photo — so every
  // pass's output is an INDEPENDENT single resample of the source, not a
  // re-warp of the previous warp (no accumulated blur). That lets us treat
  // the passes as candidates and keep whichever is best-aligned.
  //
  // Why this matters: the refinement is normally convergent (each pass
  // leaves the frame edges better aligned), but when an edge's border
  // detection is biased — a blurry or doubled edge, e.g. some full-photo
  // left edges where the dim leftmost B sub-pixel column muddies the
  // WH→LCD transition — the correction feeds back positively and that edge
  // drifts a full GB pixel pass-over-pass (observed: left edge 3.8→7.9 px,
  // producing a doubled/folded left border). The guard below detects that
  // (a later pass is worse-aligned than an earlier one) and keeps the
  // best pass instead. Well-behaved images always converge, so warp2 stays
  // the best candidate and behaviour is unchanged.
  const r1 = refineWarpWithMetrics(bgr, init.M, init.warped, scale);
  if (dbg) recordRefinementMetrics(dbg, 1, r1.metrics);
  const m0 = maxAbsEdgeCurvature(r1.metrics.edgeCurvatures); // offsets on init warp
  const r2 = refineWarpWithMetrics(bgr, r1.M, r1.refined, scale);
  if (dbg) recordRefinementMetrics(dbg, 2, r2.metrics);
  const m1 = maxAbsEdgeCurvature(r2.metrics.edgeCurvatures); // offsets on warp1
  const m2 = measureMaxEdgeOffset(r2.refined, scale);        // offsets on warp2

  // Candidates: 0 = initial warp, 1 = after pass 1, 2 = after pass 2.
  // Default to the fully-refined warp2; only fall back to an earlier pass
  // when it is better-aligned by a clear margin, so converging images
  // (every reliable photo) are completely unaffected.
  // 1.5 px = well above the ~0.5 px run-to-run noise of the edge-offset
  // measurement, well below the ~8 px (one GB pixel) divergence a blurry
  // doubled edge produces — anywhere in that range behaves the same.
  const DIVERGENCE_MARGIN = 1.5;
  const candidates: Array<{ warped: any; M: any; maxOff: number }> = [
    { warped: init.warped, M: init.M, maxOff: m0 },
    { warped: r1.refined, M: r1.M, maxOff: m1 },
    { warped: r2.refined, M: r2.M, maxOff: m2 },
  ];
  let chosen = 2;
  for (let i = 1; i >= 0; i--) {
    if (candidates[i].maxOff + DIVERGENCE_MARGIN < candidates[chosen].maxOff) {
      chosen = i;
    }
  }
  if (dbg && chosen !== 2) {
    dbg.log(
      `[warp] refinement diverged — using pass ${chosen} ` +
        `(max edge offset ${candidates[chosen].maxOff.toFixed(2)} px vs ` +
        `pass-2 ${m2.toFixed(2)} px)`,
    );
  }
  let currentWarped = candidates[chosen].warped;
  let currentM = candidates[chosen].M;
  for (let i = 0; i < candidates.length; i++) {
    if (i !== chosen) {
      candidates[i].warped.delete();
      candidates[i].M.delete();
    }
  }

  // d — Non-linear remap pass (handles residual lens-style curvature that
  // a perspective transform cannot remove). Fuses with the pass-2 perspective
  // inverse so we resample the source photo only once (avoids double-Lanczos
  // quality loss). When residual curvature is negligible the remap is skipped
  // and the pass-2 warp is used as-is.
  {
    const remapResult = nonLinearRemap(bgr, currentM, currentWarped, scale);
    if (remapResult.applied) {
      currentWarped.delete();
      currentWarped = remapResult.remapped;
    } else {
      remapResult.remapped.delete();
    }
    if (dbg) recordNonLinearMetrics(dbg, remapResult);
  }

  bgr.delete();
  currentM.delete();

  // e — Sub-pixel rectification pass: detect the BGR sub-pixel phase from the
  // WH frame columns above and below the camera area, and remap each row so
  // LCD pixel boundaries align uniformly with the 8-col output grid. Fixes
  // residual lens-distortion drift the perspective transform cannot model.
  // The per-row drift down each edge is taken from the WH-frame vertical
  // frame lines (reliable into the blurry bottom corners) rather than the
  // bottom strip (which collapses there).
  {
    const subPixelResult = subPixelRectify(currentWarped, scale);
    if (subPixelResult.applied) {
      currentWarped.delete();
      currentWarped = subPixelResult.rectified;
    } else {
      subPixelResult.rectified.delete();
    }
    if (dbg) recordSubPixelMetrics(dbg, subPixelResult);
  }

  // Render the final border-diagnostic image (so it reflects the post-remap state)
  if (dbg) renderBorderDiagnostic(dbg, currentWarped, scale);

  const rgba = new cv.Mat();
  cv.cvtColor(currentWarped, rgba, cv.COLOR_BGR2RGBA);
  currentWarped.delete();
  const result = matToImageData(rgba);
  rgba.delete();

  return result;
}

// ─── Refinement metrics recorder ───

interface RefineMetrics {
  edgeCurvatures: { top: number; bottom: number; left: number; right: number };
  cornerErrors: {
    TL: [number, number];
    TR: [number, number];
    BR: [number, number];
    BL: [number, number];
  };
  polyResidual: { top: number; bottom: number; left: number; right: number };
  subPixelGap: { left: number; right: number; top: number; bottom: number };
  refined: boolean;
}

function recordRefinementMetrics(
  dbg: DebugCollector,
  passNum: number,
  m: RefineMetrics,
): void {
  const ec = m.edgeCurvatures;
  dbg.log(
    `[warp] pass ${passNum} edge offsets (avg): ` +
      `top=${ec.top.toFixed(2)} bot=${ec.bottom.toFixed(2)} ` +
      `left=${ec.left.toFixed(2)} right=${ec.right.toFixed(2)}` +
      (m.refined ? "" : "  (refinement failed; using prior warp)"),
  );
  const ce = m.cornerErrors;
  dbg.log(
    `[warp] pass ${passNum} corner errors: ` +
      `TL=(${ce.TL[0].toFixed(1)},${ce.TL[1].toFixed(1)}) ` +
      `TR=(${ce.TR[0].toFixed(1)},${ce.TR[1].toFixed(1)}) ` +
      `BR=(${ce.BR[0].toFixed(1)},${ce.BR[1].toFixed(1)}) ` +
      `BL=(${ce.BL[0].toFixed(1)},${ce.BL[1].toFixed(1)})`,
  );
  const pr = m.polyResidual;
  dbg.log(
    `[warp] pass ${passNum} poly fit RMSE: ` +
      `top=${pr.top.toFixed(2)} bot=${pr.bottom.toFixed(2)} ` +
      `left=${pr.left.toFixed(2)} right=${pr.right.toFixed(2)}`,
  );
  const sg = m.subPixelGap;
  dbg.log(
    `[warp] pass ${passNum} sub-pixel gap width: ` +
      `left=${sg.left.toFixed(2)} right=${sg.right.toFixed(2)} ` +
      `top=${sg.top.toFixed(2)} bottom=${sg.bottom.toFixed(2)}`,
  );
  dbg.setMetric("warp", `pass${passNum}`, {
    edgeCurvatures: ec,
    cornerErrors: ce,
    polyResidual: pr,
    subPixelGap: sg,
    refined: m.refined,
  });
}

interface NonLinearResult {
  remapped: any;
  applied: boolean;
  rowShifts: number[]; // length = H, signed source-x adjustment for each row
  colShifts: number[]; // length = W, signed source-y adjustment for each col
  preResidual: { top: number; bottom: number; left: number; right: number };
  postPolyDelta: { top: number; bottom: number; left: number; right: number };
  fitRmse: { top: number; bottom: number; left: number; right: number };
}

function recordNonLinearMetrics(dbg: DebugCollector, r: NonLinearResult): void {
  const pre = r.preResidual;
  dbg.log(
    `[warp] non-linear pre-remap residuals: ` +
      `top=${pre.top.toFixed(2)} bot=${pre.bottom.toFixed(2)} ` +
      `left=${pre.left.toFixed(2)} right=${pre.right.toFixed(2)}` +
      (r.applied ? "" : "  (SKIPPED — already aligned or noisy fit)"),
  );
  const fr = r.fitRmse;
  dbg.log(
    `[warp] non-linear fit RMSE: ` +
      `top=${fr.top.toFixed(2)} bot=${fr.bottom.toFixed(2)} ` +
      `left=${fr.left.toFixed(2)} right=${fr.right.toFixed(2)}`,
  );
  const post = r.postPolyDelta;
  dbg.log(
    `[warp] non-linear shift magnitudes (max |Δ|): ` +
      `top=${post.top.toFixed(2)} bot=${post.bottom.toFixed(2)} ` +
      `left=${post.left.toFixed(2)} right=${post.right.toFixed(2)}`,
  );
  dbg.setMetric("warp", "nonLinear", {
    preResidual: pre,
    shiftMagnitudes: post,
    applied: r.applied,
  });
}

// ─── 1D Gaussian filter ───

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

// ─── Corner detection ───

type Point = [number, number];
type Corners = [Point, Point, Point, Point]; // TL, TR, BR, BL

function orderCorners(pts: Point[]): Corners {
  const sums = pts.map(([x, y]) => x + y);
  const yMinusX = pts.map(([x, y]) => y - x);
  const tlIdx = sums.indexOf(Math.min(...sums));
  const brIdx = sums.indexOf(Math.max(...sums));
  const trIdx = yMinusX.indexOf(Math.min(...yMinusX));
  const blIdx = yMinusX.indexOf(Math.max(...yMinusX));
  return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}

function scoreQuad(ordered: Corners, imgW: number, imgH: number, targetAspect = 160 / 144): number {
  const [TL, TR, BR, BL] = ordered;
  const top = Math.hypot(TR[0] - TL[0], TR[1] - TL[1]);
  const bot = Math.hypot(BR[0] - BL[0], BR[1] - BL[1]);
  const left = Math.hypot(BL[0] - TL[0], BL[1] - TL[1]);
  const right = Math.hypot(BR[0] - TR[0], BR[1] - TR[1]);
  const wAvg = (top + bot) / 2;
  const hAvg = (left + right) / 2;
  if (hAvg < 10) return 1e9;
  const aspectErr = Math.abs(wAvg / hAvg / targetAspect - 1.0);
  const parallelErr =
    Math.abs(top - bot) / Math.max(wAvg, 1) +
    Math.abs(left - right) / Math.max(hAvg, 1);
  const margin = 5;
  let clips = 0;
  if (TL[0] < margin) clips++;
  if (TL[1] < margin) clips++;
  if (TR[0] > imgW - margin) clips++;
  if (TR[1] < margin) clips++;
  if (BR[0] > imgW - margin) clips++;
  if (BR[1] > imgH - margin) clips++;
  if (BL[0] < margin) clips++;
  if (BL[1] > imgH - margin) clips++;
  return aspectErr * 2.0 + parallelErr + clips * 0.1;
}

interface CornerDetection {
  ordered: Corners;
  score: number;
  thresh: number;
  area: number;
  aspect: number;
}

function findScreenCornersWithMetrics(bgr: any, threshVal: number): CornerDetection {
  const cv = getCV();

  return withMats((track, _untrack) => {
    const gray = track(new cv.Mat());
    cv.cvtColor(bgr, gray, cv.COLOR_BGR2GRAY);
    const imgH = gray.rows;
    const imgW = gray.cols;

    const kernel = track(cv.Mat.ones(7, 7, cv.CV_8U));

    let best: { score: number; ordered: Corners; thresh: number; area: number; aspect: number } | null = null;

    for (let thresh = threshVal; thresh > 114; thresh -= 5) {
      const binary = track(new cv.Mat());
      cv.threshold(gray, binary, thresh, 255, cv.THRESH_BINARY);

      const closed = track(new cv.Mat());
      cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel);

      const contours = track(new cv.MatVector());
      const hierarchy = track(new cv.Mat());
      cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      if (contours.size() === 0) continue;

      let largestIdx = 0;
      let largestArea = 0;
      for (let i = 0; i < contours.size(); i++) {
        const a = cv.contourArea(contours.get(i));
        if (a > largestArea) {
          largestArea = a;
          largestIdx = i;
        }
      }

      const largest = contours.get(largestIdx);
      const area = largestArea;
      if (area < 1000) continue;

      const hull = track(new cv.Mat());
      cv.convexHull(largest, hull);

      const peri = cv.arcLength(hull, true);

      let quad: Point[] | null = null;
      for (const eps of [0.02, 0.03, 0.05, 0.01, 0.10]) {
        const approx = track(new cv.Mat());
        cv.approxPolyDP(hull, approx, eps * peri, true);
        if (approx.rows === 4) {
          quad = [];
          for (let i = 0; i < 4; i++) {
            quad.push([approx.intPtr(i, 0)[0], approx.intPtr(i, 0)[1]]);
          }
          break;
        }
      }

      if (quad === null) {
        const rect = cv.boundingRect(largest);
        quad = [
          [rect.x, rect.y],
          [rect.x + rect.width, rect.y],
          [rect.x + rect.width, rect.y + rect.height],
          [rect.x, rect.y + rect.height],
        ];
      }

      const ordered = orderCorners(quad);
      const score = scoreQuad(ordered, imgW, imgH);

      if (best === null || score < best.score) {
        const r = cv.boundingRect(largest);
        const aspect = r.height ? r.width / r.height : 0;
        best = { score, ordered, thresh, area, aspect };
      }

      if (score < 0.05) break;
    }

    if (best === null) throw new Error("No bright contour found");
    return best;
  });
}

// ─── Initial warp ───

interface WarpResult {
  warped: any;
  M: any;
}

function initialWarp(bgr: any, corners: Corners, scale: number): WarpResult {
  const cv = getCV();
  const W = SCREEN_W * scale;
  const H = SCREEN_H * scale;

  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    corners[0][0], corners[0][1],
    corners[1][0], corners[1][1],
    corners[2][0], corners[2][1],
    corners[3][0], corners[3][1],
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    W - 1, 0,
    W - 1, H - 1,
    0, H - 1,
  ]);

  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  srcPts.delete();
  dstPts.delete();

  const warped = new cv.Mat();
  const dsize = new cv.Size(W, H);
  cv.warpPerspective(bgr, warped, M, dsize, cv.INTER_LANCZOS4);

  return { warped, M };
}

// ─── Sub-pixel inner-border edge detection ───

/**
 * Find the position of the first DROP scanning a profile that starts HIGH
 * (white frame) and steps DOWN into the dark border. Uses quadratic
 * sub-pixel interpolation around the strongest negative gradient.
 *
 * `smoothSigma` controls Gaussian smoothing on the input profile. At
 * scale=8 we use ~scale*0.4 (≈3.2) so sub-pixel BGR ringing inside the
 * WH frame is dampened before we look for the real WH→DG transition.
 */
function firstDropFromFrame(profile: number[], smoothSigma: number): number {
  const p = gaussianFilter1d(profile, smoothSigma);
  const d: number[] = [];
  for (let i = 0; i < p.length - 1; i++) d.push(p[i + 1] - p[i]);
  let k = 0;
  let minVal = d[0];
  for (let i = 1; i < d.length; i++) {
    if (d[i] < minVal) {
      minVal = d[i];
      k = i;
    }
  }
  let delta = 0.0;
  if (k > 0 && k < d.length - 1) {
    const d0 = d[k - 1];
    const d1 = d[k];
    const d2 = d[k + 1];
    const denom = d0 - 2.0 * d1 + d2;
    if (Math.abs(denom) > 1e-10) {
      delta = Math.max(-1.0, Math.min(1.0, 0.5 * (d0 - d2) / denom));
    }
  }
  return k + 1 + delta;
}

/**
 * Find the strongest RISE scanning a profile.
 * Returns the sub-pixel index of the rise.
 */
function firstRise(profile: number[], smoothSigma: number): number {
  const p = gaussianFilter1d(profile, smoothSigma);
  const d: number[] = [];
  for (let i = 0; i < p.length - 1; i++) d.push(p[i + 1] - p[i]);
  let k = 0;
  let maxVal = d[0];
  for (let i = 1; i < d.length; i++) {
    if (d[i] > maxVal) {
      maxVal = d[i];
      k = i;
    }
  }
  let delta = 0.0;
  if (k > 0 && k < d.length - 1) {
    const d0 = d[k - 1];
    const d1 = d[k];
    const d2 = d[k + 1];
    const denom = d0 - 2.0 * d1 + d2;
    if (Math.abs(denom) > 1e-10) {
      delta = Math.max(-1.0, Math.min(1.0, 0.5 * (d2 - d0) / denom));
    }
  }
  return k + 1 + delta;
}

// ─── Profile extraction ───

function colMeans(mat: any, r1: number, r2: number, c1: number, c2: number): number[] {
  const result: number[] = [];
  for (let c = c1; c < c2; c++) {
    let sum = 0;
    let count = 0;
    for (let r = r1; r < r2; r++) {
      sum += mat.ucharAt(r, c);
      count++;
    }
    result.push(count > 0 ? sum / count : 0);
  }
  return result;
}

function rowMeans(mat: any, r1: number, r2: number, c1: number, c2: number): number[] {
  const result: number[] = [];
  for (let r = r1; r < r2; r++) {
    let sum = 0;
    let count = 0;
    for (let c = c1; c < c2; c++) {
      sum += mat.ucharAt(r, c);
      count++;
    }
    result.push(count > 0 ? sum / count : 0);
  }
  return result;
}

// ─── Border points (dense sampling, asymmetric detection) ───

interface BorderPoints {
  top: Point[];
  right: Point[];
  bottom: Point[];
  left: Point[];
  /**
   * Mean width of the dark trough between the WH frame and the DG border
   * (proxy for sub-pixel structure on each side). Right and bottom should
   * generally be wider than left and top because of BGR sub-pixel order.
   */
  gapWidth: { top: number; bottom: number; left: number; right: number };
}

const N_BORDER_POINTS = 25;
const BORDER_DETECT_BAND = 4; // ±4 GB pixels worth of search

function linspace(start: number, end: number, n: number): number[] {
  if (n <= 1) return [start];
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(start + (end - start) * i / (n - 1));
  return out;
}

/**
 * Locate a single border point by extracting a 1-D profile across the
 * expected edge position and finding both:
 *   - the WH→DG drop (outside-of-camera frame becomes inner border)
 *   - the DG→content rise (inner border ends, camera content begins)
 *
 * The DG strip lives between these two transitions. We return the
 * outer-edge position of the DG strip (which IS the inner border that we
 * want to align), plus the measured gap width.
 *
 * `direction` tells us which way the WH frame lies:
 *   - "before" — WH frame is at the START of the profile (top/left edges)
 *   - "after"  — WH frame is at the END of the profile (bottom/right edges)
 *
 * For "before" edges we look for the first DROP (WH→DG) and the next RISE
 * (DG→content). For "after" edges we reverse the profile so the WH side
 * is at the start, do the same analysis, then reflect the index back.
 */
interface BorderHit {
  outerEdgeIdx: number; // position of the outer (WH-side) edge of DG
  gapWidth: number;     // (DG inner → DG outer); also reflects sub-pixel gap if any
}

function findBorderInProfile(
  profile: number[],
  smoothSigma: number,
  direction: "before" | "after",
  scale: number,
): BorderHit {
  const prof = direction === "after" ? [...profile].reverse() : profile;
  // The strongest drop is WH→DG (going from frame into border)
  const dropIdx = firstDropFromFrame(prof, smoothSigma);
  // The strongest rise SHOULD be DG→content (going from border into camera).
  // To prevent it from being a noisy fluctuation inside the WH frame we
  // mask out everything before `dropIdx` and look only at the tail.
  const maskStart = Math.max(0, Math.floor(dropIdx) + 1);
  const tail = prof.slice(maskStart);
  let riseRel = 0;
  if (tail.length >= 3) {
    riseRel = firstRise(tail, smoothSigma);
  }
  const riseIdx = maskStart + riseRel;
  const rawGap = Math.max(0, riseIdx - dropIdx);

  // The outer DG edge — what we want — is `dropIdx` itself in the
  // forward direction. For "after" we flip back to the original index space.
  let outer = dropIdx;
  if (direction === "after") outer = (profile.length - 1) - outer - (scale - 1);
  return { outerEdgeIdx: outer, gapWidth: rawGap };
}

/**
 * Direct corner detection — averages a *half-edge* profile to locate each
 * corner. This is the historical Python approach. It tends to be very
 * accurate when edges are nearly straight (because the half-edge mean
 * suppresses per-point detection noise) but ignores curvature.
 *
 * The polynomial-based detection in `findBorderPoints` complements this by
 * capturing curvature for the downstream non-linear remap.
 */
function findBorderCornersDirect(channel: any, scale: number): CornerPts {
  const H = channel.rows;
  const W = channel.cols;
  const srch = 6 * scale;
  const smoothSigma = scale * 0.4;

  const midCol = Math.floor((INNER_LEFT + INNER_RIGHT) / 2) * scale;
  const midRow = Math.floor((INNER_TOP + INNER_BOT) / 2) * scale;

  const cLft: [number, number] = [Math.max(0, 10 * scale), midCol];
  const cRgt: [number, number] = [midCol, Math.min(W, 150 * scale)];
  const rTop: [number, number] = [Math.max(0, 10 * scale), midRow];
  const rBot: [number, number] = [midRow, Math.min(H, (SCREEN_H - 10) * scale)];

  const topY = (c0: number, c1: number): number => {
    const exp = INNER_TOP * scale;
    const r1 = Math.max(0, exp - srch);
    const r2 = Math.min(H, exp + srch);
    const profile = rowMeans(channel, r1, r2, c0, c1);
    return r1 + firstDropFromFrame(profile, smoothSigma);
  };
  const botY = (c0: number, c1: number): number => {
    const expFrame = (INNER_BOT + 1) * scale;
    const r1 = Math.max(0, expFrame - srch);
    const r2 = Math.min(H, expFrame + srch);
    const profile = rowMeans(channel, r1, r2, c0, c1);
    const reversed = [...profile].reverse();
    const idx = firstDropFromFrame(reversed, smoothSigma);
    return (r2 - 1) - idx - (scale - 1);
  };
  const leftX = (r0: number, r1_: number): number => {
    const exp = INNER_LEFT * scale;
    const c1 = Math.max(0, exp - srch);
    const c2 = Math.min(W, exp + srch);
    const profile = colMeans(channel, r0, r1_, c1, c2);
    return c1 + firstDropFromFrame(profile, smoothSigma);
  };
  const rightX = (r0: number, r1_: number): number => {
    const expFrame = (INNER_RIGHT + 1) * scale;
    const c1 = Math.max(0, expFrame - srch);
    const c2 = Math.min(W, expFrame + srch);
    const profile = colMeans(channel, r0, r1_, c1, c2);
    const reversed = [...profile].reverse();
    const idx = firstDropFromFrame(reversed, smoothSigma);
    return (c2 - 1) - idx - (scale - 1);
  };

  return {
    TL: [leftX(rTop[0], rTop[1]), topY(cLft[0], cLft[1])],
    TR: [rightX(rTop[0], rTop[1]), topY(cRgt[0], cRgt[1])],
    BR: [rightX(rBot[0], rBot[1]), botY(cRgt[0], cRgt[1])],
    BL: [leftX(rBot[0], rBot[1]), botY(cLft[0], cLft[1])],
  };
}

// Original interface (kept for type compatibility)
type CornerPts = { TL: Point; TR: Point; BR: Point; BL: Point };

export function findBorderPoints(channel: any, scale: number): BorderPoints {
  const H = channel.rows;
  const W = channel.cols;
  const srch = BORDER_DETECT_BAND * scale + scale; // give a couple extra cols for poly tails

  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;
  const expTop = INNER_TOP * scale;
  const expBottom = INNER_BOT * scale;

  const points: BorderPoints = {
    top: [], right: [], bottom: [], left: [],
    gapWidth: { top: 0, bottom: 0, left: 0, right: 0 },
  };
  const gapAccum = { top: 0, bottom: 0, left: 0, right: 0 };
  const gapCount = { top: 0, bottom: 0, left: 0, right: 0 };

  const smoothSigma = scale * 0.4; // ≈3.2 at scale=8 — kills sub-pixel ringing

  // Top edge — N points, scan a single column at a time
  for (const colFrac of linspace(0.04, 0.96, N_BORDER_POINTS)) {
    let col = Math.floor(expLeft + (expRight - expLeft) * colFrac);
    col = Math.max(0, Math.min(col, W - 1));
    const r1 = Math.max(0, expTop - srch);
    const r2 = Math.min(H, expTop + srch);
    if (r1 < r2) {
      const profile: number[] = [];
      for (let r = r1; r < r2; r++) profile.push(channel.ucharAt(r, col));
      const hit = findBorderInProfile(profile, smoothSigma, "before", scale);
      const yPos = r1 + hit.outerEdgeIdx;
      points.top.push([col, yPos]);
      gapAccum.top += hit.gapWidth;
      gapCount.top++;
    }
  }

  // Bottom edge
  for (const colFrac of linspace(0.04, 0.96, N_BORDER_POINTS)) {
    let col = Math.floor(expLeft + (expRight - expLeft) * colFrac);
    col = Math.max(0, Math.min(col, W - 1));
    const expFrame = (INNER_BOT + 1) * scale;
    const r1 = Math.max(0, expFrame - srch);
    const r2 = Math.min(H, expFrame + srch);
    if (r1 < r2) {
      const profile: number[] = [];
      for (let r = r1; r < r2; r++) profile.push(channel.ucharAt(r, col));
      const hit = findBorderInProfile(profile, smoothSigma, "after", scale);
      const yPos = r1 + hit.outerEdgeIdx;
      points.bottom.push([col, yPos]);
      gapAccum.bottom += hit.gapWidth;
      gapCount.bottom++;
    }
  }

  // Left edge
  for (const rowFrac of linspace(0.04, 0.96, N_BORDER_POINTS)) {
    let row = Math.floor(expTop + (expBottom - expTop) * rowFrac);
    row = Math.max(0, Math.min(row, H - 1));
    const c1 = Math.max(0, expLeft - srch);
    const c2 = Math.min(W, expLeft + srch);
    if (c1 < c2) {
      const profile: number[] = [];
      for (let c = c1; c < c2; c++) profile.push(channel.ucharAt(row, c));
      const hit = findBorderInProfile(profile, smoothSigma, "before", scale);
      const xPos = c1 + hit.outerEdgeIdx;
      points.left.push([xPos, row]);
      gapAccum.left += hit.gapWidth;
      gapCount.left++;
    }
  }

  // Right edge
  for (const rowFrac of linspace(0.04, 0.96, N_BORDER_POINTS)) {
    let row = Math.floor(expTop + (expBottom - expTop) * rowFrac);
    row = Math.max(0, Math.min(row, H - 1));
    const expFrame = (INNER_RIGHT + 1) * scale;
    const c1 = Math.max(0, expFrame - srch);
    const c2 = Math.min(W, expFrame + srch);
    if (c1 < c2) {
      const profile: number[] = [];
      for (let c = c1; c < c2; c++) profile.push(channel.ucharAt(row, c));
      const hit = findBorderInProfile(profile, smoothSigma, "after", scale);
      const xAdj = c1 + hit.outerEdgeIdx;
      points.right.push([xAdj, row]);
      gapAccum.right += hit.gapWidth;
      gapCount.right++;
    }
  }

  points.gapWidth.top = gapCount.top ? gapAccum.top / gapCount.top : 0;
  points.gapWidth.bottom = gapCount.bottom ? gapAccum.bottom / gapCount.bottom : 0;
  points.gapWidth.left = gapCount.left ? gapAccum.left / gapCount.left : 0;
  points.gapWidth.right = gapCount.right ? gapAccum.right / gapCount.right : 0;

  return points;
}

// ─── Polynomial fitting ───

interface Poly {
  a: number;
  b: number;
  c: number;
  d?: number; // cubic coefficient (optional)
  rmse: number;
}

function polyFit(points: Array<[number, number]>): Poly {
  // Fit f(x) = a + b*x + c*x^2 by least squares.
  if (points.length < 3) {
    // Fall back to mean / line
    if (points.length === 0) return { a: 0, b: 0, c: 0, rmse: 0 };
    const meanY = points.reduce((s, [, y]) => s + y, 0) / points.length;
    return { a: meanY, b: 0, c: 0, rmse: 0 };
  }
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  let t0 = 0, t1 = 0, t2 = 0;
  for (const [x, y] of points) {
    const x2 = x * x;
    const x3 = x2 * x;
    const x4 = x2 * x2;
    s0 += 1;
    s1 += x;
    s2 += x2;
    s3 += x3;
    s4 += x4;
    t0 += y;
    t1 += x * y;
    t2 += x2 * y;
  }
  // Solve 3x3 normal equations:
  // [s0 s1 s2] [a]   [t0]
  // [s1 s2 s3] [b] = [t1]
  // [s2 s3 s4] [c]   [t2]
  const det =
    s0 * (s2 * s4 - s3 * s3) -
    s1 * (s1 * s4 - s3 * s2) +
    s2 * (s1 * s3 - s2 * s2);
  if (Math.abs(det) < 1e-12) {
    const meanY = points.reduce((sum, [, y]) => sum + y, 0) / points.length;
    return { a: meanY, b: 0, c: 0, rmse: 0 };
  }
  const a =
    (t0 * (s2 * s4 - s3 * s3) -
      s1 * (t1 * s4 - s3 * t2) +
      s2 * (t1 * s3 - s2 * t2)) / det;
  const b =
    (s0 * (t1 * s4 - s3 * t2) -
      t0 * (s1 * s4 - s3 * s2) +
      s2 * (s1 * t2 - t1 * s2)) / det;
  const c =
    (s0 * (s2 * t2 - t1 * s3) -
      s1 * (s1 * t2 - t1 * s2) +
      t0 * (s1 * s3 - s2 * s2)) / det;
  // Compute RMSE
  let sqSum = 0;
  for (const [x, y] of points) {
    const yPred = a + b * x + c * x * x;
    sqSum += (y - yPred) ** 2;
  }
  const rmse = Math.sqrt(sqSum / points.length);
  return { a, b, c, rmse };
}

/**
 * Cubic least-squares fit: y = a + b*x + c*x^2 + d*x^3
 * Solves the 4x4 normal equations via Gaussian elimination.
 */
function cubicFit(points: Array<[number, number]>): Poly {
  if (points.length < 4) return polyFit(points);
  // Compute power sums
  const sums = new Array<number>(7).fill(0);
  const ts = new Array<number>(4).fill(0);
  for (const [x, y] of points) {
    let xk = 1;
    for (let k = 0; k <= 6; k++) {
      sums[k] += xk;
      if (k < 4) ts[k] += xk * y;
      xk *= x;
    }
  }
  // 4x4 matrix M[i][j] = sums[i+j], rhs[i] = ts[i]
  const M: number[][] = [];
  const rhs: number[] = [];
  for (let i = 0; i < 4; i++) {
    M.push([sums[i], sums[i + 1], sums[i + 2], sums[i + 3]]);
    rhs.push(ts[i]);
  }
  // Gaussian elimination with partial pivoting
  for (let i = 0; i < 4; i++) {
    let pivot = i;
    for (let r = i + 1; r < 4; r++) {
      if (Math.abs(M[r][i]) > Math.abs(M[pivot][i])) pivot = r;
    }
    if (Math.abs(M[pivot][i]) < 1e-12) return polyFit(points); // fall back
    if (pivot !== i) {
      [M[i], M[pivot]] = [M[pivot], M[i]];
      [rhs[i], rhs[pivot]] = [rhs[pivot], rhs[i]];
    }
    for (let r = i + 1; r < 4; r++) {
      const factor = M[r][i] / M[i][i];
      for (let c = i; c < 4; c++) M[r][c] -= factor * M[i][c];
      rhs[r] -= factor * rhs[i];
    }
  }
  // Back substitution
  const coef = new Array<number>(4).fill(0);
  for (let i = 3; i >= 0; i--) {
    let s = rhs[i];
    for (let j = i + 1; j < 4; j++) s -= M[i][j] * coef[j];
    coef[i] = s / M[i][i];
  }
  let sqSum = 0;
  for (const [x, y] of points) {
    const yp = coef[0] + coef[1] * x + coef[2] * x * x + coef[3] * x * x * x;
    sqSum += (y - yp) ** 2;
  }
  const rmse = Math.sqrt(sqSum / points.length);
  return { a: coef[0], b: coef[1], c: coef[2], d: coef[3], rmse };
}

function polyEval(p: Poly, x: number): number {
  const cubicPart = p.d ? p.d * x * x * x : 0;
  return p.a + p.b * x + p.c * x * x + cubicPart;
}

/**
 * Linear fit y = a + b·x. Returns a Poly with c=0.
 */
function lineFit(points: Array<[number, number]>): Poly {
  if (points.length < 2) {
    if (points.length === 0) return { a: 0, b: 0, c: 0, rmse: 0 };
    return { a: points[0][1], b: 0, c: 0, rmse: 0 };
  }
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  const n = points.length;
  for (const [x, y] of points) {
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const det = n * sxx - sx * sx;
  if (Math.abs(det) < 1e-12) {
    return { a: sy / n, b: 0, c: 0, rmse: 0 };
  }
  const b = (n * sxy - sx * sy) / det;
  const a = (sy - b * sx) / n;
  let sq = 0;
  for (const [x, y] of points) sq += (y - (a + b * x)) ** 2;
  return { a, b, c: 0, rmse: Math.sqrt(sq / n) };
}

/**
 * Outlier-rejecting fit with degree selection. Fits both a line and a
 * quadratic, runs outlier rejection on the better one, and returns the
 * quadratic only if it reduces RMSE meaningfully (>15%). Otherwise returns
 * the linear fit. This prevents fitting random sub-pixel noise to a
 * spurious quadratic curve.
 */
function robustPolyFit(points: Array<[number, number]>): Poly {
  if (points.length < 4) return polyFit(points);

  // First pass on full data
  const quadFirst = polyFit(points);
  let sumSq = 0;
  for (const [x, y] of points) {
    const yp = polyEval(quadFirst, x);
    sumSq += (y - yp) ** 2;
  }
  const sigma = Math.sqrt(sumSq / points.length);
  const cut = 2.5 * Math.max(sigma, 0.3);
  const filtered = points.filter(([x, y]) => Math.abs(y - polyEval(quadFirst, x)) <= cut);
  const usable = filtered.length >= 4 ? filtered : points;

  const quadFit = polyFit(usable);
  const linFit = lineFit(usable);

  // Quadratic justified if it cuts RMSE by ≥15% AND linear RMSE > 0.2 px
  if (linFit.rmse <= 0.2) return linFit;
  if (quadFit.rmse < linFit.rmse * 0.85) return quadFit;
  return linFit;
}

/**
 * Constrained quadratic fit: forces the polynomial to pass through two
 * given endpoints, then fits the quadratic coefficient by minimising SSE
 * on the remaining points.
 *
 * Mathematically: p(y) = line_TL_BL(y) + c·(y - y0)·(y - y1)
 * where the linear part is fixed by the two endpoints.
 *
 * This is used so the polynomial border line agrees with the
 * direct-localized corner detection at the corners — the non-linear remap
 * then only corrects mid-edge curvature, never moves the corner.
 */
function constrainedQuadFit(
  points: Array<[number, number]>,
  x0: number, y0: number,
  x1: number, y1: number,
): Poly {
  // Linear part: y = y0 + slope * (x - x0)
  const denom = x1 - x0;
  if (Math.abs(denom) < 1e-6) {
    // Endpoints stacked — fall back to the average y
    return { a: (y0 + y1) / 2, b: 0, c: 0, rmse: 0 };
  }
  const slope = (y1 - y0) / denom;
  // q(x) = (x - x0)(x - x1) — zero at endpoints
  // Solve c = Σ q(x_i) * r(x_i) / Σ q(x_i)^2
  let num = 0;
  let den = 0;
  for (const [x, y] of points) {
    const linPart = y0 + slope * (x - x0);
    const r = y - linPart;
    const q = (x - x0) * (x - x1);
    num += q * r;
    den += q * q;
  }
  const c = den > 1e-6 ? num / den : 0;
  // Expand to a + b*x + c*x^2
  // p(x) = y0 + slope*(x - x0) + c*(x - x0)*(x - x1)
  //      = y0 - slope*x0 + c*x0*x1
  //         + (slope - c*(x0 + x1)) * x
  //         + c * x^2
  const a = y0 - slope * x0 + c * x0 * x1;
  const b = slope - c * (x0 + x1);
  let sq = 0;
  for (const [x, y] of points) sq += (y - (a + b * x + c * x * x)) ** 2;
  const rmse = points.length > 0 ? Math.sqrt(sq / points.length) : 0;
  return { a, b, c, rmse };
}

/**
 * Like constrainedQuadFit but with model selection: only use the
 * quadratic correction term when it reduces RMSE by ≥15% over the line
 * through the endpoints. Otherwise the polynomial is exactly the line.
 */
function constrainedQuadFitWithSelection(
  points: Array<[number, number]>,
  x0: number, y0: number,
  x1: number, y1: number,
): Poly {
  const cFit = constrainedQuadFit(points, x0, y0, x1, y1);
  // Linear-only fit (c = 0)
  const denom = x1 - x0;
  if (Math.abs(denom) < 1e-6) return cFit;
  const slope = (y1 - y0) / denom;
  const lin: Poly = { a: y0 - slope * x0, b: slope, c: 0, rmse: 0 };
  let sq = 0;
  for (const [x, y] of points) sq += (y - (lin.a + lin.b * x)) ** 2;
  lin.rmse = points.length > 0 ? Math.sqrt(sq / points.length) : 0;
  if (lin.rmse <= 0.2) return lin;
  if (cFit.rmse < lin.rmse * 0.85) return cFit;
  return lin;
}

// ─── Build R-B channel (warm vs cool) ───

export function buildRBChannel(warped: any): any {
  const cv = getCV();
  const H = warped.rows;
  const W = warped.cols;
  return withMats((track, untrack) => {
    const rgb = track(new cv.Mat());
    cv.cvtColor(warped, rgb, cv.COLOR_BGR2RGB);
    const result = new cv.Mat(H, W, cv.CV_8UC1);
    const rgbData = rgb.data;
    const outData = result.data;
    for (let i = 0; i < H * W; i++) {
      const r = rgbData[i * 3];
      const b = rgbData[i * 3 + 2];
      outData[i] = Math.max(0, Math.min(255, r - b + 128));
    }
    return untrack(result);
  });
}

/**
 * Measure how far the four inner-border edges sit from their expected
 * positions in a warped frame — the max over edges of the mean signed
 * border-point offset. Used by the divergence guard to compare refinement
 * passes (a converging refinement shrinks this; a diverging one grows it).
 */
function measureMaxEdgeOffset(warped: any, scale: number): number {
  const rbCh = buildRBChannel(warped);
  const bp = findBorderPoints(rbCh, scale);
  rbCh.delete();
  const expTop = INNER_TOP * scale;
  const expBottom = INNER_BOT * scale;
  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;
  const meanOff = (pts: Point[], exp: number, idx: number): number =>
    pts.length > 0
      ? Math.abs(pts.reduce((s, p) => s + (p[idx] - exp), 0) / pts.length)
      : 0;
  return Math.max(
    meanOff(bp.top, expTop, 1),
    meanOff(bp.bottom, expBottom, 1),
    meanOff(bp.left, expLeft, 0),
    meanOff(bp.right, expRight, 0),
  );
}

function maxAbsEdgeCurvature(ec: RefineMetrics["edgeCurvatures"]): number {
  return Math.max(
    Math.abs(ec.top),
    Math.abs(ec.bottom),
    Math.abs(ec.left),
    Math.abs(ec.right),
  );
}

// ─── Refine warp (perspective back-projection refinement) ───

interface RefineResult {
  refined: any;
  M: any;
}

interface RefineResultWithMetrics extends RefineResult {
  metrics: RefineMetrics;
}

function refineWarpWithMetrics(
  img: any,
  currentM: any,
  warped: any,
  scale: number,
): RefineResultWithMetrics {
  const cv = getCV();
  const H = warped.rows;
  const W = warped.cols;

  const rbCh = buildRBChannel(warped);
  const borderPoints = findBorderPoints(rbCh, scale);
  const directCorners = findBorderCornersDirect(rbCh, scale);
  rbCh.delete();

  const expTop = INNER_TOP * scale;
  const expBottom = INNER_BOT * scale;
  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;

  // Fit quadratics to each edge (used for diagnostics / non-linear remap)
  const topFit = robustPolyFit(borderPoints.top);
  const botFit = robustPolyFit(borderPoints.bottom);
  const leftPts = borderPoints.left.map(([x, y]) => [y, x] as [number, number]);
  const rightPts = borderPoints.right.map(([x, y]) => [y, x] as [number, number]);
  const leftFit = robustPolyFit(leftPts);
  const rightFit = robustPolyFit(rightPts);

  // Use direct localized corner detection (averaged half-edge profile) for
  // the perspective transform input. This matches the Python pipeline and
  // is robust against per-point detection noise. The polynomial fits above
  // are used downstream by the non-linear remap to handle curvature.
  const TL: Point = [directCorners.TL[0], directCorners.TL[1]];
  const TR: Point = [directCorners.TR[0], directCorners.TR[1]];
  const BR: Point = [directCorners.BR[0], directCorners.BR[1]];
  const BL: Point = [directCorners.BL[0], directCorners.BL[1]];

  // Apply edge-curvature offset to corners (Python compatibility): pull the
  // corner positions toward the polynomial-detected edges so the perspective
  // refinement still benefits from multipoint information.
  const corrScale = 0.45;
  const meanTopOff = borderPoints.top.length > 0
    ? borderPoints.top.reduce((s, [, y]) => s + (y - expTop), 0) / borderPoints.top.length
    : 0;
  const meanBotOff = borderPoints.bottom.length > 0
    ? borderPoints.bottom.reduce((s, [, y]) => s + (y - expBottom), 0) / borderPoints.bottom.length
    : 0;
  const meanLeftOff = borderPoints.left.length > 0
    ? borderPoints.left.reduce((s, [x]) => s + (x - expLeft), 0) / borderPoints.left.length
    : 0;
  const meanRightOff = borderPoints.right.length > 0
    ? borderPoints.right.reduce((s, [x]) => s + (x - expRight), 0) / borderPoints.right.length
    : 0;
  if (Math.abs(meanTopOff) > 0.5) {
    TL[1] -= meanTopOff * corrScale;
    TR[1] -= meanTopOff * corrScale;
  }
  if (Math.abs(meanBotOff) > 0.5) {
    BL[1] -= meanBotOff * corrScale;
    BR[1] -= meanBotOff * corrScale;
  }
  if (Math.abs(meanLeftOff) > 0.5) {
    TL[0] -= meanLeftOff * corrScale;
    BL[0] -= meanLeftOff * corrScale;
  }
  if (Math.abs(meanRightOff) > 0.5) {
    TR[0] -= meanRightOff * corrScale;
    BR[0] -= meanRightOff * corrScale;
  }

  const edgeCurvatures = {
    top: borderPoints.top.length > 0
      ? borderPoints.top.reduce((s, [, y]) => s + (y - expTop), 0) / borderPoints.top.length
      : 0,
    bottom: borderPoints.bottom.length > 0
      ? borderPoints.bottom.reduce((s, [, y]) => s + (y - expBottom), 0) / borderPoints.bottom.length
      : 0,
    left: borderPoints.left.length > 0
      ? borderPoints.left.reduce((s, [x]) => s + (x - expLeft), 0) / borderPoints.left.length
      : 0,
    right: borderPoints.right.length > 0
      ? borderPoints.right.reduce((s, [x]) => s + (x - expRight), 0) / borderPoints.right.length
      : 0,
  };

  const cornerErrors = {
    TL: [TL[0] - expLeft, TL[1] - expTop] as [number, number],
    TR: [TR[0] - expRight, TR[1] - expTop] as [number, number],
    BR: [BR[0] - expRight, BR[1] - expBottom] as [number, number],
    BL: [BL[0] - expLeft, BL[1] - expBottom] as [number, number],
  };

  const polyResidual = {
    top: topFit.rmse,
    bottom: botFit.rmse,
    left: leftFit.rmse,
    right: rightFit.rmse,
  };

  const subPixelGap = {
    top: borderPoints.gapWidth.top,
    bottom: borderPoints.gapWidth.bottom,
    left: borderPoints.gapWidth.left,
    right: borderPoints.gapWidth.right,
  };

  try {
    const srcBrd = cv.matFromArray(4, 1, cv.CV_32FC2, [
      TL[0], TL[1],
      TR[0], TR[1],
      BR[0], BR[1],
      BL[0], BL[1],
    ]);
    const dstBrd = cv.matFromArray(4, 1, cv.CV_32FC2, [
      expLeft, expTop,
      expRight, expTop,
      expRight, expBottom,
      expLeft, expBottom,
    ]);

    const Hcorr = cv.getPerspectiveTransform(srcBrd, dstBrd);
    srcBrd.delete();
    dstBrd.delete();

    const HcorrInv = new cv.Mat();
    cv.invert(Hcorr, HcorrInv);
    Hcorr.delete();

    const canvas = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      W - 1, 0,
      W - 1, H - 1,
      0, H - 1,
    ]);

    const cornersInWarped = new cv.Mat();
    cv.perspectiveTransform(canvas, cornersInWarped, HcorrInv);
    HcorrInv.delete();
    canvas.delete();

    const MInv = new cv.Mat();
    cv.invert(currentM, MInv);

    const cornersInSrc = new cv.Mat();
    cv.perspectiveTransform(cornersInWarped, cornersInSrc, MInv);
    MInv.delete();
    cornersInWarped.delete();

    const srcCornerData = cornersInSrc.data32F;
    const srcCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [
      srcCornerData[0], srcCornerData[1],
      srcCornerData[2], srcCornerData[3],
      srcCornerData[4], srcCornerData[5],
      srcCornerData[6], srcCornerData[7],
    ]);
    cornersInSrc.delete();

    const dstCorners = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      W - 1, 0,
      W - 1, H - 1,
      0, H - 1,
    ]);

    const Mnew = cv.getPerspectiveTransform(srcCorners, dstCorners);
    srcCorners.delete();
    dstCorners.delete();

    const refined = new cv.Mat();
    const dsize = new cv.Size(W, H);
    cv.warpPerspective(img, refined, Mnew, dsize, cv.INTER_LANCZOS4);

    return {
      refined,
      M: Mnew,
      metrics: {
        edgeCurvatures: {
          top: Number(edgeCurvatures.top.toFixed(3)),
          bottom: Number(edgeCurvatures.bottom.toFixed(3)),
          left: Number(edgeCurvatures.left.toFixed(3)),
          right: Number(edgeCurvatures.right.toFixed(3)),
        },
        cornerErrors: {
          TL: [Number(cornerErrors.TL[0].toFixed(2)), Number(cornerErrors.TL[1].toFixed(2))],
          TR: [Number(cornerErrors.TR[0].toFixed(2)), Number(cornerErrors.TR[1].toFixed(2))],
          BR: [Number(cornerErrors.BR[0].toFixed(2)), Number(cornerErrors.BR[1].toFixed(2))],
          BL: [Number(cornerErrors.BL[0].toFixed(2)), Number(cornerErrors.BL[1].toFixed(2))],
        },
        polyResidual: {
          top: Number(polyResidual.top.toFixed(3)),
          bottom: Number(polyResidual.bottom.toFixed(3)),
          left: Number(polyResidual.left.toFixed(3)),
          right: Number(polyResidual.right.toFixed(3)),
        },
        subPixelGap: {
          top: Number(subPixelGap.top.toFixed(2)),
          bottom: Number(subPixelGap.bottom.toFixed(2)),
          left: Number(subPixelGap.left.toFixed(2)),
          right: Number(subPixelGap.right.toFixed(2)),
        },
        refined: true,
      },
    };
  } catch {
    const Mcopy = currentM.clone();
    const warpedCopy = warped.clone();
    return {
      refined: warpedCopy,
      M: Mcopy,
      metrics: {
        edgeCurvatures: {
          top: Number(edgeCurvatures.top.toFixed(3)),
          bottom: Number(edgeCurvatures.bottom.toFixed(3)),
          left: Number(edgeCurvatures.left.toFixed(3)),
          right: Number(edgeCurvatures.right.toFixed(3)),
        },
        cornerErrors: {
          TL: [Number(cornerErrors.TL[0].toFixed(2)), Number(cornerErrors.TL[1].toFixed(2))],
          TR: [Number(cornerErrors.TR[0].toFixed(2)), Number(cornerErrors.TR[1].toFixed(2))],
          BR: [Number(cornerErrors.BR[0].toFixed(2)), Number(cornerErrors.BR[1].toFixed(2))],
          BL: [Number(cornerErrors.BL[0].toFixed(2)), Number(cornerErrors.BL[1].toFixed(2))],
        },
        polyResidual: {
          top: Number(polyResidual.top.toFixed(3)),
          bottom: Number(polyResidual.bottom.toFixed(3)),
          left: Number(polyResidual.left.toFixed(3)),
          right: Number(polyResidual.right.toFixed(3)),
        },
        subPixelGap: {
          top: Number(subPixelGap.top.toFixed(2)),
          bottom: Number(subPixelGap.bottom.toFixed(2)),
          left: Number(subPixelGap.left.toFixed(2)),
          right: Number(subPixelGap.right.toFixed(2)),
        },
        refined: false,
      },
    };
  }
}

// ─── Non-linear remap (Phase 4) ───

/**
 * After the perspective refinement passes there may still be residual
 * curvature in the borders (lens distortion, panel warp, etc.). A
 * perspective transform has only 8 DoF and cannot model curvature, so we
 * apply one final non-linear correction:
 *
 *   1. Refit polynomials to the (now-near-straight) borders.
 *   2. For each output row y, define the *current* left & right border
 *      x-positions x_L(y), x_R(y). The ideal positions are expLeft and
 *      expRight. A per-row affine `x = (x_out - expLeft) * (x_R-x_L)/(expRight-expLeft) + x_L`
 *      maps output x to source x.
 *   3. Same idea per output column for y.
 *   4. Combine into a single remap field via the row-then-column composition.
 *
 * The result is fed through `cv.remap` with INTER_LANCZOS4.
 */
/**
 * Threshold: only apply the non-linear remap when the polynomial fits
 * suggest meaningful residual curvature/offset. Without this guard, fitting
 * noise to a quadratic introduces sub-pixel jitter that hurts the cleaner
 * tests more than it helps the noisier ones.
 */
const NON_LINEAR_REMAP_THRESHOLD = 0.8;
/**
 * Skip the non-linear remap if any edge's polynomial fit has RMSE above this
 * value. A noisy fit (which happens when camera content right at the border
 * resembles the DG border colour, e.g. zelda-poster-3's blue artwork) cannot
 * be trusted to drive a sub-pixel correction.
 */
const NON_LINEAR_REMAP_MAX_RMSE = 1.5;

function nonLinearRemap(
  bgrSrc: any,
  pass2M: any,
  warped: any,
  scale: number,
): NonLinearResult {
  const cv = getCV();
  const H = warped.rows;
  const W = warped.cols;

  const rbCh = buildRBChannel(warped);
  const borderPoints = findBorderPoints(rbCh, scale);
  rbCh.delete();

  const expTop = INNER_TOP * scale;
  const expBottom = INNER_BOT * scale;
  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;

  const topFit = robustPolyFit(borderPoints.top);
  const botFit = robustPolyFit(borderPoints.bottom);
  const leftPts = borderPoints.left.map(([x, y]) => [y, x] as [number, number]);
  const rightPts = borderPoints.right.map(([x, y]) => [y, x] as [number, number]);
  const leftFit = robustPolyFit(leftPts);
  const rightFit = robustPolyFit(rightPts);

  const preResidual = {
    top: borderPoints.top.length > 0
      ? borderPoints.top.reduce((s, [, y]) => s + (y - expTop), 0) / borderPoints.top.length
      : 0,
    bottom: borderPoints.bottom.length > 0
      ? borderPoints.bottom.reduce((s, [, y]) => s + (y - expBottom), 0) / borderPoints.bottom.length
      : 0,
    left: borderPoints.left.length > 0
      ? borderPoints.left.reduce((s, [x]) => s + (x - expLeft), 0) / borderPoints.left.length
      : 0,
    right: borderPoints.right.length > 0
      ? borderPoints.right.reduce((s, [x]) => s + (x - expRight), 0) / borderPoints.right.length
      : 0,
  };

  // Max polynomial deviation along each edge
  const postPolyDelta = { top: 0, bottom: 0, left: 0, right: 0 };
  const xL: number[] = new Array(H);
  const xR: number[] = new Array(H);
  for (let y = 0; y < H; y++) {
    xL[y] = polyEval(leftFit, y);
    xR[y] = polyEval(rightFit, y);
    const dL = Math.abs(xL[y] - expLeft);
    if (dL > postPolyDelta.left) postPolyDelta.left = dL;
    const dR = Math.abs(xR[y] - expRight);
    if (dR > postPolyDelta.right) postPolyDelta.right = dR;
  }
  for (let x = 0; x < W; x++) {
    const dT = Math.abs(polyEval(topFit, x) - expTop);
    if (dT > postPolyDelta.top) postPolyDelta.top = dT;
    const dB = Math.abs(polyEval(botFit, x) - expBottom);
    if (dB > postPolyDelta.bottom) postPolyDelta.bottom = dB;
  }

  // Skip remap when residual curvature is already small — protects already
  // well-aligned images from sub-pixel jitter from noisy fits.
  const maxDelta = Math.max(
    postPolyDelta.top, postPolyDelta.bottom, postPolyDelta.left, postPolyDelta.right,
  );
  const empty = new cv.Mat();
  const ROUND = (n: number) => Number(n.toFixed(3));
  const rowShifts: number[] = new Array(H);
  for (let y = 0; y < H; y++) rowShifts[y] = xL[y] - expLeft;
  const colShifts: number[] = new Array(W);
  for (let x = 0; x < W; x++) colShifts[x] = polyEval(topFit, x) - expTop;

  const maxRmse = Math.max(topFit.rmse, botFit.rmse, leftFit.rmse, rightFit.rmse);
  const fitRmse = {
    top: Number(topFit.rmse.toFixed(3)),
    bottom: Number(botFit.rmse.toFixed(3)),
    left: Number(leftFit.rmse.toFixed(3)),
    right: Number(rightFit.rmse.toFixed(3)),
  };
  if (maxDelta < NON_LINEAR_REMAP_THRESHOLD || maxRmse > NON_LINEAR_REMAP_MAX_RMSE) {
    return {
      remapped: empty,
      applied: false,
      rowShifts,
      colShifts,
      preResidual: {
        top: ROUND(preResidual.top), bottom: ROUND(preResidual.bottom),
        left: ROUND(preResidual.left), right: ROUND(preResidual.right),
      },
      postPolyDelta: {
        top: ROUND(postPolyDelta.top), bottom: ROUND(postPolyDelta.bottom),
        left: ROUND(postPolyDelta.left), right: ROUND(postPolyDelta.right),
      },
      fitRmse,
    };
  }

  // Build mapX, mapY: for each output pixel (x, y) compute the SOURCE
  // (input photo) coordinates via:
  //   (x, y) → polynomial stretch → (x_warp, y_warp) in pass-2's warped space
  //         → invM2 → (x_src, y_src) in the source photo
  // and apply cv.remap directly on the source photo so we do exactly one
  // Lanczos resample.
  const invM2 = new cv.Mat();
  cv.invert(pass2M, invM2);
  // Read the 3x3 inverse matrix entries
  const invDataF64 = invM2.data64F;
  const m00 = invDataF64[0], m01 = invDataF64[1], m02 = invDataF64[2];
  const m10 = invDataF64[3], m11 = invDataF64[4], m12 = invDataF64[5];
  const m20 = invDataF64[6], m21 = invDataF64[7], m22 = invDataF64[8];
  invM2.delete();

  const mapX = new cv.Mat(H, W, cv.CV_32FC1);
  const mapY = new cv.Mat(H, W, cv.CV_32FC1);
  const mapXData = mapX.data32F;
  const mapYData = mapY.data32F;
  const dstW = expRight - expLeft;
  const dstH = expBottom - expTop;

  for (let y = 0; y < H; y++) {
    const sxL = xL[y];
    const sxR = xR[y];
    const span = sxR - sxL;
    for (let x = 0; x < W; x++) {
      // Polynomial stretch — maps output (x, y) to coordinates in pass-2's
      // warped space (xW, yW)
      const xW = sxL + (x - expLeft) * span / dstW;
      const yT = polyEval(topFit, xW);
      const yB = polyEval(botFit, xW);
      const ySpan = yB - yT;
      const yW = yT + (y - expTop) * ySpan / dstH;
      // Apply M2^-1 to get source-photo coordinates
      const w = m20 * xW + m21 * yW + m22;
      const src_x = (m00 * xW + m01 * yW + m02) / w;
      const src_y = (m10 * xW + m11 * yW + m12) / w;
      const idx = y * W + x;
      mapXData[idx] = src_x;
      mapYData[idx] = src_y;
    }
  }

  const remapped = new cv.Mat();
  cv.remap(bgrSrc, remapped, mapX, mapY, cv.INTER_LANCZOS4, cv.BORDER_REPLICATE);
  mapX.delete();
  mapY.delete();
  empty.delete();

  return {
    remapped,
    applied: true,
    rowShifts,
    colShifts,
    preResidual: {
      top: ROUND(preResidual.top), bottom: ROUND(preResidual.bottom),
      left: ROUND(preResidual.left), right: ROUND(preResidual.right),
    },
    postPolyDelta: {
      top: ROUND(postPolyDelta.top), bottom: ROUND(postPolyDelta.bottom),
      left: ROUND(postPolyDelta.left), right: ROUND(postPolyDelta.right),
    },
    fitRmse,
  };
}

// ─── Frame-line detection (used by the sub-pixel rectifier) ───

/**
 * Track one WH-frame vertical frame line (the dim B-sub-pixel column) down the
 * image. Returns [y, x] points. At each sample row it averages a small band of
 * rows (robust to blur) and finds the luminance minimum in a sub-GB-pixel
 * window around the running estimate (so it never jumps to an adjacent line),
 * with quadratic sub-pixel interpolation.
 */
function trackFrameLine(
  data: Uint8Array,
  W: number,
  H: number,
  seedX: number,
  yTop: number,
  yBot: number,
  scale: number,
): Array<[number, number]> {
  const half = Math.max(3, Math.floor(scale / 2));
  const band = Math.max(3, Math.floor(scale * 0.6));
  const colLum = (xc: number, yc: number): Array<[number, number]> => {
    const x0 = Math.max(1, xc - half);
    const x1 = Math.min(W - 2, xc + half);
    const out: Array<[number, number]> = [];
    for (let x = x0; x <= x1; x++) {
      let s = 0, n = 0;
      const ya = Math.max(0, yc - band), yb = Math.min(H - 1, yc + band);
      for (let y = ya; y <= yb; y++) {
        const i = (y * W + x) * 3;
        s += data[i] + data[i + 1] + data[i + 2];
        n++;
      }
      out.push([x, s / n]);
    }
    return out;
  };
  let cur = seedX;
  const out: Array<[number, number]> = [];
  const step = Math.max(2, Math.floor(scale / 2));
  for (let yc = yTop; yc <= yBot; yc += step) {
    const m = colLum(Math.round(cur), yc);
    const sm = gaussianFilter1d(m.map((v) => v[1]), 1.0);
    let bi = 0;
    for (let k = 1; k < sm.length; k++) if (sm[k] < sm[bi]) bi = k;
    let xx = m[bi][0];
    if (bi > 0 && bi < sm.length - 1) {
      const y0 = sm[bi - 1], y1 = sm[bi], y2 = sm[bi + 1];
      const dn = y0 - 2 * y1 + y2;
      if (Math.abs(dn) > 1e-6) xx += 0.5 * (y0 - y2) / dn;
    }
    // Reject jumps to an adjacent frame line; keep tracking the same one.
    if (out.length === 0 || Math.abs(xx - cur) <= scale * 0.75) cur = xx;
    out.push([yc, cur]);
  }
  return out;
}

/** Fit a smooth curve to a tracked frame line, in (y-yTop)/scale units. */
function frameBendFit(
  pts: Array<[number, number]>,
  yTop: number,
  scale: number,
): Poly | null {
  if (pts.length < 5) return null;
  const norm = pts.map(([y, x]) => [(y - yTop) / scale, x] as [number, number]);
  return robustPolyFit(norm);
}

// ─── Sub-pixel rectification (Phase 5) ───

interface SubPixelResult {
  rectified: any;
  applied: boolean;
  topOffsets: number[];   // per-block G-peak offset from top WH frame
  botOffsets: number[];   // per-block G-peak offset from bottom WH frame
  maxShift: number;       // max |shift| applied
}

function recordSubPixelMetrics(dbg: DebugCollector, r: SubPixelResult): void {
  dbg.log(
    `[warp] sub-pixel rectify: max shift=${r.maxShift.toFixed(2)} px ` +
      `top G-offset range [${Math.min(...r.topOffsets).toFixed(2)}, ${Math.max(...r.topOffsets).toFixed(2)}] ` +
      `bot G-offset range [${Math.min(...r.botOffsets).toFixed(2)}, ${Math.max(...r.botOffsets).toFixed(2)}]` +
      (r.applied ? "" : "  (SKIPPED — already aligned)"),
  );
  dbg.setMetric("warp", "subPixel", {
    maxShift: Number(r.maxShift.toFixed(3)),
    topOffsets: r.topOffsets.map(v => Number(v.toFixed(3))),
    botOffsets: r.botOffsets.map(v => Number(v.toFixed(3))),
    applied: r.applied,
  });
}

/**
 * Find the sub-pixel offset of the G-subpixel peak within a GB-pixel block
 * by sampling a vertical strip of WH frame. Returns NaN if the peak isn't
 * confident (no clear winner among the columns).
 */
export function findGPeakOffset(
  warped: any,
  blockStartCol: number,
  rowStart: number,
  rowEnd: number,
  scale: number,
): number {
  const W = warped.cols;
  const data = warped.data;
  const stride = W * 3; // BGR
  const means: number[] = new Array(scale).fill(0);
  const nRows = rowEnd - rowStart;
  for (let r = rowStart; r < rowEnd; r++) {
    const rowBase = r * stride;
    for (let c = 0; c < scale; c++) {
      means[c] += data[rowBase + (blockStartCol + c) * 3 + 1];
    }
  }
  for (let c = 0; c < scale; c++) means[c] /= nRows;
  // Find max col and min
  let maxC = 0;
  let minVal = means[0];
  let maxVal = means[0];
  for (let c = 1; c < scale; c++) {
    if (means[c] > means[maxC]) maxC = c;
    if (means[c] < minVal) minVal = means[c];
    if (means[c] > maxVal) maxVal = means[c];
  }
  // Quality check: need a clear peak (range > 12) and not at boundary
  if (maxVal - minVal < 12) return NaN;
  if (maxC === 0 || maxC === scale - 1) return NaN;
  // Quadratic sub-pixel interp
  const y0 = means[maxC - 1];
  const y1 = means[maxC];
  const y2 = means[maxC + 1];
  const denom = y0 - 2 * y1 + y2;
  if (Math.abs(denom) < 1e-6) return maxC;
  const delta = Math.max(-1, Math.min(1, 0.5 * (y0 - y2) / denom));
  return maxC + delta;
}

const SUB_PIXEL_MIN_SHIFT = 0.3; // skip if max shift is below this
// A sub-pixel phase correction realigns the BGR stripe within a GB-pixel
// cell, so it can never legitimately need to move content by more than one
// GB pixel. If the computed shift exceeds this the underlying fit is
// untrustworthy (bad/sparse stripe signal) — skip the whole rectify rather
// than warp the image by a non-physical amount.
const SUB_PIXEL_MAX_SHIFT = 8; // = scale (1 GB pixel) at scale=8

function subPixelRectify(warped: any, scale: number): SubPixelResult {
  const cv = getCV();
  const H = warped.rows;
  const W = warped.cols;

  const camLeft = INNER_LEFT * scale + scale;     // 128 at scale=8
  const camRight = INNER_RIGHT * scale;           // 1152 at scale=8
  const camTop = INNER_TOP * scale + scale;       // 128
  const camBot = INNER_BOT * scale;               // 1024
  const nBlocks = (camRight - camLeft) / scale;   // 128 GB pixels horizontally
  // Detect offsets across the FULL screen width (frame area included) so the
  // polynomial fit has more sample points and is better-constrained at the
  // camera edges where lens distortion is worst.
  const detectStartCol = 0;
  const detectNBlocks = Math.floor(W / scale);    // 160 at scale=8

  // Sample strips: top WH frame (rows 16..INNER_TOP*scale-8) and bottom
  // (rows INNER_BOT*scale+scale+8..H-16). Use narrow slices closest to the
  // camera so per-row drift differences are captured accurately.
  const topR1 = Math.max(scale * 8, 0);
  const topR2 = Math.min(INNER_TOP * scale - scale, topR1 + 6 * scale);
  const botR1 = Math.max(INNER_BOT * scale + scale + scale, camBot);
  const botR2 = Math.min(H - scale * 4, botR1 + 6 * scale);

  if (topR2 <= topR1 || botR2 <= botR1) {
    return {
      rectified: new cv.Mat(),
      applied: false,
      topOffsets: [],
      botOffsets: [],
      maxShift: 0,
    };
  }

  // Detect raw G-peak offsets across the FULL detection range
  const topOffsetsRawFull = new Array<number>(detectNBlocks);
  const botOffsetsRawFull = new Array<number>(detectNBlocks);
  for (let bx = 0; bx < detectNBlocks; bx++) {
    const blockStart = detectStartCol + bx * scale;
    topOffsetsRawFull[bx] = findGPeakOffset(warped, blockStart, topR1, topR2, scale);
    botOffsetsRawFull[bx] = findGPeakOffset(warped, blockStart, botR1, botR2, scale);
  }

  // Fit polynomials using the full data; then evaluate over camera blocks
  const topPts: Array<[number, number]> = [];
  const botPts: Array<[number, number]> = [];
  for (let bx = 0; bx < detectNBlocks; bx++) {
    if (Number.isFinite(topOffsetsRawFull[bx])) topPts.push([bx, topOffsetsRawFull[bx]]);
    if (Number.isFinite(botOffsetsRawFull[bx])) botPts.push([bx, botOffsetsRawFull[bx]]);
  }
  // Also build camera-block aligned arrays for the metric output
  const topOffsetsRaw = new Array<number>(nBlocks);
  const botOffsetsRaw = new Array<number>(nBlocks);
  for (let bx = 0; bx < nBlocks; bx++) {
    const fullBx = bx + (camLeft - detectStartCol) / scale;
    topOffsetsRaw[bx] = topOffsetsRawFull[fullBx] ?? NaN;
    botOffsetsRaw[bx] = botOffsetsRawFull[fullBx] ?? NaN;
  }
  if (topPts.length < 6 || botPts.length < 6) {
    return {
      rectified: new cv.Mat(),
      applied: false,
      topOffsets: topOffsetsRaw,
      botOffsets: botOffsetsRaw,
      maxShift: 0,
    };
  }
  // Try a cubic fit and use it if it's significantly better than quadratic.
  // With 160 sample points across the full screen width, cubic can pick up
  // S-shaped lens distortion that quadratic cannot.
  const topCubic = topPts.length >= 8 ? cubicFit(topPts) : null;
  const botCubic = botPts.length >= 8 ? cubicFit(botPts) : null;
  const topQuad = robustPolyFit(topPts);
  const botQuad = robustPolyFit(botPts);
  const topFit = topCubic && topCubic.rmse < topQuad.rmse * 0.6 ? topCubic : topQuad;
  const botFit = botCubic && botCubic.rmse < botQuad.rmse * 0.6 ? botCubic : botQuad;

  const topOffsets = new Array<number>(nBlocks);
  const botOffsets = new Array<number>(nBlocks);
  const cameraBlockOffset = (camLeft - detectStartCol) / scale;
  // Only trust the polynomial within the x-range actually covered by valid
  // samples. A quadratic/cubic extrapolates to non-physical values outside
  // its data support — when the WH-frame stripe signal is only detectable
  // across part of the width (e.g. a blurry/low-contrast top strip yields
  // valid G-peaks clustered on one side), evaluating the fit over the full
  // camera span produces a runaway offset (observed: a 13-point right-side
  // fit extrapolated to −97 at the left edge → an 84px "sub-pixel" shift).
  // Clamp the evaluation x to the sampled support so unsupported blocks
  // inherit the nearest real measurement instead of an extrapolation.
  const topXs = topPts.map(([x]) => x);
  const botXs = botPts.map(([x]) => x);
  const topXmin = Math.min(...topXs), topXmax = Math.max(...topXs);
  const botXmin = Math.min(...botXs), botXmax = Math.max(...botXs);

  // Per-edge top→bottom drift from the WH-frame vertical frame lines. These
  // stay reliable into the blurry bottom corners where the bottom strip's
  // G-peak collapses, so we use them — not the bottom strip — to set the
  // top→bottom drift. The bottom strip is kept only as a fallback when a
  // frame line can't be tracked. Drift is the same in geometric px and in
  // sub-pixel phase (the content grid is rigid), so it adds directly.
  const yTopRef0 = (topR1 + topR2) / 2;
  const yBotRef0 = (botR1 + botR2) / 2;
  const lineBend = (seedX: number): number | null => {
    const pts = trackFrameLine(warped.data, W, H, seedX, Math.round(yTopRef0), Math.round(yBotRef0), scale);
    if (pts.length < 5) return null;
    const fit = frameBendFit(pts, yTopRef0, scale);
    if (!fit) return null;
    const b = polyEval(fit, (yBotRef0 - yTopRef0) / scale) - polyEval(fit, 0);
    return Math.abs(b) <= scale * 1.5 ? b : null;
  };
  const rightBend = lineBend(camRight + 2 * scale);
  const leftBend = lineBend(Math.max(2 * scale, 6 * scale));
  const rB = rightBend ?? 0;
  const lB = leftBend ?? 0;

  for (let bx = 0; bx < nBlocks; bx++) {
    const X = bx + cameraBlockOffset;
    topOffsets[bx] = polyEval(topFit, Math.max(topXmin, Math.min(topXmax, X)));
    // Bottom drift from the bottom strip (captures the per-column SHAPE, which
    // the frame lines — only measured at the two edges — cannot). But the
    // bottom strip's reading collapses in the blurry bottom CORNERS, so near
    // the left/right edges blend toward the frame-line drift (reliable there).
    const botStrip = polyEval(botFit, Math.max(botXmin, Math.min(botXmax, X)));
    const u = Math.max(0, Math.min(1, (bx * scale) / (camRight - camLeft)));
    const driftStrip = botStrip - topOffsets[bx];
    const driftLine = lB * (1 - u) + rB * u;
    // Edge weight: 0 across the central 40%, ramping to 1 in the outer 30%.
    const edge = Math.max(0, (Math.abs(u - 0.5) - 0.2) / 0.3);
    // Confidence gate: a tracked frame line carries ~1.5px of detection noise,
    // so only trust it once the measured bend clearly exceeds that — below the
    // floor the local bottom strip is the better signal and overriding it with
    // a noisy small bend adds error. Scales the correction in smoothly.
    const conf = Math.max(0, Math.min(1, (Math.max(Math.abs(rB), Math.abs(lB)) - 2) / 2));
    const w = rightBend === null ? 0 : Math.max(0, Math.min(1, edge)) * conf;
    botOffsets[bx] = topOffsets[bx] + (1 - w) * driftStrip + w * driftLine;
  }

  // Target = mean of top and bottom offsets across the image. This is the
  // global sub-pixel convention. Shifts then represent deviations from this
  // mean, fixing per-block lens-distortion drift.
  let topSum = 0, botSum = 0;
  for (let bx = 0; bx < nBlocks; bx++) {
    topSum += topOffsets[bx];
    botSum += botOffsets[bx];
  }
  const targetOffset = (topSum + botSum) / (2 * nBlocks);

  const shiftTop = new Array<number>(nBlocks);
  const shiftBot = new Array<number>(nBlocks);
  let maxShift = 0;
  for (let bx = 0; bx < nBlocks; bx++) {
    shiftTop[bx] = topOffsets[bx] - targetOffset;
    shiftBot[bx] = botOffsets[bx] - targetOffset;
    if (Math.abs(shiftTop[bx]) > maxShift) maxShift = Math.abs(shiftTop[bx]);
    if (Math.abs(shiftBot[bx]) > maxShift) maxShift = Math.abs(shiftBot[bx]);
  }

  // Skip when the correction is negligible (avoid sub-pixel jitter) or when
  // it is non-physically large (untrustworthy fit — see SUB_PIXEL_MAX_SHIFT).
  const maxPhysical = SUB_PIXEL_MAX_SHIFT * (scale / 8);
  if (maxShift < SUB_PIXEL_MIN_SHIFT || maxShift > maxPhysical) {
    return {
      rectified: new cv.Mat(),
      applied: false,
      topOffsets,
      botOffsets,
      maxShift,
    };
  }

  // Build mapX, mapY. Identity in Y. For each (x, y), compute the block bx,
  // interp shift from top to bot based on y, and src_x = x + shift.
  const mapX = new cv.Mat(H, W, cv.CV_32FC1);
  const mapY = new cv.Mat(H, W, cv.CV_32FC1);
  const mapXData = mapX.data32F;
  const mapYData = mapY.data32F;
  // y reference: top frame center ≈ (topR1+topR2)/2, bot frame center ≈ (botR1+botR2)/2
  const yTopRef = (topR1 + topR2) / 2;
  const yBotRef = (botR1 + botR2) / 2;
  const yRange = yBotRef - yTopRef;

  // Strength factor: <1 means soft rectification. This protects images
  // with fine LG/DG transitions (thing-2) from over-rectification, while
  // still doing most of the correction on images with severe drift.
  const STRENGTH = 0.9;
  for (let y = 0; y < H; y++) {
    const t = Math.max(0, Math.min(1, (y - yTopRef) / yRange));
    for (let x = 0; x < W; x++) {
      let shift = 0;
      if (x >= camLeft && x < camRight) {
        const bxFloat = (x - camLeft) / scale;
        const bxLo = Math.max(0, Math.min(nBlocks - 1, Math.floor(bxFloat)));
        const bxHi = Math.max(0, Math.min(nBlocks - 1, bxLo + 1));
        const fbx = bxFloat - bxLo;
        const sT = shiftTop[bxLo] * (1 - fbx) + shiftTop[bxHi] * fbx;
        const sB = shiftBot[bxLo] * (1 - fbx) + shiftBot[bxHi] * fbx;
        shift = (sT * (1 - t) + sB * t) * STRENGTH;
      }
      const idx = y * W + x;
      mapXData[idx] = x + shift;
      mapYData[idx] = y;
    }
  }

  const rectified = new cv.Mat();
  cv.remap(warped, rectified, mapX, mapY, cv.INTER_LANCZOS4, cv.BORDER_REPLICATE);
  mapX.delete();
  mapY.delete();

  return {
    rectified,
    applied: true,
    topOffsets,
    botOffsets,
    maxShift,
  };
}

// ─── Diagnostic image renderer ───

function renderBorderDiagnostic(
  dbg: DebugCollector,
  warpedMat: any,
  scale: number,
): void {
  const cv = getCV();

  // Build a R-B channel for re-detection (on the *current* warp state, post-remap)
  const rbCh = buildRBChannel(warpedMat);
  const borderPoints = findBorderPoints(rbCh, scale);
  rbCh.delete();

  const expTop = INNER_TOP * scale;
  const expBottom = INNER_BOT * scale;
  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;

  // Convert warped Mat to RGBA GBImageData for drawing
  const rgba = new cv.Mat();
  cv.cvtColor(warpedMat, rgba, cv.COLOR_BGR2RGBA);
  const warpedImg = matToImageData(rgba);
  rgba.delete();
  const overlay = cloneImage(warpedImg);

  // Ideal rectangle (cyan)
  const cyan: [number, number, number] = [0, 255, 255];
  drawLine(overlay, expLeft, expTop, expRight, expTop, cyan, 1);
  drawLine(overlay, expRight, expTop, expRight, expBottom, cyan, 1);
  drawLine(overlay, expRight, expBottom, expLeft, expBottom, cyan, 1);
  drawLine(overlay, expLeft, expBottom, expLeft, expTop, cyan, 1);

  // Detected points (red)
  const red: [number, number, number] = [255, 0, 0];
  const dotR = 2;
  for (const [x, y] of borderPoints.top) fillCircle(overlay, x, y, dotR, red);
  for (const [x, y] of borderPoints.bottom) fillCircle(overlay, x, y, dotR, red);
  for (const [x, y] of borderPoints.left) fillCircle(overlay, x, y, dotR, red);
  for (const [x, y] of borderPoints.right) fillCircle(overlay, x, y, dotR, red);

  // Polynomial fits (yellow)
  const yellow: [number, number, number] = [255, 255, 0];
  const topFit = robustPolyFit(borderPoints.top);
  const botFit = robustPolyFit(borderPoints.bottom);
  const leftPts = borderPoints.left.map(([x, y]) => [y, x] as [number, number]);
  const rightPts = borderPoints.right.map(([x, y]) => [y, x] as [number, number]);
  const leftFit = robustPolyFit(leftPts);
  const rightFit = robustPolyFit(rightPts);

  for (let x = expLeft; x <= expRight; x++) {
    const yT = polyEval(topFit, x);
    const yB = polyEval(botFit, x);
    if (Number.isFinite(yT) && yT >= 0 && yT < overlay.height) {
      fillCircle(overlay, x, yT, 1, yellow);
    }
    if (Number.isFinite(yB) && yB >= 0 && yB < overlay.height) {
      fillCircle(overlay, x, yB, 1, yellow);
    }
  }
  for (let y = expTop; y <= expBottom; y++) {
    const xL = polyEval(leftFit, y);
    const xR = polyEval(rightFit, y);
    if (Number.isFinite(xL) && xL >= 0 && xL < overlay.width) {
      fillCircle(overlay, xL, y, 1, yellow);
    }
    if (Number.isFinite(xR) && xR >= 0 && xR < overlay.width) {
      fillCircle(overlay, xR, y, 1, yellow);
    }
  }

  dbg.addImage("warp_b_borders", overlay);

  // Log residual deviations after remap
  const finalResidual = {
    top: borderPoints.top.length > 0
      ? borderPoints.top.reduce((s, [, y]) => s + (y - expTop), 0) / borderPoints.top.length
      : 0,
    bottom: borderPoints.bottom.length > 0
      ? borderPoints.bottom.reduce((s, [, y]) => s + (y - expBottom), 0) / borderPoints.bottom.length
      : 0,
    left: borderPoints.left.length > 0
      ? borderPoints.left.reduce((s, [x]) => s + (x - expLeft), 0) / borderPoints.left.length
      : 0,
    right: borderPoints.right.length > 0
      ? borderPoints.right.reduce((s, [x]) => s + (x - expRight), 0) / borderPoints.right.length
      : 0,
  };
  dbg.log(
    `[warp] final residuals (post-remap, signed mean): ` +
      `top=${finalResidual.top.toFixed(3)} bot=${finalResidual.bottom.toFixed(3)} ` +
      `left=${finalResidual.left.toFixed(3)} right=${finalResidual.right.toFixed(3)}`,
  );
  dbg.setMetric("warp", "finalResidual", {
    top: Number(finalResidual.top.toFixed(3)),
    bottom: Number(finalResidual.bottom.toFixed(3)),
    left: Number(finalResidual.left.toFixed(3)),
    right: Number(finalResidual.right.toFixed(3)),
  });
}
