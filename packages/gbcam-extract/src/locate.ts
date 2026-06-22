/**
 * locate.ts — Find the Game Boy Screen within a full phone photo.
 *
 * The first step in the pipeline. Takes a full original photo (e.g. ~4032×1816
 * with the GBA SP somewhere inside) and produces an approximately upright crop
 * around the Game Boy Screen, suitable for the {@link warp} step.
 *
 * Algorithm: hybrid candidate generation + Frame 02 validation. Bright
 * quadrilateral candidates are generated at a downsampled working resolution,
 * then validated against Frame 02-specific structural features (inner-border
 * ring, surrounding LCD-black ring). The highest-scoring candidate is mapped
 * back to original-image coordinates, expanded by a proportional margin, and
 * extracted as an axis-aligned image.
 */

import type { GBImageData } from "./common.js";
import { type DebugCollector, cloneImage } from "./debug.js";
import { getCV, withMats, imageDataToMat, matToImageData } from "./opencv.js";

// ─── Tunables ───
//
// Starting values, tunable empirically against the corners.json unit test
// (Task 9). If detection fails on any of the test-input-full images, revisit
// these constants.

/** Target max dimension (px) of the working-resolution image. */
const WORKING_MAX_DIM = 1000;

/**
 * Morphological-closing kernel size, in working-res pixels. The Frame 02
 * dashes (1 GB pixel ≈ 6 working-res pixels at WORKING_MAX_DIM=1000) fragment
 * the white frame into a sparse bright region; closing bridges them so the
 * frame appears as a solid filled rectangle to `findContours`.
 */
const CLOSING_KERNEL = 11;

/** Minimum candidate area as a fraction of the working-resolution image area. */
const MIN_CANDIDATE_AREA_FRAC = 0.02;

/** Maximum candidate area as a fraction of the working-resolution image area. */
const MAX_CANDIDATE_AREA_FRAC = 0.85;

/** Number of top-ranked candidates to validate. */
const TOP_N_CANDIDATES = 12;

/**
 * approxPolyDP epsilon as a fraction of contour perimeter. Loose enough to
 * collapse a noisy bright frame to 4 vertices, tight enough to reject
 * non-quad shapes.
 */
const APPROX_EPSILON_FRAC = 0.04;

/** Margin to expand the chosen rectangle by, as a fraction of its longest side. */
const MARGIN_RATIO = 0.06;

/**
 * Fixed expansion applied AFTER per-edge dash refinement. Empirically the
 * downstream `warp` step's brightness-based corner detection needs a few
 * percent of dark LCD-ring margin around the white frame to lock onto
 * stable corners — without this, even a perfectly-aligned locate crop
 * causes downstream warp to misdetect, costing accuracy.
 */
const CANNY_INSET_COMPENSATION = 0.04;

// ─── Frame 02 reference dash positions (extracted from supporting-materials/frame_ascii.txt) ───
//
// Frame 02 is a 160×144 grayscale layout. The white frame is 16 px thick on
// each side. Inside the frame, BLACK dashes provide a structural reference:
//   - 17 horizontal dashes along the top frame, with their 5-px-tall body
//     at rows 5-9 (main contrast at rows 6-7 in the template).
//   - 17 horizontal dashes along the bottom, body at rows 134-138.
//   - 14 vertical dashes along the left frame, body at cols 1-5.
//   - 14 vertical dashes along the right frame, body at cols 153-158.
// Corner dashes are fused into the L-shapes at each corner.
//
// We find the dash row/col in a warped 160×144 view by scanning each edge's
// search window and picking the row/col with maximum variance (alternating
// bright frame / dark dashes → high variance; LCD ring → low variance,
// uniformly dark; frame interior → low variance, uniformly bright). Offset
// from these expected positions tells us how the candidate is mis-aligned.

// Expected dash positions (in normalized 160×144). Tuned empirically
// against the corners.json fixtures and full-photo accuracy: the values
// here are intentionally offset by ~1 from the template's strict main-
// dash row to bias the dash refinement toward a slightly outward
// expansion that matches the dark-LCD-ring margin the downstream warp
// step prefers. Reference template (from frame_ascii.txt) has main top
// dashes at rows 6-7, bottom at 137-138, left at cols 1-5, right at
// cols ~154-158.

/** Expected dash row for top edge. */
const EXPECTED_TOP_DASH_ROW = 7;
/** Expected dash row for bottom edge. */
const EXPECTED_BOTTOM_DASH_ROW = 136;
/** Expected dash col for left edge. */
const EXPECTED_LEFT_DASH_COL = 3;
/** Expected dash col for right edge. */
const EXPECTED_RIGHT_DASH_COL = 156;

/** Scan window for finding each side's actual dash row/col (in normalized space). */
const DASH_SEARCH_WINDOW = 16;

/** Minimum total validation score to accept a candidate (0–1). */
const MIN_VALIDATION_SCORE = 0.25;

/** Target screen aspect (160 / 144). */
const TARGET_ASPECT = 160 / 144;

// ─── Public interface ───

export interface LocateOptions {
  debug?: DebugCollector;
}

/**
 * Locate the Game Boy Screen within a full phone photo and produce an
 * approximately upright crop suitable for the {@link warp} step.
 *
 * Detection: generate candidate bright quadrilaterals at a downsampled
 * working resolution, validate each against Frame 02 features
 * (inner-border ring, surrounding LCD-black ring), pick the highest-
 * scoring candidate, expand by a proportional margin, and extract the
 * rotated rectangle in original-image pixel space (no resampling beyond
 * the rotation itself).
 *
 * Already-cropped inputs pass through cleanly: with no room to expand,
 * the margin step clamps to image bounds and the output is essentially
 * the input.
 *
 * @throws if no candidate passes minimum frame validation.
 */
export function locate(input: GBImageData, options?: LocateOptions): GBImageData {
  const dbg = options?.debug;

  // Already-cropped fast path: if the input's aspect ratio is close to the
  // GB Screen's 160:144 (within 10%), the input is most likely already
  // cropped around the screen. Pass through unchanged — running detection
  // on a tightly-cropped image typically picks up arbitrary fragments and
  // produces a worse output than the input.
  const inputAspect = input.width / input.height;
  const inputAspectErr = Math.abs(inputAspect / TARGET_ASPECT - 1);
  if (inputAspectErr < 0.1) {
    if (dbg) {
      dbg.log(
        `[locate] input ${input.width}×${input.height} aspect=${inputAspect.toFixed(3)} ` +
          `≈ target ${TARGET_ASPECT.toFixed(3)}; treating as already-cropped (passThrough)`,
      );
      dbg.setMetrics("locate", {
        passThrough: true,
        inputAspect,
        outputSize: [input.width, input.height],
      });
    }
    return cloneImage(input);
  }

  const cv = getCV();
  const src = imageDataToMat(input);

  return withMats((track) => {
    track(src);

    // ── 2a. Downsample to working resolution ──
    const work = downsampleToWorking(src, WORKING_MAX_DIM);
    track(work.mat);

    // Convert to grayscale for thresholding and edge detection
    const gray = track(new cv.Mat());
    cv.cvtColor(work.mat, gray, cv.COLOR_RGBA2GRAY);

    // ── 2b. Generate candidate quads ──
    // Strategy: compute multiple binary masks and find candidate quads from
    // each. Combining strategies makes detection robust to varying lighting
    // — sometimes the screen is the brightest connected blob, sometimes
    // it's a hole in a dark surround.
    const candidatePool: Candidate[] = [];

    // Strategy 1: Otsu-thresholded bright region with morphological closing.
    // The closing bridges Frame 02 dashes so the white frame reads as solid.
    const bright = track(new cv.Mat());
    cv.threshold(gray, bright, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    const kernel = track(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(CLOSING_KERNEL, CLOSING_KERNEL)));
    const brightClosed = track(new cv.Mat());
    cv.morphologyEx(bright, brightClosed, cv.MORPH_CLOSE, kernel);
    candidatePool.push(...findCandidatesInMask(brightClosed, "bright"));

    // Strategy 2: invert the dark LCD-black ring. The screen is a hole
    // *inside* the dark blob — if we threshold dark, then close, then find
    // contours on the *holes*, we get the screen rectangle.
    const dark = track(new cv.Mat());
    cv.threshold(gray, dark, 50, 255, cv.THRESH_BINARY_INV);
    const darkClosed = track(new cv.Mat());
    cv.morphologyEx(dark, darkClosed, cv.MORPH_CLOSE, kernel);
    candidatePool.push(...findCandidateHolesInMask(darkClosed, "darkHole"));

    // Strategy 3: Canny edges + approxPolyDP. Find contours that
    // approximate to 4-sided polygons. This catches screens whose bright
    // frame doesn't form a solid blob (e.g. very bright surrounding posters).
    const blurred = track(new cv.Mat());
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    const edges = track(new cv.Mat());
    cv.Canny(blurred, edges, 50, 150);
    const edgesClosed = track(new cv.Mat());
    cv.morphologyEx(edges, edgesClosed, cv.MORPH_CLOSE, kernel);
    candidatePool.push(...findQuadCandidates(edgesClosed, "canny"));

    // Deduplicate and rank
    const candidates = dedupeAndRank(candidatePool, TOP_N_CANDIDATES);

    if (dbg) {
      dbg.addImage("locate_a_thresholded", matToImageData(track(toRgba(brightClosed))));
      dbg.log(
        `[locate] working-res ${work.mat.cols}×${work.mat.rows} ` +
          `(scale=${work.scale.toFixed(3)} from ${input.width}×${input.height})`,
      );
      dbg.setMetric("locate", "workingDim", [work.mat.cols, work.mat.rows]);
      dbg.setMetric("locate", "candidatePoolSize", candidatePool.length);
      dbg.setMetric(
        "locate",
        "candidatePool",
        candidatePool.map((c) => ({
          source: c.source,
          score: Number(c.score.toFixed(3)),
          width: Math.round(c.width),
          height: Math.round(c.height),
          centerX: Math.round((c.corners[0][0] + c.corners[2][0]) / 2),
          centerY: Math.round((c.corners[0][1] + c.corners[2][1]) / 2),
        })),
      );
    }

    if (candidates.length === 0) {
      throw new Error(
        `[locate] No candidate quadrilaterals found. ` +
          `The Game Boy Screen may be too dark or the photo too distant.`,
      );
    }

    if (dbg) {
      const workingRgba = matToImageData(work.mat);
      // Chosen index is unknown until validation runs. Draw all candidates
      // as red here; we overwrite this debug image after validation with
      // the chosen candidate highlighted in green.
      dbg.addImage("locate_b_candidates", drawCandidates(workingRgba, candidates, -1));
      dbg.log(
        `[locate] found ${candidates.length} candidate(s); ` +
          `top score=${candidates[0].score.toFixed(3)}`,
      );
      dbg.setMetric("locate", "candidateCount", candidates.length);
    }

    // ── 2c. Validate candidates against Frame 02 ──
    let bestIdx = -1;
    let bestScore: ValidationScore | null = null;
    const allScores: ValidationScore[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const score = validateCandidate(work.mat, candidates[i]);
      allScores.push(score);
      if (!bestScore || score.totalScore > bestScore.totalScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (!bestScore || bestScore.totalScore < MIN_VALIDATION_SCORE) {
      const top = bestScore
        ? `top totalScore=${bestScore.totalScore.toFixed(3)} ` +
          `(innerBorder=${bestScore.innerBorderScore.toFixed(3)}, ` +
          `darkRing=${bestScore.darkRingScore.toFixed(3)})`
        : "no candidates";
      throw new Error(
        `[locate] No candidate passed Frame 02 validation. ${top}, ` +
          `min required = ${MIN_VALIDATION_SCORE}.`,
      );
    }

    if (dbg) {
      // Re-emit the candidates debug image with the chosen one in green.
      const workingRgba = matToImageData(work.mat);
      dbg.addImage("locate_b_candidates", drawCandidates(workingRgba, candidates, bestIdx));
      // Emit the chosen-candidate validation visualization.
      dbg.addImage(
        "locate_c_validation",
        renderValidationOverlay(work.mat, candidates[bestIdx], bestScore),
      );
      dbg.log(
        `[locate] chose candidate ${bestIdx}: ` +
          `totalScore=${bestScore.totalScore.toFixed(3)} ` +
          `(innerBorder=${bestScore.innerBorderScore.toFixed(3)}, ` +
          `darkRing=${bestScore.darkRingScore.toFixed(3)})`,
      );
      dbg.setMetrics("locate", {
        chosenCandidate: {
          score: candidates[bestIdx].score,
          area: candidates[bestIdx].area,
          // corners in original-image coords are written in Task 8
          validation: bestScore,
        },
        rejectedScores: allScores.map((s, i) => ({
          index: i,
          totalScore: s.totalScore,
          innerBorderScore: s.innerBorderScore,
          darkRingScore: s.darkRingScore,
        })).filter((_, i) => i !== bestIdx),
      });
    }

    const chosen = candidates[bestIdx];

    // ── 2d. Per-edge refinement using Frame 02 dash positions ──
    // Warp the working-res grayscale to a 160×144 normalized view using the
    // candidate's corners, then find each side's actual dash row/col by
    // variance (the dash row alternates bright frame/dark dashes → high
    // variance, distinguishing it from frame interior or LCD ring). The
    // offset from the expected dash position tells us how each edge is
    // misaligned, and we expand each edge outward (in original-image
    // coordinates) by that amount.
    let refinedCorners = chosen.corners;
    if (chosen.source === "canny") {
      const warped = track(warpToNormalized(gray, chosen.corners));
      const dashOffsets = findDashOffsets(warped);
      const dashRows = findDashRowDebug(warped);
      refinedCorners = applyEdgeOffsets(chosen.corners, dashOffsets);

      if (dbg) {
        const warpedRgba = track(new cv.Mat());
        cv.cvtColor(warped, warpedRgba, cv.COLOR_GRAY2RGBA);
        // Upscale 4× for visibility
        const up = track(new cv.Mat());
        cv.resize(warpedRgba, up, new cv.Size(640, 576), 0, 0, cv.INTER_NEAREST);
        dbg.addImage("locate_e_dash_warp", matToImageData(up));
        const variances = findDashVariances(warped);
        dbg.log(
          `[locate] dash search: top@row=${dashRows.topRow} (expected ${EXPECTED_TOP_DASH_ROW}, off ${dashOffsets.top}), ` +
            `bottom@row=${dashRows.bottomRow} (expected ${EXPECTED_BOTTOM_DASH_ROW}, off ${dashOffsets.bottom}), ` +
            `left@col=${dashRows.leftCol} (expected ${EXPECTED_LEFT_DASH_COL}, off ${dashOffsets.left}), ` +
            `right@col=${dashRows.rightCol} (expected ${EXPECTED_RIGHT_DASH_COL}, off ${dashOffsets.right})`,
        );
        dbg.setMetric("locate", "dashOffsets", dashOffsets);
        dbg.setMetric("locate", "dashRows", dashRows);
        dbg.setMetric("locate", "dashVariances", variances);
      }
    }

    const workToOrig = 1 / work.scale;
    let screenCornersOrig = scaleCorners(refinedCorners, workToOrig);

    // Small safety expansion after dash refinement (the dash-variance peak
    // is at the dash centerline, ~5 px inside the frame outer edge in the
    // template, so dash-aligned corners are still slightly inside the
    // actual frame outer edge).
    if (chosen.source === "canny") {
      screenCornersOrig = expandRotatedRect(screenCornersOrig, CANNY_INSET_COMPENSATION);
    }

    // Already-cropped detection: if the chosen candidate's bounding box
    // covers most of the input image, the input is already cropped around
    // the screen. Pass through the input unchanged so the rest of the
    // pipeline gets the original full-resolution photo.
    const screenBbox = boundingBoxOfCorners(screenCornersOrig, input.width, input.height);
    const screenArea = (screenBbox.x1 - screenBbox.x0) * (screenBbox.y1 - screenBbox.y0);
    const inputArea = input.width * input.height;
    const screenFraction = screenArea / inputArea;
    const passThrough = screenFraction > 0.7;

    const expanded = expandRotatedRect(screenCornersOrig, MARGIN_RATIO);
    const clamped = clampCorners(expanded, input.width, input.height);

    const output = passThrough ? cloneImage(input) : extractRotatedRect(src, clamped);

    if (dbg) {
      dbg.addImage(
        "locate_d_output_region",
        drawOutputRegion(input, screenCornersOrig, clamped),
      );
      dbg.log(
        `[locate] output region: ${output.width}×${output.height} ` +
          `(margin=${(MARGIN_RATIO * 100).toFixed(1)}%, ` +
          `passThrough=${passThrough})`,
      );
      dbg.setMetrics("locate", {
        marginRatio: MARGIN_RATIO,
        outputCorners: clamped.map(([x, y]) => [Math.round(x), Math.round(y)]),
        outputSize: [output.width, output.height],
        passThrough,
        chosenCandidate: {
          score: chosen.score,
          area: chosen.area,
          source: chosen.source,
          corners: screenCornersOrig.map(([x, y]) => [Math.round(x), Math.round(y)]),
          validation: bestScore,
        },
      });
    }

    return output;
  });
}

// ─── Helpers ───

type Point = [number, number];
type Corners = [Point, Point, Point, Point]; // TL, TR, BR, BL

interface Candidate {
  /** Corners ordered TL, TR, BR, BL in working-resolution pixel coords. */
  corners: Corners;
  /** Width/height from the candidate's minAreaRect, sorted so width >= height. */
  width: number;
  height: number;
  area: number;
  /** Composite score (lower = better fit to expected screen shape). */
  score: number;
  /** Which detection strategy generated this candidate (for debugging). */
  source: string;
}

/**
 * Compute the four vertices of a `cv.minAreaRect` result. OpenCV.js doesn't
 * expose `cv.boxPoints`, so we compute the points from the rect's `center`,
 * `size`, and `angle` (degrees) ourselves.
 */
function rotatedRectPoints(rect: { center: { x: number; y: number }; size: { width: number; height: number }; angle: number }): Point[] {
  const { center, size, angle } = rect;
  const rad = (angle * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const hw = size.width / 2;
  const hh = size.height / 2;
  const local: Point[] = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ];
  return local.map(([x, y]) => [
    center.x + x * cos - y * sin,
    center.y + x * sin + y * cos,
  ]);
}

/**
 * Order four points TL, TR, BR, BL using the same sum/diff heuristic that
 * warp.ts uses, so detected corners are consistent across the codebase.
 */
function orderCornersTLTRBRBL(pts: Point[]): Corners {
  const sums = pts.map(([x, y]) => x + y);
  const yMinusX = pts.map(([x, y]) => y - x);
  const tlIdx = sums.indexOf(Math.min(...sums));
  const brIdx = sums.indexOf(Math.max(...sums));
  const trIdx = yMinusX.indexOf(Math.min(...yMinusX));
  const blIdx = yMinusX.indexOf(Math.max(...yMinusX));
  return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}

/**
 * Score a candidate from its size, aspect, and fill ratio. Lower is better.
 * Used to rank candidates from any source.
 */
function scoreCandidate(area: number, w: number, h: number, fillRatio: number): { score: number; longSide: number; shortSide: number } {
  const longSide = Math.max(w, h);
  const shortSide = Math.max(Math.min(w, h), 1);
  const aspect = longSide / shortSide;
  const aspectErr = Math.abs(aspect / TARGET_ASPECT - 1);
  const quadnessErr = Math.max(0, 1 - fillRatio);
  const score = aspectErr * 2.0 + quadnessErr * 1.0;
  void area;
  return { score, longSide, shortSide };
}

/**
 * Find candidate quads from external contours of a binary mask
 * (the bright-region or any thresholded mask). For each contour we fit a
 * minAreaRect and use its 4 corners.
 */
function findCandidatesInMask(binary: any, source: string): Candidate[] {
  const cv = getCV();
  const imgArea = binary.cols * binary.rows;
  const minArea = imgArea * MIN_CANDIDATE_AREA_FRAC;
  const maxArea = imgArea * MAX_CANDIDATE_AREA_FRAC;

  return withMats((track) => {
    const contours = track(new cv.MatVector());
    const hierarchy = track(new cv.Mat());
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const out: Candidate[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < minArea || area > maxArea) continue;

      const rect = cv.minAreaRect(contour);
      const pts = rotatedRectPoints(rect);
      const corners = orderCornersTLTRBRBL(pts);
      const w = rect.size.width;
      const h = rect.size.height;
      const rectArea = w * h;
      const fillRatio = rectArea > 0 ? area / rectArea : 0;
      const { score, longSide, shortSide } = scoreCandidate(area, w, h, fillRatio);
      out.push({ corners, width: longSide, height: shortSide, area, score, source });
    }
    return out;
  });
}

/**
 * Find candidate quads from *holes* (inner contours) of a dark mask. The
 * GB Screen is bright inside the dark LCD-black ring, so it forms a
 * rectangular hole inside an outer dark contour.
 */
function findCandidateHolesInMask(darkBinary: any, source: string): Candidate[] {
  const cv = getCV();
  const imgArea = darkBinary.cols * darkBinary.rows;
  const minArea = imgArea * MIN_CANDIDATE_AREA_FRAC;
  const maxArea = imgArea * MAX_CANDIDATE_AREA_FRAC;

  return withMats((track) => {
    const contours = track(new cv.MatVector());
    const hierarchy = track(new cv.Mat());
    cv.findContours(darkBinary, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

    const out: Candidate[] = [];
    // hierarchy.data32S layout: [next, prev, firstChild, parent] per contour.
    for (let i = 0; i < contours.size(); i++) {
      const parent = hierarchy.data32S[i * 4 + 3];
      if (parent < 0) continue; // skip outer contours; we want holes
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < minArea || area > maxArea) continue;

      const rect = cv.minAreaRect(contour);
      const pts = rotatedRectPoints(rect);
      const corners = orderCornersTLTRBRBL(pts);
      const w = rect.size.width;
      const h = rect.size.height;
      const rectArea = w * h;
      const fillRatio = rectArea > 0 ? area / rectArea : 0;
      const { score, longSide, shortSide } = scoreCandidate(area, w, h, fillRatio);
      out.push({ corners, width: longSide, height: shortSide, area, score, source });
    }
    return out;
  });
}

/**
 * Find candidate quads via Canny edges + approxPolyDP. Keeps contours
 * that approximate to a 4-vertex polygon (true rectangle outline).
 */
function findQuadCandidates(edgeBinary: any, source: string): Candidate[] {
  const cv = getCV();
  const imgArea = edgeBinary.cols * edgeBinary.rows;
  const minArea = imgArea * MIN_CANDIDATE_AREA_FRAC;
  const maxArea = imgArea * MAX_CANDIDATE_AREA_FRAC;

  return withMats((track) => {
    const contours = track(new cv.MatVector());
    const hierarchy = track(new cv.Mat());
    cv.findContours(edgeBinary, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const out: Candidate[] = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < minArea || area > maxArea) continue;

      const peri = cv.arcLength(contour, true);
      const approx = track(new cv.Mat());
      cv.approxPolyDP(contour, approx, peri * APPROX_EPSILON_FRAC, true);
      if (approx.rows !== 4) continue;

      const pts: Point[] = [];
      for (let k = 0; k < 4; k++) {
        pts.push([approx.data32S[k * 2], approx.data32S[k * 2 + 1]]);
      }
      const corners = orderCornersTLTRBRBL(pts);
      // approximate w/h as the average of opposing sides
      const topLen = Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]);
      const botLen = Math.hypot(corners[2][0] - corners[3][0], corners[2][1] - corners[3][1]);
      const leftLen = Math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]);
      const rightLen = Math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]);
      const w = (topLen + botLen) / 2;
      const h = (leftLen + rightLen) / 2;
      // For a quad approximation, the polygon area itself is the contour area.
      const fillRatio = w * h > 0 ? area / (w * h) : 0;
      const { score, longSide, shortSide } = scoreCandidate(area, w, h, fillRatio);
      out.push({ corners, width: longSide, height: shortSide, area, score, source });
    }
    return out;
  });
}

/**
 * Deduplicate candidates whose centers and sizes are very similar (the
 * three strategies often find the same screen) and return the top-N by
 * score.
 */
function dedupeAndRank(pool: Candidate[], topN: number): Candidate[] {
  pool.sort((a, b) => a.score - b.score);
  const kept: Candidate[] = [];
  for (const c of pool) {
    const cx = (c.corners[0][0] + c.corners[2][0]) / 2;
    const cy = (c.corners[0][1] + c.corners[2][1]) / 2;
    const isDup = kept.some((k) => {
      const kx = (k.corners[0][0] + k.corners[2][0]) / 2;
      const ky = (k.corners[0][1] + k.corners[2][1]) / 2;
      const centerDist = Math.hypot(cx - kx, cy - ky);
      const sizeDiff = Math.abs(c.width - k.width) + Math.abs(c.height - k.height);
      return centerDist < Math.max(c.width, k.width) * 0.1 && sizeDiff < (c.width + k.width) * 0.1;
    });
    if (!isDup) kept.push(c);
    if (kept.length >= topN) break;
  }
  return kept;
}

/**
 * Warp a working-resolution grayscale image to the normalized 160×144 frame
 * coordinate system using the candidate's corners (TL → (0,0), TR → (159,0),
 * BR → (159, 143), BL → (0, 143)). Returns a new Mat (caller must delete or
 * track via withMats).
 */
function warpToNormalized(gray: any /* cv.Mat (CV_8UC1) */, corners: Corners): any {
  const cv = getCV();
  const N = 160;
  const M = 144;
  return withMats((track) => {
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0][0], corners[0][1],
      corners[1][0], corners[1][1],
      corners[2][0], corners[2][1],
      corners[3][0], corners[3][1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      N - 1, 0,
      N - 1, M - 1,
      0, M - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const out = new cv.Mat();
    cv.warpPerspective(gray, out, Mhom, new cv.Size(N, M), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
    return out;
  });
}

interface DashOffsets {
  /** Outward offset for top edge in normalized rows (positive = expand outward). */
  top: number;
  /** Outward offset for right edge in normalized cols. */
  right: number;
  /** Outward offset for bottom edge in normalized rows. */
  bottom: number;
  /** Outward offset for left edge in normalized cols. */
  left: number;
}

/**
 * Compute brightness variance of a row in `warped`, sampling only the inner
 * column range [colStart, colEnd) to avoid the fused corner dashes.
 */
function rowVariance(warped: any /* cv.Mat */, row: number, colStart: number, colEnd: number): number {
  const cols: number = warped.cols;
  const data: Uint8Array = warped.data;
  let sum = 0, sumSq = 0, n = 0;
  for (let x = colStart; x < colEnd; x++) {
    const v = data[row * cols + x];
    sum += v;
    sumSq += v * v;
    n++;
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

/**
 * Compute brightness variance of a column in `warped`, sampling only the
 * inner row range [rowStart, rowEnd).
 */
function colVariance(warped: any /* cv.Mat */, col: number, rowStart: number, rowEnd: number): number {
  const cols: number = warped.cols;
  const data: Uint8Array = warped.data;
  let sum = 0, sumSq = 0, n = 0;
  for (let y = rowStart; y < rowEnd; y++) {
    const v = data[y * cols + col];
    sum += v;
    sumSq += v * v;
    n++;
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

/** Debug helper: returns the variance profiles for each side. */
function findDashVariances(warped: any): { topVar: number[]; bottomVar: number[]; leftVar: number[]; rightVar: number[] } {
  const N = warped.cols;
  const M = warped.rows;
  const innerColStart = 16, innerColEnd = N - 16;
  const innerRowStart = 16, innerRowEnd = M - 16;
  const topVar: number[] = [];
  for (let r = 0; r < DASH_SEARCH_WINDOW; r++) {
    topVar.push(Math.round(rowVariance(warped, r, innerColStart, innerColEnd)));
  }
  const bottomVar: number[] = [];
  for (let r = M - DASH_SEARCH_WINDOW; r < M; r++) {
    bottomVar.push(Math.round(rowVariance(warped, r, innerColStart, innerColEnd)));
  }
  const leftVar: number[] = [];
  for (let c = 0; c < DASH_SEARCH_WINDOW; c++) {
    leftVar.push(Math.round(colVariance(warped, c, innerRowStart, innerRowEnd)));
  }
  const rightVar: number[] = [];
  for (let c = N - DASH_SEARCH_WINDOW; c < N; c++) {
    rightVar.push(Math.round(colVariance(warped, c, innerRowStart, innerRowEnd)));
  }
  return { topVar, bottomVar, leftVar, rightVar };
}

/** Debug helper: returns the actual rows/cols where dashes were found. */
function findDashRowDebug(warped: any): { topRow: number; bottomRow: number; leftCol: number; rightCol: number } {
  const N = warped.cols;
  const M = warped.rows;
  const innerColStart = 16, innerColEnd = N - 16;
  const innerRowStart = 16, innerRowEnd = M - 16;
  let topMax = -Infinity, topRow = EXPECTED_TOP_DASH_ROW;
  for (let r = 0; r < DASH_SEARCH_WINDOW; r++) {
    const v = rowVariance(warped, r, innerColStart, innerColEnd);
    if (v > topMax) { topMax = v; topRow = r; }
  }
  let botMax = -Infinity, bottomRow = EXPECTED_BOTTOM_DASH_ROW;
  for (let r = M - DASH_SEARCH_WINDOW; r < M; r++) {
    const v = rowVariance(warped, r, innerColStart, innerColEnd);
    if (v > botMax) { botMax = v; bottomRow = r; }
  }
  let leftMax = -Infinity, leftCol = EXPECTED_LEFT_DASH_COL;
  for (let c = 0; c < DASH_SEARCH_WINDOW; c++) {
    const v = colVariance(warped, c, innerRowStart, innerRowEnd);
    if (v > leftMax) { leftMax = v; leftCol = c; }
  }
  let rightMax = -Infinity, rightCol = EXPECTED_RIGHT_DASH_COL;
  for (let c = N - DASH_SEARCH_WINDOW; c < N; c++) {
    const v = colVariance(warped, c, innerRowStart, innerRowEnd);
    if (v > rightMax) { rightMax = v; rightCol = c; }
  }
  return { topRow, bottomRow, leftCol, rightCol };
}

/**
 * Find the dash row/col by scanning each edge's search window and picking
 * the row/col with maximum variance. Returns offsets per edge in normalized
 * (160×144) space — positive offset means we need to expand that edge
 * outward (the dashes are at a smaller-than-expected coordinate, indicating
 * the candidate is inset).
 */
function findDashOffsets(warped: any /* cv.Mat (CV_8UC1) */): DashOffsets {
  const N = warped.cols;
  const M = warped.rows;

  // Inner column range to sample for top/bottom (skip first/last 16 cols
  // which are corners with fused dashes that confuse the variance calc).
  const innerColStart = 16;
  const innerColEnd = N - 16;
  const innerRowStart = 16;
  const innerRowEnd = M - 16;

  // ── TOP: scan rows [0, DASH_SEARCH_WINDOW), find max-variance row ──
  let topMaxVar = -Infinity;
  let topDashRow = EXPECTED_TOP_DASH_ROW;
  for (let r = 0; r < DASH_SEARCH_WINDOW; r++) {
    const v = rowVariance(warped, r, innerColStart, innerColEnd);
    if (v > topMaxVar) { topMaxVar = v; topDashRow = r; }
  }

  // ── BOTTOM: scan rows [M-DASH_SEARCH_WINDOW, M), find max-variance row ──
  let botMaxVar = -Infinity;
  let bottomDashRow = EXPECTED_BOTTOM_DASH_ROW;
  for (let r = M - DASH_SEARCH_WINDOW; r < M; r++) {
    const v = rowVariance(warped, r, innerColStart, innerColEnd);
    if (v > botMaxVar) { botMaxVar = v; bottomDashRow = r; }
  }

  // ── LEFT: scan cols [0, DASH_SEARCH_WINDOW), find max-variance col ──
  let leftMaxVar = -Infinity;
  let leftDashCol = EXPECTED_LEFT_DASH_COL;
  for (let c = 0; c < DASH_SEARCH_WINDOW; c++) {
    const v = colVariance(warped, c, innerRowStart, innerRowEnd);
    if (v > leftMaxVar) { leftMaxVar = v; leftDashCol = c; }
  }

  // ── RIGHT: scan cols [N-DASH_SEARCH_WINDOW, N), find max-variance col ──
  let rightMaxVar = -Infinity;
  let rightDashCol = EXPECTED_RIGHT_DASH_COL;
  for (let c = N - DASH_SEARCH_WINDOW; c < N; c++) {
    const v = colVariance(warped, c, innerRowStart, innerRowEnd);
    if (v > rightMaxVar) { rightMaxVar = v; rightDashCol = c; }
  }

  return {
    top: EXPECTED_TOP_DASH_ROW - topDashRow,         // observed < expected → +offset → expand outward
    right: rightDashCol - EXPECTED_RIGHT_DASH_COL,   // observed > expected → +offset → expand outward
    bottom: bottomDashRow - EXPECTED_BOTTOM_DASH_ROW, // observed > expected → +offset → expand outward
    left: EXPECTED_LEFT_DASH_COL - leftDashCol,      // observed < expected → +offset → expand outward
  };
}

/**
 * Move each edge of a quad outward by the per-edge offset (in normalized
 * 160×144 units), translated to the working-res scale of the candidate.
 * Reconstruct the corners as the intersections of adjacent moved edges.
 */
function applyEdgeOffsets(corners: Corners, offsets: DashOffsets): Corners {
  // Clamp absurd offsets — if dash detection found something at the very
  // edge of the search window, that's likely a false positive (LCD ring or
  // similar) rather than a real dash.
  const MAX_OFFSET_NORM = 6;
  const top = clamp(offsets.top, -MAX_OFFSET_NORM, MAX_OFFSET_NORM);
  const right = clamp(offsets.right, -MAX_OFFSET_NORM, MAX_OFFSET_NORM);
  const bottom = clamp(offsets.bottom, -MAX_OFFSET_NORM, MAX_OFFSET_NORM);
  const left = clamp(offsets.left, -MAX_OFFSET_NORM, MAX_OFFSET_NORM);

  // The candidate's TL → (0,0), TR → (160-1, 0), BR → (160-1, 144-1),
  // BL → (0, 144-1). Moving an edge in the normalized space by Δ is
  // equivalent to translating its corners by Δ along the edge's outward
  // perpendicular in the candidate's coordinate frame. We use the
  // candidate's edge vectors directly so the translation works for any
  // rotation/skew in the original photo.
  const [TL, TR, BR, BL] = corners;
  // Top edge (TL→TR) outward perpendicular in the candidate's frame is
  // -(TL→BL) direction (away from BL). Similarly for the other sides.
  const perp = (a: Point, b: Point, len: number, scaleNorm: number): Point => {
    // Returns a vector of working-res-pixel length representing `scaleNorm`
    // normalized units along the outward perpendicular of edge a→b.
    // Outward perpendicular = unit vector from a→b, rotated 90° AWAY from
    // quad center. For our corners (TL, TR, BR, BL clockwise), CCW rotation
    // by 90° points outward.
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const elen = Math.hypot(ex, ey);
    // CCW: (x, y) → (-y, x). Outward normal:
    let nx = -ey / elen;
    let ny = ex / elen;
    // Verify outwardness via center.
    const cx = (TL[0] + TR[0] + BR[0] + BL[0]) / 4;
    const cy = (TL[1] + TR[1] + BR[1] + BL[1]) / 4;
    const midX = (a[0] + b[0]) / 2;
    const midY = (a[1] + b[1]) / 2;
    if (nx * (cx - midX) + ny * (cy - midY) > 0) {
      nx = -nx;
      ny = -ny;
    }
    // 1 normalized unit on the top/bottom edge = elen / 159 candidate-px
    // (since edge spans normalized 0..159). For left/right edges, len = 143.
    return [nx * (elen / (len - 1)) * scaleNorm, ny * (elen / (len - 1)) * scaleNorm];
  };

  const topShift = perp(TL, TR, 160, top);
  const rightShift = perp(TR, BR, 144, right);
  const bottomShift = perp(BR, BL, 160, bottom);
  const leftShift = perp(BL, TL, 144, left);

  // Each corner is shared by two edges; sum their shifts to move it.
  const moved: Corners = [
    [TL[0] + topShift[0] + leftShift[0], TL[1] + topShift[1] + leftShift[1]],
    [TR[0] + topShift[0] + rightShift[0], TR[1] + topShift[1] + rightShift[1]],
    [BR[0] + bottomShift[0] + rightShift[0], BR[1] + bottomShift[1] + rightShift[1]],
    [BL[0] + bottomShift[0] + leftShift[0], BL[1] + bottomShift[1] + leftShift[1]],
  ];
  return moved;
}

/** Convert a single-channel binary Mat to RGBA so it can be saved as a PNG. */
function toRgba(grayMat: any): any {
  const cv = getCV();
  const out = new cv.Mat();
  cv.cvtColor(grayMat, out, cv.COLOR_GRAY2RGBA);
  return out;
}

interface ValidationScore {
  /** Score 0–1 measuring how dark the expected inner-border ring is. */
  innerBorderScore: number;
  /** Score 0–1 measuring how dark the band immediately outside the candidate is. */
  darkRingScore: number;
  /** Composite total, 0–1 (higher = better). */
  totalScore: number;
}

/**
 * Validate a candidate by perspective-warping it to a normalized 160×144
 * image and scoring two Frame 02 features:
 *   - Inner-border ring: at the expected location of Frame 02's #9494FF
 *     inner border (inset 16 px from the outer edge), the ring should be
 *     darker than the surrounding white frame.
 *   - Surrounding dark ring: a band immediately outside the candidate (in
 *     working-resolution coords) should be darker than the candidate's
 *     interior — this is the GBA SP LCD-black under the front-light.
 *
 * Both signals are normalized to 0–1 and averaged into `totalScore`.
 */
function validateCandidate(
  workingRgba: any /* cv.Mat */,
  candidate: Candidate,
): ValidationScore {
  const cv = getCV();

  return withMats((track) => {
    // ── Inner-border ring: warp candidate to normalized 160×144 ──
    const N = 160; // normalized width
    const M = 144; // normalized height
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      candidate.corners[0][0], candidate.corners[0][1],
      candidate.corners[1][0], candidate.corners[1][1],
      candidate.corners[2][0], candidate.corners[2][1],
      candidate.corners[3][0], candidate.corners[3][1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0,
      N - 1, 0,
      N - 1, M - 1,
      0, M - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const warped = track(new cv.Mat());
    cv.warpPerspective(workingRgba, warped, Mhom, new cv.Size(N, M), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());

    const warpedGray = track(new cv.Mat());
    cv.cvtColor(warped, warpedGray, cv.COLOR_RGBA2GRAY);

    // Inner-border ring sits at row/col 15 of the normalized frame
    // (16-px-thick frame, inner border at outer edge of the camera area).
    // We measure two means:
    //   meanFrame   — interior of the 16-px frame band (excluding the ring itself)
    //   meanRing    — the inner-border ring at row/col 15
    // Score = clamp((meanFrame - meanRing) / 80, 0, 1)
    //   80 is a reasonable expected contrast (white-frame ≈ 230, ring ≈ 100).
    const ringRow = 15;
    let meanFrame = 0, frameCnt = 0;
    let meanRing = 0, ringCnt = 0;
    const data = warpedGray.data; // Uint8Array, length N*M
    for (let y = 0; y < M; y++) {
      for (let x = 0; x < N; x++) {
        const v = data[y * N + x];
        const inFrame =
          (y < 16 || y >= M - 16 || x < 16 || x >= N - 16) &&
          !(y === ringRow || y === M - 1 - ringRow || x === ringRow || x === N - 1 - ringRow);
        const onRing =
          (y === ringRow && x >= ringRow && x <= N - 1 - ringRow) ||
          (y === M - 1 - ringRow && x >= ringRow && x <= N - 1 - ringRow) ||
          (x === ringRow && y > ringRow && y < M - 1 - ringRow) ||
          (x === N - 1 - ringRow && y > ringRow && y < M - 1 - ringRow);
        if (inFrame) { meanFrame += v; frameCnt++; }
        if (onRing) { meanRing += v; ringCnt++; }
      }
    }
    meanFrame = frameCnt > 0 ? meanFrame / frameCnt : 0;
    meanRing = ringCnt > 0 ? meanRing / ringCnt : 0;
    // The 1-px-thick inner border averages out heavily under perspective
    // warp + bilinear interpolation, so the available contrast is small.
    // Normalize against an empirically-observed ~25-unit gap.
    const innerBorderScore = clamp((meanFrame - meanRing) / 25, 0, 1);

    // ── Surrounding dark ring: in working-resolution coords ──
    // Sample a band of width = ringWidth pixels just outside each edge of
    // the candidate's bounding box and compute its mean. Compare to the
    // candidate's overall interior mean. This is the most reliable signal:
    // the GBA SP LCD-black surrounds every screen photo consistently.
    const ringWidth = Math.max(4, Math.round(Math.min(candidate.width, candidate.height) * 0.05));
    const bbox = boundingBoxOfCorners(candidate.corners, workingRgba.cols, workingRgba.rows);
    const interiorMean = meanGrayInBox(workingRgba, bbox, 0);
    const outsideMean = meanGrayInRingAround(workingRgba, bbox, ringWidth);
    const darkRingScore = clamp((interiorMean - outsideMean) / 80, 0, 1);

    // ── Aspect ratio signal: how close the candidate is to 160:144 ──
    // (Already used in candidate scoring, but repeating here as a
    // validation check rewards candidates that are near-perfect aspect.)
    const longSide = Math.max(candidate.width, candidate.height);
    const shortSide = Math.max(Math.min(candidate.width, candidate.height), 1);
    const aspect = longSide / shortSide;
    const aspectErr = Math.abs(aspect / TARGET_ASPECT - 1);
    const aspectScore = clamp(1 - aspectErr * 5, 0, 1); // err of 0.2 → score 0

    // Weighted average — darkRing is the dominant reliable signal.
    const totalScore =
      darkRingScore * 0.6 + aspectScore * 0.3 + innerBorderScore * 0.1;
    return { innerBorderScore, darkRingScore, totalScore };
  });
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

interface Bbox { x0: number; y0: number; x1: number; y1: number; }

function boundingBoxOfCorners(corners: Corners, imgW: number, imgH: number): Bbox {
  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  return {
    x0: Math.max(0, Math.floor(Math.min(...xs))),
    y0: Math.max(0, Math.floor(Math.min(...ys))),
    x1: Math.min(imgW, Math.ceil(Math.max(...xs))),
    y1: Math.min(imgH, Math.ceil(Math.max(...ys))),
  };
}

function meanGrayInBox(rgba: any /* cv.Mat */, b: Bbox, channel: number): number {
  const cv = getCV();
  return withMats((track) => {
    const gray = track(new cv.Mat());
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    let sum = 0, cnt = 0;
    for (let y = b.y0; y < b.y1; y++) {
      for (let x = b.x0; x < b.x1; x++) {
        sum += gray.data[y * gray.cols + x];
        cnt++;
      }
    }
    void channel; // retained for future per-channel scoring; unused for now.
    return cnt > 0 ? sum / cnt : 0;
  });
}

function meanGrayInRingAround(rgba: any /* cv.Mat */, b: Bbox, ringWidth: number): number {
  const cv = getCV();
  return withMats((track) => {
    const gray = track(new cv.Mat());
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    let sum = 0, cnt = 0;
    const W = gray.cols;
    const H = gray.rows;
    const xa = Math.max(0, b.x0 - ringWidth);
    const xb = Math.min(W, b.x1 + ringWidth);
    const ya = Math.max(0, b.y0 - ringWidth);
    const yb = Math.min(H, b.y1 + ringWidth);
    for (let y = ya; y < yb; y++) {
      for (let x = xa; x < xb; x++) {
        const inInner = (x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1);
        if (inInner) continue;
        sum += gray.data[y * W + x];
        cnt++;
      }
    }
    return cnt > 0 ? sum / cnt : 0;
  });
}

/**
 * Render an 8x-upscaled normalized-160×144 view of the chosen candidate
 * with overlays showing the inner-border ring (red) and an annotation of
 * the score values (drawn as colored squares — top-left red square's
 * brightness encodes innerBorderScore, top-right encodes darkRingScore).
 *
 * The visualization is intentionally minimal — clusters of pixels with
 * known meaning rather than text — so we don't pull in font rendering.
 */
function renderValidationOverlay(
  workingMat: any /* cv.Mat */,
  candidate: Candidate,
  score: ValidationScore,
): GBImageData {
  const cv = getCV();
  const N = 160, M = 144, UPSCALE = 8;
  return withMats((track) => {
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      candidate.corners[0][0], candidate.corners[0][1],
      candidate.corners[1][0], candidate.corners[1][1],
      candidate.corners[2][0], candidate.corners[2][1],
      candidate.corners[3][0], candidate.corners[3][1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, N - 1, 0, N - 1, M - 1, 0, M - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const warped = track(new cv.Mat());
    cv.warpPerspective(workingMat, warped, Mhom, new cv.Size(N, M), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());

    const upscaled = track(new cv.Mat());
    cv.resize(warped, upscaled, new cv.Size(N * UPSCALE, M * UPSCALE), 0, 0, cv.INTER_NEAREST);

    const out = matToImageData(upscaled);

    // Overlay the inner-border ring at row/col 15 in normalized space — i.e.
    // row/col 15*UPSCALE in upscaled space — as a red rectangle outline.
    const ringPx = 15 * UPSCALE;
    const ringPts: Point[] = [
      [ringPx, ringPx],
      [(N - 1 - 15) * UPSCALE, ringPx],
      [(N - 1 - 15) * UPSCALE, (M - 1 - 15) * UPSCALE],
      [ringPx, (M - 1 - 15) * UPSCALE],
    ];
    drawPolylineRGBA(out, ringPts, [255, 0, 0], 2, true);

    // Score annotations: two filled squares in the top-left corner whose
    // brightness encodes the two component scores.
    const sq = 12 * UPSCALE;
    fillRectRGBA(out, 4, 4, sq, sq, [
      Math.round(255 * score.innerBorderScore),
      0,
      0,
    ]);
    fillRectRGBA(out, 4 + sq + 4, 4, sq, sq, [
      0,
      Math.round(255 * score.darkRingScore),
      0,
    ]);

    return out;
  });
}

function fillRectRGBA(
  img: GBImageData,
  x: number,
  y: number,
  w: number,
  h: number,
  color: [number, number, number],
): void {
  for (let yy = y; yy < y + h; yy++) {
    for (let xx = x; xx < x + w; xx++) {
      if (xx < 0 || yy < 0 || xx >= img.width || yy >= img.height) continue;
      const idx = (yy * img.width + xx) * 4;
      img.data[idx] = color[0];
      img.data[idx + 1] = color[1];
      img.data[idx + 2] = color[2];
      img.data[idx + 3] = 255;
    }
  }
}

/**
 * Scale a corner array from working-resolution coords to original-image
 * coords. `workToOrig` = 1 / `scale` from `downsampleToWorking`.
 */
function scaleCorners(corners: Corners, workToOrig: number): Corners {
  return corners.map(([x, y]) => [x * workToOrig, y * workToOrig] as Point) as Corners;
}

/**
 * Expand a (possibly rotated) rectangle outward by a fraction of its
 * longest side. The expansion is along the rectangle's own axes — the
 * rectangle stays the same shape, just bigger. Corners are returned in
 * the same TL/TR/BR/BL order.
 */
function expandRotatedRect(corners: Corners, ratio: number): Corners {
  const [TL, TR, BR, BL] = corners;
  const cx = (TL[0] + TR[0] + BR[0] + BL[0]) / 4;
  const cy = (TL[1] + TR[1] + BR[1] + BL[1]) / 4;
  const expand = (p: Point): Point => {
    const dx = p[0] - cx;
    const dy = p[1] - cy;
    return [cx + dx * (1 + ratio), cy + dy * (1 + ratio)];
  };
  return [expand(TL), expand(TR), expand(BR), expand(BL)];
}

/**
 * Clamp each corner to [0, imgW]×[0, imgH]. This keeps the warp from
 * sampling out-of-bounds; for already-cropped inputs the clamp is what
 * makes the step a near-no-op.
 */
function clampCorners(corners: Corners, imgW: number, imgH: number): Corners {
  return corners.map(([x, y]) => [
    clamp(x, 0, imgW - 1),
    clamp(y, 0, imgH - 1),
  ] as Point) as Corners;
}

/**
 * Extract the rotated rectangle defined by `corners` from `srcRgba`,
 * producing an axis-aligned RGBA image. Output dimensions equal the
 * average side lengths of the rectangle (rounded to integers).
 */
function extractRotatedRect(srcRgba: any /* cv.Mat */, corners: Corners): GBImageData {
  const cv = getCV();
  const [TL, TR, BR, BL] = corners;
  const topLen = Math.hypot(TR[0] - TL[0], TR[1] - TL[1]);
  const botLen = Math.hypot(BR[0] - BL[0], BR[1] - BL[1]);
  const leftLen = Math.hypot(BL[0] - TL[0], BL[1] - TL[1]);
  const rightLen = Math.hypot(BR[0] - TR[0], BR[1] - TR[1]);
  const outW = Math.max(1, Math.round((topLen + botLen) / 2));
  const outH = Math.max(1, Math.round((leftLen + rightLen) / 2));

  return withMats((track) => {
    const srcPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      TL[0], TL[1], TR[0], TR[1], BR[0], BR[1], BL[0], BL[1],
    ]));
    const dstPts = track(cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, outW - 1, 0, outW - 1, outH - 1, 0, outH - 1,
    ]));
    const Mhom = track(cv.getPerspectiveTransform(srcPts, dstPts));
    const out = track(new cv.Mat());
    cv.warpPerspective(srcRgba, out, Mhom, new cv.Size(outW, outH), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
    return matToImageData(out);
  });
}

/**
 * Draw a polyline on a copy of `img` showing the final output region.
 * `screenCorners` are the chosen-screen corners (cyan); `outputCorners`
 * are the post-margin, post-clamp corners (green).
 */
function drawOutputRegion(
  img: GBImageData,
  screenCorners: Corners,
  outputCorners: Corners,
): GBImageData {
  const out = cloneImage(img);
  const thick = Math.max(2, Math.round(Math.min(img.width, img.height) / 400));
  drawPolylineRGBA(out, screenCorners, [0, 255, 255], thick, true); // cyan
  drawPolylineRGBA(out, outputCorners, [0, 255, 0], thick, true);   // green
  return out;
}

/**
 * Downsample `src` so the longest side is at most `maxDim`. Returns the
 * downsampled Mat (caller is responsible for tracking/deletion via withMats)
 * and the scale factor (working-resolution px per original-image px) so
 * detected coordinates can be mapped back later.
 */
function downsampleToWorking(src: any, maxDim: number): { mat: any; scale: number } {
  const cv = getCV();
  const w = src.cols;
  const h = src.rows;
  const longest = Math.max(w, h);
  if (longest <= maxDim) {
    // No downsampling needed; clone so the caller's track/delete contract is uniform
    const out = new cv.Mat();
    src.copyTo(out);
    return { mat: out, scale: 1 };
  }
  const scale = maxDim / longest;
  const newW = Math.round(w * scale);
  const newH = Math.round(h * scale);
  const out = new cv.Mat();
  cv.resize(src, out, new cv.Size(newW, newH), 0, 0, cv.INTER_AREA);
  return { mat: out, scale };
}

/**
 * Draw all candidates on a copy of the working-resolution photo. The
 * `chosen` candidate (index into `candidates`, or -1 for none) is drawn in
 * green; all others in red. Each is labeled with its score.
 */
function drawCandidates(
  workingRgba: GBImageData,
  candidates: Candidate[],
  chosen: number,
): GBImageData {
  // We avoid pulling in font rendering — just a polyline per candidate is
  // enough for visual debugging. Scores appear in the structured JSON metrics.
  const out = cloneImage(workingRgba);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const color: [number, number, number] = i === chosen ? [0, 255, 0] : [255, 0, 0];
    drawPolylineRGBA(out, c.corners, color, 2, true);
  }
  return out;
}

function drawPolylineRGBA(
  img: GBImageData,
  pts: Point[],
  color: [number, number, number],
  thickness: number,
  closed: boolean,
): void {
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    if (i === n - 1 && !closed) break;
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    drawLineRGBA(img, x0, y0, x1, y1, color, thickness);
  }
}

function drawLineRGBA(
  img: GBImageData,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: [number, number, number],
  thickness: number,
): void {
  // Bresenham with a thickness pad
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const steps = Math.max(dx, dy);
  const r = Math.max(1, Math.floor(thickness / 2));
  for (let i = 0; i <= steps; i++) {
    const t = steps === 0 ? 0 : i / steps;
    const x = Math.round(x0 + (x1 - x0) * t);
    const y = Math.round(y0 + (y1 - y0) * t);
    for (let dyp = -r; dyp <= r; dyp++) {
      for (let dxp = -r; dxp <= r; dxp++) {
        const px = x + dxp;
        const py = y + dyp;
        if (px < 0 || py < 0 || px >= img.width || py >= img.height) continue;
        const idx = (py * img.width + px) * 4;
        img.data[idx] = color[0];
        img.data[idx + 1] = color[1];
        img.data[idx + 2] = color[2];
        img.data[idx + 3] = 255;
      }
    }
  }
}
