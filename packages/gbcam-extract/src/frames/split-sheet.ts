import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";

const HOLE_W = 128;
const HOLE_H = 112;
const NORMAL_W = 160;
const NORMAL_H = 144;
/** Per-channel tolerance when matching the background colour. */
const BG_TOLERANCE = 2;
/** Per-channel tolerance when checking whether a region is uniform-coloured. */
const UNIFORM_TOLERANCE = 4;
/** Drop bounding boxes smaller than this — credits / labels. */
const MIN_FRAME_DIM = 32;

interface BBox {
  x0: number;
  y0: number;
  x1: number; // exclusive
  y1: number; // exclusive
}

/**
 * Split a frame-sheet PNG into individual frames.
 *
 * Algorithm:
 *   1. Read the top-left pixel as the background colour.
 *   2. Build a mask of background pixels (with ±BG_TOLERANCE per channel).
 *   3. Find connected components of NON-background pixels (4-connectivity).
 *   4. For each component, find the first 128 × 112 sub-rectangle that is
 *      uniform colour (handles holes that aren't the exact background colour);
 *      if found it's a frame, otherwise drop it.
 *   5. Recompute the tight frame bounding box from the hole edges so that
 *      spurious merged pixels outside the frame body do not distort the size.
 *   6. Sort the kept frames in (y, x) reading order and number per type.
 */
export function splitSheet(sheet: GBImageData, sheetStem: string): Frame[] {
  const W = sheet.width;
  const H = sheet.height;
  const data = sheet.data;

  const bgR = data[0];
  const bgG = data[1];
  const bgB = data[2];

  // bgMask[i] = 1 if pixel i is the background colour, else 0.
  const bgMask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    if (
      Math.abs(r - bgR) <= BG_TOLERANCE &&
      Math.abs(g - bgG) <= BG_TOLERANCE &&
      Math.abs(b - bgB) <= BG_TOLERANCE
    ) {
      bgMask[i] = 1;
    }
  }

  /**
   * True iff every pixel in the 128 × 112 region starting at (x0, y0) is
   * within UNIFORM_TOLERANCE of the top-left pixel's colour. This detects
   * both background-coloured holes and opaque fill holes (e.g. sprite sheets
   * where the hole is a solid non-background placeholder colour).
   */
  const isUniformHole = (x0: number, y0: number): boolean => {
    const i0 = (y0 * W + x0) * 4;
    const hr = data[i0];
    const hg = data[i0 + 1];
    const hb = data[i0 + 2];
    for (let dy = 0; dy < HOLE_H; dy++) {
      for (let dx = 0; dx < HOLE_W; dx++) {
        const i = ((y0 + dy) * W + (x0 + dx)) * 4;
        if (
          Math.abs(data[i] - hr) > UNIFORM_TOLERANCE ||
          Math.abs(data[i + 1] - hg) > UNIFORM_TOLERANCE ||
          Math.abs(data[i + 2] - hb) > UNIFORM_TOLERANCE
        ) {
          return false;
        }
      }
    }
    return true;
  };

  // Connected components of non-background pixels (4-connectivity).
  // Iterative flood fill via stack to avoid recursion.
  const seen = new Uint8Array(W * H);
  const bboxes: BBox[] = [];
  const stack: number[] = [];

  for (let i = 0; i < W * H; i++) {
    if (bgMask[i] || seen[i]) continue;
    let minX = i % W;
    let minY = (i / W) | 0;
    let maxX = minX;
    let maxY = minY;
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    while (stack.length > 0) {
      const j = stack.pop()!;
      const x = j % W;
      const y = (j / W) | 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && !bgMask[j - 1] && !seen[j - 1]) {
        seen[j - 1] = 1;
        stack.push(j - 1);
      }
      if (x < W - 1 && !bgMask[j + 1] && !seen[j + 1]) {
        seen[j + 1] = 1;
        stack.push(j + 1);
      }
      if (y > 0 && !bgMask[j - W] && !seen[j - W]) {
        seen[j - W] = 1;
        stack.push(j - W);
      }
      if (y < H - 1 && !bgMask[j + W] && !seen[j + W]) {
        seen[j + W] = 1;
        stack.push(j + W);
      }
    }
    bboxes.push({ x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1 });
  }

  // Filter to candidates that are big enough and have a 128 × 112 uniform hole.
  type Candidate = { bbox: BBox; holeX: number; holeY: number };
  const candidates: Candidate[] = [];
  for (const bbox of bboxes) {
    const w = bbox.x1 - bbox.x0;
    const h = bbox.y1 - bbox.y0;
    if (w < MIN_FRAME_DIM || h < MIN_FRAME_DIM) continue;
    if (w < HOLE_W || h < HOLE_H) continue;

    let foundX = -1;
    let foundY = -1;
    outer: for (let yy = bbox.y0; yy + HOLE_H <= bbox.y1; yy++) {
      for (let xx = bbox.x0; xx + HOLE_W <= bbox.x1; xx++) {
        if (isUniformHole(xx, yy)) {
          foundX = xx;
          foundY = yy;
          break outer;
        }
      }
    }
    if (foundX < 0) continue;

    // Recompute the tight frame bounding box by scanning outward from the
    // hole edges along a single row/column. This ensures that spurious
    // pixels that merged into the component during flood fill do not inflate
    // the frame's reported dimensions.
    let frameX0 = foundX;
    while (frameX0 > 0 && !bgMask[(foundY) * W + (frameX0 - 1)]) frameX0--;

    let frameX1 = foundX + HOLE_W;
    while (frameX1 < W && !bgMask[(foundY) * W + frameX1]) frameX1++;

    let frameY0 = foundY;
    while (frameY0 > 0 && !bgMask[(frameY0 - 1) * W + foundX]) frameY0--;

    let frameY1 = foundY + HOLE_H;
    while (frameY1 < H && !bgMask[frameY1 * W + foundX]) frameY1++;

    const tightBbox: BBox = { x0: frameX0, y0: frameY0, x1: frameX1, y1: frameY1 };
    candidates.push({
      bbox: tightBbox,
      holeX: foundX - frameX0,
      holeY: foundY - frameY0,
    });
  }

  // Sort top-to-bottom, then left-to-right.
  candidates.sort((a, b) => {
    if (a.bbox.y0 !== b.bbox.y0) return a.bbox.y0 - b.bbox.y0;
    return a.bbox.x0 - b.bbox.x0;
  });

  // Number per type and emit Frame objects.
  let normalIdx = 0;
  let wildIdx = 0;
  const frames: Frame[] = [];
  for (const c of candidates) {
    const w = c.bbox.x1 - c.bbox.x0;
    const h = c.bbox.y1 - c.bbox.y0;
    const type: "normal" | "wild" =
      w === NORMAL_W && h === NORMAL_H ? "normal" : "wild";
    const index = type === "normal" ? ++normalIdx : ++wildIdx;
    const id = `${sheetStem}:${type}:${index}`;
    const pixels = new Uint8ClampedArray(w * h);
    for (let yy = 0; yy < h; yy++) {
      for (let xx = 0; xx < w; xx++) {
        const sx = c.bbox.x0 + xx;
        const sy = c.bbox.y0 + yy;
        const di = (sy * W + sx) * 4;
        const isBg = bgMask[sy * W + sx] === 1;
        if (isBg) {
          pixels[yy * w + xx] = 255;
        } else {
          pixels[yy * w + xx] = snapToGB(data[di]);
        }
      }
    }
    frames.push({
      id,
      sheetStem,
      type,
      index,
      width: w,
      height: h,
      pixels,
      holeX: c.holeX,
      holeY: c.holeY,
    });
  }
  return frames;
}

/** Snap a 0–255 value to the nearest of {0, 82, 165, 255}. */
function snapToGB(v: number): number {
  // Midpoints between adjacent levels: (0,82)=41, (82,165)=123.5, (165,255)=210.
  if (v < 41) return 0;
  if (v < 124) return 82;
  if (v < 210) return 165;
  return 255;
}
