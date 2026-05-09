import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";

const HOLE_W = 128;
const HOLE_H = 112;
const NORMAL_W = 160;
const NORMAL_H = 144;
/** Pixel is "hole-like" if its alpha is below this. */
const TRANSPARENT_ALPHA = 128;
/** Pixel is "white" if every channel is at least this. */
const WHITE_THRESHOLD = 250;

/**
 * Load a single Game Boy Camera frame from an image where the entire image
 * is the frame body and the 128 × 112 region for the camera image is marked
 * by a uniform transparent or white rectangle.
 *
 * Algorithm:
 *   1. Build a "hole-like" mask (transparent or white pixel).
 *   2. Find the first 128 × 112 sub-rectangle that is entirely hole-like
 *      (in (y, x) reading order); that's the hole.
 *   3. Snap every non-hole-like pixel to the four GB grayscale values; hole
 *      pixels (and any other transparent/white pixels outside the hole) are
 *      stored as 255 so the frame renders with the lightest palette colour
 *      when shown alone in the picker.
 *
 * Type derives from image dimensions: 160 × 144 → "normal", anything else →
 * "wild". Each individual source produces exactly one frame, indexed 1.
 */
export function loadIndividualFrame(
  image: GBImageData,
  frameStem: string,
): Frame {
  const W = image.width;
  const H = image.height;
  const data = image.data;

  if (W < HOLE_W || H < HOLE_H) {
    throw new Error(
      `loadIndividualFrame: image ${W}x${H} is smaller than the ${HOLE_W}x${HOLE_H} hole`,
    );
  }

  const isHoleLike = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const a = data[i * 4 + 3];
    if (
      a < TRANSPARENT_ALPHA ||
      (r >= WHITE_THRESHOLD && g >= WHITE_THRESHOLD && b >= WHITE_THRESHOLD)
    ) {
      isHoleLike[i] = 1;
    }
  }

  // Row-prefix-sum trick over isHoleLike to find an all-1 128 × 112 rectangle.
  // For each (y0, x0), the number of hole-like pixels in the 128 × 112 region
  // equals HOLE_W * HOLE_H iff every pixel is hole-like.
  // Use a per-row sliding window summed across rows.
  let foundX = -1;
  let foundY = -1;
  // colSum[x] = number of hole-like pixels in the vertical strip [y0, y0 + HOLE_H)
  // at column x. Recomputed once per y0.
  outer: for (let y0 = 0; y0 + HOLE_H <= H; y0++) {
    const colSum = new Int32Array(W);
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = 0; dy < HOLE_H; dy++) {
        s += isHoleLike[(y0 + dy) * W + x];
      }
      colSum[x] = s;
    }
    // Sliding window of width HOLE_W across columns.
    let windowSum = 0;
    for (let x = 0; x < HOLE_W; x++) windowSum += colSum[x];
    const target = HOLE_W * HOLE_H;
    if (windowSum === target) {
      foundX = 0;
      foundY = y0;
      break outer;
    }
    for (let x0 = 1; x0 + HOLE_W <= W; x0++) {
      windowSum += colSum[x0 + HOLE_W - 1] - colSum[x0 - 1];
      if (windowSum === target) {
        foundX = x0;
        foundY = y0;
        break outer;
      }
    }
  }

  if (foundX < 0) {
    throw new Error(
      `loadIndividualFrame: no ${HOLE_W}x${HOLE_H} transparent or white hole found in ${frameStem}`,
    );
  }

  const pixels = new Uint8ClampedArray(W * H);
  for (let i = 0; i < W * H; i++) {
    if (isHoleLike[i]) {
      pixels[i] = 255;
    } else {
      pixels[i] = snapToGB(data[i * 4]);
    }
  }

  const type: "normal" | "wild" =
    W === NORMAL_W && H === NORMAL_H ? "normal" : "wild";
  const id = `${frameStem}:${type}:1`;
  return {
    id,
    sheetStem: frameStem,
    aliasStems: [frameStem],
    type,
    kind: "individual",
    index: 1,
    width: W,
    height: H,
    pixels,
    holeX: foundX,
    holeY: foundY,
  };
}

/** Snap a 0–255 value to the nearest of {0, 82, 165, 255}. */
function snapToGB(v: number): number {
  if (v < 41) return 0;
  if (v < 124) return 82;
  if (v < 210) return 165;
  return 255;
}
