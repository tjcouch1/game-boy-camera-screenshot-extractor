import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";

const HOLE_W = 128;
const HOLE_H = 112;
const NORMAL_W = 160;
const NORMAL_H = 144;
/** Pixel is "hole-like" if its alpha is below this. */
const TRANSPARENT_ALPHA = 128;
/** Pixel is "white" if every channel is at least this. */
const WHITE_THRESHOLD = 240;

/**
 * Load a single Game Boy Camera frame from an image where the entire image
 * is the frame body and the 128 × 112 region for the camera image is marked
 * by a uniform transparent or white rectangle.
 *
 * Algorithm:
 *   1. Build a "hole-like" mask (transparent or white pixel).
 *   2. Find the 128 × 112 sub-rectangle that is entirely hole-like and
 *      closest to the image center.
 *   3. Snap every pixel to the four GB grayscale values; however, the
 *      pixels *inside* the detected hole are forced to 255 so the frame
 *      can be correctly composed later.
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

  // Find the 128 × 112 sub-rectangle that is entirely hole-like and closest to
  // the image center. Picking the first match in (y, x) order is brittle
  // because many frame sheets have white or transparent regions in the bezel
  // (e.g. logo areas) that can form accidental 128 × 112 holes if they are
  // positioned just right.
  let bestX = -1;
  let bestY = -1;
  let minDistanceSq = Number.MAX_VALUE;

  const centerX = (W - HOLE_W) / 2;
  const centerY = (H - HOLE_H) / 2;

  for (let y0 = 0; y0 + HOLE_H <= H; y0++) {
    const colSum = new Int32Array(W);
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let dy = 0; dy < HOLE_H; dy++) {
        s += isHoleLike[(y0 + dy) * W + x];
      }
      colSum[x] = s;
    }

    let windowSum = 0;
    for (let x = 0; x < HOLE_W; x++) windowSum += colSum[x];

    const target = HOLE_W * HOLE_H;
    for (let x0 = 0; x0 + HOLE_W <= W; x0++) {
      if (x0 > 0) {
        windowSum += colSum[x0 + HOLE_W - 1] - colSum[x0 - 1];
      }

      if (windowSum === target) {
        const dx = x0 - centerX;
        const dy = y0 - centerY;
        const distSq = dx * dx + dy * dy;
        if (distSq < minDistanceSq) {
          minDistanceSq = distSq;
          bestX = x0;
          bestY = y0;
        }
      }
    }
  }

  if (bestX < 0) {
    throw new Error(
      `loadIndividualFrame: no ${HOLE_W}x${HOLE_H} transparent or white hole found in ${frameStem}`,
    );
  }

  const pixels = new Uint8ClampedArray(W * H);
  // Process bezel pixels normally (snap to nearest GB color).
  for (let i = 0; i < W * H; i++) {
    pixels[i] = snapToGB(data[i * 4]);
  }
  // Force the hole region to the hole marker (255).
  for (let dy = 0; dy < HOLE_H; dy++) {
    for (let dx = 0; dx < HOLE_W; dx++) {
      pixels[(bestY + dy) * W + (bestX + dx)] = 255;
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
    holeX: bestX,
    holeY: bestY,
  };
}

/** Snap a 0–255 value to the nearest of {0, 82, 165, 255}. */
function snapToGB(v: number): number {
  if (v < 41) return 0;
  if (v < 124) return 82;
  if (v < 210) return 165;
  return 255;
}
