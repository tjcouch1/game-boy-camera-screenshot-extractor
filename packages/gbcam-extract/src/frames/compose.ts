import type { GBImageData } from "../common.js";
import type { Frame } from "./types.js";
import { applyPalette } from "../palette.js";

const HOLE_W = 128;
const HOLE_H = 112;

/**
 * Compose a 128 × 112 grayscale image into a frame and render the result
 * with the given palette.
 *
 * Throws if the image is not 128 × 112 or if any input is malformed. The
 * caller is responsible for catching and falling back to a bare
 * `applyPalette(image, palette)` when the frame pipeline shouldn't crash
 * the UI.
 */
export function composeFrame(
  image: GBImageData,
  frame: Frame,
  palette: [string, string, string, string],
): GBImageData {
  if (image.width !== HOLE_W || image.height !== HOLE_H) {
    throw new Error(
      `composeFrame: expected ${HOLE_W}x${HOLE_H} image, got ${image.width}x${image.height}`,
    );
  }
  if (frame.pixels.length !== frame.width * frame.height) {
    throw new Error("composeFrame: frame.pixels length doesn't match dimensions");
  }

  // Build a temporary RGBA grayscale image at frame dimensions, then apply the
  // palette in one pass via the existing applyPalette logic.
  const W = frame.width;
  const H = frame.height;
  const gray = new Uint8ClampedArray(W * H * 4);

  // Fill with frame pixels.
  for (let i = 0; i < W * H; i++) {
    const v = frame.pixels[i];
    gray[i * 4 + 0] = v;
    gray[i * 4 + 1] = v;
    gray[i * 4 + 2] = v;
    gray[i * 4 + 3] = 255;
  }

  // Overwrite hole region with image grayscale values.
  for (let yy = 0; yy < HOLE_H; yy++) {
    for (let xx = 0; xx < HOLE_W; xx++) {
      const fi = ((frame.holeY + yy) * W + (frame.holeX + xx)) * 4;
      const ii = (yy * HOLE_W + xx) * 4;
      gray[fi + 0] = image.data[ii + 0];
      gray[fi + 1] = image.data[ii + 1];
      gray[fi + 2] = image.data[ii + 2];
      gray[fi + 3] = 255;
    }
  }

  return applyPalette({ data: gray, width: W, height: H }, palette);
}
