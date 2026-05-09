import type { Frame } from "gbcam-extract";

/**
 * Persisted form of a user-uploaded frame.
 *
 * The grayscale pixel values (0/82/165/255) are stored as an RGB-replicated
 * PNG data URL: each output pixel has `R = G = B = pixelValue` and `A = 255`.
 * PNG is lossless, so the four GB grayscale values survive the round-trip.
 *
 * `Frame.kind` is fixed to `"individual"` for any decoded entry — even when a
 * sheet was uploaded, each split frame is stored as its own individual-style
 * entry. Once it's in localStorage there's no value in tracking shared origin.
 */
export interface UserFrameEntry {
  id: string; // "user-frame-<timestamp>-<rand>"
  /** Sanitized filename stem; used as Frame.sheetStem and to derive the display name. */
  sheetStem: string;
  type: "normal" | "wild";
  width: number;
  height: number;
  holeX: number;
  holeY: number;
  /** PNG data URL of a single-channel grayscale image (R=G=B=pixel value, A=255). */
  pngDataUrl: string;
  /** Date.now() when the entry was added. Used for stable display ordering. */
  addedAt: number;
}

/**
 * Encode a Frame's single-channel grayscale pixel data to a PNG data URL.
 *
 * Mirrors the approach in `serialization.ts`: build an RGBA canvas, replicate
 * the grayscale value across R/G/B (alpha = 255), then `toDataURL("image/png")`.
 */
export function frameToPngDataUrl(frame: Frame): string {
  const { width, height, pixels } = frame;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2d context unavailable");
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = pixels[i];
    imageData.data[i * 4 + 0] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

/** Decode a UserFrameEntry's PNG data URL back into a Frame. */
export function pngDataUrlToFrame(entry: UserFrameEntry): Promise<Frame> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = entry.width;
        canvas.height = entry.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas 2d context unavailable"));
          return;
        }
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, entry.width, entry.height);
        const pixels = new Uint8ClampedArray(entry.width * entry.height);
        // The PNG was encoded with R = G = B = pixel value, so the R channel
        // round-trips the original grayscale value losslessly.
        for (let i = 0; i < pixels.length; i++) {
          pixels[i] = imageData.data[i * 4];
        }
        const frame: Frame = {
          id: entry.id,
          sheetStem: entry.sheetStem,
          aliasStems: [entry.sheetStem],
          type: entry.type,
          kind: "individual",
          index: 1,
          width: entry.width,
          height: entry.height,
          pixels,
          holeX: entry.holeX,
          holeY: entry.holeY,
        };
        resolve(frame);
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    };
    img.onerror = () =>
      reject(new Error(`Failed to decode user frame "${entry.sheetStem}"`));
    img.src = entry.pngDataUrl;
  });
}
