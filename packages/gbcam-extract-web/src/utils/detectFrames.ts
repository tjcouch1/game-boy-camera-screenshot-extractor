import type { Frame, GBImageData } from "gbcam-extract";
import { splitSheet, loadIndividualFrame } from "gbcam-extract";

/**
 * Sanitize a filename into a stem usable as a frame's `sheetStem`.
 *
 * - Strip the extension.
 * - Replace any character outside [A-Za-z0-9_-] with `-`.
 * - Collapse runs of `-`.
 * - Trim leading/trailing `-`.
 */
export function sanitizeFilenameStem(filename: string): string {
  const noExt = filename.replace(/\.[^.]+$/, "");
  return noExt
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Disambiguate `stem` against an existing set of stems by appending `-N`
 * where N is the smallest integer ≥ 2 that produces an unused stem. Returns
 * the original stem if it's already free.
 */
export function disambiguateStem(stem: string, taken: Set<string>): string {
  if (!taken.has(stem)) return stem;
  for (let n = 2; n < 1_000_000; n++) {
    const candidate = `${stem}-${n}`;
    if (!taken.has(candidate)) return candidate;
  }
  // Fallback: timestamp suffix. Should be unreachable in practice.
  return `${stem}-${Date.now()}`;
}

/**
 * Decode a `File` (PNG/JPEG/WebP/GIF) to RGBA `GBImageData` via canvas.
 * Throws with a human-readable message on decode failure.
 */
export async function fileToGBImageData(file: File): Promise<GBImageData> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () =>
        reject(new Error(`Couldn't decode image "${file.name}"`));
      el.src = objectUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2d context unavailable");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    return { data: imageData.data, width: img.width, height: img.height };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/**
 * Detect whether `image` is a sheet (multi-frame) or an individual frame and
 * return the resulting `Frame[]`. Always forces `kind: "individual"` on the
 * returned frames — see `UserFrameEntry` doc for the rationale.
 *
 * Dispatch rule:
 *   - splitSheet ≥ 2 frames → real sheet, trust it.
 *   - splitSheet ≤ 1 frame → ambiguous. Prefer loadIndividualFrame, because
 *     splitSheet's tight-bbox recomputation scans outward from the hole along
 *     a single row/column and clips individual frames whose bezel narrows
 *     beyond that scan (e.g. Game Boy Pocket frames with a label area below
 *     the screen). loadIndividualFrame uses the full image dimensions and
 *     preserves the entire frame body.
 *   - If both paths fail, fall back to splitSheet's single result if any, then
 *     finally throw.
 *
 * Throws an `Error` with a human-readable message if neither path produces a
 * frame so the caller can surface it as a toast.
 */
export function detectAndLoadFrames(
  image: GBImageData,
  stem: string,
): Frame[] {
  let sheetFrames: Frame[] = [];
  try {
    sheetFrames = splitSheet(image, stem);
  } catch {
    sheetFrames = [];
  }
  if (sheetFrames.length >= 2) {
    return sheetFrames.map((f) => ({ ...f, kind: "individual" as const }));
  }

  try {
    const f = loadIndividualFrame(image, stem);
    return [{ ...f, kind: "individual" as const }];
  } catch {
    // Defensive fallback: a single-frame sheet whose body has a non-white,
    // non-transparent background can defeat loadIndividualFrame's hole search
    // even though splitSheet was happy. Use what we have rather than throw.
    if (sheetFrames.length === 1) {
      return sheetFrames.map((f) => ({ ...f, kind: "individual" as const }));
    }
    throw new Error("Couldn't detect a frame in this image.");
  }
}
