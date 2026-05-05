/**
 * A single Game Boy Camera frame extracted from a sheet PNG.
 *
 * Pixel values are pre-snapped to the four GB grayscale values
 * {0, 82, 165, 255}. Hole pixels (the 128 × 112 region where the camera
 * image goes) are stored as 255 so frames render with the lightest
 * palette colour when shown alone in the picker.
 */
export interface Frame {
  /** Stable identifier of the form "<sheetStem>:<type>:<index>". */
  id: string;
  /** Stem of the source sheet (e.g. "Frames_USA"). */
  sheetStem: string;
  /** "normal" if dimensions are exactly 160 × 144, else "wild". */
  type: "normal" | "wild";
  /** 1-based index, scoped to (sheetStem, type), in (y, x) reading order. */
  index: number;
  width: number;
  height: number;
  /** length = width × height. Each value is in {0, 82, 165, 255}. */
  pixels: Uint8ClampedArray;
  /** Top-left of the 128 × 112 hole, in frame-local coords. */
  holeX: number;
  holeY: number;
}
