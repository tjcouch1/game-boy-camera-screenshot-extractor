import type { PipelineResult, Frame } from "gbcam-extract";
import { applyPalette, composeFrame } from "gbcam-extract";

export function buildOutputCanvas(
  result: PipelineResult,
  palette: [string, string, string, string],
  effectiveFrame: Frame | null,
  scale: number,
): HTMLCanvasElement | null {
  try {
    if (!result.grayscale?.data) return null;
    let rendered;
    if (effectiveFrame) {
      try {
        rendered = composeFrame(result.grayscale, effectiveFrame, palette);
      } catch {
        rendered = applyPalette(result.grayscale, palette);
      }
    } else {
      rendered = applyPalette(result.grayscale, palette);
    }
    if (!rendered?.data?.length) return null;

    const canvas = document.createElement("canvas");
    canvas.width = rendered.width * scale;
    canvas.height = rendered.height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;

    const tmp = document.createElement("canvas");
    tmp.width = rendered.width;
    tmp.height = rendered.height;
    tmp
      .getContext("2d")!
      .putImageData(
        new ImageData(
          new Uint8ClampedArray(rendered.data),
          rendered.width,
          rendered.height,
        ),
        0,
        0,
      );
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    return canvas;
  } catch {
    return null;
  }
}
