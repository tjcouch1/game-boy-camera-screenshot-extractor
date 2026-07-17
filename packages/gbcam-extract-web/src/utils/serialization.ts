import type { PipelineResult, GBImageData, PipelineIssue } from "gbcam-extract";

/**
 * Serialized form of GBImageData using PNG encoding for maximum efficiency.
 * PNG compression typically achieves 60-70% size reduction vs base64 raw encoding.
 * For a 128x112 image: ~57KB raw → ~4-5KB PNG → ~6-7KB base64-encoded PNG data URL.
 */
export interface SerializedGBImageData {
  _type: "GBImageData";
  width: number;
  height: number;
  pngData: string; // PNG image as base64 data URL (image/png;base64,...)
}

/**
 * Serialized form of PipelineResult.
 *
 * NOTE: Only the final grayscale image is persisted. Intermediate step images
 * and the `debug` payload (images, log, metrics) live in memory only — they
 * are too large for localStorage when debug mode is on (color RGBA images at
 * 1280×1152 + 8x upscales easily blow past the 5–10 MB quota). They are still
 * available on the in-memory `PipelineResult` until the page is refreshed.
 */
export interface SerializedPipelineResult {
  _type: "PipelineResult";
  grayscale: SerializedGBImageData;
  /** Processing-quality issues detected during the run (absent on items
   *  serialized by older versions). */
  issues?: PipelineIssue[];
  /** True when the input was detected as an already-processed image and
   *  passed through the pipeline untouched. */
  alreadyProcessed?: boolean;
}

/**
 * Convert Uint8ClampedArray to a canvas-based PNG data URL.
 * This performs lossless PNG compression on the grayscale image.
 */
function grayscaleToCanvasPNG(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): string {
  // Create canvas at 1x scale
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.createImageData(width, height);

  // Copy data (already in RGBA format from Uint8ClampedArray)
  imageData.data.set(data);
  ctx.putImageData(imageData, 0, 0);

  // Convert to PNG data URL
  return canvas.toDataURL("image/png");
}

/**
 * Convert PNG data URL back to Uint8ClampedArray.
 * Returns a promise that resolves when the image is loaded.
 */
function pngDataUrlToGrayscale(
  pngDataUrl: string,
  width: number,
  height: number,
): Promise<Uint8ClampedArray> {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, width, height);
        resolve(new Uint8ClampedArray(imageData.data));
      } catch (err) {
        reject(new Error(`Failed to decode PNG: ${err}`));
      }
    };

    img.onerror = () => {
      reject(new Error("Failed to load PNG image data"));
    };

    // Handle data URL
    img.src = pngDataUrl;
  });
}

/**
 * Serialize a GBImageData to PNG format.
 * @returns Serialized object with PNG data URL
 */
export function serializeGBImageData(
  img: GBImageData,
): SerializedGBImageData {
  const pngData = grayscaleToCanvasPNG(img.data, img.width, img.height);

  return {
    _type: "GBImageData",
    width: img.width,
    height: img.height,
    pngData,
  };
}

/**
 * Deserialize a GBImageData from PNG format.
 * @returns Promise that resolves to reconstructed GBImageData with proper Uint8ClampedArray
 */
export async function deserializeGBImageData(
  serialized: SerializedGBImageData,
): Promise<GBImageData> {
  const data = await pngDataUrlToGrayscale(
    serialized.pngData,
    serialized.width,
    serialized.height,
  );

  return {
    width: serialized.width,
    height: serialized.height,
    data,
  };
}

/**
 * Serialize a PipelineResult to PNG format. Only the final grayscale image
 * is included — intermediates and `debug` are dropped (see type doc comment).
 */
export function serializePipelineResult(
  result: PipelineResult,
): SerializedPipelineResult {
  return {
    _type: "PipelineResult",
    grayscale: serializeGBImageData(result.grayscale),
    ...(result.issues?.length ? { issues: result.issues } : {}),
    ...(result.alreadyProcessed ? { alreadyProcessed: true } : {}),
  };
}

/**
 * Deserialize a PipelineResult from PNG format.
 * Note: This is async because PNG decoding requires image loading.
 */
export async function deserializePipelineResult(
  serialized: SerializedPipelineResult,
): Promise<PipelineResult> {
  const grayscale = await deserializeGBImageData(serialized.grayscale);
  return {
    grayscale,
    ...(serialized.issues ? { issues: serialized.issues } : {}),
    ...(serialized.alreadyProcessed ? { alreadyProcessed: true } : {}),
  };
}

/**
 * Type guard to check if an object is a SerializedPipelineResult.
 */
export function isSerializedPipelineResult(
  obj: any,
): obj is SerializedPipelineResult {
  return (
    obj &&
    typeof obj === "object" &&
    obj._type === "PipelineResult" &&
    obj.grayscale &&
    obj.grayscale._type === "GBImageData"
  );
}

/**
 * Type guard to check if an object is a SerializedGBImageData.
 */
export function isSerializedGBImageData(
  obj: any,
): obj is SerializedGBImageData {
  return (
    obj &&
    typeof obj === "object" &&
    obj._type === "GBImageData" &&
    typeof obj.width === "number" &&
    typeof obj.height === "number" &&
    typeof obj.pngData === "string" &&
    obj.pngData.startsWith("data:image/png")
  );
}
