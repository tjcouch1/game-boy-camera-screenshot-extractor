import { describe, it } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";
import sharp from "sharp";
import { detectAndLoadFrames, fileToGBImageData } from "../../packages/gbcam-extract-web/src/utils/detectFrames";

describe("debug broken frame", () => {
  it("should output what it detects", async () => {
    const filePath = resolve(__dirname, "../../../supporting-materials/frames/broken-frame.png");
    const { data, info } = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    
    const gbImage = {
      data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
      width: info.width,
      height: info.height,
    };
    
    try {
      const frames = detectAndLoadFrames(gbImage, "broken-frame");
      console.log("DETECTED FRAMES:", frames.length);
      frames.forEach((f, i) => {
        console.log(`Frame ${i}: ${f.width}x${f.height}, hole at ${f.holeX},${f.holeY}`);
      });
    } catch (e) {
      console.error("ERROR DETECTING:", e);
    }
  });
});
