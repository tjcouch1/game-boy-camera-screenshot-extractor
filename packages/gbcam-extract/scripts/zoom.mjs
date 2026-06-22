// Quick crop+upscale helper for eyeballing warp corners.
// Usage: node scripts/zoom.mjs <in.png> <out.png> <x> <y> <w> <h> [zoom=4]
import sharp from "sharp";
const [, , inp, outp, x, y, w, h, zoom = "4"] = process.argv;
const Z = parseInt(zoom, 10);
const left = +x, top = +y, width = +w, height = +h;
await sharp(inp)
  .extract({ left, top, width, height })
  .resize({ width: width * Z, height: height * Z, kernel: "nearest" })
  .png()
  .toFile(outp);
console.log(`wrote ${outp} (${width}x${height} @${Z}x)`);
