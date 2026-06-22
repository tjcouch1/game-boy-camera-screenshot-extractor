/**
 * inspect-warp.ts — per-position diagnostic for a final `_warp.png`.
 *
 * Measures, using the SAME internals `warp.ts` uses, the inner-border ring
 * deviation along all four edges (so edge *curvature*, not just the mean,
 * is visible) and the WH-frame sub-pixel "vertical frame line" phase down
 * the left/right strips and across the top/bottom strips.
 *
 * Usage:
 *   pnpm inspect-warp -- <path-to-_warp.png> [--scale 8]
 *
 * Output is plain text: per-edge deviation curves (sampled), the
 * min/mean/max of each, and the left/right stripe-phase drift with height.
 */
import sharp from "sharp";
import { initOpenCV } from "../src/init-opencv.js";
import { imageDataToMat } from "../src/opencv.js";
import { buildRBChannel, findBorderPoints, findGPeakOffset } from "../src/warp.js";
import {
  INNER_TOP,
  INNER_BOT,
  INNER_LEFT,
  INNER_RIGHT,
  type GBImageData,
} from "../src/common.js";

async function loadImage(filePath: string): Promise<GBImageData> {
  const { data, info } = await sharp(filePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const rgba = new Uint8ClampedArray(info.width * info.height * 4);
  rgba.set(data);
  return { data: rgba, width: info.width, height: info.height };
}

function stats(vals: number[]): string {
  if (vals.length === 0) return "(none)";
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return `mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} range=${(max - min).toFixed(2)}`;
}

/** ASCII sparkline of a signed series, scaled to ±maxAbs. */
function spark(vals: number[]): string {
  const chars = "▇▆▅▄▃▂▁ ▁▂▃▄▅▆▇";
  const maxAbs = Math.max(1e-6, ...vals.map((v) => Math.abs(v)));
  return vals
    .map((v) => {
      const t = Math.max(-1, Math.min(1, v / maxAbs));
      const idx = Math.round((t + 1) / 2 * (chars.length - 1));
      return chars[idx];
    })
    .join("");
}

async function main() {
  const argv = process.argv.slice(2);
  const path = argv.find((a) => !a.startsWith("--"));
  const scaleArg = argv.indexOf("--scale");
  const scale = scaleArg >= 0 ? parseInt(argv[scaleArg + 1], 10) : 8;
  if (!path) {
    console.error("Usage: pnpm inspect-warp -- <path-to-_warp.png> [--scale 8]");
    process.exit(1);
  }

  await initOpenCV();
  const { getCV } = await import("../src/opencv.js");
  const cv = getCV();

  const img = await loadImage(path);
  const rgba = imageDataToMat(img);
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();

  const rb = buildRBChannel(bgr);
  const bp = findBorderPoints(rb, scale);
  rb.delete();

  const expTop = INNER_TOP * scale;
  const expBot = INNER_BOT * scale;
  const expLeft = INNER_LEFT * scale;
  const expRight = INNER_RIGHT * scale;

  console.log(`\n=== inspect-warp: ${path} (scale=${scale}, ${img.width}x${img.height}) ===`);
  console.log(`ideal inner border: left=${expLeft} right=${expRight} top=${expTop} bot=${expBot}\n`);

  // Edge deviations (signed: detected - ideal). For left/right the deviation
  // is in x; positive = detected border is to the RIGHT of ideal (inward on
  // left, outward on right). For top/bottom it is in y; positive = downward.
  const leftDev = bp.left.map(([x]) => x - expLeft);
  const rightDev = bp.right.map(([x]) => x - expRight);
  const topDev = bp.top.map(([, y]) => y - expTop);
  const botDev = bp.bottom.map(([, y]) => y - expBot);

  const fmtRows = (pts: [number, number][], dev: number[], axis: "row" | "col") =>
    pts
      .map(([x, y], i) => {
        const pos = axis === "row" ? Math.round(y) : Math.round(x);
        return `${pos.toString().padStart(4)}:${dev[i] >= 0 ? "+" : ""}${dev[i].toFixed(2)}`;
      })
      .join("  ");

  console.log(`LEFT  (x-dev vs ${expLeft}, +=inward):  ${stats(leftDev)}`);
  console.log(`  ${spark(leftDev)}   (top→bottom)`);
  console.log(`  ${fmtRows(bp.left, leftDev, "row")}\n`);

  console.log(`RIGHT (x-dev vs ${expRight}, +=outward): ${stats(rightDev)}`);
  console.log(`  ${spark(rightDev)}   (top→bottom)`);
  console.log(`  ${fmtRows(bp.right, rightDev, "row")}\n`);

  console.log(`TOP   (y-dev vs ${expTop}, +=down):     ${stats(topDev)}`);
  console.log(`  ${spark(topDev)}   (left→right)`);
  console.log(`  ${fmtRows(bp.top, topDev, "col")}\n`);

  console.log(`BOTTOM(y-dev vs ${expBot}, +=down):    ${stats(botDev)}`);
  console.log(`  ${spark(botDev)}   (left→right)`);
  console.log(`  ${fmtRows(bp.bottom, botDev, "col")}\n`);

  console.log(`gapWidth (dark WH→DG trough): ${JSON.stringify(bp.gapWidth)}\n`);

  // ── Vertical-frame-line (sub-pixel G-peak) phase down the LEFT and RIGHT
  // WH frame strips. Reveals stripe curvature with HEIGHT, which the current
  // subPixelRectify (top/bottom strips only) cannot see directly.
  const leftBlock = 4 * scale;             // GB px x=4, safely inside left WH frame
  const rightBlock = (INNER_RIGHT + 8) * scale; // GB px x=152, inside right WH frame
  const topStripBlocks = [Math.floor((INNER_LEFT + 4) * scale)]; // for cross-check
  void topStripBlocks;
  const nBands = 24;
  const bandH = 3 * scale;
  const leftPhase: number[] = [];
  const rightPhase: number[] = [];
  const heights: number[] = [];
  for (let i = 0; i < nBands; i++) {
    const yc = expTop + ((expBot - expTop) * i) / (nBands - 1);
    const r1 = Math.max(0, Math.round(yc - bandH / 2));
    const r2 = Math.min(img.height, r1 + bandH);
    heights.push(Math.round(yc));
    leftPhase.push(findGPeakOffset(bgr, leftBlock, r1, r2, scale));
    rightPhase.push(findGPeakOffset(bgr, rightBlock, r1, r2, scale));
  }
  const lp = leftPhase.filter(Number.isFinite);
  const rp = rightPhase.filter(Number.isFinite);
  console.log(`LEFT  WH-frame stripe phase (G-peak col within block, 0..${scale}) vs height:`);
  console.log(`  ${stats(lp)}`);
  console.log(`  ${heights.map((h, i) => `${h}:${Number.isFinite(leftPhase[i]) ? leftPhase[i].toFixed(2) : "—"}`).join(" ")}`);
  console.log(`RIGHT WH-frame stripe phase vs height:`);
  console.log(`  ${stats(rp)}`);
  console.log(`  ${heights.map((h, i) => `${h}:${Number.isFinite(rightPhase[i]) ? rightPhase[i].toFixed(2) : "—"}`).join(" ")}`);

  bgr.delete();
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
