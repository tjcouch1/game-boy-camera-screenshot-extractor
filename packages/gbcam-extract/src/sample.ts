import { type GBImageData, CAM_W, CAM_H, createGBImageData } from "./common.js";
import { type DebugCollector, upscale } from "./debug.js";

export interface SampleOptions {
  scale?: number;
  method?: "mean" | "median"; // kept for API compat; internally always uses mean (matching Python)
  marginH?: number; // ignored; replaced by subpixel col offsets
  marginV?: number;
  debug?: DebugCollector;
}

/**
 * Sample step: reduce each (scale x scale) block to a single colour value.
 *
 * The GBA SP TN LCD has BGR sub-pixels (Blue left, Green middle, Red right).
 * Sampling each channel from its own column range avoids cross-channel
 * contamination and gives values that represent each sub-pixel's actual
 * colour intensity.
 *
 * Layout at scale=8 (inner_start=1, inner_end=7, inner_w=6):
 *   B: cols [1, 3)  — blue sub-pixel columns
 *   G: cols [3, 5)  — green sub-pixel columns
 *   R: cols [5, 7)  — red sub-pixel columns
 *
 * Output: 128×112 colour RGBA PNG (R/G/B channels carry real colour data).
 * The quantize step clusters in RG colour space and requires this.
 */
export function sample(
  input: GBImageData,
  options?: SampleOptions,
): GBImageData {
  const scale = options?.scale ?? 8;
  const vMargin = options?.marginV ?? Math.max(1, Math.floor(scale / 5));
  const dbg = options?.debug;

  const expectedW = CAM_W * scale;
  const expectedH = CAM_H * scale;
  if (input.width !== expectedW || input.height !== expectedH) {
    throw new Error(
      `Unexpected input size ${input.width}x${input.height}; ` +
        `expected ${expectedW}x${expectedH} (scale=${scale})`,
    );
  }

  // Subpixel column offsets
  const innerStart = 1;
  const innerEnd = scale - 1;
  const innerW = innerEnd - innerStart;

  const output = createGBImageData(CAM_W, CAM_H);

  for (let by = 0; by < CAM_H; by++) {
    let y1 = by * scale + vMargin;
    let y2 = (by + 1) * scale - vMargin;
    // Fallback if vMargin is too large
    if (y2 <= y1) {
      y1 = by * scale;
      y2 = (by + 1) * scale;
    }

    for (let bx = 0; bx < CAM_W; bx++) {
      const x0 = bx * scale;
      const pi = by * CAM_W + bx;
      const outIdx = pi * 4;

      if (innerW < 3) {
        // Scale too small for sub-pixel columns — fall back to center pixel R channel
        const cy = by * scale + Math.floor(scale / 2);
        const cx = bx * scale + Math.floor(scale / 2);
        const v = input.data[(cy * input.width + cx) * 4];
        output.data[outIdx] = v;
        output.data[outIdx + 1] = v;
        output.data[outIdx + 2] = v;
        output.data[outIdx + 3] = 255;
        continue;
      }

      const bLo = innerStart;
      const bHi = innerStart + Math.floor(innerW / 3);
      const gLo = innerStart + Math.floor(innerW / 3);
      const gHi = innerStart + 2 * Math.floor(innerW / 3);
      // R sub-pixel center at scale=8 is at col 6.67. Sample [5,7) was
      // centered at col 5.5 — 1.17 cols left of the actual peak. Now that
      // sub-pixel rectification reliably places the LCD R sub-pixel at the
      // same column across the image, we can shift R sampling 1 col right.
      const rLo = innerStart + 2 * Math.floor(innerW / 3) + (scale >= 8 ? 1 : 0);
      const rHi = Math.min(scale, innerEnd + 1);

      let rSum = 0,
        gSum = 0,
        bSum = 0;
      let rCount = 0,
        gCount = 0,
        bCount = 0;

      for (let y = y1; y < y2; y++) {
        const rowBase = y * input.width;
        for (let dx = rLo; dx < rHi; dx++) {
          rSum += input.data[(rowBase + x0 + dx) * 4];
          rCount++;
        }
        for (let dx = gLo; dx < gHi; dx++) {
          gSum += input.data[(rowBase + x0 + dx) * 4 + 1];
          gCount++;
        }
        for (let dx = bLo; dx < bHi; dx++) {
          bSum += input.data[(rowBase + x0 + dx) * 4 + 2];
          bCount++;
        }
      }

      output.data[outIdx] = Math.round(rCount > 0 ? rSum / rCount : 0);
      output.data[outIdx + 1] = Math.round(gCount > 0 ? gSum / gCount : 0);
      output.data[outIdx + 2] = Math.round(bCount > 0 ? bSum / bCount : 0);
      output.data[outIdx + 3] = 255;
    }
  }

  // ── Directed vertical R de-bleed ──
  //
  // The GBA-SP front-light bleeds brighter pixels into dimmer ones,
  // "especially vertically" (see input-image notes). In a dithered LG/DG
  // region a DG pixel (#9494FF, low R) sandwiched between brighter warm
  // rows has its sampled R lifted toward those neighbours, pushing it
  // across the DG→LG decision boundary — by far the dominant tier-1 error
  // (out=LG, truth=DG). This pass removes that bled-in light so the dithered
  // contrast is restored before classification.
  //
  // Three constraints keep it from trading errors elsewhere (each verified
  // to matter on the test corpora):
  //   1. Directed (local-minimum only): correct a channel only where the
  //      pixel is dimmer than BOTH vertical neighbours — a genuine bleed
  //      victim. Local maxima (e.g. a DG pixel between two BK rows) are left
  //      alone, so flat/dark regions are preserved. Without this an
  //      undirected unsharp catastrophically lifts DG-between-BK into LG.
  //   2. B-gate: only correct pixels that are *bluer* than their vertical
  //      neighbours. DG carries high B; LG/WH carry low B, so a bleed-lifted
  //      DG pixel stays a vertical B-maximum while a genuinely-dim warm
  //      feature (sharp DG/LG detail) does not. This is what lets bathhouse
  //      and park improve while prison's sharp features stay protected.
  //   3. Magnitude floor: ignore sub-`MARGIN` neighbour gaps so flat-region
  //      sampling noise is never amplified.
  //
  // Strengths are physical-bleed parameters calibrated on the reference
  // corpus. (These were `process.env`-overridable while tuning in Node, but
  // the pipeline also runs in the browser where `process` is undefined, so the
  // calibrated defaults are inlined.)
  const debleedR = 0.25;
  const debleedG = 0;
  const debleedMargin = 0;
  const debleedBGate = 10;
  if (debleedR > 0 || debleedG > 0) {
    const src = output.data.slice();
    for (let by = 1; by < CAM_H - 1; by++) {
      for (let bx = 0; bx < CAM_W; bx++) {
        const o = (by * CAM_W + bx) * 4;
        const up = ((by - 1) * CAM_W + bx) * 4;
        const dn = ((by + 1) * CAM_W + bx) * 4;
        // B-gate: only de-bleed pixels that are bluer than their vertical
        // neighbours. DG (#9494FF) carries high B; LG/WH carry low B. A
        // genuine bleed victim — a DG pixel whose R was lifted by brighter
        // (warm, low-B) neighbours above/below — stays a vertical B-maximum.
        // A truly-dim warm pixel (e.g. a sharp DG/LG feature in prison) is
        // NOT bluer than its neighbours, so it is left untouched. This is
        // what separates "restore dithered DG contrast" (bathhouse/park,
        // helped) from "corrupt a sharp warm feature" (prison, protected).
        const bv = src[o + 2];
        const bIsDgLike = debleedBGate <= 0 || bv > (src[up + 2] + src[dn + 2]) / 2 + debleedBGate;
        if (bIsDgLike) {
          for (const [ch, a] of [[0, debleedR], [1, debleedG]] as const) {
            if (a <= 0) continue;
            const v = src[o + ch];
            const u = src[up + ch];
            const d = src[dn + ch];
            // Directed de-bleed: only correct vertical local minima — a pixel
            // dimmer than BOTH neighbours is a genuine bleed victim receiving
            // light from the brighter rows above and below. Subtract that
            // bled-in light, pushing it back toward its true (darker) value.
            if (v < u && v < d) {
              const neigh = (u + d) / 2;
              const gap = neigh - v;
              if (gap >= debleedMargin) {
                output.data[o + ch] = Math.max(0, Math.min(255, Math.round(v - a * gap)));
              }
            }
          }
        }
      }
    }
  }

  if (dbg) {
    // Compute per-channel min/max
    let rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
    for (let i = 0; i < CAM_W * CAM_H; i++) {
      const o = i * 4;
      const r = output.data[o];
      const g = output.data[o + 1];
      const b = output.data[o + 2];
      if (r < rMin) rMin = r; if (r > rMax) rMax = r;
      if (g < gMin) gMin = g; if (g > gMax) gMax = g;
      if (b < bMin) bMin = b; if (b > bMax) bMax = b;
    }
    dbg.log(
      `[sample] R: ${rMin}–${rMax}  G: ${gMin}–${gMax}  B: ${bMin}–${bMax}`,
    );
    const innerStartLog = 1;
    const innerEndLog = scale - 1;
    const innerWLog = innerEndLog - innerStartLog;
    const bLoLog = innerStartLog;
    const bHiLog = innerStartLog + Math.floor(innerWLog / 3);
    const gLoLog = bHiLog;
    const gHiLog = innerStartLog + 2 * Math.floor(innerWLog / 3);
    const rLoLog = gHiLog + (scale >= 8 ? 1 : 0);
    const rHiLog = scale;
    dbg.log(
      `[sample] subpixel cols (scale=${scale}): ` +
        `B=[${bLoLog},${bHiLog}) G=[${gLoLog},${gHiLog}) R=[${rLoLog},${rHiLog}) vMargin=${vMargin}`,
    );
    dbg.setMetrics("sample", {
      ranges: {
        R: [rMin, rMax],
        G: [gMin, gMax],
        B: [bMin, bMax],
      },
      subpixelCols: {
        B: [bLoLog, bHiLog],
        G: [gLoLog, gHiLog],
        R: [rLoLog, rHiLog],
      },
      vMargin,
    });
    // 8x upscale for visual inspection
    dbg.addImage("sample_a_8x", upscale(output, 8));
  }

  return output;
}
