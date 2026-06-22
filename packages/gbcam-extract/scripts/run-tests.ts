/**
 * run-tests.ts — Pipeline regression test runner mirroring run_tests.py.
 *
 * 1. Runs all sample pictures through the full pipeline.
 * 2. Runs test cases against reference images and reports accuracy.
 * 3. Writes a summary log to test-output/test-summary.log.
 *
 * Usage:
 *   pnpm test:pipeline
 */

import { resolve, join, basename, extname, relative } from "path";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "fs";
import sharp from "sharp";
import { initOpenCV } from "../src/init-opencv.js";
import { processPicture } from "../src/index.js";
import { applyPalette } from "../src/palette.js";
import type { GBImageData } from "../src/common.js";
import { GB_COLORS, CAM_W, CAM_H } from "../src/common.js";

// "Down" palette (matches the GBA SP screen colors used as input).
const DOWN_PALETTE: [string, string, string, string] = [
  "#FFFFA5",
  "#FF9494",
  "#9494FF",
  "#000000",
];

// ─── Paths ───

const SCRIPT_DIR = resolve(import.meta.dirname ?? ".");
const PKG_DIR = resolve(SCRIPT_DIR, "..");
const REPO_ROOT = resolve(PKG_DIR, "..", "..");

const SAMPLE_PICTURES_DIR = join(REPO_ROOT, "sample-pictures");
const SAMPLE_PICTURES_OUT = join(REPO_ROOT, "sample-pictures-out");
const TEST_INPUT_DIR = join(REPO_ROOT, "test-input");
const TEST_OUTPUT_DIR = join(REPO_ROOT, "test-output");
const TEST_INPUT_FULL_DIR = join(REPO_ROOT, "test-input-full");
const TEST_OUTPUT_FULL_DIR = join(REPO_ROOT, "test-output-full");
const SAMPLE_PICTURES_FULL_DIR = join(REPO_ROOT, "sample-pictures-full");
const SAMPLE_PICTURES_OUT_FULL = join(REPO_ROOT, "sample-pictures-out-full");
const SAMPLE_PICTURES_PRIVATE_DIR = join(REPO_ROOT, "sample-pictures-private");
const SAMPLE_PICTURES_OUT_PRIVATE = join(
  REPO_ROOT,
  "sample-pictures-out-private",
);
const TEST_INPUT_PRIVATE_DIR = join(REPO_ROOT, "test-input-private");
const TEST_OUTPUT_PRIVATE_DIR = join(REPO_ROOT, "test-output-private");
const REFERENCE_SUFFIX = "-output-corrected.png";

const DEFAULT_SCALE = 8;

// ─── Color constants ───

const COLOR_NAMES: Record<number, string> = {
  0: "BK  #000000",
  82: "DG  #9494FF",
  165: "LG  #FF9494",
  255: "WH  #FFFFA5",
};

// RGBA palette for diagnostic images
const RGBA_PALETTE: Record<number, [number, number, number, number]> = {
  0: [0, 0, 0, 255],
  82: [148, 148, 255, 255],
  165: [255, 148, 148, 255],
  255: [255, 255, 165, 255],
};

// ─── Image helpers ───

async function loadImage(filePath: string): Promise<GBImageData> {
  // Auto-orient: applies any EXIF rotation so loaded pixels match visual
  // orientation. Phone photos in test-input-full/ may store landscape
  // images with EXIF rotation; locate's detection runs in pixel-storage
  // coords, so without this it sees a flipped image.
  const img = sharp(filePath).rotate().removeAlpha().ensureAlpha();
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });

  const rgba = new Uint8ClampedArray(info.width * info.height * 4);
  for (let i = 0; i < info.width * info.height; i++) {
    rgba[i * 4] = data[i * 4];
    rgba[i * 4 + 1] = data[i * 4 + 1];
    rgba[i * 4 + 2] = data[i * 4 + 2];
    rgba[i * 4 + 3] = data[i * 4 + 3];
  }

  return { data: rgba, width: info.width, height: info.height };
}

async function saveImage(img: GBImageData, outPath: string): Promise<void> {
  const dir = resolve(outPath, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  await sharp(Buffer.from(img.data.buffer), {
    raw: { width: img.width, height: img.height, channels: 4 },
  })
    .png()
    .toFile(outPath);
}

/** Load grayscale 128x112 image, snap stray values to palette. */
async function loadReference(filePath: string): Promise<Uint8Array> {
  const { data, info } = await sharp(filePath)
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  if (info.width !== CAM_W || info.height !== CAM_H) {
    throw new Error(
      `Reference image is ${info.width}x${info.height}, expected ${CAM_W}x${CAM_H}`,
    );
  }

  // Snap to nearest palette value
  const snapped = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    let minDist = 256;
    let closest = 0;
    for (const c of GB_COLORS) {
      const d = Math.abs(data[i] - c);
      if (d < minDist) {
        minDist = d;
        closest = c;
      }
    }
    snapped[i] = closest;
  }
  return snapped;
}

/** Extract grayscale values from pipeline output (RGBA -> grayscale using R channel). */
function extractGrayscale(img: GBImageData): Uint8Array {
  const gray = new Uint8Array(img.width * img.height);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = img.data[i * 4];
  }
  // Snap to palette
  for (let i = 0; i < gray.length; i++) {
    let minDist = 256;
    let closest = 0;
    for (const c of GB_COLORS) {
      const d = Math.abs(gray[i] - c);
      if (d < minDist) {
        minDist = d;
        closest = c;
      }
    }
    gray[i] = closest;
  }
  return gray;
}

// ─── Comparison ───

interface ComparisonResult {
  total: number;
  matches: number;
  wrongs: number;
  matchPct: number;
  wrongPct: number;
  passed: boolean;
}

function compare(
  result: Uint8Array,
  reference: Uint8Array,
  outputDir: string,
  stem: string,
  log: (msg: string) => void,
): ComparisonResult {
  const total = result.length;
  let matches = 0;
  for (let i = 0; i < total; i++) {
    if (result[i] === reference[i]) matches++;
  }
  const wrongs = total - matches;
  const matchPct = (100 * matches) / total;
  const wrongPct = (100 * wrongs) / total;

  log(`\n${"=".repeat(70)}`);
  log("COMPARISON SUMMARY");
  log("=".repeat(70));
  log(`  Total pixels : ${total}`);
  log(`  Matching     : ${matches}  (${matchPct.toFixed(2)}%)`);
  log(`  Different    : ${wrongs}   (${wrongPct.toFixed(2)}%)`);

  // Per-color distribution
  log(`\n${"-".repeat(70)}`);
  log("COLOR DISTRIBUTION");
  log("-".repeat(70));
  log(
    `  ${"Color".padEnd(22)}  ${"Result".padStart(8)}  ${"Reference".padStart(10)}  ${"Diff".padStart(8)}`,
  );
  for (const v of GB_COLORS) {
    let rCnt = 0,
      gCnt = 0;
    for (let i = 0; i < total; i++) {
      if (result[i] === v) rCnt++;
      if (reference[i] === v) gCnt++;
    }
    const diff = rCnt - gCnt;
    log(
      `  ${COLOR_NAMES[v].padEnd(22)}  ${String(rCnt).padStart(8)}  ${String(gCnt).padStart(10)}  ${(diff >= 0 ? "+" : "") + diff}`,
    );
  }

  // Confusion matrix
  log(`\n${"-".repeat(70)}`);
  log("CONFUSION MATRIX  (rows = pipeline result, cols = reference)");
  log("-".repeat(70));
  let hdr = `  ${"Result / Ref".padEnd(18)}`;
  for (const v of GB_COLORS) {
    hdr += `  ${COLOR_NAMES[v].padStart(16)}`;
  }
  hdr += "   TOTAL";
  log(hdr);

  for (const rv of GB_COLORS) {
    let row = `  ${COLOR_NAMES[rv].padEnd(18)}`;
    let totalR = 0;
    for (const cv of GB_COLORS) {
      let cnt = 0;
      for (let i = 0; i < total; i++) {
        if (result[i] === rv && reference[i] === cv) cnt++;
      }
      if (rv === cv) totalR += cnt;
      else totalR += cnt;
      const mark = rv === cv ? " v" : cnt === 0 ? "  " : " X";
      row += `  ${String(cnt).padStart(15)}${mark}`;
    }
    // Recount totalR properly
    let trueTotal = 0;
    for (let i = 0; i < total; i++) {
      if (result[i] === rv) trueTotal++;
    }
    row += `  ${String(trueTotal).padStart(6)}`;
    log(row);
  }

  // Error breakdown
  if (wrongs > 0) {
    log(`\n${"-".repeat(70)}`);
    log("ERROR BREAKDOWN  (result -> reference)");
    log("-".repeat(70));
    for (const rv of GB_COLORS) {
      for (const cv of GB_COLORS) {
        if (rv === cv) continue;
        let cnt = 0;
        for (let i = 0; i < total; i++) {
          if (result[i] === rv && reference[i] === cv) cnt++;
        }
        if (cnt > 0) {
          log(`  ${COLOR_NAMES[rv]}  ->  ${COLOR_NAMES[cv]} : ${cnt} px`);
        }
      }
    }
  }

  // Reprint summary
  log(`\n${"=".repeat(70)}`);
  log("COMPARISON SUMMARY (reprint)");
  log("=".repeat(70));
  log(`  Total pixels : ${total}`);
  log(`  Matching     : ${matches}  (${matchPct.toFixed(2)}%)`);
  log(`  Different    : ${wrongs}   (${wrongPct.toFixed(2)}%)`);

  const passed = wrongs === 0;
  log(`\n${"=".repeat(70)}`);
  if (passed) {
    log("RESULT: PASS — all pixels match the reference.");
  } else {
    log("RESULT: FAIL — see diagnostics above and in the output directory.");
  }
  log("=".repeat(70));

  return { total, matches, wrongs, matchPct, wrongPct, passed };
}

/** Save diagnostic error map image. */
async function saveErrorMap(
  result: Uint8Array,
  reference: Uint8Array,
  outputDir: string,
  stem: string,
  scale: number = 1,
  suffix: string = "",
): Promise<void> {
  const w = CAM_W;
  const h = CAM_H;
  const outW = w * scale;
  const outH = h * scale;
  const buf = Buffer.alloc(outW * outH * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const r = result[idx];
      const ref = reference[idx];
      let rgba: [number, number, number, number];

      if (r === ref) {
        rgba = [255, 255, 255, 255]; // white = correct
      } else {
        // Color-code by error type (red = generic error)
        rgba = [255, 0, 0, 255];
      }

      // Fill the scale x scale block
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) {
          const px = x * scale + dx;
          const py = y * scale + dy;
          const oi = (py * outW + px) * 4;
          buf[oi] = rgba[0];
          buf[oi + 1] = rgba[1];
          buf[oi + 2] = rgba[2];
          buf[oi + 3] = rgba[3];
        }
      }
    }
  }

  const outPath = join(outputDir, `${stem}_diag_error_map${suffix}.png`);
  await sharp(buf, { raw: { width: outW, height: outH, channels: 4 } })
    .png()
    .toFile(outPath);
}

/** Save palette-rendered diagnostic image. */
async function savePaletteImage(
  gray: Uint8Array,
  outputDir: string,
  filename: string,
): Promise<void> {
  const w = CAM_W;
  const h = CAM_H;
  const scale = 4;
  const outW = w * scale;
  const outH = h * scale;
  const buf = Buffer.alloc(outW * outH * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      const rgba = RGBA_PALETTE[v] ?? [128, 128, 128, 255];
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) {
          const px = x * scale + dx;
          const py = y * scale + dy;
          const oi = (py * outW + px) * 4;
          buf[oi] = rgba[0];
          buf[oi + 1] = rgba[1];
          buf[oi + 2] = rgba[2];
          buf[oi + 3] = rgba[3];
        }
      }
    }
  }

  const outPath = join(outputDir, filename);
  await sharp(buf, { raw: { width: outW, height: outH, channels: 4 } })
    .png()
    .toFile(outPath);
}

// ─── Corpus config ───

interface CorpusConfig {
  /** Human-readable name shown in summary logs. */
  name: string;
  /** Absolute path to the input directory. */
  inputDir: string;
  /** Absolute path to the output directory. */
  outputDir: string;
  /**
   * Comparison mode:
   *  - "reference":   compare against hand-corrected refs in test-input/
   *                   (uses `<baseName>-output-corrected.png`)
   *  - "self":        compare against `referenceFromOutputDir`'s outputs
   *  - "none":        no comparison (extraction only)
   */
  comparison: "reference" | "self" | "none";
  /** When comparison === "self", which output dir to read references from. */
  referenceFromOutputDir?: string;
}

/**
 * Collect input files for a corpus. Includes .jpg/.jpeg/.png; skips reference
 * images (those ending in `-output-corrected.png`).
 */
function collectCorpusInputs(inputDir: string): string[] {
  if (!existsSync(inputDir)) return [];
  const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png"]);
  return readdirSync(inputDir)
    .filter((f) => {
      if (f.endsWith(REFERENCE_SUFFIX)) return false;
      return IMAGE_EXTS.has(extname(f).toLowerCase());
    })
    .sort()
    .map((f) => join(inputDir, f));
}

/** Find the reference path for an input photo, or null if none. */
function findReferenceFor(inputStem: string, inputDir: string): string | null {
  // The reference uses the *base name* (e.g. "thing" or "zelda-poster"),
  // derived by stripping the trailing "-<number>" off the input stem.
  const m = inputStem.match(/^(.*)-\d+$/);
  if (!m) return null;
  const baseName = m[1];
  const refPath = join(inputDir, `${baseName}${REFERENCE_SUFFIX}`);
  if (!existsSync(refPath)) return null;
  return refPath;
}

/**
 * Run every input in a corpus through the pipeline. Returns the per-image
 * test results (used for the final summary).
 */
async function runCorpus(config: CorpusConfig): Promise<TestResult[]> {
  const inputs = collectCorpusInputs(config.inputDir);
  if (inputs.length === 0) {
    console.log(`[${config.name}] no inputs found in ${config.inputDir}`);
    return [];
  }

  if (!existsSync(config.outputDir))
    mkdirSync(config.outputDir, { recursive: true });

  console.log(`\n${"=".repeat(70)}`);
  console.log(`CORPUS: ${config.name}  (${inputs.length} file(s))`);
  console.log("=".repeat(70));

  const results: TestResult[] = [];
  for (const inputPath of inputs) {
    const inputFilename = basename(inputPath);
    const stem = basename(inputPath, extname(inputPath));

    let perImageOutDir: string;
    if (config.comparison === "reference" || config.comparison === "self") {
      perImageOutDir = join(config.outputDir, stem);
      if (!existsSync(perImageOutDir))
        mkdirSync(perImageOutDir, { recursive: true });
    } else {
      perImageOutDir = config.outputDir;
    }

    console.log(`\n  [${config.name}] ${inputFilename}`);

    const logPath = join(perImageOutDir, `${stem}.log`);
    const logLines: string[] = [];
    const log = (msg: string) => {
      console.log(msg);
      logLines.push(msg);
    };

    try {
      log(`PIPELINE RUN`);
      log(`  Input:      ${relative(REPO_ROOT, inputPath)}`);
      log(`  Output dir: ${relative(REPO_ROOT, perImageOutDir)}`);

      const input = await loadImage(inputPath);
      const result = await processPicture(input, {
        scale: DEFAULT_SCALE,
        debug: true,
        locate: true,
        onProgress: (step, pct) => {
          if (pct === 0) process.stdout.write(`    ${step}...`);
          if (pct === 100) process.stdout.write(" done\n");
        },
      });

      await saveImage(
        result.grayscale,
        join(perImageOutDir, `${stem}_gbcam.png`),
      );
      const rgb = applyPalette(result.grayscale, DOWN_PALETTE);
      await saveImage(rgb, join(perImageOutDir, `${stem}_gbcam_rgb.png`));
      await writeDebugArtifacts(result, perImageOutDir, stem);

      if (result.debug?.log.length) {
        log(`\nPIPELINE DIAGNOSTICS`);
        for (const line of result.debug.log) log(`  ${line}`);
      }

      // ── Comparison ──
      if (config.comparison === "none") {
        results.push({
          name: stem,
          matchN: null,
          matchPct: null,
          diffN: null,
          diffPct: null,
          verdict: "OK",
        });
        writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
        continue;
      }

      // Resolve reference path
      let refPath: string | null;
      if (config.comparison === "reference") {
        refPath = findReferenceFor(stem, config.inputDir);
      } else {
        // "self": reference is `<referenceFromOutputDir>/<stem>_gbcam.png`
        // for flat corpora (sample-pictures-out is flat), or
        // `<referenceFromOutputDir>/<stem>/<stem>_gbcam.png` for per-image-dir
        // corpora.
        const flat = join(config.referenceFromOutputDir!, `${stem}_gbcam.png`);
        const nested = join(
          config.referenceFromOutputDir!,
          stem,
          `${stem}_gbcam.png`,
        );
        refPath = existsSync(flat) ? flat : existsSync(nested) ? nested : null;
      }

      if (!refPath) {
        log(`\n  No reference image found — skipping comparison.`);
        results.push({
          name: stem,
          matchN: null,
          matchPct: null,
          diffN: null,
          diffPct: null,
          verdict: "NO REF",
        });
        writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
        continue;
      }

      const resultGray = extractGrayscale(result.grayscale);
      const referenceGray = await loadReference(refPath);

      const cmp = compare(resultGray, referenceGray, perImageOutDir, stem, log);

      const debugDir = join(perImageOutDir, "debug");
      if (!existsSync(debugDir)) mkdirSync(debugDir, { recursive: true });
      await saveErrorMap(resultGray, referenceGray, debugDir, stem, 1);
      await saveErrorMap(
        resultGray,
        referenceGray,
        debugDir,
        stem,
        DEFAULT_SCALE,
        "_scaled",
      );
      await savePaletteImage(resultGray, debugDir, `${stem}_diag_result.png`);
      await savePaletteImage(
        referenceGray,
        debugDir,
        `${stem}_diag_reference.png`,
      );

      writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");

      results.push({
        name: stem,
        matchN: cmp.matches,
        matchPct: cmp.matchPct,
        diffN: cmp.wrongs,
        diffPct: cmp.wrongPct,
        verdict: cmp.passed ? "PASS" : "FAIL",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ERROR: ${msg}`);
      if (err instanceof Error) console.error(err.stack);
      logLines.push(`PIPELINE ERROR: ${msg}`);
      writeFileSync(logPath, logLines.join("\n") + "\n", "utf-8");
      results.push({
        name: stem,
        matchN: null,
        matchPct: null,
        diffN: null,
        diffPct: null,
        verdict: "ERROR",
      });
    }
  }

  return results;
}

/**
 * Write all debug artifacts for a pipeline run, all under `<outputDir>/debug/`:
 *   <stem>_<step>.png       — base step intermediates (warp/correct/crop/sample)
 *   <stem>_<dbgname>.png    — per-step debug images (e.g. warp_a_corners)
 *   <stem>_debug.json       — structured metrics + chronological log
 */
async function writeDebugArtifacts(
  result: {
    intermediates?: Record<string, GBImageData>;
    debug?: {
      images: Record<string, GBImageData>;
      log: string[];
      metrics: Record<string, Record<string, unknown>>;
    };
  },
  outputDir: string,
  stem: string,
): Promise<void> {
  if (!result.intermediates && !result.debug) return;
  const debugDir = join(outputDir, "debug");
  if (!existsSync(debugDir)) mkdirSync(debugDir, { recursive: true });

  if (result.intermediates) {
    for (const [stepName, img] of Object.entries(result.intermediates)) {
      await saveImage(img, join(debugDir, `${stem}_${stepName}.png`));
    }
  }
  if (result.debug?.images) {
    for (const [name, img] of Object.entries(result.debug.images)) {
      await saveImage(img, join(debugDir, `${stem}_${name}.png`));
    }
  }
  if (result.debug) {
    writeFileSync(
      join(debugDir, `${stem}_debug.json`),
      JSON.stringify(
        { metrics: result.debug.metrics, log: result.debug.log },
        null,
        2,
      ),
      "utf-8",
    );
  }
}

// ─── Test result shape ───

interface TestResult {
  name: string;
  matchN: number | null;
  matchPct: number | null;
  diffN: number | null;
  diffPct: number | null;
  verdict: string;
}

// ─── Summary ───

function writeCorpusSummary(corpus: CorpusConfig, results: TestResult[]): void {
  const lines: string[] = [];
  lines.push("=".repeat(70));
  lines.push(`CORPUS SUMMARY — ${corpus.name}`);
  lines.push(`  inputDir:   ${relative(REPO_ROOT, corpus.inputDir)}`);
  lines.push(`  outputDir:  ${relative(REPO_ROOT, corpus.outputDir)}`);
  lines.push(
    `  comparison: ${corpus.comparison}` +
      (corpus.referenceFromOutputDir
        ? `  (referenceFromOutputDir: ${relative(REPO_ROOT, corpus.referenceFromOutputDir)})`
        : ""),
  );
  lines.push("=".repeat(70));
  lines.push("");

  if (results.length === 0) {
    lines.push("  (no inputs found)");
  } else {
    const colW = Math.max(...results.map((r) => r.name.length));
    const header = `  ${"Test".padEnd(colW)}   ${"Matching".padEnd(18)}  ${"Different".padEnd(18)}  Verdict`;
    lines.push(header);
    lines.push("  " + "-".repeat(header.length - 2));
    for (const r of results) {
      const fmt = (n: number | null, pct: number | null): string => {
        if (n === null) return "       N/A       ";
        return `${String(n).padStart(5)} (${pct!.toFixed(2).padStart(6)}%)`;
      };
      lines.push(
        `  ${r.name.padEnd(colW)}   ${fmt(r.matchN, r.matchPct)}   ${fmt(r.diffN, r.diffPct)}   ${r.verdict}`,
      );
    }
    lines.push("");
    const passed = results.filter((r) => r.verdict === "PASS").length;
    const total = results.filter((r) => r.verdict !== "OK").length;
    if (total > 0) lines.push(`  ${passed}/${total} passed`);
  }

  lines.push("");
  const text = lines.join("\n") + "\n";
  console.log("\n" + text);

  if (!existsSync(corpus.outputDir))
    mkdirSync(corpus.outputDir, { recursive: true });
  writeFileSync(join(corpus.outputDir, "test-summary.log"), text, "utf-8");
}

// ─── Corpus catalogs ───
//
// Two test-run modes are supported, each running a different subset of
// corpora. Both share `runCorpus` etc. above.
//
// `quick` (default) — focused day-to-day validation: just test-input and
// test-input-full extraction. Skips the tier-2 self-consistency corpora.
//
// `full` — every corpus, in dependency order.

/** Corpora used by `pnpm test:pipeline` (the quick day-to-day run). */
function quickCorpora(): CorpusConfig[] {
  const corpora: CorpusConfig[] = [
    {
      name: "test-input",
      inputDir: TEST_INPUT_DIR,
      outputDir: TEST_OUTPUT_DIR,
      comparison: "reference",
    },
    {
      name: "test-input-full",
      inputDir: TEST_INPUT_FULL_DIR,
      outputDir: TEST_OUTPUT_FULL_DIR,
      comparison: "reference",
    },
  ];

  if (existsSync(TEST_INPUT_PRIVATE_DIR)) {
    corpora.push({
      name: "test-input-private",
      inputDir: TEST_INPUT_PRIVATE_DIR,
      outputDir: TEST_OUTPUT_PRIVATE_DIR,
      comparison: "reference",
    });
  }

  return corpora;
}

/** All corpora used by `pnpm test:pipeline:all`. */
function allCorpora(): CorpusConfig[] {
  const corpora: CorpusConfig[] = [
    {
      name: "sample-pictures",
      inputDir: SAMPLE_PICTURES_DIR,
      outputDir: SAMPLE_PICTURES_OUT,
      comparison: "none",
    },
    {
      name: "test-input",
      inputDir: TEST_INPUT_DIR,
      outputDir: TEST_OUTPUT_DIR,
      comparison: "reference",
    },
    {
      name: "test-input-full",
      inputDir: TEST_INPUT_FULL_DIR,
      outputDir: TEST_OUTPUT_FULL_DIR,
      comparison: "reference",
    },
    {
      name: "sample-pictures-full [self-consistency]",
      inputDir: SAMPLE_PICTURES_FULL_DIR,
      outputDir: SAMPLE_PICTURES_OUT_FULL,
      comparison: "self",
      referenceFromOutputDir: SAMPLE_PICTURES_OUT,
    },
  ];

  if (existsSync(SAMPLE_PICTURES_PRIVATE_DIR)) {
    corpora.push({
      name: "sample-pictures-private",
      inputDir: SAMPLE_PICTURES_PRIVATE_DIR,
      outputDir: SAMPLE_PICTURES_OUT_PRIVATE,
      comparison: "none",
    });
  }

  if (existsSync(TEST_INPUT_PRIVATE_DIR)) {
    corpora.push({
      name: "test-input-private",
      inputDir: TEST_INPUT_PRIVATE_DIR,
      outputDir: TEST_OUTPUT_PRIVATE_DIR,
      comparison: "reference",
    });
  }

  return corpora;
}

// ─── Main ───

async function main() {
  // Parse mode from CLI args (default: quick).
  const argv = process.argv.slice(2);
  let mode: "quick" | "all" = "quick";
  for (const arg of argv) {
    if (arg === "--mode=quick" || arg === "--quick") mode = "quick";
    else if (arg === "--mode=all" || arg === "--all") mode = "all";
    else if (arg === "--help" || arg === "-h") {
      console.log(
        "Usage: run-tests [--mode=quick|all]\n" +
          "  --mode=quick  Run test-input + test-input-full (default)\n" +
          "  --mode=all    Run all corpora",
      );
      return;
    }
  }

  console.log("Initializing OpenCV...");
  await initOpenCV();
  console.log(
    `OpenCV ready. Running ${mode === "all" ? "ALL" : "QUICK"} corpora.\n`,
  );

  const corpora = mode === "all" ? allCorpora() : quickCorpora();

  const allResults: { corpus: CorpusConfig; results: TestResult[] }[] = [];
  let anyError = false;

  for (const corpus of corpora) {
    const results = await runCorpus(corpus);
    writeCorpusSummary(corpus, results);
    allResults.push({ corpus, results });
    if (results.some((r) => r.verdict === "ERROR")) anyError = true;
  }

  // Top-level fail/pass: existing tier-1 ("reference") corpora must all PASS.
  // Tier-2 self-consistency corpora are soft signals.
  let allPassed = true;
  for (const { corpus, results } of allResults) {
    if (corpus.comparison !== "reference") continue;
    if (results.some((r) => r.verdict !== "PASS")) allPassed = false;
  }

  if (!allPassed || anyError) process.exit(1);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
