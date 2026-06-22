/**
 * extract.ts — CLI script to extract Game Boy Camera images from phone photos.
 *
 * Usage:
 *   pnpm extract -- [options] [input files...]
 */

import { resolve, join, basename, extname, dirname } from "path";
import { existsSync, mkdirSync, readdirSync, unlinkSync } from "fs";
import sharp from "sharp";
import { initOpenCV } from "../src/init-opencv.js";
import { processPicture } from "../src/index.js";
import { locate } from "../src/locate.js";
import { warp } from "../src/warp.js";
import { correct } from "../src/correct.js";
import { crop } from "../src/crop.js";
import { sample } from "../src/sample.js";
import { quantize } from "../src/quantize.js";
import type { GBImageData, StepName } from "../src/common.js";
import { STEP_ORDER } from "../src/common.js";

// ─── Helpers ───

function printHelp() {
  console.log(`
Game Boy Camera image extractor — TypeScript pipeline CLI

USAGE
  pnpm extract -- [options] [input files...]

POSITIONAL ARGUMENTS
  input files       One or more image files to process (.jpg, .jpeg, .png)

OPTIONS
  -d, --dir DIR     Directory of input images to glob for .jpg/.jpeg/.png files
  -o, --output-dir DIR
                    Output directory (created if needed). Default: same as input.
  --scale N         Working scale (default: 8)
  --start STEP      Start pipeline at this step
                    (locate/warp/correct/crop/sample/quantize)
  --end STEP        End pipeline at this step
  --clean-steps     Delete intermediate files after pipeline completes
  --debug           Save intermediate step images
  --help            Show this help message

STEPS (in order): locate -> warp -> correct -> crop -> sample -> quantize

  locate    Find the Game Boy Screen within a full phone photo and produce
            an upright crop with margin (input to warp). Already-cropped
            photos pass through cleanly. Skip with --start warp.

EXAMPLES
  pnpm extract -- --dir ../../sample-pictures-full -o ../../sample-pictures-out
  pnpm extract -- photo.jpg -o ./out
  pnpm extract -- --start warp --dir ../../test-input -o ../../test-output
  pnpm extract -- --start warp photo_already_cropped.jpg -o ./out
  pnpm extract -- --start quantize --dir ./out -o ./out
  pnpm extract -- --dir ./photos -o ./out --debug --clean-steps
`);
}

function stripStepSuffix(stem: string): string {
  for (const step of STEP_ORDER) {
    const suffix = `_${step === "quantize" ? "gbcam" : step}`;
    if (stem.endsWith(suffix)) {
      return stem.slice(0, -suffix.length);
    }
  }
  return stem;
}

const STEP_SUFFIX: Record<string, string> = {
  locate: "_locate",
  warp: "_warp",
  correct: "_correct",
  crop: "_crop",
  sample: "_sample",
  quantize: "_gbcam",
};

const STEP_INPUT_SUFFIX: Record<string, string> = {
  warp: "_locate",
  correct: "_warp",
  crop: "_correct",
  sample: "_crop",
  quantize: "_sample",
};

function collectInputFiles(positionalArgs: string[], dir?: string): string[] {
  const files: string[] = [];
  const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png"]);

  for (const f of positionalArgs) {
    const abs = resolve(f);
    if (existsSync(abs)) {
      files.push(abs);
    } else {
      console.error(`WARNING: file not found: ${f}`);
    }
  }

  if (dir) {
    const absDir = resolve(dir);
    if (existsSync(absDir)) {
      const entries = readdirSync(absDir);
      for (const entry of entries) {
        if (IMAGE_EXTS.has(extname(entry).toLowerCase())) {
          files.push(join(absDir, entry));
        }
      }
    } else {
      console.error(`WARNING: directory not found: ${dir}`);
    }
  }

  // Deduplicate
  return [...new Set(files)];
}

function collectForStart(positionalArgs: string[], dir: string | undefined, startStep: string): string[] {
  if (startStep === "locate" || startStep === "warp") {
    return collectInputFiles(positionalArgs, dir);
  }

  const suffix = STEP_INPUT_SUFFIX[startStep];
  const files: string[] = [];

  for (const f of positionalArgs) {
    const abs = resolve(f);
    if (existsSync(abs)) {
      files.push(abs);
    }
  }

  if (dir) {
    const absDir = resolve(dir);
    if (existsSync(absDir)) {
      const entries = readdirSync(absDir);
      for (const entry of entries) {
        const stem = basename(entry, extname(entry));
        if (stem.endsWith(suffix) && extname(entry).toLowerCase() === ".png") {
          files.push(join(absDir, entry));
        }
      }
    }
  }

  return [...new Set(files)];
}

async function loadImage(filePath: string): Promise<GBImageData> {
  // Auto-orient: applies any EXIF rotation so loaded pixels match visual
  // orientation. Phone photos may store landscape images with EXIF rotation.
  const img = sharp(filePath).rotate().removeAlpha().ensureAlpha();
  const { width, height } = await sharp(filePath).rotate().metadata() as { width: number; height: number };
  const { data, info } = await img
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Convert to RGBA Uint8ClampedArray
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
  const dir = dirname(outPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  await sharp(Buffer.from(img.data.buffer), {
    raw: { width: img.width, height: img.height, channels: 4 },
  })
    .png()
    .toFile(outPath);
}

// ─── Step runners ───

const STEP_FUNCTIONS: Record<string, (input: GBImageData, scale: number) => GBImageData> = {
  locate: (input, _scale) => locate(input),
  warp: (input, scale) => warp(input, { scale }),
  correct: (input, scale) => correct(input, { scale }),
  crop: (input, scale) => crop(input, { scale }),
  sample: (input, scale) => sample(input, { scale }),
  quantize: (input, _scale) => quantize(input),
};

// ─── CLI arg parsing ───

interface CLIArgs {
  inputs: string[];
  dir?: string;
  outputDir?: string;
  scale: number;
  start: StepName;
  end: StepName;
  cleanSteps: boolean;
  debug: boolean;
  help: boolean;
}

function parseArgs(argv: string[]): CLIArgs {
  const args: CLIArgs = {
    inputs: [],
    scale: 8,
    start: "locate",
    end: "quantize",
    cleanSteps: false,
    debug: false,
    help: false,
  };

  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    switch (arg) {
      case "--help":
      case "-h":
        args.help = true;
        break;
      case "--dir":
      case "-d":
        args.dir = argv[++i];
        break;
      case "--output-dir":
      case "-o":
        args.outputDir = argv[++i];
        break;
      case "--scale":
        args.scale = parseInt(argv[++i], 10);
        break;
      case "--start":
        args.start = argv[++i] as StepName;
        break;
      case "--end":
        args.end = argv[++i] as StepName;
        break;
      case "--clean-steps":
        args.cleanSteps = true;
        break;
      case "--debug":
        args.debug = true;
        break;
      case "--":
        // Some pnpm versions forward the argument separator literally; ignore it.
        break;
      default:
        if (!arg.startsWith("-")) {
          args.inputs.push(arg);
        } else {
          console.error(`Unknown option: ${arg}`);
          process.exit(1);
        }
    }
    i++;
  }

  return args;
}

// ─── Main ───

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printHelp();
    process.exit(0);
  }

  // Validate start/end
  const startIdx = STEP_ORDER.indexOf(args.start);
  const endIdx = STEP_ORDER.indexOf(args.end);
  if (startIdx < 0) {
    console.error(`Invalid --start step: ${args.start}. Choices: ${STEP_ORDER.join(", ")}`);
    process.exit(1);
  }
  if (endIdx < 0) {
    console.error(`Invalid --end step: ${args.end}. Choices: ${STEP_ORDER.join(", ")}`);
    process.exit(1);
  }
  if (startIdx > endIdx) {
    console.error(`--start ${args.start} comes after --end ${args.end} in the pipeline.`);
    process.exit(1);
  }

  // Collect input files
  const inputFiles = collectForStart(args.inputs, args.dir, args.start);
  if (inputFiles.length === 0) {
    console.error("No input files found.");
    printHelp();
    process.exit(1);
  }

  const activeSteps = STEP_ORDER.slice(startIdx, endIdx + 1);
  console.log(
    `Pipeline: ${activeSteps.join(" -> ")}  |  scale=${args.scale}  |  ${inputFiles.length} input file(s)`
  );

  // Initialize OpenCV
  console.log("Initializing OpenCV...");
  await initOpenCV();
  console.log("OpenCV ready.");

  let errors = 0;
  const intermediateFiles: string[] = [];

  for (let fi = 0; fi < inputFiles.length; fi++) {
    const inputPath = inputFiles[fi];
    const inputStem = stripStepSuffix(basename(inputPath, extname(inputPath)));
    const outDir = args.outputDir ? resolve(args.outputDir) : dirname(inputPath);

    if (!existsSync(outDir)) {
      mkdirSync(outDir, { recursive: true });
    }

    console.log(`\n[${fi + 1}/${inputFiles.length}] ${basename(inputPath)}`);

    try {
      let current = await loadImage(inputPath);

      for (let si = 0; si < activeSteps.length; si++) {
        const stepName = activeSteps[si];
        const isFinal = si === activeSteps.length - 1;
        const outPath = join(outDir, `${inputStem}${STEP_SUFFIX[stepName]}.png`);

        console.log(`  ${stepName}...`);
        const stepFn = STEP_FUNCTIONS[stepName];
        current = stepFn(current, args.scale);

        // Save output
        await saveImage(current, outPath);

        if (args.debug && !isFinal) {
          // Save debug copy
          const debugDir = join(outDir, "debug");
          if (!existsSync(debugDir)) {
            mkdirSync(debugDir, { recursive: true });
          }
        }

        if (!isFinal) {
          intermediateFiles.push(outPath);
        }

        // Load step output for next step (use the saved version)
        if (!isFinal) {
          current = await loadImage(outPath);
        }
      }

      console.log(`  -> ${inputStem}${STEP_SUFFIX[activeSteps[activeSteps.length - 1]]}.png`);
    } catch (err) {
      console.error(`  ERROR: ${err instanceof Error ? err.message : String(err)}`);
      if (args.debug && err instanceof Error) {
        console.error(err.stack);
      }
      errors++;
    }
  }

  // Clean up intermediate files
  if (args.cleanSteps && intermediateFiles.length > 0) {
    let removed = 0;
    let moved = 0;
    for (const filePath of intermediateFiles) {
      if (!existsSync(filePath)) continue;
      if (args.debug && args.outputDir) {
        const debugDir = join(resolve(args.outputDir), "debug");
        if (!existsSync(debugDir)) {
          mkdirSync(debugDir, { recursive: true });
        }
        const dest = join(debugDir, basename(filePath));
        const { renameSync } = await import("fs");
        renameSync(filePath, dest);
        moved++;
      } else {
        unlinkSync(filePath);
        removed++;
      }
    }
    if (moved) console.log(`  [clean-steps] Moved ${moved} intermediate file(s) to debug/`);
    if (removed) console.log(`  [clean-steps] Deleted ${removed} intermediate file(s).`);
  }

  const succeeded = inputFiles.length - errors;
  console.log(`\nDone — ${succeeded} succeeded, ${errors} failed.`);
  if (errors > 0) process.exit(1);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
