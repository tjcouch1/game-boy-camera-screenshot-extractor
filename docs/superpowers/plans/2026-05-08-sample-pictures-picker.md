# Sample Pictures Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Sample Pictures" popover next to the Camera Capture button that lets users select bundled sample pictures from `<repo>/sample-pictures/` (with localStorage-persisted selection, defaulting to all on first load) and run them through the existing pipeline as if uploaded.

**Architecture:** A new build-time script copies `sample-pictures/` into `public/sample-pictures/` and generates a manifest module, mirroring `copy-frames.ts`. `package.json` consolidates the three asset-build scripts behind a single `build:assets` script. A new `SamplePicturePicker` React component reads the manifest, manages selection via `useLocalStorage`, fetches selected images on submit, and forwards them to the same `onImagesSelected` callback the upload buttons use.

**Tech Stack:** TypeScript, React 19, base-ui Popover (via `@/shadcn/components/popover`), `useLocalStorage` hook, `sonner` for toasts, lucide-react icons, Node.js fs APIs for the build script.

**Spec:** `docs/superpowers/specs/2026-05-08-sample-pictures-picker-design.md`

**Notes for executor:**

- All paths in this plan are relative to the repo root: `C:\Users\tj_co\source\repos-p\game-boy-camera-picture-extractor-2`.
- The web package has no automated UI test suite, so verification is `pnpm typecheck` plus a manual dev-server pass at the end. Do NOT invent a test framework — match the codebase.
- Always use `pnpm` (not npm/yarn). The repo is a pnpm workspace.
- The shell is PowerShell on Windows. Bash is also available via the Bash tool. Either is fine for the commands below.
- Do NOT edit auto-generated files by hand. `src/generated/SamplePictures.ts` is produced by the build script.
- Follow project conventions from `AGENTS.md`: lucide-react icons (no emojis), semantic Tailwind tokens (`bg-muted`, `text-muted-foreground`, etc.), logical Tailwind classes (`ms-*` / `me-*` / `start-*` / `end-*`), `cn()` from `@/shadcn/utils/utils` for conditional classes, base-ui composition (`render={<Component />}`, not `asChild`).

---

## File Structure

**New files:**

- `packages/gbcam-extract-web/scripts/copy-sample-pictures.ts` — build-time script that copies `sample-pictures/` to `public/sample-pictures/` and writes the manifest module.
- `packages/gbcam-extract-web/src/components/SamplePicturePicker.tsx` — popover component with the selection grid and submit footer.
- `packages/gbcam-extract-web/src/generated/SamplePictures.ts` — auto-generated, gitignored. Created by the script in Task 1.
- `packages/gbcam-extract-web/public/sample-pictures/*` — auto-copied, gitignored. Created by the script in Task 1.

**Edited files:**

- `.gitignore` — ignore the new generated module and public assets.
- `packages/gbcam-extract-web/package.json` — add `build:sample-pictures` + consolidating `build:assets`; replace repeated chains.
- `packages/gbcam-extract-web/src/components/ImageInput.tsx` — mount `SamplePicturePicker` in the existing button row.

---

## Task 1: Add the build script and gitignore entries

**Files:**

- Create: `packages/gbcam-extract-web/scripts/copy-sample-pictures.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Create `copy-sample-pictures.ts`**

Create `packages/gbcam-extract-web/scripts/copy-sample-pictures.ts` with this exact content. It mirrors `copy-frames.ts` but targets `sample-pictures/` and emits `SAMPLE_PICTURES` instead of `FRAME_SHEETS`.

```ts
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const projectRoot = path.join(__dirname, "../../../");
const sourceRoot = path.join(projectRoot, "sample-pictures");
const destPublic = path.join(__dirname, "../public/sample-pictures");
const destManifest = path.join(
  __dirname,
  "../src/generated/SamplePictures.ts",
);

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);

interface SampleEntry {
  url: string;
  filename: string;
}

function walk(dir: string, out: string[] = []): string[] {
  if (!fs.existsSync(dir)) return out;
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function main() {
  if (!fs.existsSync(sourceRoot)) {
    console.warn(
      `[copy-sample-pictures] No sample pictures at ${sourceRoot}; skipping.`,
    );
    fs.mkdirSync(path.dirname(destManifest), { recursive: true });
    fs.writeFileSync(
      destManifest,
      `// auto-generated — do not edit\nexport interface SamplePictureEntry { url: string; filename: string; }\nexport const SAMPLE_PICTURES: ReadonlyArray<SamplePictureEntry> = [];\n`,
      "utf-8",
    );
    return;
  }

  fs.mkdirSync(destPublic, { recursive: true });
  fs.mkdirSync(path.dirname(destManifest), { recursive: true });

  const all = walk(sourceRoot);
  const entries: SampleEntry[] = [];
  for (const src of all) {
    const ext = path.extname(src).toLowerCase();
    if (!IMAGE_EXTS.has(ext)) continue;
    const rel = path.relative(sourceRoot, src);
    const dest = path.join(destPublic, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
    const url = "./sample-pictures/" + rel.split(path.sep).join("/");
    const filename = path.basename(rel);
    entries.push({ url, filename });
    console.log(`[copy-sample-pictures] ${src} -> ${dest}`);
  }

  // Stable order: alphabetical by filename.
  entries.sort((a, b) => a.filename.localeCompare(b.filename));

  const manifest = `// auto-generated by scripts/copy-sample-pictures.ts — do not edit
export interface SamplePictureEntry {
  /** URL the browser fetches (relative to the deployed root). */
  url: string;
  /** Source filename including extension, e.g. "20260313_213416.jpg". */
  filename: string;
}

export const SAMPLE_PICTURES: ReadonlyArray<SamplePictureEntry> = ${JSON.stringify(entries, null, 2)};
`;
  fs.writeFileSync(destManifest, manifest, "utf-8");
  console.log(
    `[copy-sample-pictures] wrote ${entries.length} entries to ${destManifest}`,
  );
}

main();
```

- [ ] **Step 2: Append the new gitignore entries**

Open `.gitignore` and append at the very end (after the existing `# Frame sheets copied into web/public and the generated manifest` block):

```gitignore

# Sample pictures copied into web/public and the generated manifest
packages/gbcam-extract-web/public/sample-pictures
packages/gbcam-extract-web/src/generated/SamplePictures.ts
```

- [ ] **Step 3: Run the script and verify output**

Run from the web package directory:

```bash
cd packages/gbcam-extract-web && node scripts/copy-sample-pictures.ts
```

Expected console output (order of "->" lines may vary):

```
[copy-sample-pictures] <repo>\sample-pictures\20260313_213416.jpg -> <repo>\packages\gbcam-extract-web\public\sample-pictures\20260313_213416.jpg
[copy-sample-pictures] <repo>\sample-pictures\20260313_213430.jpg -> ...
[copy-sample-pictures] <repo>\sample-pictures\20260313_213443.jpg -> ...
[copy-sample-pictures] <repo>\sample-pictures\20260313_213457.jpg -> ...
[copy-sample-pictures] <repo>\sample-pictures\20260313_213510.jpg -> ...
[copy-sample-pictures] wrote 5 entries to <repo>\packages\gbcam-extract-web\src\generated\SamplePictures.ts
```

Then verify:

```bash
ls packages/gbcam-extract-web/public/sample-pictures
```

Expected: the same 5 `.jpg` filenames currently in `<repo>/sample-pictures/`.

Read `packages/gbcam-extract-web/src/generated/SamplePictures.ts` and confirm:
- It exports `SAMPLE_PICTURES` with 5 entries.
- Each entry has `url` (e.g. `"./sample-pictures/20260313_213416.jpg"`) and `filename` (e.g. `"20260313_213416.jpg"`).
- Entries are sorted alphabetically by filename.

- [ ] **Step 4: Confirm the generated artifacts are gitignored**

Run from the repo root:

```bash
git status --short
```

Expected: only `.gitignore` and `packages/gbcam-extract-web/scripts/copy-sample-pictures.ts` show up under "untracked" / "modified". Neither `packages/gbcam-extract-web/public/sample-pictures/` nor `packages/gbcam-extract-web/src/generated/SamplePictures.ts` should appear.

- [ ] **Step 5: Commit**

```bash
git add .gitignore packages/gbcam-extract-web/scripts/copy-sample-pictures.ts
git commit -m "Add copy-sample-pictures build script

Mirrors copy-frames.ts: copies sample-pictures/ into web/public and
generates a typed manifest module. Both outputs gitignored.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Consolidate asset-build scripts in package.json

**Files:**

- Modify: `packages/gbcam-extract-web/package.json`

- [ ] **Step 1: Edit the `scripts` block**

Open `packages/gbcam-extract-web/package.json` and replace the existing `scripts` block with the following. The diff: add `build:sample-pictures` and `build:assets`; collapse the repeated `build:instructions && build:frames` chains in `dev`/`dev:host`/`build`/`preview`/`preview:host` into a single `pnpm build:assets`; rewrite `postinstall` to use `build:assets` plus the install-only `generate-licenses.ts`:

```json
  "scripts": {
    "build:instructions": "node scripts/generate-instructions.ts",
    "build:frames": "node scripts/copy-frames.ts",
    "build:sample-pictures": "node scripts/copy-sample-pictures.ts",
    "build:assets": "pnpm build:instructions && pnpm build:frames && pnpm build:sample-pictures",
    "dev": "pnpm build:assets && vite",
    "dev:host": "pnpm build:assets && vite --host",
    "build": "pnpm build:assets && tsc -b && vite build",
    "preview": "pnpm build:assets && vite preview",
    "preview:host": "pnpm build:assets && vite preview --host",
    "serve": "node scripts/serve-dist.ts",
    "typecheck": "tsc --noEmit",
    "postinstall": "pnpm build:assets && node scripts/generate-licenses.ts"
  },
```

- [ ] **Step 2: Run `pnpm build:assets` end-to-end**

```bash
cd packages/gbcam-extract-web && pnpm build:assets
```

Expected: it executes the three asset builders in sequence with no errors. You should see the per-script logs from `generate-instructions`, `copy-frames`, and `copy-sample-pictures`, ending with the sample-pictures wrote-N-entries line.

- [ ] **Step 3: Run typecheck (sanity)**

```bash
cd packages/gbcam-extract-web && pnpm typecheck
```

Expected: exits 0 with no output. (Nothing has imported the new generated module yet, so this is just a sanity check that the package.json edit didn't break tooling.)

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract-web/package.json
git commit -m "Consolidate asset-build scripts into build:assets

Adds build:sample-pictures and a combined build:assets that runs all
three asset builders. Replaces the repeated build:instructions &&
build:frames chains in dev/build/preview/postinstall scripts.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Add the SamplePicturePicker component

**Files:**

- Create: `packages/gbcam-extract-web/src/components/SamplePicturePicker.tsx`

- [ ] **Step 1: Create the component**

Create `packages/gbcam-extract-web/src/components/SamplePicturePicker.tsx` with this exact content:

```tsx
import { useMemo, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/shadcn/components/button";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/shadcn/components/popover";
import { cn } from "@/shadcn/utils/utils";
import { useLocalStorage } from "../hooks/useLocalStorage.js";
import { SAMPLE_PICTURES } from "../generated/SamplePictures.js";

interface SamplePicturePickerProps {
  onImagesSelected: (files: File[]) => void;
  disabled?: boolean;
}

const STORAGE_KEY = "gbcam-sample-picture-selections";

function guessMimeFromName(filename: string): string {
  const ext = filename.toLowerCase().match(/\.[^.]+$/)?.[0];
  switch (ext) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

export function SamplePicturePicker({
  onImagesSelected,
  disabled,
}: SamplePicturePickerProps) {
  const [open, setOpen] = useState(false);
  const [storedSelection, setStoredSelection] = useLocalStorage<
    string[] | null
  >(STORAGE_KEY, null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validFilenames = useMemo(
    () => new Set(SAMPLE_PICTURES.map((s) => s.filename)),
    [],
  );

  const effectiveSelected = useMemo(
    () =>
      storedSelection === null
        ? SAMPLE_PICTURES.map((s) => s.filename)
        : storedSelection.filter((f) => validFilenames.has(f)),
    [storedSelection, validFilenames],
  );

  const selectedSet = useMemo(
    () => new Set(effectiveSelected),
    [effectiveSelected],
  );

  if (SAMPLE_PICTURES.length === 0) return null;

  const toggle = (filename: string) => {
    setStoredSelection((prev) => {
      const base =
        prev === null
          ? SAMPLE_PICTURES.map((s) => s.filename)
          : prev.filter((f) => validFilenames.has(f));
      return base.includes(filename)
        ? base.filter((f) => f !== filename)
        : [...base, filename];
    });
  };

  const handleSubmit = async () => {
    const selectedEntries = SAMPLE_PICTURES.filter((s) =>
      selectedSet.has(s.filename),
    );
    if (selectedEntries.length === 0) return;

    setIsSubmitting(true);
    try {
      const settled = await Promise.allSettled(
        selectedEntries.map(async (entry) => {
          const res = await fetch(entry.url);
          if (!res.ok) {
            throw new Error(
              `Failed to fetch ${entry.filename}: ${res.status}`,
            );
          }
          const blob = await res.blob();
          const type = blob.type || guessMimeFromName(entry.filename);
          return new File([blob], entry.filename, { type });
        }),
      );

      const files: File[] = [];
      const errors: string[] = [];
      for (const result of settled) {
        if (result.status === "fulfilled") files.push(result.value);
        else
          errors.push(
            result.reason instanceof Error
              ? result.reason.message
              : String(result.reason),
          );
      }

      if (errors.length > 0) {
        toast.error(
          `Failed to load ${errors.length} sample picture${
            errors.length === 1 ? "" : "s"
          }`,
          { description: errors.join("\n") },
        );
      }

      if (files.length > 0) {
        onImagesSelected(files);
        setOpen(false);
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedCount = effectiveSelected.length;
  const totalCount = SAMPLE_PICTURES.length;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={<Button variant="secondary" disabled={disabled} />}
      >
        Sample Pictures
        <ChevronDown data-icon="inline-end" />
      </PopoverTrigger>
      <PopoverContent
        align="start"
        side="bottom"
        className="flex flex-col w-[min(28rem,90vw)] max-h-[min(80vh,32rem)] gap-2"
      >
        <PopoverHeader>
          <PopoverTitle>Sample pictures</PopoverTitle>
          <PopoverDescription>
            {selectedCount} of {totalCount} selected
          </PopoverDescription>
        </PopoverHeader>
        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-2 gap-2">
            {SAMPLE_PICTURES.map((entry) => {
              const isSelected = selectedSet.has(entry.filename);
              return (
                <button
                  key={entry.filename}
                  type="button"
                  onClick={() => toggle(entry.filename)}
                  aria-pressed={isSelected}
                  aria-label={`Toggle ${entry.filename}`}
                  className={cn(
                    "relative rounded-md border bg-muted p-1 ring-2 ring-transparent transition-colors text-start",
                    isSelected && "ring-primary border-primary",
                  )}
                >
                  <img
                    src={entry.url}
                    loading="lazy"
                    alt=""
                    className="block w-full h-24 object-contain"
                  />
                  <span className="block text-xs text-muted-foreground truncate mt-1">
                    {entry.filename}
                  </span>
                  {isSelected && (
                    <span
                      aria-hidden="true"
                      className="absolute top-1 end-1 inline-flex items-center justify-center rounded-full bg-primary text-primary-foreground size-5"
                    >
                      <Check className="size-3" />
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex items-center justify-end gap-2 pt-2 border-t">
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || selectedCount === 0}
          >
            {isSubmitting
              ? "Loading…"
              : `Process ${selectedCount} picture${
                  selectedCount === 1 ? "" : "s"
                }`}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
```

Notes for the executor on why specific things are written this way:

- All hooks (`useState`, `useLocalStorage`, `useMemo`) appear unconditionally before the `if (SAMPLE_PICTURES.length === 0) return null` early return — required by the React rules of hooks even though the condition is build-time constant.
- `useLocalStorage<string[] | null>(..., null)` uses `null` as the sentinel for "user has never made a selection on this device". On first load, `effectiveSelected` falls back to "all filenames" so the picker comes preselected. Once the user toggles anything, `storedSelection` becomes a concrete array (possibly empty) and the null branch is no longer used.
- `validFilenames` filtering means that if a sample picture is renamed or deleted (and the rebuilt manifest no longer lists it), its stale entry is silently dropped on read.
- `toast` is imported from `sonner`; `<Toaster />` is already mounted at the top of `App.tsx`, so toasts will render.
- `PopoverTrigger` uses `render={<Button ... />}` — base-ui composition pattern, matching the existing usage in `App.tsx`.
- Each tile is a real `<button type="button">` with `aria-pressed` for a11y; the whole tile is clickable.

- [ ] **Step 2: Typecheck**

```bash
cd packages/gbcam-extract-web && pnpm typecheck
```

Expected: exits 0 with no output.

If TypeScript complains that `SAMPLE_PICTURES` or its module cannot be found, the generated file from Task 1 wasn't written. Re-run `pnpm build:sample-pictures` and try again.

- [ ] **Step 3: Commit**

```bash
git add packages/gbcam-extract-web/src/components/SamplePicturePicker.tsx
git commit -m "Add SamplePicturePicker popover component

Popover with a 2-column grid of sample-picture thumbnails. Selections
persist to localStorage; on first load (no stored entry) every picture
is preselected. Submit fetches the selected images, wraps them in File
objects, and forwards to the existing onImagesSelected callback.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Mount the picker in ImageInput

**Files:**

- Modify: `packages/gbcam-extract-web/src/components/ImageInput.tsx`

- [ ] **Step 1: Add the import**

Open `packages/gbcam-extract-web/src/components/ImageInput.tsx`. After the existing imports at the top of the file, add:

```tsx
import { SamplePicturePicker } from "./SamplePicturePicker.js";
```

- [ ] **Step 2: Mount the picker after the Camera Capture button**

In the same file, locate the existing button row:

```tsx
      <div className="flex gap-3 mt-3">
        <Button
          variant="secondary"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
        >
          Choose Files
        </Button>
        <Button
          variant="secondary"
          onClick={() => cameraInputRef.current?.click()}
          disabled={disabled}
        >
          Camera Capture
        </Button>
      </div>
```

Add the picker as the last child inside the row, immediately after the Camera Capture button:

```tsx
      <div className="flex gap-3 mt-3">
        <Button
          variant="secondary"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
        >
          Choose Files
        </Button>
        <Button
          variant="secondary"
          onClick={() => cameraInputRef.current?.click()}
          disabled={disabled}
        >
          Camera Capture
        </Button>
        <SamplePicturePicker
          onImagesSelected={onImagesSelected}
          disabled={disabled}
        />
      </div>
```

No other changes to `ImageInput.tsx`. The parent `App.handleImagesSelected` already runs the archive-then-process flow, so submitting from the picker behaves identically to a fresh upload.

- [ ] **Step 3: Typecheck**

```bash
cd packages/gbcam-extract-web && pnpm typecheck
```

Expected: exits 0 with no output.

- [ ] **Step 4: Commit**

```bash
git add packages/gbcam-extract-web/src/components/ImageInput.tsx
git commit -m "Mount SamplePicturePicker in ImageInput button row

Sits to the right of Camera Capture, sharing the same disabled flag
and onImagesSelected callback so submissions flow through the existing
archive-then-process handler unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Manual verification with the dev server

**Files:** none (read-only verification)

This task does NOT produce a commit unless something fails and needs fixing. If you find a defect, fix it, re-run the relevant typecheck, and commit the fix as a separate commit before proceeding.

- [ ] **Step 1: Start the dev server in the background**

```bash
cd packages/gbcam-extract-web && pnpm dev
```

Expected: Vite eventually prints something like `Local: http://localhost:5173/`. The terminal stays attached. Run this in the background (`run_in_background: true`) so you can keep using other tools.

- [ ] **Step 2: Confirm initial selection state**

Use the Bash tool to confirm the build succeeded:

```bash
ls packages/gbcam-extract-web/public/sample-pictures
```

Expected: 5 `.jpg` files matching `<repo>/sample-pictures/`.

```bash
cat packages/gbcam-extract-web/src/generated/SamplePictures.ts
```

Expected: 5 entries in `SAMPLE_PICTURES`, each with `./sample-pictures/<filename>` URLs, sorted alphabetically.

The remaining checks are in-browser. **The executor cannot directly drive the browser**, so report each of the following as "needs human verification" with the URL and the checklist below, unless a Playwright/Puppeteer integration is available in the harness — in that case use it.

Manual checklist (state these explicitly in the final summary):

1. "Sample Pictures" button appears next to "Camera Capture" with a chevron-down icon.
2. Clicking it opens a popover with 5 thumbnails, all selected (filled ring + check badge), and the description reads "5 of 5 selected".
3. Toggling a tile updates the count and the visible ring/badge state.
4. The Submit button reads `Process N picture(s)` and stays anchored at the bottom of the popover even when the grid scrolls.
5. Reloading the page preserves the toggled selection.
6. Clicking Submit closes the popover and runs the pipeline on each selected image; current results archive into history; the new batch appears.
7. Re-opening the popover after submit shows the same selection state.

- [ ] **Step 3: Stop the dev server**

If you started it as a background process, kill it via the harness's process-management tools. (PowerShell: `Stop-Process -Name node -Force` is too aggressive; prefer killing the specific background process by its PID.)

- [ ] **Step 4: Final summary**

Report to the user:

- Each of the 4 commits made (titles).
- The output of `git status` (should be clean).
- The manual-verification checklist above and which items the executor was/was not able to verify directly.

No commit in this task unless a defect was fixed.

---

## Self-review (executor-internal)

Before declaring the plan complete, double-check:

- All four code commits exist on the current branch (`add-frames`).
- `pnpm typecheck` from `packages/gbcam-extract-web` exits 0.
- `pnpm build:assets` from `packages/gbcam-extract-web` exits 0 and produces both `public/sample-pictures/` and `src/generated/SamplePictures.ts`.
- `git status` is clean (no stray untracked files outside the gitignored generated outputs).
- No part of the spec is left unimplemented.
