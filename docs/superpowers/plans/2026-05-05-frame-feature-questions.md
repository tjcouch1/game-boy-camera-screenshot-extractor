# Frame feature — questions I had to guess on

You asked me to record decisions where I would normally have asked you, with my guessed answer, so you can review them at the end.

## Process / housekeeping

1. **Branch name** — Picked `add-frames`. (You said "do this on a branch" without specifying a name.)
2. **Spec / plan locations** — Used the default from the brainstorming skill: `docs/superpowers/specs/2026-05-05-frame-feature-design.md` and `docs/superpowers/plans/2026-05-05-frame-feature-plan.md`. New top-level `docs/superpowers/` directory.
3. **Push / open PR** — Did **not** push or create a PR. Branch is local, on `add-frames`, ready when you want it.
4. **Version bumps** — Did not bump `gbcam-extract` or `gbcam-extract-web` package versions. Recent commit history shows version bumps live in their own PR (`c6a8d7d Bump versions: …`). Left at `0.4.0` for both.
5. **Pre-existing `pipeline.test.ts` failures** — 6 pixel-accuracy tests in `packages/gbcam-extract/tests/pipeline.test.ts` were already failing on `main` (last touched in commit `83d3f25`). I did not touch them. They show up in `pnpm test` output as failures unrelated to this branch.
6. **The dev server I started for the smoke test is still running on port 5174.** I tried to stop it but my shell wasn't allowed to kill all `node.exe` processes broadly. Please stop it yourself (`Ctrl+C` in the terminal that ran it, or kill the relevant process via Task Manager).

## Algorithmic / design decisions made by the implementer

7. **Splitter's hole-detection algorithm changed during implementation.** The plan said "find the 128×112 sub-rectangle that's entirely *background-coloured*". The real PNG sheets fill the hole with a different uniform colour (`(200, 191, 231)`) — close to but not exactly the sheet background `(200, 180, 255)`. The implementer (with my approval) replaced the check with "find a 128×112 sub-rectangle that is *any single uniform colour* (within ±4 per channel)". Synthetic test (where the hole is bg-coloured) still passes. Real sheets now split cleanly. **If this is wrong, the fix is to revert the `isUniformHole` change in `split-sheet.ts` and override the bg-tolerance constants.**
8. **Splitter recomputes a tight bounding box by scanning outward from the hole edges,** because in the synthetic test a spurious extra rectangle could have flood-filled into the same connected component. For real sheets where every frame is cleanly separated by background, this scan-outward is redundant but harmless. Net result: robust to "decorations near a frame" without changing real-sheet output.
9. **Frame counts.** The splitter discovered:
   - `Frames_USA.png`: 18 normal + 7 wild = **25 frames**.
   - `Frames_JPN.png`: 18 normal + 8 wild = **26 frames** (the JPN sheet has one more wild frame; consistent with the README's note about CoroCoro extras).
   - After dedup: **36 unique frames** (15 USA frames are exact duplicates of JPN frames; JPN wins the tiebreak alphabetically). 26 from JPN + 10 from USA.
10. **Wild frame dimensions.** All wild frames in both sheets are 160×224 with the hole at (16, 40). I had assumed they might be of varied widths from a quick visual look at the sheets, but the splitter found uniform dimensions. If you expected some wild frames to be different aspect ratios, please verify the snapshots in `tests/frames/split-sheet.test.ts`.

## UX decisions

11. **Final reviewer's I-1 fix.** `setDefaultFrame` now `console.warn`s instead of silently dropping a `kind: "default"` change. The "Default" tile is hidden in the global default picker, so this branch should never fire — but if it ever does, you'll see it in the console.
12. **Final reviewer's I-2 fix.** Added `framePickerDisabled` prop to `ResultCard` so per-result pickers are also disabled while the catalog is loading (matching the global default picker's behaviour).
13. **Final reviewer's I-3 fix.** Added an `<Alert variant="destructive">` for `catalog.status === "error"`, with copy explaining frames are unavailable but processing still works.
14. **Final reviewer's I-4** (stale frame IDs in localStorage). Skipped: the code already gracefully degrades to no-frame via `getFrameById(id) ?? null`. Not adding a comment since the user already understands the trade-off.
15. **Final reviewer's M-3** (dead `?? FRAME_SELECTION_NONE` because `useAppSettings` already merges DEFAULTS). Kept as belt-and-braces. Negligible.
16. **Final reviewer's M-6** (`README.md` for Spriters Resource has no trailing newline). Skipped.

## Things that need your attention

17. **License (Critical).** Final reviewer flagged that The Spriters Resource Terms of Use prohibit commercial use, including "anything 100% free being published to an established market place (e.g. Steam, Apple's App Store, or Google Play)". The PWA is currently deployed only to GitHub Pages, which is fine. **If you ever plan to ship this to the App Store or Play Store, you'll need to either (a) get explicit permission from the artist (`rabidrodent`) or (b) remove the bundled Spriters Resource frames and let users supply their own.** The licence/attribution content is in `packages/gbcam-extract-web/additional-licenses/additional-licenses.json` and `supporting-materials/frames/the-spriters-resource/README.md`.
18. **The pre-existing pipeline accuracy gap** is unrelated to this branch but you should know it exists if `pnpm test` will be run as part of CI/release gates. Branch passes everything else (typecheck, build, all frame tests).
