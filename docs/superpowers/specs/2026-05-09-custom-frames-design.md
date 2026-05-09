# Custom Frames Upload — Design Spec

Date: 2026-05-09
Branch: `add-custom-frames` (off `add-frames`)

## Goal

Let a user upload their own frame images in the web app. The picker auto-detects whether the upload is an individual frame or a sheet, splits sheets into individual frames, dedupes against the existing catalog, and persists the new frames to localStorage as PNGs. Users can delete custom frames after a confirmation.

## Non-goals

- Editing custom frames in place.
- Sharing custom frames between users / across devices.
- Reordering custom frames.
- Renaming custom frames after upload (the inferred name from the filename is used as-is).

## User flow

1. User opens the Frame picker.
2. A new "Custom frames" section sits below "Wild frames", with a header that includes an `Upload` button (lucide `Upload` icon) and is shown even when empty.
3. Empty state: a short note ("Upload a Game Boy Camera frame PNG. Sheets and individual frames are both supported.").
4. Clicking `Upload` opens a hidden `<input type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple>` picker.
5. For each selected file:
   - Decode → `GBImageData` (RGBA).
   - Try `splitSheet`. If it yields ≥1 frames, keep them.
   - Otherwise try `loadIndividualFrame`. If it succeeds, keep that frame.
   - On any other outcome, surface a toast with the filename and the reason.
6. Frames pass through `appendDeduped(catalog, candidates)`. Dropped duplicates show a single toast summarising the count. New frames are encoded to PNG data URLs and persisted via `useUserFrames`.
7. Each custom-frame tile shows a small `Trash2` icon button in its top-end corner. Click → confirmation `Dialog` ("Delete this frame? This can't be undone.") → on confirm, the entry is removed from localStorage and the catalog refreshes.

## Architecture

### Data model

`UserFrameEntry` (lives in `useUserFrames.ts`):

```ts
interface UserFrameEntry {
  id: string;            // "user-frame-<timestamp>-<rand>"
  sheetStem: string;     // sanitized filename stem, used in display name
  type: "normal" | "wild";
  width: number;
  height: number;
  holeX: number;
  holeY: number;
  /** PNG data URL of a single-channel grayscale image (R=G=B=pixel value, A=255). */
  pngDataUrl: string;
  addedAt: number;       // Date.now() — used for stable ordering
}
```

The `Frame.kind` for any decoded user entry is fixed to `"individual"`. Even when uploaded as part of a sheet, each split frame is stored as its own individual-style entry — once it's in localStorage there is no value in tracking that several frames came from the same sheet.

### Storage

- localStorage key: `gbcam-user-frames` (JSON array of `UserFrameEntry`).
- Version key: `gbcam-user-frames-version` = `"1"`. Module-load reset matches the `useUserPalettes` pattern (drop the data key if version mismatches, then write the current version).
- PNG encoding mirrors `serialization.ts`'s `grayscaleToCanvasPNG`: build an RGBA canvas (`R = G = B = pixelValue`, `A = 255`), call `toDataURL("image/png")`. Decoding loads via `<img>`, draws to a canvas, reads back, and grabs the R channel.
- Storing the grayscale value as RGB-replicated keeps round-trip lossless (PNG is lossless; the four GB grayscale values survive intact).

A 160×144 frame compresses to ~3–4 KB as a base64 data URL — well within typical localStorage budgets.

### Hook: `useUserFrames`

```ts
{
  entries: UserFrameEntry[];
  decodedFrames: Frame[];                 // Frame[] objects ready for the catalog
  status: "loading" | "ready" | "error";
  addFrames(frames: Frame[]): { added: number; skippedDuplicates: number };
  deleteFrame(id: string): void;
}
```

- Decode happens in `useEffect` whenever `entries` changes; `decodedFrames` is memoized.
- `addFrames` is the reverse — encode to PNG, append, persist. Dedup happens *upstream* (before `addFrames` is called) so this hook stays storage-only.

### Catalog integration

`useFrameCatalog` is refactored:

- Built-in frames (sheets + individuals) are still cached at module scope so the parse runs once per session.
- The hook reads `useUserFrames().decodedFrames` and merges via `appendDeduped(builtIns, userFrames)` on every render where either changes.
- The cached `byId` map is rebuilt whenever the merged list changes.
- `getFrameById` continues to work for both built-in and custom frames since IDs are unique by construction.

A new helper `isUserFrame(frame, userEntries)` (exported from the hook) lets the picker tell whether to render the delete button on a tile.

### Detection

A new helper `detectAndLoadFrames(image: GBImageData, stem: string): Frame[]` lives in `packages/gbcam-extract-web/src/utils/detectFrames.ts`:

1. Try `splitSheet(image, stem)`.
   - If it returns ≥1 frame → return them.
2. Else try `loadIndividualFrame(image, stem)`.
   - On success → wrap the single frame in an array.
3. Else throw an `Error` with a human-readable message.

This is web-package-local because it only orchestrates existing exports from `gbcam-extract`. Each call's result has its `kind` overwritten to `"individual"` before being passed to `addFrames`.

### Filename → stem

- Strip the extension.
- Replace any character outside `[A-Za-z0-9_-]` with `-`.
- Collapse runs of `-`.
- If the resulting stem collides with a built-in or existing custom frame's `sheetStem`, append `-N` where N is the smallest integer ≥ 2 that disambiguates.

### Frame picker UI changes

- Add an upload button to the picker body, below "Wild frames" and above the (new) custom-frame grid. The button is a secondary `Button` with a `Upload` icon. Mobile: same button, just lives in the drawer body.
- Add a "Custom frames" section header (matching the existing `Normal frames` / `Wild frames` styling).
- Each custom tile reuses the existing tile component but adds an icon-only `Button` (`Trash2`, `aria-label="Delete frame"`, `data-icon="inline-start"` not applicable since icon-only) absolutely positioned at `top-2 end-2`.
- Clicking the trash button opens a `Dialog` with two buttons: "Cancel" (default variant) and "Delete" (destructive variant). The dialog quotes the frame's display name in the body.
- Toast feedback uses the existing `sonner` integration.

The picker continues to receive `frames: Frame[]` from `useFrameCatalog`. To know which tiles are custom (and thus need a trash button), the picker also receives a `userFrameIds: Set<string>` — derived from `useUserFrames` at the parent layer (`App.tsx`) and threaded through the existing prop.

### shadcn

- Reuse: `Button`, `Dialog`, `Drawer`, `Popover`, `sonner` (already installed).
- The delete confirmation uses the generic `Dialog` (not `AlertDialog`, which is not currently installed). Adding `alert-dialog` is a separate concern; `Dialog` with two buttons is sufficient and matches the available primitives.

## Error handling

- Unsupported file type → toast: "‹filename›: Unsupported image type."
- Decoding fails → toast: "‹filename›: Couldn't decode image."
- Detection fails (no hole found, dimensions off, etc.) → toast: "‹filename›: Couldn't detect a frame in this image."
- localStorage quota exceeded → toast: "Out of storage. Delete some frames or images and try again." Storage write is wrapped in try/catch; on failure, the in-memory state is rolled back so the catalog stays consistent with disk.
- A single batch can mix successes and failures; each file produces at most one toast (success batch is summarised: "Added N frame(s). Skipped M duplicate(s).").

## Testing

- Unit test for `detectAndLoadFrames` in `packages/gbcam-extract-web/src/utils/detectFrames.test.ts`:
  - Sheet input → multiple frames returned.
  - Individual input → single frame returned.
  - Unrecognisable input → throws.
- Unit test for the PNG encode/decode round-trip in `useUserFrames` (or a sibling `frameCodec.ts` helper).
- Manual smoke test in `pnpm dev`:
  - Upload an existing built-in frame file → "skipped duplicate" toast.
  - Upload an unrelated PNG → success, custom tile appears, can be selected, applied, and deleted.

## Out-of-scope follow-ups (intentionally deferred)

- Renaming custom frames.
- Exporting / importing custom frame collections.
- Drag-and-drop upload (stick to button-triggered file input for v1).
- Showing storage usage in the UI.
