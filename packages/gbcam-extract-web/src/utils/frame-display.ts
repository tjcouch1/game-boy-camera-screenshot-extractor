import type { Frame } from "gbcam-extract";

/**
 * Extract a short region tag from a sheet stem. e.g. "Frames_USA" → "USA",
 * "Frames_JPN" → "JPN". Falls back to the full stem if no trailing tag is
 * found so unknown sheets still surface useful information.
 */
function regionFromStem(stem: string): string {
  const match = stem.match(/_([A-Za-z0-9]+)$/);
  return match ? match[1] : stem;
}

/**
 * Cosmetic clean-up for an individual frame's file stem: replace hyphens
 * with spaces and uppercase the first character. Existing capitalisation in
 * the rest of the stem is preserved so proper-noun-style names (e.g.
 * `wild-megaman-BOICHOT`) keep their casing.
 */
function prettifyStem(stem: string): string {
  const spaced = stem.replace(/-/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Regional sheet stems whose frames participate in USA/JPN tagging. */
function regionStemsOf(frame: Frame): { inUsa: boolean; inJpn: boolean } {
  const stems = new Set([frame.sheetStem, ...frame.aliasStems]);
  return { inUsa: stems.has("Frames_USA"), inJpn: stems.has("Frames_JPN") };
}

/**
 * Human-readable name for a frame, used in the picker UI and download
 * filenames.
 *
 * A region tag is only shown when the two regions actually render the same
 * slot *differently* — i.e. a USA-exclusive frame and a JPN-exclusive frame
 * collide at the same (type, index). Otherwise there's no ambiguity, so no
 * tag:
 *   - `Frame 3` — shared between USA and JPN (identical → one entry), OR only
 *     one region has been uploaded, OR no other-region frame collides here.
 *   - `Frame 3 (USA)` / `Frame 3 (JPN)` — the two regions have a *different*
 *     frame at index 3, so both are tagged to tell them apart.
 *   - `Standard matrix` — individual frame; just the cleaned file stem.
 *
 * `allFrames` is the catalog used to detect cross-region collisions; without
 * it, exclusive frames render untagged.
 */
export function frameDisplayName(frame: Frame, allFrames?: Frame[]): string {
  if (frame.kind === "individual") return prettifyStem(frame.sheetStem);
  const prefix = frame.type === "wild" ? "Wild Frame" : "Frame";

  const { inUsa, inJpn } = regionStemsOf(frame);

  // Shared across both regions → never ambiguous, no tag.
  if (inUsa && inJpn) return `${prefix} ${frame.index}`;

  // Non-regional sheet (e.g. a multi-frame custom upload): disambiguate by the
  // sheet's own region tag/stem.
  if (!inUsa && !inJpn)
    return `${prefix} ${frame.index} (${regionFromStem(frame.sheetStem)})`;

  // Exclusive to one region. Only tag if a *different* regional frame collides
  // at the same (type, index) — that's the only case where USA vs JPN actually
  // diverge. If only one region is present, nothing collides → no tag.
  const collides =
    allFrames?.some((f) => {
      if (f.id === frame.id) return false;
      if (f.type !== frame.type || f.index !== frame.index) return false;
      const other = regionStemsOf(f);
      return other.inUsa || other.inJpn;
    }) ?? false;

  if (!collides) return `${prefix} ${frame.index}`;
  return `${prefix} ${frame.index} (${inUsa ? "USA" : "JPN"})`;
}

/**
 * Sanitize a frame display name for use inside a filename. Mirrors
 * {@link sanitizePaletteName} but kept separate because frame names contain
 * parentheses and spaces that we want to flatten to underscores rather than
 * drop entirely (so "Frame 3 (USA)" → "Frame_3_USA" rather than "Frame3USA").
 */
export function sanitizeFrameName(name: string): string {
  return name
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}
