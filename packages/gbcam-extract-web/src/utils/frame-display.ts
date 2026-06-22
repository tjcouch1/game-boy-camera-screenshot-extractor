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

/**
 * Human-readable name for a frame, used in the picker UI and download
 * filenames.
 *
 * Regional (USA/JPN) frames are tagged by which region(s) they belong to,
 * derived purely from the frame's stems — order-independent and not reliant on
 * the rest of the catalog:
 *   - `Frame 3` — shared between USA and JPN (its `aliasStems` cover both)
 *   - `Frame 3 (USA)` — exclusive to the USA sheet
 *   - `Frame 3 (JPN)` — exclusive to the JPN sheet
 *   - `Standard matrix` — individual frame; just the cleaned file stem
 */
export function frameDisplayName(frame: Frame): string {
  if (frame.kind === "individual") return prettifyStem(frame.sheetStem);
  const prefix = frame.type === "wild" ? "Wild Frame" : "Frame";

  const stems = new Set([frame.sheetStem, ...frame.aliasStems]);
  const inUsa = stems.has("Frames_USA");
  const inJpn = stems.has("Frames_JPN");

  // Shared between both regions → no tag. Exclusive to one → that region's tag.
  if (inUsa && inJpn) return `${prefix} ${frame.index}`;
  if (inUsa) return `${prefix} ${frame.index} (USA)`;
  if (inJpn) return `${prefix} ${frame.index} (JPN)`;

  // Non-regional sheet (e.g. a multi-frame custom upload): disambiguate by the
  // sheet's own region tag/stem.
  return `${prefix} ${frame.index} (${regionFromStem(frame.sheetStem)})`;
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
