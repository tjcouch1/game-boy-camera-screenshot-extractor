/**
 * A frame choice. The discriminated union keeps the per-result override
 * semantics clean: "default" means follow the global default, "none" is
 * an explicit "no frame" override, and "frame" pins a specific frame ID.
 *
 * Persisted shape — only `id` is stored, never pixel data.
 */
export type FrameSelection =
  | { kind: "default" }
  | { kind: "none" }
  | { kind: "frame"; id: string };

/** Default for new per-result overrides. */
export const FRAME_SELECTION_DEFAULT: FrameSelection = { kind: "default" };

/** Default for the global default-frame setting (no global frame). */
export const FRAME_SELECTION_NONE: FrameSelection = { kind: "none" };
