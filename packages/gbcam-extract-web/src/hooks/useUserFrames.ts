import { useCallback, useEffect, useState } from "react";
import type { Frame } from "gbcam-extract";
import { useLocalStorage } from "./useLocalStorage.js";
import {
  frameToPngDataUrl,
  pngDataUrlToFrame,
  type UserFrameEntry,
} from "./frameCodec.js";

const STORAGE_KEY = "gbcam-user-frames";
const STORAGE_VERSION = "1";
const STORAGE_VERSION_KEY = "gbcam-user-frames-version";

export type UserFramesStatus = "loading" | "ready" | "error";

export interface UseUserFramesResult {
  entries: UserFrameEntry[];
  decodedFrames: Frame[];
  status: UserFramesStatus;
  /**
   * Encode the given frames and append them to storage. Dedup is the caller's
   * responsibility — this hook is storage-only. Returns counts so the caller
   * can summarise the batch in a toast.
   *
   * Each frame's `Frame.kind` is forced to `"individual"` and `aliasStems` is
   * trimmed to `[sheetStem]` before encoding (sheet-origin metadata isn't
   * useful once the frame is decoupled from its sheet).
   */
  addFrames(frames: Frame[]): { added: number };
  deleteFrame(id: string): void;
}

function generateId(): string {
  return `user-frame-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

// On module load: if the stored schema version doesn't match, drop stored
// frames so useLocalStorage falls back to its initial value. Then write the
// current version so future loads round-trip cleanly. Mirrors the pattern in
// useUserPalettes.
try {
  if (localStorage.getItem(STORAGE_VERSION_KEY) !== STORAGE_VERSION) {
    localStorage.removeItem(STORAGE_KEY);
  }
  localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
} catch {
  // localStorage may be unavailable (private mode, etc.) — ignore.
}

/**
 * Persists user-uploaded frames to localStorage and provides a decoded
 * `Frame[]` view for the catalog. The hook is storage-only: callers must
 * dedup against the existing catalog before calling `addFrames`.
 *
 * Decoding individual entries is best-effort — if a stored entry fails to
 * decode (corrupt data URL, etc.) we log a warning and skip it rather than
 * surfacing an error, so a single bad entry can't take down the picker.
 */
export function useUserFrames(): UseUserFramesResult {
  const [entries, setEntries] = useLocalStorage<UserFrameEntry[]>(
    STORAGE_KEY,
    [],
  );
  const [decodedFrames, setDecodedFrames] = useState<Frame[]>([]);
  const [status, setStatus] = useState<UserFramesStatus>(
    entries.length === 0 ? "ready" : "loading",
  );

  // Decode entries to Frames whenever the persisted list changes. Each entry
  // requires an `<img>` round-trip, so this is genuinely async.
  useEffect(() => {
    let cancelled = false;
    if (entries.length === 0) {
      setDecodedFrames([]);
      setStatus("ready");
      return () => {
        cancelled = true;
      };
    }
    setStatus("loading");
    Promise.allSettled(entries.map((e) => pngDataUrlToFrame(e)))
      .then((results) => {
        if (cancelled) return;
        const frames: Frame[] = [];
        for (let i = 0; i < results.length; i++) {
          const r = results[i];
          if (r.status === "fulfilled") {
            frames.push(r.value);
          } else {
            console.warn(
              `useUserFrames: failed to decode entry ${entries[i].id}`,
              r.reason,
            );
          }
        }
        setDecodedFrames(frames);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("useUserFrames: decode batch failed", err);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [entries]);

  const addFrames = useCallback(
    (frames: Frame[]): { added: number } => {
      if (frames.length === 0) return { added: 0 };
      const newEntries: UserFrameEntry[] = [];
      for (const f of frames) {
        try {
          newEntries.push({
            id: generateId(),
            sheetStem: f.sheetStem,
            type: f.type,
            width: f.width,
            height: f.height,
            holeX: f.holeX,
            holeY: f.holeY,
            pngDataUrl: frameToPngDataUrl(f),
            addedAt: Date.now(),
          });
        } catch (err) {
          console.error(
            `useUserFrames: failed to encode frame ${f.sheetStem}`,
            err,
          );
        }
      }
      if (newEntries.length === 0) return { added: 0 };

      // Persist with rollback on quota errors. The useLocalStorage helper
      // swallows write errors silently, so we drive the write directly here
      // to surface failures and roll back the in-memory state.
      const previous = entries;
      const next = [...previous, ...newEntries];
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        setEntries(next);
        return { added: newEntries.length };
      } catch (err) {
        // Restore on-disk state if a partial write occurred and surface the
        // error to the caller via a thrown exception.
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(previous));
        } catch {
          // ignore secondary failure
        }
        throw err;
      }
    },
    [entries, setEntries],
  );

  const deleteFrame = useCallback(
    (id: string) => {
      setEntries((prev) => prev.filter((e) => e.id !== id));
    },
    [setEntries],
  );

  return { entries, decodedFrames, status, addFrames, deleteFrame };
}
