import { useCallback, useEffect, useState } from "react";
import type { Frame } from "gbcam-extract";
import { useLocalStorage } from "./useLocalStorage.js";
import {
  frameToPngDataUrl,
  pngDataUrlToFrame,
  type UserFrameEntry,
} from "./frameCodec.js";

/** Calculate a fast fingerprint (hash) of the pixel data to identify duplicates. */
export function frameFingerprint(frame: Frame): string {
  const bytes = new Uint8Array(frame.pixels.buffer, frame.pixels.byteOffset, frame.pixels.byteLength);
  let h = 0x811c9dc5; // FNV-1a 32-bit offset basis
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193); // FNV-1a 32-bit prime
  }
  return (h >>> 0).toString(16);
}

const STORAGE_KEY = "gbcam-original-frames";
const STORAGE_VERSION = "1";
const STORAGE_VERSION_KEY = "gbcam-original-frames-version";

export type OriginalFramesStatus = "loading" | "ready" | "error";

export interface UseOriginalFramesResult {
  entries: UserFrameEntry[];
  decodedFrames: Frame[];
  status: OriginalFramesStatus;
  addFrames(frames: Frame[]): { added: number };
}

function generateId(): string {
  return `original-frame-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

try {
  if (localStorage.getItem(STORAGE_VERSION_KEY) !== STORAGE_VERSION) {
    localStorage.removeItem(STORAGE_KEY);
  }
  localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
} catch {
  // localStorage may be unavailable
}

export function useOriginalFrames(): UseOriginalFramesResult {
  const [entries, setEntries] = useLocalStorage<UserFrameEntry[]>(
    STORAGE_KEY,
    [],
  );
  const [decodedFrames, setDecodedFrames] = useState<Frame[]>([]);
  const [status, setStatus] = useState<OriginalFramesStatus>(
    entries.length === 0 ? "ready" : "loading",
  );

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
              `useOriginalFrames: failed to decode entry ${entries[i].id}`,
              r.reason,
            );
          }
        }
        setDecodedFrames(frames);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("useOriginalFrames: decode batch failed", err);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [entries]);

  const addFrames = useCallback(
    (newFrames: Frame[]): { added: number } => {
      if (newFrames.length === 0) return { added: 0 };

      // Dedupe against existing frames *from the same sheet* only. A frame
      // shared between the USA and JPN sheets is pixel-identical but must be
      // stored under both stems so the catalog can mark it as shared
      // (see useFrameCatalog). Keying on `sheetStem:fingerprint` blocks
      // re-uploading the same sheet while preserving cross-region duplicates.
      const key = (f: Frame) => `${f.sheetStem}:${frameFingerprint(f)}`;
      const existingKeys = new Set(decodedFrames.map(key));
      const seen = new Set<string>();
      const uniqueNew = newFrames.filter((f) => {
        const k = key(f);
        if (existingKeys.has(k) || seen.has(k)) return false;
        seen.add(k);
        return true;
      });
      if (uniqueNew.length === 0) return { added: 0 };

      const newEntries: UserFrameEntry[] = [];
      for (const f of uniqueNew) {
        try {
          newEntries.push({
            id: generateId(),
            sheetStem: f.sheetStem,
            aliasStems: f.aliasStems,
            type: f.type,
            kind: f.kind,
            index: f.index,
            width: f.width,
            height: f.height,
            holeX: f.holeX,
            holeY: f.holeY,
            pngDataUrl: frameToPngDataUrl(f),
            addedAt: Date.now(),
          });
        } catch (err) {
          console.error(
            `useOriginalFrames: failed to encode frame ${f.sheetStem}`,
            err,
          );
        }
      }
      if (newEntries.length === 0) return { added: 0 };

      const previous = entries;
      const next = [...previous, ...newEntries];
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        setEntries(next);
        return { added: newEntries.length };
      } catch (err) {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(previous));
        } catch {
          // ignore
        }
        throw err;
      }
    },
    [entries, setEntries],
  );

  return { entries, decodedFrames, status, addFrames };
}
