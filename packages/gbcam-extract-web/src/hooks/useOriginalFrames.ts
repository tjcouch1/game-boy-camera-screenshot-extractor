import { useCallback, useEffect, useState } from "react";
import type { Frame } from "gbcam-extract";
import { useLocalStorage } from "./useLocalStorage.js";
import {
  frameToPngDataUrl,
  pngDataUrlToFrame,
  type UserFrameEntry,
} from "./frameCodec.js";

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
    (frames: Frame[]): { added: number } => {
      if (frames.length === 0) return { added: 0 };
      const newEntries: UserFrameEntry[] = [];
      for (const f of frames) {
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
