import { useState, useCallback, useEffect } from "react";
import type { PipelineResult } from "gbcam-extract";
import {
  serializePipelineResult,
  deserializePipelineResult,
  isSerializedPipelineResult,
} from "../utils/serialization.js";
import { useLocalStorage } from "./useLocalStorage.js";
import type { ProcessingResult } from "./useProcessing.js";
import type { FrameSelection } from "../types/frame-selection.js";

export interface ImageHistoryBatch {
  id: string;
  timestamp: number;
  results: ProcessingResult[];
}

export interface HistorySettings {
  maxSize: number;
}

const HISTORY_STORAGE_KEY = "gbcam-image-history";
const HISTORY_SETTINGS_KEY = "gbcam-history-settings";
const DEFAULT_MAX_SIZE = 10;
const MAX_BATCH_SIZE = 10; // Don't store more than 10 images per batch

function generateId(): string {
  return `batch-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Serialized form on disk (each result has a serialized PipelineResult).
type SerializedHistory = Array<{
  id: string;
  timestamp: number;
  results: Array<{
    filename: string;
    processingTime: number;
    result: unknown;
  }>;
}>;

async function deserializeHistoryBatches(
  raw: SerializedHistory,
): Promise<ImageHistoryBatch[]> {
  return Promise.all(
    raw.map(async (batch) => ({
      ...batch,
      results: await Promise.all(
        batch.results.map(async (item) => ({
          ...item,
          result: isSerializedPipelineResult(item.result)
            ? await deserializePipelineResult(item.result)
            : (item.result as PipelineResult),
        })),
      ),
    })),
  );
}

function serializeHistoryBatches(
  history: ImageHistoryBatch[],
): SerializedHistory {
  return history.map((batch) => ({
    ...batch,
    results: batch.results.map((item) => ({
      ...item,
      result: serializePipelineResult(item.result),
    })),
  }));
}

export function useImageHistory() {
  // Raw serialized form persisted via the localStorage hook.
  const [serializedHistory, setSerializedHistory] =
    useLocalStorage<SerializedHistory>(HISTORY_STORAGE_KEY, []);
  const [settings, setSettings] = useLocalStorage<HistorySettings>(
    HISTORY_SETTINGS_KEY,
    { maxSize: DEFAULT_MAX_SIZE },
  );

  // Deserialized in-memory form (async deserialize on mount).
  const [history, setHistory] = useState<ImageHistoryBatch[]>([]);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(false);

  useEffect(() => {
    let mounted = true;
    deserializeHistoryBatches(serializedHistory)
      .then((deserialized) => {
        if (mounted) {
          setHistory(deserialized);
          setIsHistoryLoaded(true);
        }
      })
      .catch(() => {
        if (mounted) {
          setHistory([]);
          setIsHistoryLoaded(true);
          setSerializedHistory([]);
        }
      });
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only re-deserialize on mount; in-memory updates flow the other way.

  // Re-serialize and persist whenever in-memory history changes after load.
  useEffect(() => {
    if (!isHistoryLoaded) return;
    setSerializedHistory(serializeHistoryBatches(history));
  }, [history, isHistoryLoaded, setSerializedHistory]);

  // Add current results to history (moves them from current to history)
  const archiveResults = useCallback(
    (results: ProcessingResult[]) => {
      if (results.length === 0) return;

      const newBatch: ImageHistoryBatch = {
        id: generateId(),
        timestamp: Date.now(),
        results: results.slice(0, MAX_BATCH_SIZE), // Limit batch size
      };

      setHistory((prev) => {
        let updated = [newBatch, ...prev];

        // Calculate total number of images
        let totalImages = updated.reduce(
          (sum, batch) => sum + batch.results.length,
          0,
        );

        // Remove oldest batches if total exceeds max size
        while (totalImages > settings.maxSize && updated.length > 0) {
          const lastBatch = updated[updated.length - 1];
          totalImages -= lastBatch.results.length;
          updated = updated.slice(0, -1);
        }

        return updated;
      });
    },
    [settings],
  );

  // Delete a specific result from history
  const deleteFromHistory = useCallback(
    (batchId: string, resultIndex: number) => {
      setHistory(
        (prev) =>
          prev
            .map((batch) => {
              if (batch.id === batchId) {
                return {
                  ...batch,
                  results: batch.results.filter((_, i) => i !== resultIndex),
                };
              }
              return batch;
            })
            .filter((batch) => batch.results.length > 0), // Remove empty batches
      );
    },
    [],
  );

  // Delete all results from a specific batch
  const deleteBatch = useCallback((batchId: string) => {
    setHistory((prev) => prev.filter((batch) => batch.id !== batchId));
  }, []);

  // Delete all history
  const deleteAllHistory = useCallback(() => {
    setHistory([]);
  }, []);

  // Update history settings
  const updateSettings = useCallback(
    (newSettings: Partial<HistorySettings>) => {
      setSettings((prev) => ({ ...prev, ...newSettings }));
    },
    [setSettings],
  );

  // Prune history based on current max size
  const pruneHistory = useCallback(() => {
    setHistory((prev) => {
      let updated = [...prev];
      let totalImages = updated.reduce(
        (sum, batch) => sum + batch.results.length,
        0,
      );

      while (totalImages > settings.maxSize && updated.length > 0) {
        const lastBatch = updated[updated.length - 1];
        totalImages -= lastBatch.results.length;
        updated = updated.slice(0, -1);
      }

      return updated;
    });
  }, [settings]);

  const updateFrameOverride = useCallback(
    (batchId: string, resultIndex: number, override: FrameSelection) => {
      setHistory((prev) =>
        prev.map((batch) =>
          batch.id === batchId
            ? {
                ...batch,
                results: batch.results.map((r, i) =>
                  i === resultIndex ? { ...r, frameOverride: override } : r,
                ),
              }
            : batch,
        ),
      );
    },
    [],
  );

  return {
    history,
    settings,
    isHistoryExpanded,
    setIsHistoryExpanded,
    archiveResults,
    deleteFromHistory,
    deleteBatch,
    deleteAllHistory,
    updateSettings,
    pruneHistory,
    updateFrameOverride,
  };
}
