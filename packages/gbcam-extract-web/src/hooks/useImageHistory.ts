import { useState, useCallback, useEffect, useMemo } from "react";
import type { PipelineResult } from "gbcam-extract";
import {
  serializePipelineResult,
  deserializePipelineResult,
  isSerializedPipelineResult,
} from "../utils/serialization.js";
import { useLocalStorage } from "./useLocalStorage.js";
import type { FrameSelection } from "../types/frame-selection.js";

/**
 * A single processed image. This is the app's single source of truth for
 * every image — both the "current results" shown at the top of the page and
 * the history grid render (subsets of) the same item list, so an update made
 * from either place is reflected in both.
 */
export interface HistoryItem {
  id: string;
  timestamp: number;
  filename: string;
  processingTime: number;
  result: PipelineResult;
  /** Per-image frame override. Undefined = follow global default. */
  frameOverride?: FrameSelection;
  /**
   * Whether the user collapsed this result's processing-quality warning.
   * Persisted so the warning stays collapsed across page reloads.
   */
  warningCollapsed?: boolean;
}

/** The fields needed to add a freshly processed image to the store. */
export interface ProcessedImage {
  result: PipelineResult;
  filename: string;
  processingTime: number;
}

export interface HistorySettings {
  maxSize: number;
}

const HISTORY_STORAGE_KEY = "gbcam-image-history";
const HISTORY_SETTINGS_KEY = "gbcam-history-settings";
const LEGACY_CURRENT_RESULTS_KEY = "gbcam-current-results";
const DEFAULT_MAX_SIZE = 10;

function generateId(): string {
  return `img-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

// ─── Storage schema ───

interface SerializedHistoryItem {
  id: string;
  timestamp: number;
  filename: string;
  processingTime: number;
  result: unknown;
  frameOverride?: FrameSelection;
  warningCollapsed?: boolean;
}

/** v2 storage: flat item list plus the ids currently open at the top. */
interface StoredHistoryV2 {
  version: 2;
  items: SerializedHistoryItem[];
  currentIds: string[];
}

/** v1 (legacy) storage: array of batches of results. */
type LegacyStoredHistory = Array<{
  id: string;
  timestamp: number;
  results: Array<{
    filename: string;
    processingTime: number;
    result: unknown;
    frameOverride?: FrameSelection;
    warningCollapsed?: boolean;
  }>;
}>;

type StoredHistory = StoredHistoryV2 | LegacyStoredHistory;

/** In-memory store: the deserialized items plus the ids open at the top. */
interface HistoryStore {
  items: HistoryItem[];
  currentIds: string[];
}

/** Build a SerializedHistoryItem from a legacy result record. */
function toSerializedItem(
  r: {
    filename: string;
    processingTime: number;
    result: unknown;
    frameOverride?: FrameSelection;
    warningCollapsed?: boolean;
  },
  id: string,
  timestamp: number,
): SerializedHistoryItem {
  return {
    id,
    timestamp,
    filename: r.filename,
    processingTime: r.processingTime,
    result: r.result,
    ...(r.frameOverride ? { frameOverride: r.frameOverride } : {}),
    ...(r.warningCollapsed !== undefined
      ? { warningCollapsed: r.warningCollapsed }
      : {}),
  };
}

/**
 * Normalize whatever was on disk into v2 shape. Handles:
 * - v2 object storage (returned as-is),
 * - legacy batch arrays (flattened, newest batch first),
 * - the legacy separate "current results" key (prepended as newest items and
 *   marked current, preserving the pre-migration view — checked regardless of
 *   the history key's shape, since a legacy user may have current results but
 *   no history).
 *
 * Pure with respect to localStorage: the caller is responsible for persisting
 * the result and retiring the legacy key (in that order, so an interruption
 * between the two can't lose data).
 */
function migrateStored(stored: unknown): {
  v2: StoredHistoryV2;
  changed: boolean;
} {
  let items: SerializedHistoryItem[];
  let currentIds: string[];
  let changed: boolean;

  if (Array.isArray(stored)) {
    const legacy = stored as LegacyStoredHistory;
    items = legacy.flatMap((batch) =>
      batch.results.map((r, i) =>
        toSerializedItem(r, `${batch.id}-${i}`, batch.timestamp),
      ),
    );
    currentIds = [];
    changed = true;
  } else if (
    stored &&
    typeof stored === "object" &&
    (stored as StoredHistoryV2).version === 2
  ) {
    const v2 = stored as StoredHistoryV2;
    items = v2.items;
    currentIds = v2.currentIds;
    changed = false;
  } else {
    items = [];
    currentIds = [];
    changed = false;
  }

  try {
    const rawCurrent = localStorage.getItem(LEGACY_CURRENT_RESULTS_KEY);
    if (rawCurrent) {
      const parsed = JSON.parse(rawCurrent) as Array<{
        filename: string;
        processingTime: number;
        result: unknown;
        frameOverride?: FrameSelection;
        warningCollapsed?: boolean;
      }>;
      const now = Date.now();
      const migrated = parsed.map((r) =>
        toSerializedItem(r, generateId(), now),
      );
      items = [...migrated, ...items];
      currentIds = [...currentIds, ...migrated.map((m) => m.id)];
      changed = true;
    }
  } catch {
    // Ignore a corrupt legacy key — history still migrates.
  }

  return { v2: { version: 2, items, currentIds }, changed };
}

/**
 * Serialized results keyed by the deserialized PipelineResult object. Results
 * are immutable once created (item patches spread the item, never the
 * result), so caching lets the persist effect re-encode only results it has
 * never seen instead of PNG-encoding every stored image on every change.
 */
const serializedResultCache = new WeakMap<PipelineResult, unknown>();

function serializeResultCached(result: PipelineResult): unknown {
  let serialized = serializedResultCache.get(result);
  if (!serialized) {
    serialized = serializePipelineResult(result);
    serializedResultCache.set(result, serialized);
  }
  return serialized;
}

/**
 * Deserialize stored items, dropping (only) items that fail to decode so one
 * corrupt entry can't take the whole history down with it.
 */
async function deserializeItems(
  stored: SerializedHistoryItem[],
): Promise<HistoryItem[]> {
  const results = await Promise.all(
    stored.map(async (item): Promise<HistoryItem | null> => {
      try {
        let result: PipelineResult;
        if (isSerializedPipelineResult(item.result)) {
          result = await deserializePipelineResult(item.result);
          // Seed the cache so loaded items are never re-encoded on persist.
          serializedResultCache.set(result, item.result);
        } else {
          result = item.result as PipelineResult;
        }
        return { ...item, result };
      } catch {
        return null;
      }
    }),
  );
  return results.filter((item): item is HistoryItem => item !== null);
}

function serializeItems(items: HistoryItem[]): SerializedHistoryItem[] {
  return items.map((item) => ({
    ...item,
    result: serializeResultCached(item.result),
  }));
}

/**
 * Drop the oldest non-current items until the store fits `maxSize`. Items
 * currently open at the top are never pruned out from under the user.
 */
function pruneItems(
  items: HistoryItem[],
  currentIds: string[],
  maxSize: number,
): HistoryItem[] {
  if (items.length <= maxSize) return items;
  const current = new Set(currentIds);
  const removable = items.length - maxSize;
  let removed = 0;
  const kept: HistoryItem[] = [];
  // Items are stored newest-first, so walk from the end (oldest) backwards.
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (removed < removable && !current.has(item.id)) {
      removed++;
      continue;
    }
    kept.unshift(item);
  }
  return kept;
}

export function useImageHistory() {
  const [, setStored] = useLocalStorage<StoredHistory>(HISTORY_STORAGE_KEY, {
    version: 2,
    items: [],
    currentIds: [],
  });
  const [settings, setSettings] = useLocalStorage<HistorySettings>(
    HISTORY_SETTINGS_KEY,
    { maxSize: DEFAULT_MAX_SIZE },
  );

  // Deserialized in-memory store (async deserialize on mount).
  const [store, setStore] = useState<HistoryStore>({
    items: [],
    currentIds: [],
  });
  const [isLoaded, setIsLoaded] = useState(false);
  const [isHistoryExpanded, setIsHistoryExpanded] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      let v2: StoredHistoryV2 = { version: 2, items: [], currentIds: [] };
      try {
        // Read fresh from localStorage (rather than the hook's mount-time
        // snapshot) so the migration is idempotent under StrictMode's
        // double-mount: the second run sees the v2 store the first run wrote.
        const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
        const migrated = migrateStored(raw ? JSON.parse(raw) : null);
        v2 = migrated.v2;
        if (migrated.changed) {
          // Persist the migrated store BEFORE retiring the legacy key so an
          // interruption between the two can't lose data.
          setStored(v2);
          try {
            localStorage.removeItem(LEGACY_CURRENT_RESULTS_KEY);
          } catch {
            // Ignore — worst case the retired key lingers.
          }
        }
      } catch {
        // Corrupt storage: start empty in memory but leave the stored value
        // untouched rather than overwriting it with an empty store.
      }
      const loaded = await deserializeItems(v2.items);
      if (!mounted) return;
      setStore((prev) => {
        // Merge under anything added (via addResult) while deserialization
        // was in flight instead of clobbering it.
        const prevIds = new Set(prev.items.map((item) => item.id));
        const items = [
          ...prev.items,
          ...loaded.filter((item) => !prevIds.has(item.id)),
        ];
        const ids = new Set(items.map((item) => item.id));
        const currentIds = [
          ...prev.currentIds,
          ...v2.currentIds.filter((id) => !prev.currentIds.includes(id)),
        ].filter((id) => ids.has(id));
        return { items, currentIds };
      });
      setIsLoaded(true);
    })();
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only deserialize on mount; in-memory updates flow the other way.

  // Re-serialize and persist whenever the in-memory store changes after load.
  useEffect(() => {
    if (!isLoaded) return;
    setStored({
      version: 2,
      items: serializeItems(store.items),
      currentIds: store.currentIds,
    });
  }, [store, isLoaded, setStored]);

  const { items, currentIds } = store;

  const itemsById = useMemo(
    () => new Map(items.map((item) => [item.id, item])),
    [items],
  );

  /** The items currently open as result cards at the top, in display order. */
  const currentItems = useMemo(
    () =>
      currentIds
        .map((id) => itemsById.get(id))
        .filter((item): item is HistoryItem => item !== undefined),
    [currentIds, itemsById],
  );

  /** Add a freshly processed image to history and open it at the top. */
  const addResult = useCallback(
    (processed: ProcessedImage) => {
      const item: HistoryItem = {
        id: generateId(),
        timestamp: Date.now(),
        ...processed,
      };
      setStore((prev) => {
        const currentIdsNext = [...prev.currentIds, item.id];
        return {
          items: pruneItems(
            [item, ...prev.items],
            currentIdsNext,
            settings.maxSize,
          ),
          currentIds: currentIdsNext,
        };
      });
    },
    [settings.maxSize],
  );

  /** Close all result cards at the top (items stay in history). */
  const clearCurrent = useCallback(() => {
    setStore((prev) => ({
      // Closed items lose their prune protection, so re-enforce maxSize.
      items: pruneItems(prev.items, [], settings.maxSize),
      currentIds: [],
    }));
  }, [settings.maxSize]);

  /** Open a history item as a result card at the top (moves it first). */
  const openItem = useCallback((id: string) => {
    setStore((prev) => ({
      ...prev,
      currentIds: [id, ...prev.currentIds.filter((x) => x !== id)],
    }));
  }, []);

  /** Close one result card at the top (the item stays in history). */
  const closeItem = useCallback(
    (id: string) => {
      setStore((prev) => {
        const currentIds = prev.currentIds.filter((x) => x !== id);
        return {
          // Closed items lose their prune protection, so re-enforce maxSize.
          items: pruneItems(prev.items, currentIds, settings.maxSize),
          currentIds,
        };
      });
    },
    [settings.maxSize],
  );

  /** Permanently delete an item from history (and the top, if open). */
  const deleteItem = useCallback((id: string) => {
    setStore((prev) => ({
      items: prev.items.filter((item) => item.id !== id),
      currentIds: prev.currentIds.filter((x) => x !== id),
    }));
  }, []);

  const deleteAllHistory = useCallback(() => {
    setStore({ items: [], currentIds: [] });
  }, []);

  /** Patch an item's user-editable fields (frame override, warning state). */
  const updateItem = useCallback(
    (
      id: string,
      patch: Partial<Pick<HistoryItem, "frameOverride" | "warningCollapsed">>,
    ) => {
      setStore((prev) => ({
        ...prev,
        items: prev.items.map((item) =>
          item.id === id ? { ...item, ...patch } : item,
        ),
      }));
    },
    [],
  );

  const updateSettings = useCallback(
    (newSettings: Partial<HistorySettings>) => {
      setSettings((prev) => ({ ...prev, ...newSettings }));
      if (newSettings.maxSize !== undefined) {
        const maxSize = newSettings.maxSize;
        setStore((prev) => ({
          ...prev,
          items: pruneItems(prev.items, prev.currentIds, maxSize),
        }));
      }
    },
    [setSettings],
  );

  /**
   * Reset any item whose frameOverride points to `frameId` back to
   * `{kind: "default"}` so it follows the global default. Called when a user
   * frame is deleted so stale references don't render as "unknown frame".
   */
  const purgeFrameOverride = useCallback((frameId: string) => {
    setStore((prev) => ({
      ...prev,
      items: prev.items.map((item) =>
        item.frameOverride?.kind === "frame" &&
        item.frameOverride.id === frameId
          ? { ...item, frameOverride: { kind: "default" } }
          : item,
      ),
    }));
  }, []);

  return {
    items,
    currentIds,
    currentItems,
    isLoaded,
    settings,
    isHistoryExpanded,
    setIsHistoryExpanded,
    addResult,
    clearCurrent,
    openItem,
    closeItem,
    deleteItem,
    deleteAllHistory,
    updateItem,
    updateSettings,
    purgeFrameOverride,
  };
}
