import { memo, useEffect, useRef } from "react";
import type { Frame } from "gbcam-extract";
import { X, TriangleAlert } from "lucide-react";
import { Button } from "@/shadcn/components/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/shadcn/components/tooltip";
import { cn } from "@/shadcn/utils/utils";
import type { HistoryItem } from "../hooks/useImageHistory.js";
import { buildOutputCanvas } from "../utils/buildOutputCanvas.js";

interface HistoryGridProps {
  items: HistoryItem[];
  /** Ids of items currently open as result cards at the top. */
  currentIds: string[];
  palette: [string, string, string, string];
  /** Resolve an item's effective frame (already includes the global default). */
  resolveFrame: (item: HistoryItem) => Frame | null;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

const HistoryThumbnail = memo(function HistoryThumbnail({
  item,
  palette,
  frame,
  isOpen,
  onOpen,
  onDelete,
}: {
  item: HistoryItem;
  palette: [string, string, string, string];
  frame: Frame | null;
  isOpen: boolean;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hasIssues = !!item.result.issues?.length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rendered = buildOutputCanvas(item.result, palette, frame, 1);
    if (!rendered) return;
    canvas.width = rendered.width;
    canvas.height = rendered.height;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(rendered, 0, 0);
  }, [item.result, palette, frame]);

  return (
    <div className="group relative">
      <Tooltip>
        <TooltipTrigger
          render={
            <button
              type="button"
              onClick={() => onOpen(item.id)}
              className={cn(
                "block w-full rounded border bg-muted/40 p-1 transition-colors",
                "hover:border-primary/60 focus-visible:outline-2 focus-visible:outline-ring",
                isOpen && "border-primary ring-2 ring-primary/40",
              )}
              aria-label={`Open ${item.filename}`}
            />
          }
        >
          <canvas
            ref={canvasRef}
            className="w-full h-auto rounded-sm"
            style={{ imageRendering: "pixelated" }}
          />
        </TooltipTrigger>
        <TooltipContent>
          <p>{item.filename}</p>
          <p className="text-muted-foreground">
            {new Date(item.timestamp).toLocaleString()}
          </p>
        </TooltipContent>
      </Tooltip>
      {hasIssues && (
        <TriangleAlert
          className="absolute start-1.5 top-1.5 size-4 text-warning pointer-events-none"
          aria-label="Possible processing quality issues"
        />
      )}
      <Button
        variant="destructive"
        size="icon"
        onClick={() => onDelete(item.id)}
        aria-label={`Delete ${item.filename} from history`}
        className={cn(
          "absolute end-1 top-1 size-5 opacity-0 transition-opacity",
          "group-hover:opacity-100 focus-visible:opacity-100",
          // Touch devices have no hover — keep the delete button visible.
          "pointer-coarse:opacity-100",
        )}
      >
        <X />
      </Button>
    </div>
  );
});

/**
 * Grid of small previews of every image in history (rendered with the
 * active palette and each image's effective frame). Clicking a preview
 * opens that image as a result card at the top of the page.
 */
export function HistoryGrid({
  items,
  currentIds,
  palette,
  resolveFrame,
  onOpen,
  onDelete,
}: HistoryGridProps) {
  const current = new Set(currentIds);
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(6.5rem,1fr))] gap-2">
      {items.map((item) => (
        <HistoryThumbnail
          key={item.id}
          item={item}
          palette={palette}
          frame={resolveFrame(item)}
          isOpen={current.has(item.id)}
          onOpen={onOpen}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}
