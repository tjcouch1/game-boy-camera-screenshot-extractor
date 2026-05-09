import { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/shadcn/components/button";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/shadcn/components/popover";
import { cn } from "@/shadcn/utils/utils";
import { useLocalStorage } from "../hooks/useLocalStorage.js";
import { SAMPLE_PICTURES } from "../generated/SamplePictures.js";

interface SamplePicturePickerProps {
  onImagesSelected: (files: File[]) => void;
  disabled?: boolean;
}

const STORAGE_KEY = "gbcam-sample-picture-selections";

function guessMimeFromName(filename: string): string {
  const ext = filename.toLowerCase().match(/\.[^.]+$/)?.[0];
  switch (ext) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

export function SamplePicturePicker({
  onImagesSelected,
  disabled,
}: SamplePicturePickerProps) {
  const [open, setOpen] = useState(false);
  const [storedSelection, setStoredSelection] = useLocalStorage<
    string[] | null
  >(STORAGE_KEY, null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  const validFilenames = useMemo(
    () => new Set(SAMPLE_PICTURES.map((s) => s.filename)),
    [],
  );

  const effectiveSelected = useMemo(
    () =>
      storedSelection === null
        ? SAMPLE_PICTURES.map((s) => s.filename)
        : storedSelection.filter((f) => validFilenames.has(f)),
    [storedSelection, validFilenames],
  );

  const selectedSet = useMemo(
    () => new Set(effectiveSelected),
    [effectiveSelected],
  );

  if (SAMPLE_PICTURES.length === 0) return null;

  const toggle = (filename: string) => {
    setStoredSelection((prev) => {
      const base =
        prev === null
          ? SAMPLE_PICTURES.map((s) => s.filename)
          : prev.filter((f) => validFilenames.has(f));
      return base.includes(filename)
        ? base.filter((f) => f !== filename)
        : [...base, filename];
    });
  };

  const handleSubmit = async () => {
    const selectedEntries = SAMPLE_PICTURES.filter((s) =>
      selectedSet.has(s.filename),
    );
    if (selectedEntries.length === 0) return;

    setIsSubmitting(true);
    try {
      const settled = await Promise.allSettled(
        selectedEntries.map(async (entry) => {
          const res = await fetch(entry.url);
          if (!res.ok) {
            throw new Error(
              `Failed to fetch ${entry.filename}: ${res.status}`,
            );
          }
          const blob = await res.blob();
          const type = blob.type || guessMimeFromName(entry.filename);
          return new File([blob], entry.filename, { type });
        }),
      );

      const files: File[] = [];
      const errors: string[] = [];
      for (const result of settled) {
        if (result.status === "fulfilled") files.push(result.value);
        else
          errors.push(
            result.reason instanceof Error
              ? result.reason.message
              : String(result.reason),
          );
      }

      if (errors.length > 0) {
        toast.error(
          `Failed to load ${errors.length} sample picture${
            errors.length === 1 ? "" : "s"
          }`,
          { description: errors.join("; ") },
        );
      }

      if (files.length > 0) {
        onImagesSelected(files);
        setOpen(false);
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedCount = effectiveSelected.length;
  const totalCount = SAMPLE_PICTURES.length;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={<Button variant="secondary" disabled={disabled} />}
      >
        Sample Pictures
        <ChevronDown data-icon="inline-end" />
      </PopoverTrigger>
      <PopoverContent
        align="start"
        side="bottom"
        className="flex flex-col w-[min(28rem,90vw)] max-h-[min(80vh,32rem)] gap-2"
      >
        <PopoverHeader>
          <PopoverTitle>Sample pictures</PopoverTitle>
          <PopoverDescription>
            {selectedCount} of {totalCount} selected
          </PopoverDescription>
        </PopoverHeader>
        <div className="flex-1 overflow-y-auto">
          <div className="grid grid-cols-2 gap-2">
            {SAMPLE_PICTURES.map((entry) => {
              const isSelected = selectedSet.has(entry.filename);
              return (
                <button
                  key={entry.filename}
                  type="button"
                  onClick={() => toggle(entry.filename)}
                  aria-pressed={isSelected}
                  aria-label={`Toggle ${entry.filename}`}
                  className={cn(
                    "relative rounded-md border bg-muted p-1 ring-2 ring-transparent transition-colors text-start",
                    isSelected && "ring-primary border-primary",
                  )}
                >
                  <img
                    src={entry.url}
                    loading="lazy"
                    alt=""
                    className="block w-full h-24 object-contain"
                  />
                  <span className="block text-xs text-muted-foreground truncate mt-1">
                    {entry.filename}
                  </span>
                  {isSelected && (
                    <span
                      aria-hidden="true"
                      className="absolute top-1 end-1 inline-flex items-center justify-center rounded-full bg-primary text-primary-foreground size-5"
                    >
                      <Check className="size-3" />
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex items-center justify-end gap-2 pt-2 border-t">
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || selectedCount === 0}
          >
            {isSubmitting
              ? "Loading…"
              : `Process ${selectedCount} picture${
                  selectedCount === 1 ? "" : "s"
                }`}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
