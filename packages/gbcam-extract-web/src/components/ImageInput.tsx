import { useRef, useState, useCallback } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/shadcn/components/button";
import { cn } from "@/shadcn/utils/utils";
import { SamplePicturePicker } from "./SamplePicturePicker.js";

interface ImageInputProps {
  onImagesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export function ImageInput({ onImagesSelected, disabled }: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const imageFiles = Array.from(files).filter((f) =>
        f.type.startsWith("image/"),
      );
      if (imageFiles.length > 0) {
        onImagesSelected(imageFiles);
      }
    },
    [onImagesSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      handleFiles(e.dataTransfer.files);
    },
    [disabled, handleFiles],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setDragOver(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  return (
    <div>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={cn(
          "rounded-lg border-2 border-dashed p-8 text-center transition-colors",
          disabled
            ? "border-border text-muted-foreground/50 cursor-not-allowed"
            : dragOver
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-muted-foreground hover:border-foreground/40 cursor-pointer",
        )}
      >
        <div className="flex flex-col items-center gap-3">
          <Upload className="opacity-50 size-10" />
          <p className="text-sm">Drag and drop images here, or click to browse</p>
          <p className="text-xs text-muted-foreground">Supports multiple files</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 mt-3">
        <Button
          variant="secondary"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
        >
          Choose Files
        </Button>
        <Button
          variant="secondary"
          onClick={() => cameraInputRef.current?.click()}
          disabled={disabled}
        >
          Camera Capture
        </Button>
        <SamplePicturePicker
          onImagesSelected={onImagesSelected}
          disabled={disabled}
        />
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
