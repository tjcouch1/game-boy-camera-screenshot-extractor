import { useState } from "react";
import {
  BUTTON_COMBO_PALETTES,
  BG_PRESETS,
  ADDITIONAL_PALETTES,
  FUN_PALETTES_EXPORT,
} from "../data/palettes.js";
import type { PaletteEntry } from "../data/palettes.js";
import {
  useUserPalettes,
  type UserPaletteEntry,
} from "../hooks/useUserPalettes.js";
import { usePaletteSectionState } from "../hooks/usePaletteSectionState.js";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/shadcn/components/accordion";
import { Button } from "@/shadcn/components/button";
import { Card, CardContent } from "@/shadcn/components/card";
import { cn } from "@/shadcn/utils/utils";
import {
  Field,
  FieldDescription,
  FieldLabel,
} from "@/shadcn/components/field";
import { Input } from "@/shadcn/components/input";
import { useClipboardPaletteCheck } from "../hooks/useClipboardPalette.js";
import {
  PALETTE_COLOR_LABELS,
  PALETTE_TEXT_CLASS,
  PALETTE_LABEL_CLASS,
  PALETTE_INPUT_CLASS,
} from "../utils/paletteUI.js";
import {
  writePaletteToClipboard,
  readPaletteFromClipboard,
} from "../utils/paletteClipboard.js";
import { toast } from "sonner";
import {
  ClipboardPaste,
  Copy as CopyIcon,
  Pencil,
  Plus,
} from "lucide-react";

interface PalettePickerProps {
  selected: PaletteEntry;
  onSelectWithName: (entry: PaletteEntry) => void;
  clipboardEnabled?: boolean;
}

function PaletteSwatch({
  entry,
  isSelected,
  doesMatchColors,
  isBuiltIn,
  isEditing,
  onClick,
  onEdit,
}: {
  entry: PaletteEntry | UserPaletteEntry;
  isSelected: boolean;
  doesMatchColors: boolean;
  isBuiltIn: boolean;
  isEditing?: boolean;
  onClick: () => void;
  onEdit?: () => void;
}) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className={cn(
        "justify-start gap-2 h-auto px-2 py-1.5",
        isSelected &&
          "bg-primary/25 hover:bg-primary/35 border-primary inset-ring-2 inset-ring-primary dark:bg-primary/25 dark:hover:bg-primary/35 dark:border-primary",
        doesMatchColors &&
          !isSelected &&
          "bg-primary/10 hover:bg-primary/20 border-primary dark:bg-primary/15 dark:hover:bg-primary/25 dark:border-primary",
      )}
    >
      <div className="flex shrink-0">
        {entry.colors.map((c, i) => (
          <div
            key={i}
            className="size-4 first:rounded-s last:rounded-e"
            style={{ backgroundColor: c }}
          />
        ))}
      </div>
      <span className="truncate">{entry.name}</span>
      {!isBuiltIn && onEdit && (
        <span
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          className="ms-auto text-primary hover:text-primary/80 cursor-pointer"
          title="Edit palette"
        >
          <Pencil className="size-3" />
        </span>
      )}
    </Button>
  );
}

function PaletteSectionItem({
  title,
  entries,
  selected,
  onSelectWithName,
  onEdit,
  isBuiltIn,
}: {
  title: string;
  entries: (PaletteEntry | UserPaletteEntry)[];
  selected: PaletteEntry;
  onSelectWithName: (entry: PaletteEntry) => void;
  onEdit?: (id: string, entry: UserPaletteEntry) => void;
  isBuiltIn?: boolean;
}) {
  if (entries.length === 0) return null;

  return (
    <AccordionItem value={title}>
      <AccordionTrigger>
        {title} ({entries.length})
      </AccordionTrigger>
      <AccordionContent>
        <div className="grid grid-cols-2 gap-1.5 ms-3 sm:grid-cols-3">
          {entries.map((entry, i) => {
            const isUserPalette = "id" in entry;
            const isSelected =
              entry.name === selected.name &&
              entry.colors.every((c, j) => c === selected.colors[j]);
            const doesMatchColors = entry.colors.every(
              (c, j) => c === selected.colors[j],
            );
            const isEditing =
              isUserPalette && "isEditing" in entry && entry.isEditing;

            return (
              <PaletteSwatch
                key={isUserPalette ? (entry as UserPaletteEntry).id : i}
                entry={entry}
                isSelected={isSelected}
                doesMatchColors={doesMatchColors}
                isBuiltIn={!!isBuiltIn}
                isEditing={isEditing}
                onClick={() => {
                  onSelectWithName(
                    "id" in entry
                      ? { name: entry.name, colors: entry.colors }
                      : entry,
                  );
                }}
                onEdit={
                  isUserPalette && onEdit && !isBuiltIn
                    ? () => onEdit(i.toString(), entry as UserPaletteEntry)
                    : undefined
                }
              />
            );
          })}
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}

export function PalettePicker({
  selected,
  onSelectWithName,
  clipboardEnabled = false,
}: PalettePickerProps) {
  const {
    palettes: userPalettes,
    createPaletteInEditMode,
    updatePalette,
    savePalette,
    cancelPaletteEdit,
    deletePalette,
    startEditingPalette,
  } = useUserPalettes();

  const { isExpanded, toggleExpanded } = usePaletteSectionState();
  const { hasClipboardPalette } = useClipboardPaletteCheck(clipboardEnabled);

  const [selectedEditingPaletteId, setSelectedEditingPaletteId] = useState<
    string | undefined
  >();
  const [editingPaletteErrors, setEditingPaletteErrors] = useState<
    Record<string, string>
  >({});
  // Get the currently editing palette (if any)
  const editingPalette =
    selectedEditingPaletteId &&
    userPalettes.find((p) => p.id === selectedEditingPaletteId);

  // Helper: check if a palette is the selected one (by name + colors match)
  const isPaletteSelected = (palette: UserPaletteEntry): boolean => {
    return (
      palette.name === selected.name &&
      palette.colors.every((c, j) => c === selected.colors[j])
    );
  };

  // Validate palette edits
  const validatePaletteName = (id: string, name: string): string => {
    if (!name.trim()) {
      return "Palette name cannot be empty";
    }
    // Check against ALL other palettes (both saved and editing) except the one being validated
    const isDuplicate = userPalettes.some(
      (p) => p.id !== id && p.name.toLowerCase() === name.toLowerCase(),
    );
    if (isDuplicate) {
      return "A palette with this name already exists";
    }
    return "";
  };

  const handleCreateCustom = () => {
    const newPalette = createPaletteInEditMode(selected.name, selected.colors);
    setSelectedEditingPaletteId(newPalette.id);
    // Select the newly created palette using its generated name (e.g. "Down custom 1")
    onSelectWithName({ name: newPalette.name, colors: newPalette.colors });
  };

  const handleCopyPaletteToClipboard = async (palette: UserPaletteEntry) => {
    const success = await writePaletteToClipboard({
      name: palette.name,
      colors: palette.colors,
    });
    if (success) {
      toast.success("Palette copied to clipboard");
    } else {
      toast.error("Copy failed — check browser permissions");
    }
  };

  const handlePastePaletteColors = async (paletteId: string) => {
    const palette = userPalettes.find((p) => p.id === paletteId);
    if (!palette) return;
    const paletteData = await readPaletteFromClipboard();
    if (paletteData) {
      updatePalette(paletteId, { colors: paletteData.colors });
      if (isPaletteSelected(palette)) {
        onSelectWithName({ name: palette.name, colors: paletteData.colors });
      }
      toast.success("Palette colors pasted");
    } else {
      toast.info("Clipboard does not contain a palette");
    }
  };

  const handlePasteNewPalette = async () => {
    const paletteData = await readPaletteFromClipboard();
    if (paletteData) {
      const newPalette = createPaletteInEditMode(
        paletteData.name,
        paletteData.colors,
      );
      setSelectedEditingPaletteId(newPalette.id);
      onSelectWithName({ name: newPalette.name, colors: newPalette.colors });
      toast.success("Palette pasted");
    } else {
      toast.info("Clipboard does not contain a palette");
    }
  };

  const handleStartEdit = (paletteId: string) => {
    startEditingPalette(paletteId);
    setSelectedEditingPaletteId(paletteId);
  };

  const handleSavePalette = (id: string) => {
    const palette = userPalettes.find((p) => p.id === id);
    if (palette) {
      const error = validatePaletteName(id, palette.name);
      if (error) {
        setEditingPaletteErrors((prev) => ({ ...prev, [id]: error }));
        return;
      }
    }
    savePalette(id);
    setSelectedEditingPaletteId(undefined);
    setEditingPaletteErrors((prev) => {
      const updated = { ...prev };
      delete updated[id];
      return updated;
    });
    if (!isExpanded("User Palettes")) {
      toggleExpanded("User Palettes");
    }
  };

  const handleCancelEdit = (id: string) => {
    cancelPaletteEdit(id);
    setSelectedEditingPaletteId(undefined);
    setEditingPaletteErrors((prev) => {
      const updated = { ...prev };
      delete updated[id];
      return updated;
    });
  };

  const handleDeletePalette = (id: string) => {
    deletePalette(id);
    setSelectedEditingPaletteId(undefined);
    setEditingPaletteErrors((prev) => {
      const updated = { ...prev };
      delete updated[id];
      return updated;
    });
  };

  const handlePaletteNameChange = (id: string, newName: string) => {
    const palette = userPalettes.find((p) => p.id === id);
    updatePalette(id, { name: newName });
    const error = validatePaletteName(id, newName);
    setEditingPaletteErrors((prev) => ({
      ...prev,
      [id]: error,
    }));
    // If this palette is currently selected, update the selection to reflect new name
    if (palette && isPaletteSelected(palette)) {
      onSelectWithName({ name: newName, colors: palette.colors });
    }
  };

  const handlePaletteColorChange = (
    id: string,
    colorIndex: number,
    newColor: string,
  ) => {
    const palette = userPalettes.find((p) => p.id === id);
    if (palette) {
      const newColors = [...palette.colors] as [string, string, string, string];
      newColors[colorIndex] = newColor;
      updatePalette(id, { colors: newColors });
      // If this palette is currently selected, update the selection to reflect new colors
      if (isPaletteSelected(palette)) {
        onSelectWithName({ name: palette.name, colors: newColors });
      }
    }
  };

  const handleSelectEditingPalette = (id: string) => {
    const palette = userPalettes.find((p) => p.id === id);
    if (palette) {
      onSelectWithName({ name: palette.name, colors: palette.colors });
      setSelectedEditingPaletteId(id);
    }
  };

  const editingPalettes = userPalettes.filter((p) => p.isEditing);
  const savedUserPalettes = userPalettes.filter((p) => !p.isEditing);

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold">Palette</h2>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex">
            {selected.colors.map((c: string, i: number) => (
              <div
                key={i}
                className="size-5 first:rounded-s last:rounded-e border"
                style={{ backgroundColor: c }}
              />
            ))}
          </div>
          <Button variant="secondary" size="sm" onClick={handleCreateCustom}>
            <Plus data-icon="inline-start" />
            Custom
          </Button>
          {clipboardEnabled && (
            <Button
              variant="secondary"
              size="icon-sm"
              onClick={handlePasteNewPalette}
              disabled={!hasClipboardPalette}
              aria-label={
                hasClipboardPalette
                  ? "Paste palette from clipboard"
                  : "Clipboard does not contain a palette"
              }
            >
              <ClipboardPaste />
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {/* Editing Palettes Section */}
        {editingPalettes.length > 0 && (
          <div>
            <div className="text-sm font-medium text-muted-foreground mb-1 flex items-center gap-1">
              <span className="text-xs">v</span>
              <Pencil className="size-3" />
              EDITING ({editingPalettes.length})
            </div>
            <div className="ms-3 space-y-2">
              {editingPalettes.map((palette) => {
                const isSelected = isPaletteSelected(palette);
                const doesMatchColors = palette.colors.every(
                  (c, i) =>
                    c.toUpperCase() === selected.colors[i].toUpperCase(),
                );
                return (
                  <div
                    key={palette.id}
                    className={cn(
                      "p-3 rounded border-2 cursor-pointer transition-colors",
                      isSelected
                        ? "border-primary bg-primary/20 ring-2 ring-primary"
                        : doesMatchColors
                          ? "border-primary/60 bg-primary/10 hover:bg-primary/20"
                          : "border-border bg-card hover:bg-muted/30",
                    )}
                    onClick={() => {
                      handleSelectEditingPalette(palette.id);
                      setSelectedEditingPaletteId(palette.id);
                    }}
                  >
                    {/* Color pickers */}
                    <div className="flex flex-wrap items-center gap-2 mb-2">
                      {palette.colors.map((c, i) => (
                        <label
                          key={i}
                          className="flex flex-col items-center gap-1"
                        >
                          <input
                            type="color"
                            value={c}
                            onChange={(e) => {
                              e.stopPropagation();
                              handlePaletteColorChange(
                                palette.id,
                                i,
                                e.target.value,
                              );
                            }}
                            className="size-8 cursor-pointer rounded bg-transparent p-0 border border-input"
                          />
                          <span className={PALETTE_LABEL_CLASS}>
                            {PALETTE_COLOR_LABELS[i]}
                          </span>
                        </label>
                      ))}
                      <div className="flex gap-1 ms-auto">
                        {clipboardEnabled && (
                          <>
                            <Button
                              variant="secondary"
                              size="icon-sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCopyPaletteToClipboard(palette);
                              }}
                              aria-label="Copy palette colors to clipboard"
                            >
                              <CopyIcon />
                            </Button>
                            <Button
                              variant="secondary"
                              size="icon-sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handlePastePaletteColors(palette.id);
                              }}
                              disabled={!hasClipboardPalette}
                              aria-label={
                                hasClipboardPalette
                                  ? "Paste palette colors from clipboard"
                                  : "Clipboard does not contain a palette"
                              }
                            >
                              <ClipboardPaste />
                            </Button>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Name input */}
                    <Field
                      className="mb-2"
                      data-invalid={
                        editingPaletteErrors[palette.id] ? true : undefined
                      }
                    >
                      <FieldLabel
                        htmlFor={`palette-name-${palette.id}`}
                        className="sr-only"
                      >
                        Palette name
                      </FieldLabel>
                      <Input
                        id={`palette-name-${palette.id}`}
                        type="text"
                        value={palette.name}
                        placeholder="Palette name"
                        aria-invalid={
                          editingPaletteErrors[palette.id] ? true : undefined
                        }
                        onChange={(e) => {
                          e.stopPropagation();
                          handlePaletteNameChange(palette.id, e.target.value);
                        }}
                      />
                      {editingPaletteErrors[palette.id] && (
                        <FieldDescription className="text-destructive text-[10px]">
                          {editingPaletteErrors[palette.id]}
                        </FieldDescription>
                      )}
                    </Field>

                    {/* Action buttons */}
                    <div className="flex gap-1 justify-end">
                      {palette.savedName && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelEdit(palette.id);
                          }}
                        >
                          Cancel
                        </Button>
                      )}
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeletePalette(palette.id);
                        }}
                      >
                        Delete
                      </Button>
                      <Button
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSavePalette(palette.id);
                        }}
                        disabled={!!editingPaletteErrors[palette.id]}
                      >
                        Save
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {(() => {
          const sectionTitles = [
            "User Palettes",
            "Button Combos",
            "BG Presets",
            "Additional",
            "Fun",
          ];
          const expandedValues = sectionTitles.filter((t) => isExpanded(t));

          return (
            <Accordion
              multiple
              value={expandedValues}
              onValueChange={(next: string[]) => {
                const current = new Set(expandedValues);
                const target = new Set(next);
                sectionTitles.forEach((title) => {
                  if (current.has(title) !== target.has(title)) {
                    toggleExpanded(title);
                  }
                });
              }}
            >
              <PaletteSectionItem
                title="User Palettes"
                entries={savedUserPalettes}
                selected={selected}
                onSelectWithName={onSelectWithName}
                onEdit={(_, palette) => {
                  handleStartEdit((palette as UserPaletteEntry).id);
                }}
              />
              <PaletteSectionItem
                title="Button Combos"
                entries={BUTTON_COMBO_PALETTES}
                selected={selected}
                onSelectWithName={onSelectWithName}
                isBuiltIn
              />
              <PaletteSectionItem
                title="BG Presets"
                entries={BG_PRESETS}
                selected={selected}
                onSelectWithName={onSelectWithName}
                isBuiltIn
              />
              <PaletteSectionItem
                title="Additional"
                entries={ADDITIONAL_PALETTES}
                selected={selected}
                onSelectWithName={onSelectWithName}
                isBuiltIn
              />
              <PaletteSectionItem
                title="Fun"
                entries={FUN_PALETTES_EXPORT}
                selected={selected}
                onSelectWithName={onSelectWithName}
                isBuiltIn
              />
            </Accordion>
          );
        })()}
      </div>
    </Card>
  );
}
