import { ChevronDown } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/shadcn/components/collapsible";
import { Button } from "@/shadcn/components/button";
import { useLocalStorage } from "../hooks/useLocalStorage.js";
import { MarkdownRenderer } from "./MarkdownRenderer.js";

const INSTRUCTIONS_STORAGE_KEY = "gbcam-instructions-open";

export function CollapsibleInstructions({ markdown }: { markdown: string }) {
  const [isOpen, setIsOpen] = useLocalStorage<boolean>(
    INSTRUCTIONS_STORAGE_KEY,
    true,
  );

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="mb-6 rounded-lg border bg-card text-card-foreground"
    >
      <CollapsibleTrigger
        render={
          <Button
            variant="ghost"
            className="w-full justify-between rounded-b-none px-4 py-3 font-semibold"
          />
        }
      >
        Instructions
        <ChevronDown
          className="transition-transform data-[state=open]:rotate-180"
          data-icon="inline-end"
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="max-h-96 overflow-y-auto p-4">
          <MarkdownRenderer markdown={markdown} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
