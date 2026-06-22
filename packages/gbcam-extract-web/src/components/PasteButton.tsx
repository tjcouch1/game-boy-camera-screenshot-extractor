import type { ComponentProps, ReactNode } from "react";
import { ClipboardPaste } from "lucide-react";
import { Button } from "@/shadcn/components/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/shadcn/components/tooltip";
import type { ClipboardPermissionState } from "../hooks/useClipboardPermission.js";

type ButtonProps = ComponentProps<typeof Button>;

export interface PasteButtonProps extends Omit<ButtonProps, "onClick"> {
  /** Current clipboard-read permission. When "denied" the button is disabled. */
  permission: ClipboardPermissionState;
  /** Invoked when the (enabled) button is clicked. */
  onPaste: () => void;
  /**
   * Action phrase for the "denied" tooltip, e.g. "paste frames" →
   * "Enable Clipboard permissions to paste frames".
   */
  deniedReason: string;
  /** Button content. Defaults to a clipboard-paste icon. */
  children?: ReactNode;
}

/**
 * A clipboard-paste button that is automatically disabled — with an
 * explanatory tooltip — when clipboard-read permission has been denied. Every
 * paste affordance in the app routes through this so the disabled/tooltip
 * behaviour stays consistent.
 */
export function PasteButton({
  permission,
  onPaste,
  deniedReason,
  children,
  ...buttonProps
}: PasteButtonProps) {
  const content = children ?? <ClipboardPaste />;

  if (permission === "denied") {
    return (
      <Tooltip>
        {/* Disabled buttons swallow pointer events, so the tooltip is anchored
            to a wrapper element instead of the button itself. */}
        <TooltipTrigger render={<div />}>
          <Button {...buttonProps} disabled>
            {content}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          Enable Clipboard permissions to {deniedReason}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Button
      {...buttonProps}
      onClick={(e) => {
        // Some paste buttons live inside clickable containers (e.g. an editable
        // palette card); keep the click from selecting the container.
        e.stopPropagation();
        onPaste();
      }}
    >
      {content}
    </Button>
  );
}
