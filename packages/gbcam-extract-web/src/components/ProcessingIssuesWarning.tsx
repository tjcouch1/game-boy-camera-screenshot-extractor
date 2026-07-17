import { useState } from "react";
import type { PipelineIssue } from "gbcam-extract";
import { TriangleAlert, ChevronUp } from "lucide-react";
import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/shadcn/components/alert";
import { Button } from "@/shadcn/components/button";

/** Join reason phrases into natural English: "a", "a and b", "a, b, and c". */
function formatReasons(reasons: string[]): string {
  if (reasons.length <= 1) return reasons[0] ?? "";
  if (reasons.length === 2) return `${reasons[0]} and ${reasons[1]}`;
  return `${reasons.slice(0, -1).join(", ")}, and ${reasons[reasons.length - 1]}`;
}

/**
 * Collapsible warning about processing-quality issues detected by the
 * pipeline. Expanded by default when issues exist; collapses to a single
 * warning-icon button so it takes almost no room once acknowledged.
 * Renders nothing when there are no issues.
 */
export function ProcessingIssuesWarning({
  issues,
}: {
  issues?: PipelineIssue[];
}) {
  const [open, setOpen] = useState(true);
  if (!issues?.length) return null;

  if (!open) {
    return (
      <Button
        variant="ghost"
        size="icon"
        aria-label="Show processing quality warning"
        title="Possible processing quality issues"
        onClick={() => setOpen(true)}
        className="size-7 text-destructive"
      >
        <TriangleAlert />
      </Button>
    );
  }

  return (
    <Alert>
      <TriangleAlert />
      <AlertTitle>Possible low accuracy</AlertTitle>
      <AlertDescription>
        This picture may have processed with low accuracy due to{" "}
        {formatReasons(issues.map((issue) => issue.reason))}. Try taking
        another picture to see if it processes more accurately.
      </AlertDescription>
      <AlertAction>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Collapse processing quality warning"
          onClick={() => setOpen(false)}
          className="size-6"
        >
          <ChevronUp />
        </Button>
      </AlertAction>
    </Alert>
  );
}
