import type { PipelineIssue } from "gbcam-extract";
import { TriangleAlert, ChevronUp } from "lucide-react";
import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/shadcn/components/alert";
import { Button } from "@/shadcn/components/button";
import { cn } from "@/shadcn/utils/utils";

/** Join reason phrases into natural English: "a", "a and b", "a, b, and c". */
function formatReasons(reasons: string[]): string {
  if (reasons.length <= 1) return reasons[0] ?? "";
  if (reasons.length === 2) return `${reasons[0]} and ${reasons[1]}`;
  return `${reasons.slice(0, -1).join(", ")}, and ${reasons[reasons.length - 1]}`;
}

/**
 * Expanded warning about processing-quality issues detected by the pipeline.
 * The collapse control hands visibility back to the parent, which renders
 * {@link ProcessingIssuesIcon} in its place.
 */
export function ProcessingIssuesAlert({
  issues,
  onCollapse,
}: {
  issues: PipelineIssue[];
  onCollapse: () => void;
}) {
  return (
    // text-warning colors the icon (via the alert's text-current svg rule)
    // and the title; the description keeps its muted foreground.
    <Alert className="text-warning border-warning/40">
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
          onClick={onCollapse}
          className="size-6"
        >
          <ChevronUp />
        </Button>
      </AlertAction>
    </Alert>
  );
}

/**
 * Collapsed form of the processing-quality warning: a single warning-icon
 * button that re-expands the alert.
 */
export function ProcessingIssuesIcon({
  onExpand,
  className,
}: {
  onExpand: () => void;
  className?: string;
}) {
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Show processing quality warning"
      title="Possible processing quality issues"
      onClick={onExpand}
      className={cn("size-7 text-warning", className)}
    >
      <TriangleAlert />
    </Button>
  );
}
