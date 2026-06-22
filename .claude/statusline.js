#!/usr/bin/env node
// Claude Code status line — reads JSON on stdin, prints one line.
// Cross-platform: requires only Node (already used by this repo).

const { execSync } = require("node:child_process");
const path = require("node:path");

function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
  });
}

function formatContext(size) {
  if (!size) return "? context";
  if (size >= 1_000_000) {
    const m = size / 1_000_000;
    return `${m % 1 === 0 ? m : m.toFixed(1)}M context`;
  }
  if (size >= 1000) return `${Math.round(size / 1000)}k context`;
  return `${size} context`;
}

function gitBranch(cwd) {
  try {
    return execSync("git rev-parse --abbrev-ref HEAD", {
      cwd,
      stdio: ["ignore", "pipe", "ignore"],
      encoding: "utf8",
    }).trim();
  } catch {
    return "";
  }
}

(async () => {
  let input = {};
  try {
    const raw = await readStdin();
    if (raw.trim()) input = JSON.parse(raw);
  } catch {
    // fall through with empty input
  }

  const model =
    input?.model?.display_name || input?.model?.id || "Unknown model";

  const ctxSize = input?.context_window?.context_window_size || 0;
  const usedTokens = input?.context_window?.total_input_tokens || 0;
  const usedPct = input?.context_window?.used_percentage;

  const ctxLabel = formatContext(ctxSize);
  const usedK = `${Math.round(usedTokens / 1000)}k/${Math.round(ctxSize / 1000)}k`;
  const pctDisplay =
    typeof usedPct === "number" ? `${Math.round(usedPct)}%` : "?%";

  const cwd =
    input?.cwd || input?.workspace?.current_dir || process.cwd();
  const folder = path.basename(cwd);

  const branch = gitBranch(cwd);

  const parts = [
    `${model} (${ctxLabel})`,
    pctDisplay,
    usedK,
    folder,
  ];
  if (branch && branch !== "HEAD") parts.push(branch);

  process.stdout.write(parts.join(" | "));
})();
