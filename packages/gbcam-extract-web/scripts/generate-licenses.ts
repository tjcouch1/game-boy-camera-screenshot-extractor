#!/usr/bin/env node

/**
 * Generate license file for all dependencies using license-checker
 * Transforms the license data into a beautiful HTML page
 * This script runs on postinstall to keep licenses up-to-date
 */

import * as licenseChecker from "license-checker";
import type { ModuleInfos, ModuleInfo } from "license-checker";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const packageDir = path.join(__dirname, "..");
// The location where all dependencies are installed (recursively) in a pnpm workspace
const workspacePackageDir = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "node_modules",
  ".pnpm",
);
const publicDir = path.join(packageDir, "public");
const configPath = path.join(packageDir, "license-checker.config.json");
const outputFile = path.join(publicDir, "licenses.html");
const augmentedDataFile = path.join(packageDir, "licenses.json");
const additionalLicensesPath = path.join(
  packageDir,
  "additional-licenses",
  "additional-licenses.json",
);

/* Prefix to ensure additional licenses sort to top */
const ADDITIONAL_LICENSES_PREFIX = "00_";

interface LicenseData extends ModuleInfos {}

interface AugmentedLicense {
  id?: string;
  name: string;
  version: string;
  licenses: string | string[];
  copyright: string;
  repository?: string;
  publisher?: string;
  email?: string;
  url?: string;
  description?: string;
  licenseText: string;
}

// Extract copyright from license text (look for real copyright statements)
function extractCopyright(licenseText: string | undefined): string {
  const lines = licenseText?.split("\n") ?? [];

  // First pass: look for lines that clearly start with "Copyright"
  // These are most likely actual copyright statements
  // Pattern: "Copyright" followed by optional (c), ©, or year
  const strictPattern = /^Copyright\s*(?:\([cC]\)|©)?\s*/;

  for (const line of lines) {
    const trimmed = line.trim();
    if (strictPattern.test(trimmed)) {
      return trimmed;
    }
  }

  // Second pass: look for "Copyright" with a year (YYYY)
  // This catches variations like "Copyright YYYY Name" or "Copyright 2018 Google Inc."
  const yearPattern = /^[^C]*Copyright\s+(\d{4})/i;

  for (const line of lines) {
    const trimmed = line.trim();
    if (yearPattern.test(trimmed) && !isBodeLine(trimmed)) {
      return trimmed;
    }
  }

  // Third pass: look for lines containing "copyright" that are NOT part of license body
  const bodyExclusions = [
    /shall\s+(be|include|retain)/i,
    /copyright\s+(notice|owner|statement|claim)/i,
    /reproduce.*copyright/i,
    /notice.*attached/i,
    /notice.*included/i,
    /notice.*appear/i,
    /licensor.*copyright/i,
    /copyright.*licensor/i,
  ];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.toLowerCase().includes("copyright")) {
      if (!bodyExclusions.some((pattern) => pattern.test(trimmed))) {
        return trimmed;
      }
    }
  }

  return "";
}

// Helper to check if a line is part of the license body
function isBodeLine(line: string): boolean {
  const bodyIndicators = [
    /shall\s+(be|include|retain)/i,
    /notice.*attached/i,
    /notice.*included/i,
    /licensor.*shall/i,
  ];
  return bodyIndicators.some((pattern) => pattern.test(line));
}

// Load additional licenses from the supplementary JSON file
async function loadAdditionalLicenses(): Promise<AugmentedLicense[]> {
  try {
    const content = await fs.readFile(additionalLicensesPath, "utf-8");
    const data = JSON.parse(content);

    if (!data.additionalLicenses || !Array.isArray(data.additionalLicenses)) {
      throw new Error(
        "additionalLicenses field must be an array in additional-licenses.json",
      );
    }

    // Validate each license has required fields
    for (const license of data.additionalLicenses) {
      if (
        !license.name ||
        !license.version ||
        !license.licenses ||
        !license.copyright ||
        !license.licenseText
      ) {
        throw new Error(
          `Invalid license entry in additional-licenses.json: missing required fields (name, version, licenses, copyright, licenseText)`,
        );
      }
    }

    console.log(
      `✓ Loaded ${data.additionalLicenses.length} additional licenses from ${additionalLicensesPath}`,
    );
    return data.additionalLicenses;
  } catch (error) {
    console.error(
      "Error loading additional licenses:",
      error instanceof Error ? error.message : String(error),
    );
    throw error;
  }
}

// Augment license data with extracted copyright and filter to relevant fields
function augmentLicenseData(
  licenses: LicenseData,
): Record<string, AugmentedLicense> {
  const augmented: Record<string, AugmentedLicense> = {};

  for (const [key, license] of Object.entries(licenses)) {
    const copyright = extractCopyright(license.licenseText);

    augmented[key] = {
      name: license.name || key.split("@")[0],
      version: license.version || "",
      licenses: license.licenses || "Unknown",
      copyright,
      licenseText: license.licenseText || "",
      ...(license.repository && { repository: license.repository }),
      ...(license.publisher && { publisher: license.publisher }),
      ...(license.email && { email: license.email }),
      ...(license.url && { url: license.url }),
      ...(license.description && { description: license.description }),
    };
  }

  return augmented;
}

// Transform augmented license data into HTML format
function generateHtmlLicensesPage(
  augmentedLicenses: Record<string, AugmentedLicense>,
): string {
  // Convert to array and sort by package name
  const licenseEntries = Object.entries(augmentedLicenses).sort(
    ([aKey, aLicense], [bKey, bLicense]) => {
      if (
        aKey.startsWith(ADDITIONAL_LICENSES_PREFIX) &&
        !bKey.startsWith(ADDITIONAL_LICENSES_PREFIX)
      )
        return -1; // Additional licenses first
      if (
        !aKey.startsWith(ADDITIONAL_LICENSES_PREFIX) &&
        bKey.startsWith(ADDITIONAL_LICENSES_PREFIX)
      )
        return 1;
      if (
        aKey.startsWith(ADDITIONAL_LICENSES_PREFIX) &&
        bKey.startsWith(ADDITIONAL_LICENSES_PREFIX)
      )
        return 0; // Maintain order of additional licenses as they appear in the file
      return aLicense.name.localeCompare(bLicense.name);
    },
  );

  const licenseList = licenseEntries
    .map(([key, license]) => {
      const licenseText = license.licenseText || "License text not available";
      const escaped = licenseText
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");

      // Build package info lines
      const infoLines: string[] = [];

      // Package name: use link for NPM packages, plain text for additional licenses
      const isAdditionalLicense = key.startsWith(ADDITIONAL_LICENSES_PREFIX);
      const packageLink = isAdditionalLicense
        ? license.name
        : `<a href="https://www.npmjs.com/package/${license.name}" target="_blank" rel="noopener noreferrer">${license.name}</a>`;

      // Add version if available
      if (license.version) {
        infoLines.push(
          `<span class="package-info-label">Version:</span> <span class="package-info-value">${license.version}</span>`,
        );
      }

      // Add description if available
      if (license.description) {
        infoLines.push(
          `<span class="package-info-label">Description:</span> <span class="package-info-value">${license.description}</span>`,
        );
      }

      // Add publisher/author if available
      if (license.publisher) {
        infoLines.push(
          `<span class="package-info-label">Publisher:</span> <span class="package-info-value">${license.publisher}</span>`,
        );
      }

      // Add email if available
      if (license.email) {
        infoLines.push(
          `<span class="package-info-label">Email:</span> <span class="package-info-value"><a href="mailto:${license.email}">${license.email}</a></span>`,
        );
      }

      // Add URL if available
      if (license.url) {
        infoLines.push(
          `<span class="package-info-label">URL:</span> <span class="package-info-value"><a href="${license.url}" target="_blank" rel="noopener noreferrer">${license.url}</a></span>`,
        );
      }

      // Add repository if available
      if (license.repository) {
        infoLines.push(
          `<span class="package-info-label">Repository:</span> <span class="package-info-value"><a href="${license.repository}" target="_blank" rel="noopener noreferrer">${license.repository}</a></span>`,
        );
      }

      const infoHtml =
        infoLines.length > 0
          ? `<div class="package-info">${infoLines.join("<br>")}</div>`
          : "";

      const entryId = license.id || license.name.toLowerCase().replace(/\s+/g, "-");

      return `
    <div class="license-entry" id="${entryId}">
      <h3>${packageLink}</h3>
      <p class="license-type">${license.licenses || "Unknown"}</p>
      ${license.copyright ? `<p class="copyright">${license.copyright}</p>` : ""}
      ${infoHtml}
      <details>
        <summary>View License Text</summary>
        <pre><code>${escaped}</code></pre>
      </details>
    </div>`;
    })
    .join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Open Source Licenses - Game Boy Camera Picture Extractor</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      background-color: #111827;
      color: #f3f4f6;
      line-height: 1.6;
    }

    .container {
      max-width: 48rem;
      margin: 0 auto;
      padding: 2rem 1rem;
    }

    header {
      margin-bottom: 2rem;
      border-bottom: 1px solid #374151;
      padding-bottom: 2rem;
    }

    h1 {
      font-size: 2rem;
      font-weight: bold;
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    h1 img {
      width: 2rem;
      height: 2rem;
    }

    p {
      color: #d1d5db;
      margin-bottom: 1rem;
    }

    a {
      color: #60a5fa;
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    .back-link {
      display: inline-block;
      margin-bottom: 1rem;
      padding: 0.5rem 1rem;
      background-color: #1f2937;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      color: #9ca3af;
      text-decoration: none;
      transition: all 0.2s;
    }

    .back-link:hover {
      background-color: #374151;
      color: #f3f4f6;
      text-decoration: none;
    }

    .license-entry {
      margin-bottom: 2rem;
      padding: 1.5rem;
      background-color: #1f2937;
      border: 1px solid #374151;
      border-radius: 0.5rem;
    }

    .license-entry h3 {
      font-size: 1.125rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }

    .license-entry h3 a {
      color: #60a5fa;
    }

    .license-entry h3 a:hover {
      text-decoration: underline;
    }

    .license-type {
      font-size: 0.875rem;
      color: #9ca3af;
      margin-bottom: 0.5rem;
    }

    .copyright {
      font-size: 0.875rem;
      color: #9ca3af;
      margin-bottom: 0.5rem;
      font-style: italic;
      white-space: pre-wrap;
    }

    .package-info {
      font-size: 0.875rem;
      color: #9ca3af;
      margin-bottom: 1rem;
      padding: 0.75rem;
      background-color: #111827;
      border-radius: 0.375rem;
    }

    .package-info-label {
      font-weight: 600;
      color: #d1d5db;
      margin-right: 0.5rem;
    }

    .package-info-value {
      color: #9ca3af;
    }

    .package-info-value a {
      color: #60a5fa;
    }

    .package-info-value a:hover {
      text-decoration: underline;
    }

    .package-info br {
      margin-bottom: 0.5rem;
    }

    details {
      margin-top: 1rem;
    }

    summary {
      cursor: pointer;
      padding: 0.5rem;
      margin: -0.5rem;
      border-radius: 0.25rem;
      user-select: none;
      color: #60a5fa;
      transition: all 0.2s;
    }

    summary:hover {
      background-color: #111827;
    }

    pre {
      margin-top: 0.75rem;
      padding: 1rem;
      background-color: #111827;
      border-radius: 0.375rem;
      overflow-x: auto;
      font-size: 0.75rem;
      line-height: 1.5;
      max-height: 400px;
    }

    code {
      font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', 'Menlo',
        'Courier', monospace;
      color: #d1d5db;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    footer {
      margin-top: 3rem;
      padding-top: 2rem;
      border-top: 1px solid #374151;
      text-align: center;
      font-size: 0.875rem;
      color: #6b7280;
    }

    @media (prefers-color-scheme: light) {
      body {
        background-color: #f9fafb;
        color: #111827;
      }

      .container {
        color: #111827;
      }

      p {
        color: #4b5563;
      }

      header {
        border-bottom-color: #e5e7eb;
      }

      .back-link {
        background-color: #f3f4f6;
        color: #6b7280;
      }

      .back-link:hover {
        background-color: #e5e7eb;
        color: #111827;
      }

      .license-entry {
        background-color: #f3f4f6;
        border-color: #e5e7eb;
      }

      .license-type {
        color: #6b7280;
      }

      .copyright {
        color: #6b7280;
      }

      .package-info {
        background-color: #f9fafb;
      }

      .package-info-label {
        color: #111827;
      }

      .package-info-value {
        color: #6b7280;
      }

      summary:hover {
        background-color: #f9fafb;
      }

      pre {
        background-color: #f9fafb;
      }

      code {
        color: #4b5563;
      }

      footer {
        border-top-color: #e5e7eb;
        color: #9ca3af;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="/" class="back-link">← Back to App</a>

    <header>
      <h1>
        <img src="./icon.svg" alt="">
        Open Source Licenses
      </h1>
      <p>
        This application uses the following ${licenseEntries.length} open source dependencies. Thank you to all the maintainers!
      </p>
    </header>

    <main>
      ${licenseList}
    </main>

    <footer>
      <p>Generated from package dependencies. Last updated on ${new Date().toLocaleString()}</p>
    </footer>
  </div>
</body>
</html>`;
}

async function main() {
  try {
    // Ensure public directory exists
    await fs.mkdir(publicDir, { recursive: true });

    // Load additional licenses first
    const additionalLicenses = await loadAdditionalLicenses();

    // Use license-checker to get all licenses
    console.log("Resolving licenses...");

    const licenseData = await new Promise<LicenseData>((resolve, reject) => {
      licenseChecker.init(
        {
          start: workspacePackageDir,
          customPath: configPath,
          json: true,
        },
        (err, packages) => {
          if (err) {
            reject(err);
          } else {
            resolve(packages as LicenseData);
          }
        },
      );
    });

    // Write the raw license data to a JSON file for debugging
    await fs.writeFile(
      path.join(packageDir, "licenses-raw.json"),
      JSON.stringify(licenseData, null, 2),
      "utf-8",
    );

    // Augment license data with extracted copyright and filter fields
    const augmentedData = augmentLicenseData(licenseData);

    // Prepend additional licenses to augmented data
    // Create a merged object with additional licenses first
    const mergedAugmentedData: Record<string, AugmentedLicense> = {};

    // Add additional licenses first (they'll appear at the top after sorting)
    // We give them a special prefix to ensure they sort to the top
    for (const license of additionalLicenses) {
      // Use a numeric prefix to sort to top: "0_" ensures these come before packages
      const key = `${ADDITIONAL_LICENSES_PREFIX}${license.name.toLowerCase().replace(/\s+/g, "-")}`;
      mergedAugmentedData[key] = license;
    }

    // Then add all the npm packages
    Object.assign(mergedAugmentedData, augmentedData);

    // Write augmented data to licenses.json
    await fs.writeFile(
      augmentedDataFile,
      JSON.stringify(mergedAugmentedData, null, 2),
      "utf-8",
    );

    // Generate HTML from merged license data
    const htmlContent = generateHtmlLicensesPage(mergedAugmentedData);

    // Write to file
    await fs.writeFile(outputFile, htmlContent, "utf-8");

    console.log(`✓ Generated licenses page at ${outputFile}`);
    console.log(`✓ Generated augmented license data at ${augmentedDataFile}`);
  } catch (error) {
    console.error(
      "Error generating licenses:",
      error instanceof Error ? error.message : String(error),
    );
    process.exit(1);
  }
}

main();
