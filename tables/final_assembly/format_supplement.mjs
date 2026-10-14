import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { FileBlob, SpreadsheetFile } from "../workbook.mjs";

import { inputPath as sourceFile, outputPath as targetFile, outputDirectory } from "../inputs.mjs";
const inputPath = sourceFile("supplementExpanded");
const outputDir = outputDirectory("supplementFormatted");
const outputPath = targetFile("supplementFormatted");
const auditDir = outputDirectory("supplementFormatAudit");
const replacements = {
  "Notes!B5": "Table S6 reports training-reference tertile boundaries, incremental C-index, and endpoint-excluded MBP agreement.",
  "Data_dictionary!F12": "Training-reference tertile boundaries are reported across outer folds; no direct identifiers.",
};

function sheetSnapshot(sheet) {
  const used = sheet.getUsedRange(true);
  const values = used ? used.values : [];
  const formulas = used ? used.formulas : [];
  return {
    values,
    formulas,
    usedRows: values.length,
    usedCols: values.reduce((m, row) => Math.max(m, row.length), 0),
    sha256: crypto.createHash("sha256").update(JSON.stringify({ values, formulas })).digest("hex"),
  };
}

function valueAt(matrix, row, col) {
  const value = matrix?.[row]?.[col];
  return value === undefined ? null : value;
}

function equalValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function cellAddress(row, col) {
  let n = col + 1;
  let letters = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    letters = String.fromCharCode(65 + rem) + letters;
    n = Math.floor((n - 1) / 26);
  }
  return `${letters}${row + 1}`;
}

function compareSnapshots(beforeMap, afterMap, sheetNames) {
  const valueDiffs = [];
  const formulaDiffs = [];
  const dimensionDiffs = [];
  for (const name of sheetNames) {
    const before = beforeMap.get(name);
    const after = afterMap.get(name);
    if (!before || !after) {
      dimensionDiffs.push({ sheet: name, reason: !before ? "missing before" : "missing after" });
      continue;
    }
    if (before.usedRows !== after.usedRows || before.usedCols !== after.usedCols) {
      dimensionDiffs.push({
        sheet: name,
        before: [before.usedRows, before.usedCols],
        after: [after.usedRows, after.usedCols],
      });
    }
    const rows = Math.max(before.usedRows, after.usedRows);
    const cols = Math.max(before.usedCols, after.usedCols);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const bv = valueAt(before.values, r, c);
        const av = valueAt(after.values, r, c);
        if (!equalValue(bv, av)) valueDiffs.push({ sheet: name, cell: cellAddress(r, c), before: bv, after: av });
        const bf = valueAt(before.formulas, r, c);
        const af = valueAt(after.formulas, r, c);
        if (!equalValue(bf, af)) formulaDiffs.push({ sheet: name, cell: cellAddress(r, c), before: bf, after: af });
      }
    }
  }
  return { valueDiffs, formulaDiffs, dimensionDiffs };
}

function searchCells(workbook, regex) {
  const matches = [];
  for (const sheet of workbook.worksheets.items) {
    const snap = sheetSnapshot(sheet);
    for (let r = 0; r < snap.values.length; r += 1) {
      for (let c = 0; c < (snap.values[r]?.length ?? 0); c += 1) {
        const value = snap.values[r][c];
        if (typeof value === "string" && regex.test(value)) {
          matches.push({ sheet: sheet.name, cell: cellAddress(r, c), value });
        }
        regex.lastIndex = 0;
      }
    }
  }
  return matches;
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(auditDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const beforeNames = workbook.worksheets.items.map((sheet) => sheet.name);
const beforeMap = new Map(workbook.worksheets.items.map((sheet) => [sheet.name, sheetSnapshot(sheet)]));

workbook.worksheets.getItem("Notes").getRange("B5").values = [[replacements["Notes!B5"]]];
workbook.worksheets.getItem("Data_dictionary").getRange("F12").values = [[replacements["Data_dictionary!F12"]]];

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const afterNames = saved.worksheets.items.map((sheet) => sheet.name);
const afterMap = new Map(saved.worksheets.items.map((sheet) => [sheet.name, sheetSnapshot(sheet)]));
const comparisons = compareSnapshots(beforeMap, afterMap, beforeNames);

const targetValues = {
  "Notes!B5": saved.worksheets.getItem("Notes").getRange("B5").values?.[0]?.[0] ?? null,
  "Data_dictionary!F12": saved.worksheets.getItem("Data_dictionary").getRange("F12").values?.[0]?.[0] ?? null,
};

const bannedHistoryMatches = searchCells(saved, /S4f|retired|weight-perturbation/i);
const formulaErrors = await saved.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 16000,
});

const notesInspect = await saved.inspect({
  kind: "table",
  range: "Notes!A1:F10",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8,
  maxChars: 12000,
});
const dictionaryInspect = await saved.inspect({
  kind: "table",
  range: "Data_dictionary!A1:H16",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 10,
  maxChars: 18000,
});

const notesPreview = await saved.render({
  sheetName: "Notes",
  range: "A1:F10",
  scale: 2,
  format: "png",
});
const notesPreviewPath = path.join(auditDir, "after_Notes_A1_F10.png");
await fs.writeFile(notesPreviewPath, new Uint8Array(await notesPreview.arrayBuffer()));

const dictionaryPreview = await saved.render({
  sheetName: "Data_dictionary",
  range: "A1:H16",
  scale: 2,
  format: "png",
});
const dictionaryPreviewPath = path.join(auditDir, "after_Data_dictionary_A1_H16.png");
await fs.writeFile(dictionaryPreviewPath, new Uint8Array(await dictionaryPreview.arrayBuffer()));

const outputBytes = await fs.readFile(outputPath);
const outputSha256 = crypto.createHash("sha256").update(outputBytes).digest("hex").toUpperCase();
const expectedValueDiffs = ["Notes!B5", "Data_dictionary!F12"].sort();
const actualValueDiffs = comparisons.valueDiffs.map((d) => `${d.sheet}!${d.cell}`).sort();
const formulaErrorCount = formulaErrors.ndjson.includes("matched 0 entries") ? 0 : null;

const audit = {
  inputPath,
  outputPath,
  outputSha256,
  sheetCountBefore: beforeNames.length,
  sheetCountAfter: afterNames.length,
  sheetNamesUnchanged: JSON.stringify(beforeNames) === JSON.stringify(afterNames),
  usedRangesUnchanged: comparisons.dimensionDiffs.length === 0,
  sheetDimensions: afterNames.map((name) => ({
    sheet: name,
    rows: afterMap.get(name).usedRows,
    columns: afterMap.get(name).usedCols,
  })),
  targetValues,
  expectedValueDiffs,
  actualValueDiffs,
  onlyRequestedValuesChanged: JSON.stringify(expectedValueDiffs) === JSON.stringify(actualValueDiffs),
  valueDiffDetails: comparisons.valueDiffs,
  formulaDiffCount: comparisons.formulaDiffs.length,
  formulaDiffDetails: comparisons.formulaDiffs,
  bannedHistoryMatchCount: bannedHistoryMatches.length,
  bannedHistoryMatches,
  formulaErrorCount,
  formulaErrorInspect: formulaErrors.ndjson,
  notesInspect: notesInspect.ndjson,
  dictionaryInspect: dictionaryInspect.ndjson,
  previews: [notesPreviewPath, dictionaryPreviewPath],
  qaPass:
    beforeNames.length === 37 &&
    afterNames.length === 37 &&
    JSON.stringify(beforeNames) === JSON.stringify(afterNames) &&
    comparisons.dimensionDiffs.length === 0 &&
    JSON.stringify(expectedValueDiffs) === JSON.stringify(actualValueDiffs) &&
    comparisons.formulaDiffs.length === 0 &&
    bannedHistoryMatches.length === 0 &&
    formulaErrorCount === 0 &&
    targetValues["Notes!B5"] === replacements["Notes!B5"] &&
    targetValues["Data_dictionary!F12"] === replacements["Data_dictionary!F12"],
};

const auditPath = path.join(auditDir, "supplement_format_validation.json");
await fs.writeFile(auditPath, JSON.stringify(audit, null, 2), "utf8");


