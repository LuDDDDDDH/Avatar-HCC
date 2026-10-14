import fs from "node:fs/promises";
import crypto from "node:crypto";
import { FileBlob, SpreadsheetFile } from "../workbook.mjs";

import { inputPath as sourceFile, outputPath as targetFile, outputDirectory } from "../inputs.mjs";
const input = sourceFile("baselineBuilt");
const output = targetFile("baselineUpdated");
const auditDir = outputDirectory("baselineUpdateAudit");
await fs.mkdir(auditDir, { recursive: true });

const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const frozenRadiomicsMedianPfs = 8.016427104722792;
wb.worksheets.getItem("Cohort overview").getRange("H37").values = [[frozenRadiomicsMedianPfs]];
wb.worksheets.getItem("Survival summary").getRange("D6").values = [[frozenRadiomicsMedianPfs]];
wb.worksheets.getItem("Analysis data").getRange("F6").values = [[frozenRadiomicsMedianPfs]];
wb.worksheets.getItem("Statistical methods").getRange("A41").values = [["Antiangiogenic agent"]];

const blob = await SpreadsheetFile.exportXlsx(wb);
await blob.save(output);

const check = await SpreadsheetFile.importXlsx(await FileBlob.load(output));
for (const [sheet, cell, expected] of [
  ["Cohort overview", "H37", frozenRadiomicsMedianPfs],
  ["Survival summary", "D6", frozenRadiomicsMedianPfs],
  ["Analysis data", "F6", frozenRadiomicsMedianPfs],
  ["Statistical methods", "A41", "Antiangiogenic agent"],
]) {
  const actual = check.worksheets.getItem(sheet).getRange(cell).values[0][0];
  if (actual !== expected) throw new Error(`${sheet}!${cell}: expected ${expected}, found ${actual}`);
}

const sheets = [];
for (const sheet of check.worksheets.items) {
  const used = sheet.getUsedRange();
  const previewRows = Math.min(24, used.rowCount);
  const previewCols = Math.min(20, used.columnCount);
  const letters = (n) => { let s=""; for(let x=n;x>0;x=Math.floor((x-1)/26))s=String.fromCharCode(65+(x-1)%26)+s; return s; };
  const range = `A1:${letters(previewCols)}${previewRows}`;
  const inspected = await check.inspect({kind:"table", range:`${sheet.name}!${range}`, include:"values,formulas", tableMaxRows:previewRows, tableMaxCols:previewCols, maxChars:14000});
  const rendered = await check.render({sheetName:sheet.name, range, scale:1.0, format:"png"});
  const file = `${sheet.name.replace(/[^A-Za-z0-9_-]+/g,"_")}.png`;
  await fs.writeFile(`${auditDir}/${file}`, new Uint8Array(await rendered.arrayBuffer()));
  sheets.push({name:sheet.name, usedRange:used.address, inspectPass:Boolean(inspected.ndjson), render:file});
}
const errors = await check.inspect({kind:"match", searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options:{useRegex:true,maxResults:200}, summary:"Table 1 v002 formula error scan", maxChars:6000});
const bytes = await fs.readFile(output);
const report = {
  file: output.split("/").at(-1),
  sha256: crypto.createHash("sha256").update(bytes).digest("hex").toUpperCase(),
  frozenRadiomicsMedianPfs,
  terminology: "Antiangiogenic agent",
  sheets,
  formulaErrorScan: errors.ndjson,
};
await fs.writeFile(`${auditDir}/baseline_update_validation.json`, JSON.stringify(report,null,2), "utf8");

