import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import { FileBlob, SpreadsheetFile, Workbook } from "../workbook.mjs";

import { inputPath as sourceFile, outputPath as targetFile, outputDirectory, label as sourceLabel } from "../inputs.mjs";
const outDir = outputDirectory("baselineBuild");
const auditDir = outputDirectory("baselineAudit");
const table1Source = sourceFile("baselineSource");
const table1Stats = sourceFile("baselineStatistics");
const suppSource = sourceFile("supplementSource");
const figure5Patients = sourceFile("patientResults");
const table1Out = targetFile("baselineBuilt");
const suppOut = targetFile("supplementBuilt");
await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(auditDir, { recursive: true });
const deepClone = (x) => JSON.parse(JSON.stringify(x));
const isBlank = (x) => x === null || x === undefined || x === "";
const norm = (x) => String(x ?? "").trim().toLowerCase();
const numberOrText = (x) => {
  if (x === null || x === undefined || x === "") return null;
  if (typeof x === "number") return x;
  const s = String(x).trim();
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) return Number(s);
  return x;
};

function colName(n) {
  let s = "";
  for (let x = n; x > 0; x = Math.floor((x - 1) / 26)) s = String.fromCharCode(65 + ((x - 1) % 26)) + s;
  return s;
}

function lastAddress(rows, cols) {
  return `A1:${colName(cols)}${rows}`;
}

async function sha256(path) {
  const b = await fs.readFile(path);
  return createHash("sha256").update(b).digest("hex").toUpperCase();
}

async function csvValues(path) {
  const text = await fs.readFile(path, "utf8");
  const wb = await Workbook.fromCSV(text, { sheetName: "Data" });
  return wb.worksheets.getItemAt(0).getUsedRange().values;
}

function replaceBritishEnglish(values) {
  const pairs = [
    [/Tumor/g, "Tumour"], [/tumor/g, "tumour"], [/Etiology/g, "Aetiology"], [/summarized/g, "summarised"],
    [/harmonized/g, "harmonised"], [/analyzed/g, "analysed"], [/modeled/g, "modelled"],
  ];
  return values.map((row) => row.map((cell) => {
    if (typeof cell !== "string") return cell;
    let s = cell;
    for (const [a, b] of pairs) s = s.replace(a, b);
    return s;
  }));
}

function styleSheet(sheet, rows, cols, { headerRows = [1], titleRows = [], freezeRows = 1, wrapBody = false } = {}) {
  sheet.showGridLines = false;
  if (freezeRows > 0) sheet.freezePanes.freezeRows(freezeRows);
  const full = sheet.getRange(lastAddress(rows, cols));
  full.format.font = { name: "Arial", size: 9, color: "#1F2937" };
  full.format.verticalAlignment = "top";
  if (wrapBody) full.format.wrapText = true;
  for (const row of headerRows) {
    const r = sheet.getRange(`A${row}:${colName(cols)}${row}`);
    r.format = {
      fill: "#244A73",
      font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      wrapText: true,
      borders: { bottom: { style: "thin", color: "#17324D" } },
    };
    r.format.rowHeightPx = 36;
  }
  for (const row of titleRows) {
    const r = sheet.getRange(`A${row}:${colName(cols)}${row}`);
    r.format = {
      fill: "#E8F1F8",
      font: { name: "Arial", size: row === 1 ? 12 : 9, bold: row === 1, color: "#17324D" },
      verticalAlignment: "center",
      wrapText: true,
    };
  }
  const sampleRows = Math.min(rows, 250);
  const values = sheet.getRange(`A1:${colName(cols)}${sampleRows}`).values;
  for (let c = 0; c < cols; c++) {
    let maxLen = 8;
    for (let r = 0; r < values.length; r++) {
      const v = values[r]?.[c];
      if (!isBlank(v)) maxLen = Math.max(maxLen, Math.min(48, String(v).replace(/\n/g, " ").length));
    }
    const px = Math.max(78, Math.min(330, 8.0 * maxLen + 18));
    sheet.getRange(`${colName(c + 1)}1:${colName(c + 1)}${rows}`).format.columnWidthPx = px;
  }
}

function addMatrixSheet(wb, name, matrix, options = {}) {
  if (!matrix.length || !matrix[0].length) throw new Error(`Empty matrix for ${name}`);
  const cols = Math.max(...matrix.map((r) => r.length));
  const padded = matrix.map((r) => [...r, ...Array(cols - r.length).fill(null)]);
  const sheet = wb.worksheets.add(name);
  sheet.getRange(lastAddress(padded.length, cols)).values = padded;
  styleSheet(sheet, padded.length, cols, options);
  return sheet;
}
const stats = JSON.parse(await fs.readFile(table1Stats, "utf8"));
const table1 = await SpreadsheetFile.importXlsx(await FileBlob.load(table1Source));
for (const sheet of table1.worksheets.items) {
  const used = sheet.getUsedRange();
  used.values = replaceBritishEnglish(used.values);
}

const t1 = table1.worksheets.getItem("Table 1");
t1.getRange("A1").values = [["Table 1 | Baseline characteristics, treatment and outcomes across Avatar-HCC cohorts"]];
t1.getRange("B4:F4").values = [[
  "Internal multimodal\n(N=109)",
  "External radiomics\n(N=200)",
  "External urine metabolomics\n(N=33)",
  "External plasma metabolomics\n(N=72)",
  "External metagenomics\n(N=133)",
]];
t1.getRange("G7").values = [[0.060770834184785974]];
t1.getRange("A40").values = [["Locoregional therapy"]];
t1.getRange("A43").values = [["    TACE + HAIC"]];
t1.getRange("A44").values = [["Immune checkpoint inhibitor"]];
t1.getRange("A45").values = [["    Cadonilimab (AK104)"]];
t1.getRange("A53").values = [["Antiangiogenic agent"]];
t1.getRange("A63").values = [["Median follow-up (reverse Kaplan–Meier), months"]];
t1.getRange("A67").values = [["Continuous variables are summarised as mean ± s.d. for age and median (interquartile range) for all other continuous variables. Continuous variables were compared using Kruskal–Wallis tests."]];
t1.getRange("A68").values = [["Categorical variables were compared using Pearson χ² tests; sparse tables used the Fisher–Freeman–Halton exact test with 20,000 fixed-margin Monte Carlo samples. All P values are two-sided and descriptive; no multiplicity correction was applied."]];
t1.getRange("A69").values = [["Median PFS and OS were estimated by Kaplan–Meier analysis. Median follow-up was estimated by reverse Kaplan–Meier analysis based on OS follow-up. No between-cohort P value was assigned to the Kaplan–Meier medians."]];
t1.getRange("A70").values = [["External survival times were converted from days to months using 30.4375 days per month. External tumour-number coding was harmonised as 0 = single and 1 = multiple; internal counts were harmonised as 1 = single and >1 = multiple."]];
t1.getRange("A71").values = [["Abbreviations: AFP, alpha-fetoprotein; ALBI, albumin–bilirubin; BCLC, Barcelona Clinic Liver Cancer; ECOG, Eastern Cooperative Oncology Group; HAIC, hepatic arterial infusion chemotherapy; OS, overall survival; PFS, progression-free survival; PVTT, portal vein tumour thrombosis; TACE, transarterial chemoembolisation."]];
t1.getRange("A72:G72").clear({ applyTo: "all" });
t1.getRange("G6:G63").format.numberFormat = "0.000";
t1.getRange("G12").values = [["<0.001"]];
t1.getRange("G21").values = [["<0.001"]];
t1.getRange("G25").values = [["<0.001"]];
t1.getRange("G36").values = [["<0.001"]];
t1.getRange("G40").values = [["<0.001"]];
t1.getRange("G44").values = [["<0.001"]];
t1.getRange("G53").values = [["<0.001"]];
const pDisplay = t1.getRange("G6:G63").values.map(([v]) => [typeof v === "number" ? (v < 0.001 ? "<0.001" : v.toFixed(3)) : v]);
t1.getRange("G6:G63").values = pDisplay;
t1.getRange("G6:G63").format = { fill: "#FFFFFF" };
for (const r of [9, 24, 39, 57]) t1.getRange(`G${r}`).format = { fill: "#E8F1F8" };
for (const r of [13, 16, 25, 29, 33, 40, 44, 53]) t1.getRange(`G${r}`).format = { fill: "#F3F6F8" };
t1.getRange("A1:G71").format.font = { name: "Arial" };

const methods = table1.worksheets.getItem("Statistical methods");
methods.getRange("B8").values = [["Continuous variables were compared using Kruskal–Wallis tests."]];
methods.getRange("B13").values = [["Agent names were normalised for spelling and capitalisation; combination records were mapped to their component agents."]];
methods.getRange("A16:D16").clear({ applyTo: "all" });
methods.getRange("B22:D22").values = [["Kruskal–Wallis test", 0.060770834184785974, "0.061"]];
methods.getRange("A1:D44").format.font = { name: "Arial" };

const surv = table1.worksheets.getItem("Survival summary");
surv.getRange("A5:A9").values = [["Internal multimodal"], ["External radiomics"], ["External urine metabolomics"], ["External plasma metabolomics"], ["External metagenomics"]];
surv.getRange("A1:H13").format.font = { name: "Arial" };

const qc = table1.worksheets.getItem("Data QC");
qc.getRange("A5:A9").values = [["Internal multimodal"], ["External radiomics"], ["External urine metabolomics"], ["External plasma metabolomics"], ["External metagenomics"]];
qc.getRange("A14:H14").clear({ applyTo: "all" });
qc.getRange("A1:H19").format.font = { name: "Arial" };

const analysis = table1.worksheets.getItem("Analysis data");
analysis.getRange("A5:B9").values = [
  ["Internal multimodal", "Internal"], ["External radiomics", "Radiomics"],
  ["External urine metabolomics", "Urine"], ["External plasma metabolomics", "Plasma"],
  ["External metagenomics", "Metagenomics"],
];
analysis.getRange("A1:J9").format.font = { name: "Arial" };

const overview = table1.worksheets.getItem("Cohort overview");
overview.getRange("A36:A40").values = [["Internal"], ["Radiomics"], ["Urine"], ["Plasma"], ["Metagenomics"]];
overview.getRange("D36:D40").values = [["Internal"], ["Radiomics"], ["Urine"], ["Plasma"], ["Metagenomics"]];
overview.getRange("G36:G40").values = [["Internal"], ["Radiomics"], ["Urine"], ["Plasma"], ["Metagenomics"]];
overview.getRange("A1:T40").format.font = { name: "Arial" };

const table1Blob = await SpreadsheetFile.exportXlsx(table1);
await table1Blob.save(table1Out);
const srcSupp = await SpreadsheetFile.importXlsx(await FileBlob.load(suppSource));
const patientCsv = await csvValues(figure5Patients);
const patientHeader = patientCsv[0].map(String);
const patientRows = patientCsv.slice(1).filter((r) => !isBlank(r[0]));
if (patientRows.length !== 109) throw new Error(`Expected 109 internal patients; found ${patientRows.length}`);
const originalIds = patientRows.map((r) => norm(r[0]));
if (new Set(originalIds).size !== 109) throw new Error("Internal patient IDs are not unique.");
const internalMap = new Map(originalIds.map((id, i) => [id, `Internal_${String(i + 1).padStart(3, "0")}`]));

const supp = Workbook.create();
const contents = [
  ["Supplementary table", "Worksheet", "Contents"],
  ["S1", "Table_S1_Cohort_composition", "Centre-level source populations and locked analytic cohorts"],
  ["S2", "Table_S2_Model_performance", "Internal multimodal model performance"],
  ["S3", "Table_S3_Selected_models", "Selected model configurations"],
  ["S4", "Table_S4_External_validation", "Single-modality external validation"],
  ["S5", "Table_S5_MBP_phenotype; S5_missingness", "Anonymised internal patient-level MBP phenotype and missingness audit"],
  ["S6", "Table_S6_MBP_robustness", "MBP robustness analyses; retired S4f block excluded"],
  ["S7", "Table_S7_Modality_contribution", "Modality contribution analyses"],
  ["S8", "Table_S8_Feature_stability; S8_attribution_ranking; S8_SHAP_long", "Feature stability, attribution ranking and labelled long-form display SHAP"],
  ["S9", "Table_S9_Module_dictionary; S9_axis_membership; S9_patient_data", "Module definitions, axis membership and anonymised patient-level values"],
];
addMatrixSheet(supp, "Contents", contents, {wrapBody:true});

const s1 = [
  ["Centre", "Centre-provided source records", "Locked analytic cohort", "Available data domains", "Final n"],
  ["SCH", 1120, "Internal multimodal cohort", "Clinical, arterial-phase CT radiomics, plasma metabolomics, urine metabolomics and stool metagenomic KO profiles", 109],
  ["QLH", 1050, "External radiomics; external plasma metabolomics; external urine metabolomics", "Clinical and the corresponding modality-specific data", "200; 72; 33"],
  ["PUMCH", 930, "External metagenomics cohort", "Clinical and stool metagenomic KO profiles", 133],
  ["All centres", 3100, "Five locked analytic cohorts", "One internal multimodal cohort and four modality-specific external cohorts", 547],
];
addMatrixSheet(supp, "Table_S1_Cohort_composition", s1, {wrapBody:true});

for (const name of ["Table_S2_Model_performance", "Table_S3_Selected_models", "Table_S4_External_validation"]) {
  const values = srcSupp.worksheets.getItem(name).getUsedRange().values;
  addMatrixSheet(supp, name, values);
}

const oldS5 = srcSupp.worksheets.getItem("Table_S5_MBP_phenotype").getUsedRange().values;
const s5BaseHeader = oldS5[0].slice(0, 79);
const s5Data = oldS5.slice(110, 219).map((r, i) => [internalMap.get(norm(patientRows[i][0])), ...r.slice(0, 79)]);
if (s5Data.some((r) => !r[0])) throw new Error("S5 anonymisation failed.");
const responseColOld = s5BaseHeader.indexOf("Response_pred");
const responseColCsv = patientHeader.indexOf("Response_pred");
for (let i = 0; i < 109; i++) {
  const a = Number(s5Data[i][1 + responseColOld]);
  const b = Number(patientRows[i][responseColCsv]);
  if (!(Math.abs(a - b) < 1e-12)) throw new Error(`S5 patient-order check failed at row ${i + 1}`);
}
addMatrixSheet(supp, "Table_S5_MBP_phenotype", [["Patient ID", ...s5BaseHeader], ...s5Data]);
const s5Missing = oldS5.slice(219).filter((r) => !isBlank(r[79])).map((r) => r.slice(79, 82));
addMatrixSheet(supp, "S5_missingness", [["Variable", "n non-missing", "n missing"], ...s5Missing]);

const oldS6 = srcSupp.worksheets.getItem("Table_S6_MBP_robustness").getRange("A1:N505").values;
addMatrixSheet(supp, "Table_S6_MBP_robustness", oldS6);
const oldS7 = srcSupp.worksheets.getItem("Table_S7_Modality_contribution").getUsedRange().values;
addMatrixSheet(supp, "Table_S7_Modality_contribution", oldS7);

const oldS8 = srcSupp.worksheets.getItem("Table_S8_Feature_attribution").getUsedRange().values;
const stability = oldS8.slice(1).filter((r) => !isBlank(r[0])).map((r) => r.slice(0, 10));
addMatrixSheet(supp, "Table_S8_Feature_stability", [oldS8[0].slice(0, 10), ...stability]);
const ranks = oldS8.slice(1).filter((r) => !isBlank(r[10])).map((r) => r.slice(10, 16));
addMatrixSheet(supp, "S8_attribution_ranking", [oldS8[0].slice(10, 16), ...ranks]);

const shapLong = [["Endpoint", "Modality", "Feature", "Patient ID", "MBP", "Row-scaled SHAP", "Row-scale denominator", "Patient order"]];
for (const endpoint of ["Response", "PFS", "OS"]) {
  const matrix = await csvValues(sourceFile(`shapMatrix_${endpoint.toLowerCase()}`));
  const scaling = await csvValues(sourceFile(`shapScaling_${endpoint.toLowerCase()}`));
  const order = await csvValues(sourceFile(`shapOrder_${endpoint.toLowerCase()}`));
  const scaleMap = new Map(scaling.slice(1).map((r) => [String(r[0]), { modality: String(r[1]), denominator: Number(r[2]) }]));
  const orderMap = new Map(order.slice(1).map((r, i) => [norm(r[0]), { mbp: String(r[1]), order: i + 1 }]));
  const ids = matrix[0].slice(1).map((x) => norm(x));
  if (ids.length !== 109 || ids.some((id) => !internalMap.has(id))) throw new Error(`S8 ${endpoint} patient IDs do not match the internal cohort.`);
  for (const row of matrix.slice(1)) {
    const feature = String(row[0]);
    const meta = scaleMap.get(feature);
    if (!meta) throw new Error(`Missing S8 scaling metadata for ${endpoint}: ${feature}`);
    for (let j = 0; j < ids.length; j++) {
      const pm = orderMap.get(ids[j]);
      shapLong.push([endpoint, meta.modality, feature, internalMap.get(ids[j]), pm?.mbp ?? null, Number(row[j + 1]), meta.denominator, pm?.order ?? null]);
    }
  }
}
if (shapLong.length !== 12754) throw new Error(`Expected 12,753 S8 SHAP observations; found ${shapLong.length - 1}`);
addMatrixSheet(supp, "S8_SHAP_long", shapLong);

const oldS9 = srcSupp.worksheets.getItem("Table_S9_MBP_SHAP_module").getUsedRange().values;
const moduleDictionary = oldS9.slice(1, 28).map((r) => r.slice(0, 8));
addMatrixSheet(supp, "Table_S9_Module_dictionary", [oldS9[0].slice(0, 8), ...moduleDictionary]);
const axisMembership = oldS9.slice(28, 31).map((r) => [r[3], r[8], r[9]]);
addMatrixSheet(supp, "S9_axis_membership", [["Axis", "Module IDs", "n modules"], ...axisMembership]);
const mbpRawIndex = patientHeader.indexOf("MBP_raw");
const s9Patients = patientRows.filter((r) => ["MBP-I", "MBP-III"].includes(String(r[mbpRawIndex]))).map((r) => [internalMap.get(norm(r[0])), ...r.slice(1).map(numberOrText)]);
if (s9Patients.length !== 73) throw new Error(`Expected 73 MBP-I/III patients for S9; found ${s9Patients.length}`);
addMatrixSheet(supp, "S9_patient_data", [["Patient ID", ...patientHeader.slice(1)], ...s9Patients]);

const notes = [
  ["Topic", "Note"],
  ["Scope", sourceLabel("baselineAssemblyNote")],
  ["S1", sourceLabel("baselineCohortNote")],
  ["S5", sourceLabel("baselineScopeNote")],
  ["S6", sourceLabel("baselineRobustnessNote")],
  ["S8", sourceLabel("baselineAttributionNote")],
  ["S9", sourceLabel("baselineSelectedPatientsNote")],
  ["Privacy", sourceLabel("baselinePrivacyNote")],
];
addMatrixSheet(supp, "Notes", notes, {wrapBody:true});

const suppBlob = await SpreadsheetFile.exportXlsx(supp);
await suppBlob.save(suppOut);
async function qaWorkbook(path, label, originalIdSet = new Set()) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const sheetQa = [];
  let leaked = [];
  for (const sheet of wb.worksheets.items) {
    const used = sheet.getUsedRange();
    const values = used.values;
    const idColumns = [];
    for (let c = 0; c < (values[0]?.length ?? 0); c++) {
      const h = norm(values[0][c]);
      if (h === "patient id" || h === "sample" || h === "sample_id" || h === "sample id") idColumns.push(c);
    }
    for (const c of idColumns) {
      for (let r = 1; r < values.length; r++) {
        const v = norm(values[r][c]);
        if (originalIdSet.has(v)) leaked.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
      }
    }
    const highRisk = new Set([sourceLabel("identifierCheck1"), sourceLabel("identifierCheck2"), sourceLabel("identifierCheck3")]);
    for (let r = 1; r < values.length; r++) for (let c = 0; c < values[r].length; c++) {
      if (highRisk.has(norm(values[r][c]))) leaked.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
    }
    const topEnd = Math.min(20, used.rowCount);
    const top = await wb.inspect({kind:"table", range:`${sheet.name}!A1:${colName(Math.min(used.columnCount, 20))}${topEnd}`, include:"values,formulas", tableMaxRows:20, tableMaxCols:20, maxChars:12000});
    let bottom = null;
    if (used.rowCount > 40) {
      const start = Math.max(1, used.rowCount - 9);
      bottom = await wb.inspect({kind:"table", range:`${sheet.name}!A${start}:${colName(Math.min(used.columnCount, 20))}${used.rowCount}`, include:"values,formulas", tableMaxRows:10, tableMaxCols:20, maxChars:10000});
    }
    const preview = await wb.render({sheetName: sheet.name, range:`A1:${colName(Math.min(used.columnCount, 18))}${Math.min(used.rowCount, 24)}`, scale:1.0, format:"png"});
    const safe = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
    const previewPath = `${auditDir}/${label}_${safe}.png`;
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    sheetQa.push({name:sheet.name, usedRange:used.address, rows:used.rowCount, columns:used.columnCount, top:top.ndjson, bottom:bottom?.ndjson ?? null, preview:previewPath.split("/").at(-1)});
  }
  const errors = await wb.inspect({kind:"match", searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options:{useRegex:true,maxResults:500}, summary:`${label} formula error scan`, maxChars:12000});
  if (leaked.length) throw new Error(`${label}: source patient identifiers remain at ${leaked.slice(0, 20).join(", ")}`);
  return {file:path.split("/").at(-1), sha256:await sha256(path), sheets:sheetQa, formulaErrorScan:errors.ndjson, privacyLeakCount:leaked.length};
}

const report = {
  scope: sourceLabel("baselineReportScope"),
  table1: await qaWorkbook(table1Out, "Table1"),
  supplementary: await qaWorkbook(suppOut, "SuppTables", new Set(originalIds)),
  validation: {
    internalPatients: 109,
    s5PatientRows: s5Data.length,
    s8ShapRows: shapLong.length - 1,
    s9PatientRows: s9Patients.length,
    retiredS4fIncluded: false,
    duplicateS5BlockIncluded: false,
    duplicateS9BlockIncluded: false,
  },
  sourceHashes: {
    table1Template: await sha256(table1Source),
    table1StatisticalPreparation: await sha256(table1Stats),
    supplementaryS2toS9: await sha256(suppSource),
    figure5PatientTable: await sha256(figure5Patients),
  },
};
await fs.writeFile(`${auditDir}/baseline_validation.json`, JSON.stringify(report, null, 2), "utf8");

