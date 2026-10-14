import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import { FileBlob, SpreadsheetFile, Workbook } from "../workbook.mjs";

import { inputPath as sourceFile, outputPath as targetFile, outputDirectory, label } from "../inputs.mjs";
const outputDir = outputDirectory("supplementBuild");
const inputPath = sourceFile("supplementBuilt");
const outputPath = targetFile("supplementExpanded");
const auditDir = outputDirectory("supplementAudit");
const fig5Patients = sourceFile("patientResults");
const isBlank = (x) => x === null || x === undefined || x === "";
const norm = (x) => String(x ?? "").trim().toLowerCase();

function colName(n) {
  let s = "";
  for (let x = n; x > 0; x = Math.floor((x - 1) / 26)) {
    s = String.fromCharCode(65 + ((x - 1) % 26)) + s;
  }
  return s;
}

function lastAddress(rows, cols) {
  return `A1:${colName(cols)}${rows}`;
}

function typedCell(x) {
  if (x === null || x === undefined || x === "") return null;
  if (typeof x === "number" || typeof x === "boolean") return x;
  const s = String(x).trim();
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) return Number(s);
  if (s === "TRUE") return true;
  if (s === "FALSE") return false;
  return x;
}

function normaliseModelLabel(x) {
  if (typeof x !== "string") return x;
  return x
    .replace(/Combo2_Blood\+Urine/g, "Combo2_Plasma+Urine")
    .replace(/Combo3_Metagenomics\+Blood\+Urine/g, "Combo3_Metagenomics+Plasma+Urine")
    .replace(/Combo3_Radiomics\+Blood\+Urine/g, "Combo3_Radiomics+Plasma+Urine")
    .replace(/Metagenomics\+Radiomics\+Blood\+Urine/g, "Metagenomics+Radiomics+Plasma+Urine")
    .replace(/Blood metabolomics/g, "Plasma metabolomics")
    .replace(/^Blood$/g, "Plasma")
    .replace(/\bBlood\+Urine\b/g, "Plasma+Urine");
}

async function csvMatrix(path, { normaliseModels = false } = {}) {
  const text = await fs.readFile(path, "utf8");
  const temp = await Workbook.fromCSV(text, { sheetName: "Data" });
  let values = temp.worksheets.getItemAt(0).getUsedRange().values;
  values = values.map((r) => r.map(typedCell));
  if (values.length && values[0].length && typeof values[0][0] === "string") {
    values[0][0] = values[0][0].replace(/^\uFEFF/, "");
  }
  if (normaliseModels) values = values.map((r) => r.map(normaliseModelLabel));
  return values;
}

function objectsFromMatrix(matrix) {
  const headers = matrix[0].map(String);
  return matrix.slice(1).filter((r) => r.some((x) => !isBlank(x))).map((r) => {
    const o = {};
    headers.forEach((h, i) => { o[h] = r[i] ?? null; });
    return o;
  });
}

function styleSheet(sheet, matrix, { wrapBody = false, freezeFirstColumn = true } = {}) {
  const rows = matrix.length;
  const cols = Math.max(...matrix.map((r) => r.length));
  const full = sheet.getRange(lastAddress(rows, cols));
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  if (freezeFirstColumn && cols > 3) sheet.freezePanes.freezeColumns(1);
  full.format.font = { name: "Arial", size: 9, color: "#1F2937" };
  full.format.verticalAlignment = "top";
  full.format.wrapText = wrapBody;
  const header = sheet.getRange(`A1:${colName(cols)}1`);
  header.format = {
    fill: "#244A73",
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "thin", color: "#17324D" } },
  };
  header.format.rowHeightPx = 38;

  const sampleRows = Math.min(rows, 250);
  for (let c = 0; c < cols; c++) {
    let maxLen = 8;
    for (let r = 0; r < sampleRows; r++) {
      const v = matrix[r]?.[c];
      if (!isBlank(v)) maxLen = Math.max(maxLen, Math.min(52, String(v).replace(/\n/g, " ").length));
    }
    const px = Math.max(82, Math.min(340, maxLen * 7.4 + 20));
    sheet.getRange(`${colName(c + 1)}1:${colName(c + 1)}${rows}`).format.columnWidthPx = px;
  }

  const headers = matrix[0].map((x) => String(x ?? ""));
  for (let c = 0; c < cols; c++) {
    const h = headers[c];
    const range = sheet.getRange(`${colName(c + 1)}2:${colName(c + 1)}${rows}`);
    if (/^n$|^n[ _]|^n_|^n\d+$|_n$|(^|_)(count|events|rank|order|folds|patients|responders|features|families)($|_)|^valid_bootstrap_complete$|semantic_sign|module_type_rank|(^|_)(tp|fp|tn|fn)($|_)/i.test(h)) {
      range.format.numberFormat = "0";
    } else if (/(^p$|_p$|p_value|p_raw|p_fdr|q_value|_q$|_q_|ci_lower|ci_upper)/i.test(h)) {
      range.format.numberFormat = "0.000E+00";
    } else {
      range.format.numberFormat = "0.0000";
    }
  }
}

function addMatrixSheet(workbook, name, matrix, options = {}) {
  if (!matrix.length || !matrix[0].length) throw new Error(`Empty matrix for ${name}`);
  if (name.length > 31) throw new Error(`Worksheet name exceeds 31 characters: ${name}`);
  const cols = Math.max(...matrix.map((r) => r.length));
  const padded = matrix.map((r) => [...r, ...Array(cols - r.length).fill(null)]);
  const sheet = workbook.worksheets.add(name);
  sheet.getRange(lastAddress(padded.length, cols)).values = padded;
  styleSheet(sheet, padded, options);
  return sheet;
}

function anonymiseMatrix(matrix, internalMap, sampleHeader = "sample") {
  const headers = matrix[0].map(String);
  const index = headers.findIndex((x) => norm(x) === norm(sampleHeader));
  if (index < 0) throw new Error(`No ${sampleHeader} column found for anonymisation.`);
  const out = matrix.map((r) => [...r]);
  out[0][index] = "Patient ID";
  for (let i = 1; i < out.length; i++) {
    if (isBlank(out[i][index])) continue;
    const anon = internalMap.get(norm(out[i][index]));
    if (!anon) throw new Error(`Unmapped patient identifier in row ${i + 1}: ${out[i][index]}`);
    out[i][index] = anon;
  }
  return out;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function sha256(path) {
  const bytes = await fs.readFile(path);
  return createHash("sha256").update(bytes).digest("hex").toUpperCase();
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(auditDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const patientMatrix = await csvMatrix(fig5Patients);
const patientRows = patientMatrix.slice(1).filter((r) => !isBlank(r[0]));
assert(patientRows.length === 109, `Expected 109 patients, found ${patientRows.length}`);
const originalIds = patientRows.map((r) => norm(r[0]));
assert(new Set(originalIds).size === 109, "Patient identifiers are not unique.");
const internalMap = new Map(originalIds.map((id, i) => [id, `Internal_${String(i + 1).padStart(3, "0")}`]));
const landscape = objectsFromMatrix(await csvMatrix(sourceFile("patientLandscape")))
  .sort((a, b) => Number(a.patient_order) - Number(b.patient_order));
assert(landscape.length === 109, `Expected 109 landscape patients, found ${landscape.length}`);
for (let i = 0; i < 109; i++) {
  assert(norm(landscape[i].sample) === originalIds[i], `Patient-order mismatch at position ${i + 1}`);
}
const s2 = objectsFromMatrix(workbook.worksheets.getItem("Table_S2_Model_performance").getUsedRange().values);
const selected = objectsFromMatrix(workbook.worksheets.getItem("Table_S3_Selected_models").getUsedRange().values);
const brier = objectsFromMatrix(await csvMatrix(sourceFile("selectedBrier"), { normaliseModels: true }));
const timeAuc = objectsFromMatrix(await csvMatrix(sourceFile("selectedTimeAuc"), { normaliseModels: true }));
const s2ByShort = new Map(s2.map((r) => [String(r.short), r]));
const brierByShort = new Map(brier.map((r) => [String(r.short), r]));
const timeByKey = new Map(timeAuc.map((r) => [`${r.endpoint}|${r.short}`, r]));
const s3Performance = [[
  "Role", "Model", "Modalities", "Short label", "N patients", "N features",
  "Response AUROC mean", "Response AUROC s.d.", "Response AUPRC mean", "Response AUPRC s.d.",
  "Response Brier", "PFS C-index mean", "PFS C-index s.d.", "OS C-index mean", "OS C-index s.d.",
  "PFS mean time-dependent AUC", "OS mean time-dependent AUC",
]];
for (const row of selected) {
  const short = String(row.short);
  const perf = s2ByShort.get(short);
  const br = brierByShort.get(short);
  const pfs = timeByKey.get(`PFS|${short}`);
  const os = timeByKey.get(`OS|${short}`);
  assert(perf && br && pfs && os, `Incomplete S3 frozen sources for ${short}: perf=${Boolean(perf)}, brier=${Boolean(br)}, pfs=${Boolean(pfs)}, os=${Boolean(os)}`);
  s3Performance.push([
    row.role, perf.model, perf.modalities, short, br.n_patients, perf.n_features,
    perf.Response_AUC_mean, perf.Response_AUC_sd, perf.Response_AUPRC_mean, perf.Response_AUPRC_sd,
    br.Response_Brier, perf.PFS_Cindex_mean, perf.PFS_Cindex_sd, perf.OS_Cindex_mean, perf.OS_Cindex_sd,
    pfs.mean_AUC, os.mean_AUC,
  ]);
}
addMatrixSheet(workbook, "S3_selected_performance", s3Performance);
const s3Threshold = await csvMatrix(sourceFile("responseThresholds"), { normaliseModels: true });
addMatrixSheet(workbook, "S3_threshold_metrics", s3Threshold);
const extPerf = objectsFromMatrix(await csvMatrix(sourceFile("externalTransfer"), { normaliseModels: true }));
const extResponse = objectsFromMatrix(await csvMatrix(sourceFile("externalResponse"), { normaliseModels: true }));
const extBrier = objectsFromMatrix(await csvMatrix(sourceFile("externalBrier"), { normaliseModels: true }));
const responseByShort = new Map(extResponse.map((r) => [String(r.short), r]));
const extBrierByShort = new Map(extBrier.map((r) => [String(r.short), r]));
const s4Performance = [[
  "Model", "Short label", "Display name", "N external", "N responders", "Response prevalence",
  "Internal CV Response AUROC mean", "Internal CV Response AUROC s.d.",
  "External Response AUROC", "Response AUROC CI lower", "Response AUROC CI upper", "External Response AUPRC", "External Response Brier",
  "Internal CV PFS C-index mean", "Internal CV PFS C-index s.d.", "External PFS C-index", "PFS C-index CI lower", "PFS C-index CI upper",
  "Internal CV OS C-index mean", "Internal CV OS C-index s.d.", "External OS C-index", "OS C-index CI lower", "OS C-index CI upper",
]];
for (const short of ["M", "R", "B", "U"]) {
  const p = extPerf.find((r) => String(r.short) === short);
  const r = responseByShort.get(short);
  const b = extBrierByShort.get(short);
  assert(p && r && b, `Incomplete S4 frozen sources for ${short}`);
  assert(Math.abs(Number(p.Response_AUROC) - Number(r.Response_AUROC)) < 1e-12, `S4 AUROC mismatch for ${short}`);
  s4Performance.push([
    p.model, short, p.display_name_4d, p.n_external, r.n_responders, r.response_prevalence,
    p.Response_AUC_mean, p.Response_AUC_sd,
    p.Response_AUROC, p.Response_AUROC_CI_lower, p.Response_AUROC_CI_upper, r.Response_AUPRC, b.Response_Brier,
    p.PFS_Cindex_mean, p.PFS_Cindex_sd, p.PFS_Cindex, p.PFS_Cindex_CI_lower, p.PFS_Cindex_CI_upper,
    p.OS_Cindex_mean, p.OS_Cindex_sd, p.OS_Cindex, p.OS_Cindex_CI_lower, p.OS_Cindex_CI_upper,
  ]);
}
addMatrixSheet(workbook, "S4_all_performance", s4Performance);
const s4Risk = await csvMatrix(sourceFile("externalRisk"), { normaliseModels: true });
addMatrixSheet(workbook, "S4_risk_stratification", s4Risk);
const s6Incremental = await csvMatrix(sourceFile("incrementalBenefit"));
addMatrixSheet(workbook, "S6_incremental_summary", s6Incremental);
const s6EndpointExcluded = await csvMatrix(sourceFile("endpointAgreement"));
addMatrixSheet(workbook, "S6_endpoint_excl_agreement", s6EndpointExcluded);
const s8Simplex = await csvMatrix(sourceFile("endpointPreference"));
addMatrixSheet(workbook, "S8_endpoint_simplex", s8Simplex);
const s8ConcordanceSummary = await csvMatrix(sourceFile("concordanceSummary"));
addMatrixSheet(workbook, "S8_concordance_summary", s8ConcordanceSummary);
const s8ConcordancePatient = anonymiseMatrix(
  await csvMatrix(sourceFile("patientConcordance")),
  internalMap,
);
addMatrixSheet(workbook, "S8_concordance_patient", s8ConcordancePatient);
addMatrixSheet(workbook, "S9_KO_feature_mapping", await csvMatrix(sourceFile("functionalMapping")), { wrapBody: true });
addMatrixSheet(workbook, "S9_metabolite_mapping", await csvMatrix(sourceFile("metaboliteMapping")), { wrapBody: true });
addMatrixSheet(workbook, "S9_radiomics_feature_map", await csvMatrix(sourceFile("imagingMapping")), { wrapBody: true });
addMatrixSheet(workbook, "S9_metabolite_module_map", await csvMatrix(sourceFile("metabolicModuleMapping")), { wrapBody: true });
addMatrixSheet(workbook, "S9_module_SHAP", await csvMatrix(sourceFile("moduleAttribution")), { wrapBody: true });
addMatrixSheet(workbook, "S9_axis_scores", anonymiseMatrix(await csvMatrix(sourceFile("axisScores")), internalMap));
addMatrixSheet(workbook, "S9_axis_stats", await csvMatrix(sourceFile("axisStatistics")));
addMatrixSheet(workbook, "S9_axis_pairwise", await csvMatrix(sourceFile("axisPairwise")));
addMatrixSheet(workbook, "S9_radiomics_module_scores", anonymiseMatrix(await csvMatrix(sourceFile("patientImagingScores")), internalMap));

const flowSpecs = [
  ["Amino Acids & TME Regulation Axis", "MBP-I", "aminoInterfaceLow"],
  ["Amino Acids & TME Regulation Axis", "MBP-III", "aminoInterfaceHigh"],
  ["Lipid & 1-Carbon Metabolism Axis", "MBP-I", "lipidInterfaceLow"],
  ["Lipid & 1-Carbon Metabolism Axis", "MBP-III", "lipidInterfaceHigh"],
  ["SCFA & BCAA Energy Metabolism Axis", "MBP-I", "energyInterfaceLow"],
  ["SCFA & BCAA Energy Metabolism Axis", "MBP-III", "energyInterfaceHigh"],
];
let interfaceMatrix = null;
for (const [axis, mbp, file] of flowSpecs) {
  const m = await csvMatrix(sourceFile(file));
  if (!interfaceMatrix) interfaceMatrix = [["Axis", "MBP group", ...m[0]]];
  else assert(JSON.stringify(m[0]) === JSON.stringify(interfaceMatrix[0].slice(2)), `Interface header mismatch: ${file}`);
  for (const row of m.slice(1)) interfaceMatrix.push([axis, mbp, ...row]);
}
addMatrixSheet(workbook, "S9_ecosystem_interfaces", interfaceMatrix, { wrapBody: true });
addMatrixSheet(workbook, "S9_metab_radiomics_links", await csvMatrix(sourceFile("metabolicImagingLinks")), { wrapBody: true });
const contentsRows = [
  ["Supplementary table", "Worksheet", "Contents"],
  ["S1", "Table_S1_Cohort_composition", "Centre-level source populations and locked analytic cohorts"],
  ["S2", "Table_S2_Model_performance", "Internal multimodal model performance"],
  ["S3", "Table_S3_Selected_models; S3_selected_performance; S3_threshold_metrics", "Selected model configurations, performance, Brier scores and frozen response thresholds"],
  ["S4", "Table_S4_External_validation; S4_all_performance; S4_risk_stratification", "Single-modality external response and survival performance, plus frozen-threshold risk stratification"],
  ["S5", "Table_S5_MBP_phenotype; S5_missingness", "Anonymised internal patient-level MBP phenotype and missingness audit"],
  ["S6", "Table_S6_MBP_robustness; S6_incremental_summary; S6_endpoint_excl_agreement", "Training-fold cut-offs, incremental C-index with bootstrap intervals, and endpoint-excluded MBP agreement"],
  ["S7", "Table_S7_Modality_contribution", "Total and feature-count-adjusted modality contribution analyses"],
  ["S8", "Table_S8_Feature_stability; S8_attribution_ranking; S8_SHAP_long; S8_endpoint_simplex; S8_concordance_summary; S8_concordance_patient", "Feature stability, attribution, endpoint preference and within-patient cross-endpoint concordance"],
  ["S9", "Table_S9_Module_dictionary; S9_axis_membership; S9_patient_data; S9_KO_feature_mapping; S9_metabolite_mapping; S9_radiomics_feature_map; S9_metabolite_module_map; S9_module_SHAP; S9_axis_scores; S9_axis_stats; S9_axis_pairwise; S9_radiomics_module_scores; S9_ecosystem_interfaces; S9_metab_radiomics_links", "Feature-to-module maps, module attribution, ecosystem-axis summaries and cross-modal interfaces"],
  ["Guide", "Data_dictionary", "Worksheet-level provenance, units of record, coding and privacy notes"],
];
const contentsSheet = workbook.worksheets.getItem("Contents");
contentsSheet.getRange("A1:C11").values = contentsRows;
styleSheet(contentsSheet, contentsRows, { wrapBody: true, freezeFirstColumn: false });

const dictionary = [
  ["Worksheet", "Supplementary table", "Unit of record", "Key content", "Frozen source", "Coding / privacy note"],
  ["Table_S1_Cohort_composition", "S1", "Centre or all-centre summary", "Centre-provided source counts and final analytic cohorts", "Supplementary Table S1 locked cohort summary", "Aggregate counts only"],
  ["Table_S2_Model_performance", "S2", "Candidate modality combination", "Mean and s.d. over 25 outer test folds", "Locked S2 model-performance table", "B denotes plasma metabolomics"],
  ["Table_S3_Selected_models", "S3", "Selected model", "Representative single, pair, MRU and full models", "Locked S3 selection table", "B denotes plasma metabolomics"],
  ["S3_selected_performance", "S3", "Selected model", "AUROC, AUPRC, Brier, C-index and mean time-dependent AUC", label("sourceLabel04"), "No model refitting"],
  ["S3_threshold_metrics", "S3", "Selected model", "Frozen Youden threshold and classification metrics", label("sourceLabel05"), "Thresholds derived from internally aggregated OOF predictions"],
  ["Table_S4_External_validation", "S4", "External modality-specific cohort", "Response AUROC and AUPRC", "Locked S4 external-validation table", "External cohorts are non-overlapping"],
  ["S4_all_performance", "S4", "External modality-specific cohort", "Response, PFS and OS performance with confidence intervals", label("sourceLabel06"), "B/Blood in frozen model labels denotes plasma metabolomics"],
  ["S4_risk_stratification", "S4", "Endpoint × external modality", "Frozen internal cut-off, group counts, events, HR and P values", label("sourceLabel07"), "Higher versus lower risk; cut-offs were not re-estimated externally"],
  ["Table_S5_MBP_phenotype", "S5", "Internal patient", "Clinical variables, OOF projections, benefit components and MBP", "Figure5 rebuilt patient-level table and locked S5", "Patient ID is a stable anonymous study ID; Sex 1=male, 0=female; status 1=event, 0=censored"],
  ["S5_missingness", "S5", "Variable", "Non-missing and missing counts", "Locked S5 missingness audit", "Counts refer to 109 internal patients"],
  ["Table_S6_MBP_robustness", "S6", "Outer fold or cut-off summary", "Training-reference tertile boundaries", "Locked retained S6 A1:N505", label("robustnessSourceScope")],
  ["S6_incremental_summary", "S6", "Endpoint × clinical context", "Incremental C-index and patient-level bootstrap interval", label("sourceLabel08"), "Pretreatment and post-mRECIST contexts; no recalculation"],
  ["S6_endpoint_excl_agreement", "S6", "Excluded endpoint", "Exact agreement, quadratic weighted kappa and category shift", label("sourceLabel09"), "Compares endpoint-excluded with primary MBP"],
  ["Table_S7_Modality_contribution", "S7", "Modality, patient-pair record or stratum summary", "Total and per-feature-normalised SHAP shares and concordance support", "Locked S7 export", "No direct identifiers"],
  ["Table_S8_Feature_stability", "S8", "Feature × endpoint", "Bootstrap top-k stability and rank summaries", "Locked S8 export", "KO identifiers are biological features, not patient IDs"],
  ["S8_attribution_ranking", "S8", "Selected attribution feature", "Shared and endpoint-specific SHAP ranks", "Locked S8 export", "No direct identifiers"],
  ["S8_SHAP_long", "S8", "Endpoint × feature × internal patient", "Row-scaled benefit-oriented SHAP display values", "Frozen endpoint-specific SHAP matrices", "Patient ID anonymised; row-scale denominator retained"],
  ["S8_endpoint_simplex", "S8", "Feature", "Normalised endpoint coordinates, entropy, preference margin and dominant endpoint", label("sourceLabel10"), "Coordinates are relative attribution shares"],
  ["S8_concordance_summary", "S8", "Overall, MBP or modality stratum × endpoint pair", "Median, IQR, mean and s.d. of within-patient Spearman rho", label("sourceLabel11"), "Descriptive frozen summary"],
  ["S8_concordance_patient", "S8", "Internal patient × endpoint pair", "Within-patient 39-feature Spearman rho", label("sourceLabel12"), "Patient ID anonymised"],
  ["Table_S9_Module_dictionary", "S9", "Module", "Module identifier, type, axis and display order", "Locked S9 module dictionary", "Hex values are display colours"],
  ["S9_axis_membership", "S9", "Ecosystem axis", "Module membership", "Locked S9 axis membership", "Imaging modules are shared across axes"],
  ["S9_patient_data", "S9", "Selected internal patient", "MBP-I and MBP-III benefit projections", "Figure5 rebuilt patient-level table", "Patient ID anonymised"],
  ["S9_KO_feature_mapping", "S9", "Selected KO feature", "KO-to-axis and microbial subpathway mapping", label("sourceLabel13"), "Original KO IDs retained"],
  ["S9_metabolite_mapping", "S9", "Selected metabolite feature", "Metabolite-to-axis and metabolic-module mapping", label("sourceLabel14"), "B/Blood in frozen feature labels denotes plasma"],
  ["S9_radiomics_feature_map", "S9", "Radiomic feature", "Radiomic feature-to-phenotype-module mapping and orientation", label("sourceLabel15"), "Original radiomic feature IDs retained"],
  ["S9_metabolite_module_map", "S9", "Metabolite feature", "Metabolic feature-to-module loading and QC", label("sourceLabel16"), "Frozen module construction output"],
  ["S9_module_SHAP", "S9", "Axis × module × endpoint", "Module-level benefit-oriented SHAP contribution", label("sourceLabel17"), "plot_value is a display-scaled value, not a percentage"],
  ["S9_axis_scores", "S9", "Internal patient × ecosystem axis", "KO, metabolic, imaging and integrated axis scores", label("sourceLabel18"), "Patient ID anonymised; positive means more responder-like"],
  ["S9_axis_stats", "S9", "Ecosystem axis", "Trend and global MBP tests with BH-adjusted q values", label("sourceLabel19"), "Frozen descriptive/inferential output"],
  ["S9_axis_pairwise", "S9", "Axis × MBP comparison", "Pairwise P and q values", label("sourceLabel20"), "Frozen pairwise output"],
  ["S9_radiomics_module_scores", "S9", "Internal patient × radiomics module", "Radiomics phenotype-module scores", label("sourceLabel21"), "Patient ID anonymised"],
  ["S9_ecosystem_interfaces", "S9", "Axis × MBP group × KO–metabolite edge", "Cross-modal microbial–metabolic interfaces", "Amino/Lipid/SCFA FeatureFlow frozen files", "MBP-I and MBP-III edge lists; no patient identifiers"],
  ["S9_metab_radiomics_links", "S9", "Axis × metabolite × radiomics module", "Metabolite–radiomics module correlations", label("sourceLabel22"), "B/Blood in frozen feature labels denotes plasma"],
  ["Notes", "Guide", "Topic", "Scope, reconstruction and privacy notes", "Workbook integration record", "No source-to-anonymous crosswalk included"],
];
addMatrixSheet(workbook, "Data_dictionary", dictionary, { wrapBody: true, freezeFirstColumn: true });

const notes = workbook.worksheets.getItem("Notes");
const noteRows = [
  ["S3 v002", label("supplementSelectionNote")],
  ["S4 v002", label("supplementExternalNote")],
  ["S6 v002", label("supplementBenefitNote")],
  ["S8 v002", label("supplementAttributionNote")],
  ["S9 v002", label("supplementModuleNote")],
];
notes.getRange("A9:B13").values = noteRows;
const noteMatrix = notes.getRange("A1:B13").values;
styleSheet(notes, noteMatrix, { wrapBody: true, freezeFirstColumn: false });
const outputBlob = await SpreadsheetFile.exportXlsx(workbook);
await outputBlob.save(outputPath);
const checked = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const required = [
  "S3_selected_performance", "S3_threshold_metrics", "S4_all_performance", "S4_risk_stratification",
  "S6_incremental_summary", "S6_endpoint_excl_agreement", "S8_endpoint_simplex",
  "S8_concordance_summary", "S8_concordance_patient", "S9_KO_feature_mapping",
  "S9_metabolite_mapping", "S9_radiomics_feature_map", "S9_metabolite_module_map", "S9_module_SHAP",
  "S9_axis_scores", "S9_axis_stats", "S9_axis_pairwise", "S9_radiomics_module_scores",
  "S9_ecosystem_interfaces", "S9_metab_radiomics_links", "Data_dictionary",
];
for (const name of required) assert(checked.worksheets.items.some((s) => s.name === name), `Missing required worksheet: ${name}`);

const highRisk = new Set([label("identifierCheck1"), label("identifierCheck2"), label("identifierCheck3")]);
let leakedIds = [];
let absolutePaths = [];
let crosswalkRows = [];
const sheetAudit = [];

for (const sheet of checked.worksheets.items) {
  const used = sheet.getUsedRange();
  const values = used.values;
  const headers = (values[0] ?? []).map(norm);
  const idColumns = headers.map((h, i) => ({ h, i })).filter(({ h }) => h === "patient id" || h === "sample" || h === "sample id" || h === "sample_id").map(({ i }) => i);
  for (let r = 1; r < values.length; r++) {
    for (let c = 0; c < values[r].length; c++) {
      const v = values[r][c];
      if (typeof v !== "string") continue;
      const n = norm(v);
      if (highRisk.has(n)) leakedIds.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
      if (/[A-Za-z]:[\\/]/.test(v) || /[\\/]Users[\\/]/i.test(v)) absolutePaths.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
      if (/internal_\d{3}/i.test(v)) {
        const tokens = n.split(/[^a-z0-9_]+/).filter(Boolean);
        if (tokens.some((token) => originalIds.includes(token))) {
          crosswalkRows.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
        }
      }
    }
    for (const c of idColumns) {
      const v = norm(values[r][c]);
      if (originalIds.includes(v)) leakedIds.push(`${sheet.name}!${colName(c + 1)}${r + 1}`);
      if (v && !/^internal_\d{3}$/.test(v)) leakedIds.push(`${sheet.name}!${colName(c + 1)}${r + 1}:unexpected-id`);
    }
  }

  const cols = Math.min(used.columnCount, 18);
  const topRows = Math.min(used.rowCount, 20);
  const topRange = `A1:${colName(cols)}${topRows}`;
  const top = await checked.inspect({
    kind: "table", range: `${sheet.name}!${topRange}`, include: "values,formulas",
    tableMaxRows: 20, tableMaxCols: 18, tableMaxCellChars: 120, maxChars: 16000,
  });
  let bottomText = null;
  if (used.rowCount > 40) {
    const bottomStart = Math.max(2, used.rowCount - 7);
    const bottom = await checked.inspect({
      kind: "table", range: `${sheet.name}!A${bottomStart}:${colName(cols)}${used.rowCount}`,
      include: "values,formulas", tableMaxRows: 8, tableMaxCols: 18, tableMaxCellChars: 120, maxChars: 10000,
    });
    bottomText = bottom.ndjson;
  }
  const preview = await checked.render({ sheetName: sheet.name, range: topRange, scale: 0.9, format: "png" });
  const safe = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  const previewName = `${String(sheetAudit.length + 1).padStart(2, "0")}_${safe}.png`;
  await fs.writeFile(`${auditDir}/${previewName}`, new Uint8Array(await preview.arrayBuffer()));
  sheetAudit.push({
    sheet: sheet.name,
    usedRange: used.address,
    rows: used.rowCount,
    columns: used.columnCount,
    inspectTop: top.ndjson,
    inspectBottom: bottomText,
    render: previewName,
  });
}

const s4RightInspect = await checked.inspect({
  kind: "table", range: "S4_all_performance!S1:W5", include: "values,formulas",
  tableMaxRows: 5, tableMaxCols: 5, tableMaxCellChars: 120, maxChars: 8000,
});
const s4FullPreview = await checked.render({
  sheetName: "S4_all_performance", range: "A1:W5", scale: 0.9, format: "png",
});
await fs.writeFile(`${auditDir}/external_performance.png`, new Uint8Array(await s4FullPreview.arrayBuffer()));

assert(leakedIds.length === 0, `Patient identifier/privacy QA failed: ${leakedIds.slice(0, 20).join(", ")}`);
assert(absolutePaths.length === 0, `Absolute path QA failed: ${absolutePaths.slice(0, 20).join(", ")}`);
assert(crosswalkRows.length === 0, `Crosswalk QA failed: ${crosswalkRows.slice(0, 20).join(", ")}`);

const formulaErrors = await checked.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 500 },
  summary: "v002 formula error scan",
  maxChars: 16000,
});

const rowExpectations = {
  S3_selected_performance: 5,
  S3_threshold_metrics: 5,
  S4_all_performance: 5,
  S4_risk_stratification: 9,
  S6_incremental_summary: 5,
  S6_endpoint_excl_agreement: 4,
  S8_endpoint_simplex: 40,
  S8_concordance_summary: 22,
  S8_concordance_patient: 328,
  S9_module_SHAP: 100,
  S9_axis_scores: 328,
  S9_axis_stats: 4,
  S9_axis_pairwise: 10,
  S9_radiomics_module_scores: 328,
  S9_ecosystem_interfaces: 81,
};
for (const [name, expected] of Object.entries(rowExpectations)) {
  const actual = checked.worksheets.getItem(name).getUsedRange().rowCount;
  assert(actual === expected, `${name}: expected ${expected} rows, found ${actual}`);
}

await fs.writeFile(`${auditDir}/supplement_validation.json`, JSON.stringify({
  input: label("sourceLabel23"),
  output: label("sourceLabel24"),
  modelOrStatisticalRecalculation: false,
  patientOrderVerifiedAgainstFigure5ALandscape: true,
  requiredSheetsPresent: true,
  patientIdentifierLeakCount: leakedIds.length,
  absolutePathCount: absolutePaths.length,
  crosswalkLeakCount: crosswalkRows.length,
  formulaErrorScan: formulaErrors.ndjson,
  outputSha256: await sha256(outputPath),
  s4RightSideInspect: s4RightInspect.ndjson,
  s4FullWidthRender: "external_performance.png",
  rowExpectations,
  sheets: sheetAudit,
}, null, 2), "utf8");

await sha256(outputPath);

