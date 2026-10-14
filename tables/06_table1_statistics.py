from __future__ import annotations
from pathlib import Path
import os
import sys
_release_root = Path(os.environ.get('AVATAR_RELEASE_ROOT', Path.cwd())).resolve()
while not (_release_root / 'avatar_paths.py').is_file():
    if _release_root == _release_root.parent:
        raise RuntimeError('Open this notebook from the code package directory.')
    _release_root = _release_root.parent
if str(_release_root) not in sys.path:
    sys.path.insert(0, str(_release_root))
from avatar_paths import project_path, get_value, get_column, get_path, get_label
from collections import Counter
from pathlib import Path
import json
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(str(project_path(get_value('science.paths.directory_169', 'directory_169'))))
CLINICAL_DIR = ROOT / get_value('science.paths.directory_170', 'directory_170') / get_value('science.paths.directory_171', 'directory_171') / get_value('science.paths.directory_172', 'directory_172') / get_value('science.paths.directory_173', 'directory_173')
OUT_DIR = ROOT / get_value('science.paths.directory_174', 'directory_174') / get_value('science.paths.directory_175', 'directory_175') / get_value('science.paths.directory_176', 'directory_176')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260808
N_PERMUTATIONS = 5000
COHORTS = [('Internal multimodal', pd.read_excel(CLINICAL_DIR / get_value('science.tables.clinical_023', 'clinical_023.xlsx')), 'internal'), ('External radiomics', pd.read_excel(CLINICAL_DIR / get_value('science.tables.result_225', 'result_225.xlsx'), sheet_name='200'), 'external'), ('External urine metabolomics', pd.read_excel(CLINICAL_DIR / get_value('science.tables.result_225', 'result_225.xlsx'), sheet_name='33'), 'external'), ('External plasma metabolomics', pd.read_excel(CLINICAL_DIR / get_value('science.tables.result_225', 'result_225.xlsx'), sheet_name='72'), 'external'), ('External metagenomics', pd.read_excel(CLINICAL_DIR / get_value('science.tables.result_225', 'result_225.xlsx'), sheet_name='133'), 'external')]

def clean_number(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    return value

def fmt_p(p: float | None) -> str | None:
    if p is None or not np.isfinite(p):
        return None
    return '<0.001' if p < 0.001 else f'{p:.3f}'

def fmt_count(n: int, denominator: int) -> str:
    return f'{n} ({100 * n / denominator:.1f}%)'

def fmt_mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors='coerce').dropna()
    return f'{values.mean():.1f} ± {values.std(ddof=1):.1f}'

def fmt_median_iqr(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors='coerce').dropna()
    return f'{values.median():.1f} ({values.quantile(0.25):.1f}–{values.quantile(0.75):.1f})'

def kw_p(column: str) -> float:
    arrays = [pd.to_numeric(frame[column], errors='coerce').dropna().to_numpy() for _, frame, _ in COHORTS]
    return float(stats.kruskal(*arrays).pvalue)

def tumour_multiplicity(frame: pd.DataFrame, kind: str) -> pd.Series:
    values = pd.to_numeric(frame['Number'], errors='coerce')
    return values.gt(1) if kind == 'internal' else values.eq(1)

def categorical_values(extractor):
    output = []
    for _, frame, kind in COHORTS:
        values = []
        for _, row in frame.iterrows():
            value = extractor(row, kind)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            values.append(value)
        output.append(values)
    return output

def categorical_p(extractor):
    grouped = categorical_values(extractor)
    categories = sorted(set().union(*(set(x) for x in grouped)), key=lambda x: str(x))
    table = np.array([[Counter(values).get(category, 0) for category in categories] for values in grouped], dtype=int)
    if table.shape[1] < 2:
        return (None, 'Not testable', table.tolist(), [str(x) for x in categories])
    observed = float(stats.chi2_contingency(table, correction=False).statistic)
    _, pearson_p, _, expected = stats.chi2_contingency(table, correction=False)
    if expected.min() >= 1 and np.mean(expected < 5) <= 0.2:
        return (float(pearson_p), 'Pearson chi-squared', table.tolist(), [str(x) for x in categories])
    rng = random.Random(SEED)
    labels = []
    pooled = []
    for group_index, values in enumerate(grouped):
        labels.extend([group_index] * len(values))
        pooled.extend(values)
    cat_index = {value: idx for idx, value in enumerate(categories)}
    count_ge = 0
    shuffled = list(pooled)
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(shuffled)
        perm = np.zeros_like(table)
        for group_index, value in zip(labels, shuffled):
            perm[group_index, cat_index[value]] += 1
        try:
            statistic = float(stats.chi2_contingency(perm, correction=False).statistic)
        except ValueError:
            continue
        if statistic >= observed - 1e-12:
            count_ge += 1
    p = (count_ge + 1) / (N_PERMUTATIONS + 1)
    return (float(p), f'Permutation chi-squared ({N_PERMUTATIONS} permutations; seed {SEED})', table.tolist(), [str(x) for x in categories])

def norm_text(value) -> str:
    return str(value).strip().casefold().replace(get_value('science.translation.open_parenthesis', '('), '(').replace(get_value('science.translation.close_parenthesis', ')'), ')')

def normalize_ici(value):
    z = norm_text(value)
    mapping = {'ak104': 'Cadonilimab (AK104)', 'atezolizumab': 'Atezolizumab', 'camrelizumab': 'Camrelizumab', 'durvalumab': 'Durvalumab', 'pembrolizumab': 'Pembrolizumab', 'sintilimab': 'Sintilimab', 'tislelizumab': 'Tislelizumab', 'toripalimab': 'Toripalimab'}
    for needle, label in mapping.items():
        if needle in z:
            return label
    return 'Other/unknown'

def normalize_target(value):
    z = norm_text(value)
    mapping = {'apatinib': 'Apatinib', 'bevacizumab': 'Bevacizumab', 'lenvatinib': 'Lenvatinib'}
    for needle, label in mapping.items():
        if needle in z:
            return label
    return 'Other/unknown'

def cat_extractor(column, transform=lambda x, _: x):
    return lambda row, kind: transform(row[column], kind) if pd.notna(row[column]) else None
tests = {}

def add_test(key, p, method, table=None, categories=None):
    tests[key] = {'p_value': clean_number(p), 'p_display': fmt_p(p), 'method': method, 'contingency_table': table, 'categories': categories}
for key, column in [('Age', 'Age'), ('BMI', 'BMI'), ('ALT', 'ALT'), ('AST', 'AST'), ('AFP', 'AFP'), ('Tumour size', 'Size')]:
    add_test(key, kw_p(column), 'Kruskal–Wallis')
categorical_specs = {'Sex': cat_extractor('Sex', lambda x, _: int(x)), 'HBV': cat_extractor('Hepatitis B', lambda x, _: int(x)), 'HCV': cat_extractor('HCV', lambda x, _: int(x)), 'Cirrhosis': cat_extractor('Cirrhosis', lambda x, _: int(x)), 'Child–Pugh class': cat_extractor('Child-Pugh', lambda x, _: str(x).strip()), 'ALBI grade': cat_extractor('ALBI_grade', lambda x, _: int(x)), 'AFP category': cat_extractor('AFP', lambda x, _: '>400' if float(x) > 400 else '≤400'), 'ECOG performance status': cat_extractor('ECOG', lambda x, _: int(x)), 'BCLC stage': cat_extractor('BCLC', lambda x, _: str(x).strip()), 'Tumour multiplicity': lambda row, kind: 'Multiple' if (float(row['Number']) > 1 if kind == 'internal' else float(row['Number']) == 1) else 'Single', 'PVTT': cat_extractor('PVTT', lambda x, _: int(x)), 'Extrahepatic metastasis': cat_extractor('Metastasis', lambda x, _: int(x)), 'Objective response': cat_extractor('Response', lambda x, _: int(x)), 'Locoregional treatment': cat_extractor(get_column('science.columns.locoregional_treatment', 'Locoregional treatment'), lambda x, _: str(x).strip().upper()), 'Immune checkpoint inhibitor': cat_extractor(get_column('science.columns.immune_checkpoint_inhibitor', 'Immune checkpoint inhibitor'), lambda x, _: normalize_ici(x)), 'Antiangiogenic agent': cat_extractor(get_column('science.columns.antiangiogenic_agent', 'Antiangiogenic agent'), lambda x, _: normalize_target(x))}
for key, extractor in categorical_specs.items():
    p, method, table, categories = categorical_p(extractor)
    add_test(key, p, method, table, categories)
add_test('PFS', 0.293, 'Five-group log-rank')
add_test('OS', 0.984, 'Five-group log-rank')
cohort_names = [name for name, _, _ in COHORTS]
cohort_sizes = [len(frame) for _, frame, _ in COHORTS]

def categorical_row(label, column, predicate, test_key=None):
    values = []
    for _, frame, kind in COHORTS:
        source = frame[column]
        mask = predicate(source, kind)
        values.append(fmt_count(int(mask.sum()), int(mask.notna().sum())))
    return {'type': 'data', 'label': label, 'values': values, 'test': test_key}
rows = []

def section(label):
    rows.append({'type': 'section', 'label': label, 'values': [None] * 5, 'test': None})

def data(label, values, test=None):
    rows.append({'type': 'data', 'label': label, 'values': values, 'test': test})
section('Demographic characteristics')
data('Male', [fmt_count(int((f['Sex'] == 1).sum()), len(f)) for _, f, _ in COHORTS], 'Sex')
data('Female', [fmt_count(int((f['Sex'] == 0).sum()), len(f)) for _, f, _ in COHORTS])
data('Age, years', [fmt_mean_sd(f['Age']) for _, f, _ in COHORTS], 'Age')
data('BMI, kg m⁻²', [fmt_median_iqr(f['BMI']) for _, f, _ in COHORTS], 'BMI')
section('Aetiology and liver function')
data('Hepatitis B virus infection', [fmt_count(int((f['Hepatitis B'] == 1).sum()), len(f)) for _, f, _ in COHORTS], 'HBV')
data('Hepatitis C virus infection', [fmt_count(int((f['HCV'] == 1).sum()), len(f)) for _, f, _ in COHORTS], 'HCV')
data('Cirrhosis', [fmt_count(int((f['Cirrhosis'] == 1).sum()), len(f)) for _, f, _ in COHORTS], 'Cirrhosis')
data('Child–Pugh class A', [fmt_count(int((f['Child-Pugh'].astype(str).str.strip() == 'A').sum()), len(f)) for _, f, _ in COHORTS], 'Child–Pugh class')
data('Child–Pugh class B', [fmt_count(int((f['Child-Pugh'].astype(str).str.strip() == 'B').sum()), len(f)) for _, f, _ in COHORTS])
data('ALBI grade 1', [fmt_count(int((pd.to_numeric(f['ALBI_grade']) == 1).sum()), len(f)) for _, f, _ in COHORTS], 'ALBI grade')
data('ALBI grade 2', [fmt_count(int((pd.to_numeric(f['ALBI_grade']) == 2).sum()), len(f)) for _, f, _ in COHORTS])
data('ALBI grade 3', [fmt_count(int((pd.to_numeric(f['ALBI_grade']) == 3).sum()), len(f)) for _, f, _ in COHORTS])
data('ALT, U l⁻¹', [fmt_median_iqr(f['ALT']) for _, f, _ in COHORTS], 'ALT')
data('AST, U l⁻¹', [fmt_median_iqr(f['AST']) for _, f, _ in COHORTS], 'AST')
data('AFP, ng ml⁻¹', [fmt_median_iqr(f['AFP']) for _, f, _ in COHORTS], 'AFP')
data('AFP ≤400 ng ml⁻¹', [fmt_count(int((pd.to_numeric(f['AFP']) <= 400).sum()), len(f)) for _, f, _ in COHORTS], 'AFP category')
data('AFP >400 ng ml⁻¹', [fmt_count(int((pd.to_numeric(f['AFP']) > 400).sum()), len(f)) for _, f, _ in COHORTS])
section('Tumour characteristics')
data('ECOG performance status 0', [fmt_count(int((pd.to_numeric(f['ECOG']) == 0).sum()), len(f)) for _, f, _ in COHORTS], 'ECOG performance status')
data('ECOG performance status ≥1', [fmt_count(int((pd.to_numeric(f['ECOG']) >= 1).sum()), len(f)) for _, f, _ in COHORTS])
data('BCLC stage A', [fmt_count(int((f['BCLC'].astype(str).str.strip() == 'A').sum()), len(f)) for _, f, _ in COHORTS], 'BCLC stage')
data('BCLC stage B', [fmt_count(int((f['BCLC'].astype(str).str.strip() == 'B').sum()), len(f)) for _, f, _ in COHORTS])
data('BCLC stage C', [fmt_count(int((f['BCLC'].astype(str).str.strip() == 'C').sum()), len(f)) for _, f, _ in COHORTS])
data('Maximum tumour diameter, cm', [fmt_median_iqr(f['Size']) for _, f, _ in COHORTS], 'Tumour size')
data('Multiple tumours (>1)', [fmt_count(int(tumour_multiplicity(f, k).sum()), len(f)) for _, f, k in COHORTS], 'Tumour multiplicity')
data('Portal vein tumour thrombosis', [fmt_count(int((pd.to_numeric(f['PVTT']) == 1).sum()), len(f)) for _, f, _ in COHORTS], 'PVTT')
data('Extrahepatic metastasis', [fmt_count(int((pd.to_numeric(f['Metastasis']) == 1).sum()), len(f)) for _, f, _ in COHORTS], 'Extrahepatic metastasis')
section('Treatment characteristics')
data('Locoregional therapy received', [fmt_count(len(f), len(f)) for _, f, _ in COHORTS])
data('Immune checkpoint inhibitor received', [fmt_count(len(f), len(f)) for _, f, _ in COHORTS])
data('Antiangiogenic therapy received', [fmt_count(len(f), len(f)) for _, f, _ in COHORTS])
section('Clinical outcomes and follow-up')
data('Objective response', [fmt_count(int((pd.to_numeric(f['Response']) == 1).sum()), len(f)) for _, f, _ in COHORTS], 'Objective response')
frozen_pfs = [9.3, 8.0, 10.0, 7.8, 8.0]
frozen_os = [24.8, 26.3, 25.0, 24.0, 25.1]
frozen_followup = [31.6, 33.8, 30.7, 29.9, 29.5]
data('Median PFS, months', [f'{x:.1f}' for x in frozen_pfs], 'PFS')
data('Median OS, months', [f'{x:.1f}' for x in frozen_os], 'OS')
data('Median follow-up, months', [f'{x:.1f}' for x in frozen_followup])
table_rows = []
for row in rows:
    table_rows.append([row['label'], *row['values'], tests[row['test']]['p_display'] if row['test'] else None, row['type'], tests[row['test']]['method'] if row['test'] else None])

def treatment_counts(column, normalizer=lambda value: str(value).strip().upper()):
    labels = sorted(set().union(*(set(frame[column].dropna().map(normalizer)) for _, frame, _ in COHORTS)))
    output = []
    for label in labels:
        values = []
        for _, frame, _ in COHORTS:
            normalized = frame[column].dropna().map(normalizer)
            values.append(fmt_count(int((normalized == label).sum()), len(frame)))
        output.append([label, *values])
    return output
treatment_tables = {'Locoregional therapy': treatment_counts(get_column('science.columns.locoregional_treatment', 'Locoregional treatment')), 'Immune checkpoint inhibitor': treatment_counts(get_column('science.columns.immune_checkpoint_inhibitor', 'Immune checkpoint inhibitor'), normalize_ici), 'Antiangiogenic agent': treatment_counts(get_column('science.columns.antiangiogenic_agent', 'Antiangiogenic agent'), normalize_target)}
survival_rows = [['Internal multimodal', 109, 98, 89.9, 9.3, 67, 61.5, 24.8, 31.6], ['External cohorts combined', 438, 408, 93.2, 8.1, 261, 59.6, 25.3, 32.7], ['External radiomics', 200, 192, 96.0, 8.0, 126, 63.0, 26.3, 33.8], ['External urine metabolomics', 33, 30, 90.9, 10.0, 18, 54.5, 25.0, 30.7], ['External plasma metabolomics', 72, 66, 91.7, 7.8, 42, 58.3, 24.0, 29.9], ['External metagenomics', 133, 120, 90.2, 8.0, 75, 56.4, 25.1, 29.5]]
statistical_rows = [[key, value['method'], value['p_value'], value['p_display']] for key, value in tests.items()]
payload = {'cohorts': [{'name': n, 'n': size} for n, size in zip(cohort_names, cohort_sizes)], 'table1_headers': ['Characteristic', *[f'{n} (n={size})' for n, size in zip(cohort_names, cohort_sizes)], 'P value', 'row_type', 'method'], 'table1_rows': table_rows, 'treatment_tables': treatment_tables, 'survival_headers': ['Cohort', 'N', 'PFS events', 'PFS events (%)', 'Median PFS (months)', 'OS deaths', 'OS deaths (%)', 'Median OS (months)', 'Median follow-up (months)'], 'survival_rows': survival_rows, 'statistical_headers': ['Variable', 'Statistical test', 'P value (numeric)', 'P value (display)'], 'statistical_rows': statistical_rows, 'tests': tests, 'notes': ['Data are n (%) unless otherwise stated. Age is shown as mean ± s.d.; other continuous variables are shown as median (interquartile range).', 'P values are two-sided omnibus comparisons across the five cohorts. Continuous variables used Kruskal–Wallis tests; categorical variables used Pearson chi-squared tests or permutation chi-squared tests for sparse tables; PFS and OS used log-rank tests.', 'Tumour multiplicity was harmonised as >1 tumour in the internal cohort and the recorded multiple-tumour category in external cohorts.', 'Median follow-up was estimated using the reverse Kaplan–Meier method at the 10 March 2026 data cutoff.', 'ALBI, albumin–bilirubin; AFP, alpha-fetoprotein; ALT, alanine aminotransferase; AST, aspartate aminotransferase; BCLC, Barcelona Clinic Liver Cancer; BMI, body mass index; ECOG, Eastern Cooperative Oncology Group; PFS, progression-free survival; OS, overall survival.']}
(OUT_DIR / get_value('science.tables.result_226', 'result_226.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT_DIR / get_value('science.tables.result_227', 'result_227.json')).write_text(json.dumps({'source_files': [str(CLINICAL_DIR / get_value('science.tables.clinical_023', 'clinical_023.xlsx')), str(CLINICAL_DIR / get_value('science.tables.result_225', 'result_225.xlsx'))], 'cohort_rows': {n: len(f) for n, f, _ in COHORTS}, 'duplicate_ids': {n: int(f['sample_id'].duplicated().sum()) for n, f, _ in COHORTS}, 'missing_cells': {n: int(f.isna().sum().sum()) for n, f, _ in COHORTS}, 'tests': tests, 'frozen_survival_source': str(ROOT / get_value('science.paths.directory_177', 'directory_177') / get_value('science.paths.directory_178', 'directory_178') / get_value('science.paths.directory_179', 'directory_179') / get_value('science.paths.directory_180', 'directory_180'))}, ensure_ascii=False, indent=2), encoding='utf-8')
