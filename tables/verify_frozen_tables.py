from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avatar_paths import get_value, get_column, get_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
F = get_value('science.paths.directory_181', 'directory_181')
STAGE = get_value('science.paths.directory_182', 'directory_182')
REGISTRY_PATH = get_path('tables.source_registry')
with REGISTRY_PATH.open(encoding='utf-8') as stream:
    SOURCES = json.load(stream)['sources']
if not isinstance(SOURCES, dict) or not all((isinstance(key, str) and isinstance(value, str) for key, value in SOURCES.items())):
    raise ValueError('Invalid table input configuration')

def equal_column(a, b):
    a, b = (a.reset_index(drop=True), b.reset_index(drop=True))
    an, bn = (pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce'))
    numeric = an.notna() & bn.notna()
    eq = (a.fillna('').astype(str) == b.fillna('').astype(str)).to_numpy().copy()
    eq[numeric.to_numpy()] = np.isclose(an[numeric], bn[numeric], rtol=1e-10, atol=1e-12)
    return int((~eq).sum())

def verify(project, workbook):
    book = pd.ExcelFile(workbook)
    pts = pd.read_csv(project / SOURCES['Table_S5_MBP_phenotype'])
    mapping = {str(s).strip().lower(): f'Internal_{i + 1:03d}' for i, s in enumerate(pts.iloc[:, 0])}
    records = []
    for sheet, relative in SOURCES.items():
        path = project / relative
        row = {'sheet': sheet, 'source': relative, 'source_exists': path.exists()}
        if not path.exists():
            records.append(row)
            continue
        source = pd.read_csv(path)
        target = pd.read_excel(book, sheet_name=sheet)
        if sheet == 'Table_S2_Model_performance':
            layout = pd.read_csv(project / (F + get_value('science.tables.metrics_084', 'metrics_084.csv')))
            source = source.merge(layout[['model', 'short']], on='model', how='left', validate='one_to_one')
        if 'Patient ID' in target and 'Patient ID' not in source:
            ids = next((c for c in source if str(c).lower() in ['sample', 'sample_id', 'sample id']), None)
            if ids:
                source = source.rename(columns={ids: 'Patient ID'})
                source['Patient ID'] = source['Patient ID'].astype(str).str.strip().str.lower().map(mapping)
        if sheet in ['Table_S2_Model_performance', 'S3_threshold_metrics', 'S4_risk_stratification']:
            for col in source:
                source[col] = source[col].map(lambda x: x.replace('Blood metabolomics', 'Plasma metabolomics').replace('Blood', 'Plasma') if isinstance(x, str) else x)
        for key in ['model', 'Patient ID']:
            if key in source and key in target and source[key].is_unique and target[key].is_unique and (set(source[key]) == set(target[key])):
                source = source.set_index(key).loc[target[key]].reset_index()
                break
        row.update(source_rows=len(source), target_rows=len(target), missing_target_columns=[c for c in target if c not in source], extra_source_columns=[c for c in source if c not in target])
        if len(source) == len(target):
            row['different_cells_by_column'] = {c: equal_column(source[c], target[c]) for c in target if c in source}
            row['different_cells'] = sum(row['different_cells_by_column'].values())
            row['matched'] = not row['missing_target_columns'] and row['different_cells'] == 0
        else:
            row['matched'] = False
        records.append(row)
    covered = set(SOURCES)
    return {'source_mapping': 'explicit', 'analysis_recomputed': False, 'clinical_provenance_verified': False, 'checks': records, 'other_sheets': [s for s in book.sheet_names if s not in covered]}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.project_root, args.workbook)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
