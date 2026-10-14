from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
REASON = 'additional_longitudinal_sample'

def apply_exclusions(matrix, exclusions, expected_exclusions=2, expected_retained=133):
    if matrix.index.isna().any():
        raise ValueError('Use the pre-exclusion matrix with all original sample keys present.')
    result = matrix.copy()
    result.index = result.index.astype(str).str.strip().str.casefold()
    if not result.index.is_unique or (result.index == '').any():
        raise ValueError('Input sample keys must be nonempty and unique.')
    required = {'sample_id', 'reason'}
    if not required.issubset(exclusions.columns):
        raise ValueError('Exclusion list requires sample_id and reason columns.')
    if exclusions['sample_id'].isna().any():
        raise ValueError('Missing sample key in exclusion list.')
    ids = exclusions['sample_id'].astype(str).str.strip().str.casefold()
    if len(ids) != expected_exclusions or not ids.is_unique:
        raise ValueError('Unexpected exclusion count or duplicate exclusion keys.')
    if not exclusions['reason'].eq(REASON).all():
        raise ValueError('This workflow accepts documented additional longitudinal samples only.')
    if not ids.isin(result.index).all():
        raise ValueError('Exclusion list contains keys absent from the input matrix.')
    result = result.loc[~result.index.isin(ids)].copy()
    if len(result) != expected_retained:
        raise ValueError('Unexpected retained cohort size.')
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--exclusions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new output path.')
    frame = pd.read_csv(args.input, index_col=0)
    exclusions = pd.read_csv(args.exclusions, dtype=str)
    kept = apply_exclusions(frame, exclusions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.output, index=True)
