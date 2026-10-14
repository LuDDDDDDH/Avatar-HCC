from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avatar_paths import get_path, get_value, get_column
import base64
import html
import math
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
try:
    st.set_page_config(page_title='Avatar-Agent Projection Interface', page_icon='🧬', layout='wide', initial_sidebar_state='expanded')
    RELEASE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = get_path('agent.project_root', os.environ.get('AVATAR_PROJECT_DIR', str(RELEASE_ROOT / 'work')))
    APP_DIR = get_path('agent.asset_root', os.environ.get('AVATAR_ASSET_DIR', 'assets'), root=PROJECT_ROOT)
    ANALYSIS_ROOT = get_path('agent.analysis_root', os.environ.get('AVATAR_ANALYSIS_DIR', 'analysis'), root=PROJECT_ROOT)
    ENDPOINTS = ['Response', 'PFS', 'OS']
    MRU_MODALITIES = ['Radiomics', 'Metagenomics', 'Urine']
    EXPECTED_MRU_FEATURE_COUNT = 39
    EXPECTED_MRU_MODALITY_COUNTS = {'Metagenomics': 15, 'Radiomics': 9, 'Urine': 15}
    AXIS_ORDER = ['Amino Acids & TME Regulation Axis', 'Lipid & 1-Carbon Metabolism Axis', 'SCFA & BCAA Energy Metabolism Axis']
    AXIS_DISPLAY = {'Amino Acids & TME Regulation Axis': 'Amino acids & TME regulation', 'Lipid & 1-Carbon Metabolism Axis': 'Lipid & one-carbon metabolism', 'SCFA & BCAA Energy Metabolism Axis': 'SCFA & BCAA energy metabolism'}
    AXIS_SHORT = {'Amino Acids & TME Regulation Axis': 'Amino & TME', 'Lipid & 1-Carbon Metabolism Axis': 'Lipid & 1-carbon', 'SCFA & BCAA Energy Metabolism Axis': 'SCFA & BCAA'}
    AXIS_DESCRIPTIONS = {'Amino Acids & TME Regulation Axis': 'Amino-acid utilization and tryptophan/TME-associated regulation', 'Lipid & 1-Carbon Metabolism Axis': 'Methylation-related functions, membrane remodeling & metabolic organization', 'SCFA & BCAA Energy Metabolism Axis': 'SCFA-related microbial energy functions & host metabolic state'}

    def require_file(path, label='required input'):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {label}.') from None
        return path

    def require_dir_with_file(directory, required_filename, label='required stage'):
        directory = Path(directory)
        if not (directory / required_filename).is_file():
            raise FileNotFoundError(f'Missing {label}.') from None
        return directory
    for _label, _path in {'project root': PROJECT_ROOT, 'avatar assets folder': APP_DIR, 'analysis results folder': ANALYSIS_ROOT}.items():
        if not _path.is_dir():
            raise FileNotFoundError(f'Missing {_label}.') from None
    MBP_PATIENT_PATH = require_file(get_path('agent.patient_predictions', 'inputs/patient_predictions.csv', root=PROJECT_ROOT), label='MRU/MBP patient table')
    MRU_SHAP_DIR = require_dir_with_file(get_path('agent.feature_attribution_dir', 'inputs/feature_attribution', root=PROJECT_ROOT), get_value('agent.signed_response_filename', 'signed_response.csv.gz'), label='final MRU SHAP directory')
    MRU_META_PATH = get_path('agent.feature_metadata', 'features.csv', root=MRU_SHAP_DIR)
    MRU_IMPORTANCE_PATH = get_path('agent.feature_importance', 'importance.csv', root=MRU_SHAP_DIR)
    MRU_GLOBAL_MODALITY_CONTRIB_PATH = get_path('agent.modality_contribution', 'modality_contribution.csv', root=MRU_SHAP_DIR)
    MRU_SHAP_SIGNED_PATHS = {task: get_path(f'agent.feature_shap_signed_{task.lower()}', f'signed_{task.lower()}.csv.gz', root=MRU_SHAP_DIR) for task in ENDPOINTS}
    MRU_SHAP_ABS_PATHS = {task: get_path(f'agent.feature_shap_absolute_{task.lower()}', f'absolute_{task.lower()}.csv.gz', root=MRU_SHAP_DIR) for task in ENDPOINTS}
    ECOSYSTEM_AXIS_PATH = require_file(get_path('agent.ecosystem_axis_scores', 'inputs/ecosystem_axis_scores.csv', root=PROJECT_ROOT), label='Figure-6 ecosystem-axis scores')
    MODULE_SCORE_PATH = get_path('agent.module_scores', 'inputs/module_scores.csv', root=PROJECT_ROOT)
    MODULE_META_PATH = get_path('agent.module_metadata', 'inputs/module_metadata.csv', root=PROJECT_ROOT)
    MODULE_SHAP_BENEFIT_PATHS = {task: get_path(f'agent.module_shap_benefit_{task.lower()}', f'inputs/module_benefit_{task.lower()}.csv.gz', root=PROJECT_ROOT) for task in ENDPOINTS}
    MODULE_SHAP_CONTRIB_PATH = get_path('agent.module_contribution', 'inputs/module_contribution.csv', root=PROJECT_ROOT)
    ASSET_DIR = APP_DIR
    BODY_IMG_PATH = get_path('agent.body_illustration', 'body.png', root=APP_DIR)
    REQUIRED_INPUTS = {'MRU/MBP patient table': MBP_PATIENT_PATH, 'MRU feature metadata': MRU_META_PATH, 'MRU signed SHAP — Response': MRU_SHAP_SIGNED_PATHS['Response'], 'MRU signed SHAP — PFS': MRU_SHAP_SIGNED_PATHS['PFS'], 'MRU signed SHAP — OS': MRU_SHAP_SIGNED_PATHS['OS'], 'MRU absolute SHAP — Response': MRU_SHAP_ABS_PATHS['Response'], 'MRU absolute SHAP — PFS': MRU_SHAP_ABS_PATHS['PFS'], 'MRU absolute SHAP — OS': MRU_SHAP_ABS_PATHS['OS'], 'Figure-6 ecosystem-axis scores': ECOSYSTEM_AXIS_PATH}
    OPTIONAL_INPUTS = {'MRU global importance': MRU_IMPORTANCE_PATH, 'MRU global modality contribution': MRU_GLOBAL_MODALITY_CONTRIB_PATH, 'Figure-6 module score matrix': MODULE_SCORE_PATH, 'Figure-6 module metadata': MODULE_META_PATH, 'Figure-6 module SHAP — Response': MODULE_SHAP_BENEFIT_PATHS['Response'], 'Figure-6 module SHAP — PFS': MODULE_SHAP_BENEFIT_PATHS['PFS'], 'Figure-6 module SHAP — OS': MODULE_SHAP_BENEFIT_PATHS['OS'], 'Figure-6 module SHAP contribution': MODULE_SHAP_CONTRIB_PATH}

    def require_analysis_inputs():
        missing = [label for label, path in REQUIRED_INPUTS.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError('Required analysis inputs are unavailable: ' + ', '.join(missing)) from None
    require_analysis_inputs()

    def esc(x):
        if x is None:
            return ''
        try:
            if pd.isna(x):
                return ''
        except Exception:
            pass
        return html.escape(str(x))

    def as_float(x, default=np.nan):
        try:
            if pd.isna(x):
                return default
            return float(x)
        except Exception:
            return default

    def clamp(x, lo=0.0, hi=100.0):
        try:
            return max(lo, min(hi, float(x)))
        except Exception:
            return np.nan

    def fmt_num(x, digits=2):
        x = as_float(x)
        return '—' if not np.isfinite(x) else f'{x:.{digits}f}'

    def fmt_prob(x):
        x = as_float(x)
        return '—' if not np.isfinite(x) else f'{x:.2f}'

    def fmt_pct(x, digits=0):
        x = as_float(x)
        return '—' if not np.isfinite(x) else f'{x:.{digits}f}%'

    def fmt_months(x):
        x = as_float(x)
        if not np.isfinite(x):
            return '—'
        if abs(x - round(x)) < 1e-09:
            return f'{x:.0f} months'
        return f'{x:.1f} months'

    def normalize_sample_id(x):
        s = str(x).strip()
        if s.lower() in {'nan', 'none', ''}:
            return ''
        return s.upper()

    def find_col(df, candidates, required=False):
        if df is None:
            if required:
                raise KeyError(f'Cannot find any of {candidates}; dataframe is None.')
            return None
        cols = list(df.columns)
        lower = {str(c).lower(): c for c in cols}
        for cand in candidates:
            if cand in cols:
                return cand
            if str(cand).lower() in lower:
                return lower[str(cand).lower()]
        for cand in candidates:
            token = str(cand).lower()
            for c in cols:
                if token in str(c).lower():
                    return c
        if required:
            raise KeyError(f'Cannot find any of columns: {candidates}')
        return None

    def get_first(row, candidates, default=np.nan):
        for c in candidates:
            if c in row.index:
                return row[c]
        lower = {str(c).lower(): c for c in row.index}
        for c in candidates:
            if str(c).lower() in lower:
                return row[lower[str(c).lower()]]
        return default

    def read_table(path, required=True):
        path = Path(path)
        if not path.exists():
            if required:
                raise FileNotFoundError('A required analysis table is unavailable.') from None
            return None
        try:
            frame = pd.read_csv(path, encoding='utf-8-sig')
            column_keys = {'sample': 'sample_identifier', 'sample_id': 'sample_identifier_alternative', 'patient_id': 'patient_identifier', 'id': 'identifier', 'MBP_raw': 'phenotype_assignment', 'MBP': 'phenotype_assignment_alternative', 'MRU_benefit_score': 'integrated_benefit_score', 'Response_pred': 'response_probability', 'Response_true': 'observed_response', 'Response': 'observed_response_alternative', 'PFS_risk': 'progression_risk', 'PFS_time': 'progression_time', 'PFS_status': 'progression_status', 'OS_risk': 'survival_risk', 'OS_time': 'survival_time', 'OS_status': 'survival_status', 'feature': 'feature_identifier', 'selected_feature': 'feature_identifier_alternative', 'feature_name': 'feature_name', 'modal': 'modality_identifier', 'modality': 'modality_name', 'feature_type': 'modality_type', 'original_feature': 'original_feature_label', 'original': 'original_label', 'feature_label': 'feature_label', 'axis': 'ecosystem_axis', 'integrated_score': 'axis_integrated_score', 'ko_score': 'microbial_function_score', 'metabolic_module_score': 'metabolic_module_score', 'shared_imaging_score': 'imaging_phenotype_score', 'module_id': 'module_identifier', 'display_label': 'module_display_label', 'label': 'module_label_alternative', 'module_label': 'module_label', 'module_type': 'module_type', 'type': 'module_type_alternative'}
            aliases = {}
            for canonical, key in column_keys.items():
                actual = get_column('agent.columns.' + key, canonical)
                if actual != canonical and actual in frame.columns:
                    if canonical in frame.columns or actual in aliases:
                        raise ValueError('Configured analysis columns are ambiguous.')
                    aliases[actual] = canonical
            return frame.rename(columns=aliases) if aliases else frame
        except Exception:
            raise ValueError('An analysis table could not be read.') from None

    def read_indexed_csv(path, required=True):
        df = read_table(path, required=required)
        if df is None:
            return None
        id_col = next((column for candidate in ['sample', 'sample_id', 'patient_id', 'id'] for column in df.columns if str(column).strip().lower() == candidate), None)
        if id_col is None:
            first = df.columns[0]
            if str(first).lower().startswith('unnamed'):
                id_col = first
            else:
                raise KeyError('An analysis table has no sample identifier column.') from None
        df = df.copy()
        df[id_col] = df[id_col].map(normalize_sample_id)
        df = df[df[id_col] != ''].copy()
        df = df.drop_duplicates(subset=[id_col], keep='first')
        df = df.set_index(id_col)
        df.index.name = 'sample'
        return df

    def image_to_data_uri(path):
        path = Path(path)
        if not path.exists():
            return ''
        suffix = path.suffix.lower()
        mime = 'image/jpeg' if suffix in {'.jpg', '.jpeg'} else 'image/webp' if suffix == '.webp' else 'image/png'
        b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f'data:{mime};base64,{b64}'

    def percentile_series(values, higher_is_higher=True):
        x = pd.to_numeric(values, errors='coerce')
        pct = x.rank(method='average', pct=True) * 100.0
        if not higher_is_higher:
            pct = 100.0 - pct
        return pct

    def infer_observed_response(x):
        if pd.isna(x):
            return '—'
        s = str(x).strip().lower()
        if s in {'1', 'responder', 'response', 'pr', 'cr', 'yes', 'true', 'r'}:
            return 'Responder'
        if s in {'0', 'non-responder', 'nonresponder', 'sd', 'pd', 'no', 'false', 'nr'}:
            return 'Non-responder'
        return str(x)

    def status_text(x, event_word='event', censor_word='censored'):
        if pd.isna(x):
            return '—'
        s = str(x).strip().lower()
        if s in {'1', 'event', 'death', 'dead', 'progression'}:
            return event_word
        if s in {'0', 'censored', 'alive', 'no'}:
            return censor_word
        return str(x)

    def canonical_mbp(row):
        raw = get_first(row, ['MBP_raw', 'MBP'], 'Unclassified')
        s = str(raw).upper().replace('\n', ' ')
        if 'MBP-III' in s or 'MBP III' in s:
            return 'MBP-III'
        if 'MBP-II' in s or 'MBP II' in s:
            return 'MBP-II'
        if 'MBP-I' in s or 'MBP I' in s:
            return 'MBP-I'
        return str(raw)

    def benefit_state_from_mbp(mbp):
        s = str(mbp).upper()
        if 'III' in s:
            return 'Low projected benefit'
        if 'II' in s:
            return 'Intermediate projected benefit'
        if 'I' in s:
            return 'High projected benefit'
        return 'Unclassified benefit state'

    def mbp_color(mbp):
        s = str(mbp).upper()
        if 'III' in s:
            return '#7B61FF'
        if 'II' in s:
            return '#F6C343'
        if 'I' in s:
            return '#13A6A6'
        return '#8AA0B5'

    def infer_modality_from_feature(feature):
        s = str(feature)
        prefix = s.split('__', 1)[0] if '__' in s else ''
        low = prefix.lower()
        if 'radio' in low or low == 'ct':
            return 'Radiomics'
        if 'meta' in low or 'micro' in low or low == 'ko':
            return 'Metagenomics'
        if 'urine' in low:
            return 'Urine'
        return prefix if prefix else 'Unknown'

    def short_feature_name(feature):
        s = str(feature)
        return s.split('__', 1)[1] if '__' in s else s

    def validate_mru_feature_package(mru_meta, signed_by_endpoint, abs_by_endpoint):
        feature_col = find_col(mru_meta, ['feature', 'selected_feature', 'feature_name'], required=True)
        modality_col = find_col(mru_meta, ['modal', 'modality', 'feature_type'], required=True)
        metadata_features = mru_meta[feature_col].astype(str).str.strip()
        metadata_modalities = mru_meta[modality_col].astype(str).str.strip()
        if len(metadata_features) != EXPECTED_MRU_FEATURE_COUNT or metadata_features.eq('').any() or metadata_features.nunique(dropna=False) != EXPECTED_MRU_FEATURE_COUNT:
            raise RuntimeError(f'MRU feature-package validation failed: metadata must contain exactly {EXPECTED_MRU_FEATURE_COUNT} unique, non-empty features.')
        observed_modality_counts = {str(k): int(v) for k, v in metadata_modalities.value_counts(dropna=False).to_dict().items()}
        if observed_modality_counts != EXPECTED_MRU_MODALITY_COUNTS:
            raise RuntimeError('MRU feature-package validation failed: modality counts must be Metagenomics=15, Radiomics=9 and Urine=15.')
        expected_features = set(metadata_features)
        for matrix_kind, matrices in {'signed': signed_by_endpoint, 'absolute': abs_by_endpoint}.items():
            for endpoint in ENDPOINTS:
                matrix = matrices.get(endpoint)
                if matrix is None:
                    raise RuntimeError(f'MRU feature-package validation failed: a required SHAP matrix is unavailable ({matrix_kind}, {endpoint}).')
                matrix_features = [str(c).strip() for c in matrix.columns]
                if len(matrix_features) != EXPECTED_MRU_FEATURE_COUNT or len(set(matrix_features)) != EXPECTED_MRU_FEATURE_COUNT or set(matrix_features) != expected_features:
                    raise RuntimeError('MRU feature-package validation failed: every endpoint and SHAP orientation must match the locked 39-feature metadata.')

    def load_current_outputs():
        mbp_df = read_table(MBP_PATIENT_PATH, required=True)
        sample_col = find_col(mbp_df, ['sample', 'sample_id', 'patient_id', 'id'], required=True)
        mbp_df = mbp_df.rename(columns={sample_col: 'sample'}).copy()
        mbp_df['sample'] = mbp_df['sample'].map(normalize_sample_id)
        mbp_df = mbp_df[mbp_df['sample'] != ''].drop_duplicates('sample').copy()
        if 'Response_pred' in mbp_df.columns:
            mbp_df['Response_pred_percentile'] = percentile_series(mbp_df['Response_pred'], True)
        if 'PFS_risk' in mbp_df.columns:
            mbp_df['PFS_risk_percentile'] = percentile_series(mbp_df['PFS_risk'], True)
        if 'OS_risk' in mbp_df.columns:
            mbp_df['OS_risk_percentile'] = percentile_series(mbp_df['OS_risk'], True)
        if 'MRU_benefit_score' in mbp_df.columns:
            mbp_df['MRU_percentile'] = percentile_series(mbp_df['MRU_benefit_score'], True)
        mru_signed_raw = {task: read_indexed_csv(path, required=True) for task, path in MRU_SHAP_SIGNED_PATHS.items()}
        mru_abs = {task: read_indexed_csv(path, required=True) for task, path in MRU_SHAP_ABS_PATHS.items()}
        mru_signed_benefit = {}
        for task in ENDPOINTS:
            mru_signed_benefit[task] = mru_signed_raw[task].copy()
            if task in {'PFS', 'OS'}:
                mru_signed_benefit[task] = -mru_signed_benefit[task]
        mru_meta = read_table(MRU_META_PATH, required=True)
        mru_importance = read_table(MRU_IMPORTANCE_PATH, required=False)
        mru_global_contrib = read_table(MRU_GLOBAL_MODALITY_CONTRIB_PATH, required=False)
        validate_mru_feature_package(mru_meta, mru_signed_raw, mru_abs)
        meta_feature_col = find_col(mru_meta, ['feature', 'selected_feature', 'feature_name'], required=True)
        meta_modal_col = find_col(mru_meta, ['modal', 'modality', 'feature_type'], required=True)
        meta_original_col = find_col(mru_meta, ['original_feature', 'original', 'feature_label'])
        meta_lookup = pd.DataFrame({'feature': mru_meta[meta_feature_col].astype(str), 'modality': mru_meta[meta_modal_col].astype(str)})
        if meta_original_col is not None:
            meta_lookup['original_feature'] = mru_meta[meta_original_col].astype(str)
        else:
            meta_lookup['original_feature'] = meta_lookup['feature'].map(short_feature_name)
        meta_lookup = meta_lookup.drop_duplicates('feature').set_index('feature')
        axis_df = read_table(ECOSYSTEM_AXIS_PATH, required=True)
        axis_sample_col = find_col(axis_df, ['sample', 'sample_id'], required=True)
        axis_df = axis_df.rename(columns={axis_sample_col: 'sample'}).copy()
        axis_df['sample'] = axis_df['sample'].map(normalize_sample_id)
        axis_df['integrated_score'] = pd.to_numeric(axis_df['integrated_score'], errors='coerce')
        axis_df['axis_percentile'] = axis_df.groupby('axis')['integrated_score'].rank(method='average', pct=True) * 100.0
        AXIS_COMPONENT_COLUMNS = {'ko_score': 'ko_percentile', 'metabolic_module_score': 'metabolic_percentile', 'shared_imaging_score': 'imaging_percentile'}
        for score_col, pct_col in AXIS_COMPONENT_COLUMNS.items():
            if score_col in axis_df.columns:
                axis_df[score_col] = pd.to_numeric(axis_df[score_col], errors='coerce')
                axis_df[pct_col] = axis_df.groupby('axis')[score_col].rank(method='average', pct=True) * 100.0
        module_meta = read_table(MODULE_META_PATH, required=False)
        module_score = read_indexed_csv(MODULE_SCORE_PATH, required=False)
        module_shap = {}
        for task, path in MODULE_SHAP_BENEFIT_PATHS.items():
            module_shap[task] = read_indexed_csv(path, required=False)
        module_contrib = read_table(MODULE_SHAP_CONTRIB_PATH, required=False)
        return {'mbp': mbp_df, 'mru_signed_raw': mru_signed_raw, 'mru_signed_benefit': mru_signed_benefit, 'mru_abs': mru_abs, 'mru_meta': mru_meta, 'meta_lookup': meta_lookup, 'mru_importance': mru_importance, 'mru_global_contrib': mru_global_contrib, 'axis': axis_df, 'module_meta': module_meta, 'module_score': module_score, 'module_shap': module_shap, 'module_contrib': module_contrib}
    try:
        DATA = load_current_outputs()
    except Exception:
        st.error('The analysis inputs could not be validated. The application has stopped.')
        st.stop()
    mbp_df = DATA['mbp']
    mru_signed_benefit = DATA['mru_signed_benefit']
    mru_abs = DATA['mru_abs']
    meta_lookup = DATA['meta_lookup']
    axis_df = DATA['axis']
    module_meta = DATA['module_meta']
    module_shap = DATA['module_shap']

    def patient_feature_frame(sample_id, endpoint):
        sample_id = normalize_sample_id(sample_id)
        signed = mru_signed_benefit[endpoint]
        abs_df = mru_abs[endpoint]
        if sample_id not in signed.index or sample_id not in abs_df.index:
            return pd.DataFrame()
        common_features = [c for c in signed.columns if c in abs_df.columns]
        s = pd.to_numeric(signed.loc[sample_id, common_features], errors='coerce')
        a = pd.to_numeric(abs_df.loc[sample_id, common_features], errors='coerce')
        out = pd.DataFrame({'feature': common_features, 'shap_value': s.values, 'abs_shap': a.values})

        def _meta(feat, field):
            if feat in meta_lookup.index:
                return meta_lookup.loc[feat, field]
            if field == 'modality':
                return infer_modality_from_feature(feat)
            return short_feature_name(feat)
        out['modality'] = out['feature'].map(lambda x: _meta(x, 'modality'))
        out['original_feature'] = out['feature'].map(lambda x: _meta(x, 'original_feature'))
        out['direction'] = np.where(out['shap_value'] >= 0, 'Favorable', 'Unfavorable')
        out = out[np.isfinite(out['abs_shap'])].copy()
        return out.sort_values('abs_shap', ascending=False)

    def patient_top_signals(sample_id, endpoint, n=12):
        return patient_feature_frame(sample_id, endpoint).head(n).copy()

    def patient_modality_contribution(sample_id, endpoint):
        d = patient_feature_frame(sample_id, endpoint)
        if d.empty:
            return pd.DataFrame(columns=['modality', 'contribution_percent'])
        g = d.groupby('modality', as_index=False)['abs_shap'].sum()
        total = g['abs_shap'].sum()
        g['contribution_percent'] = 100.0 * g['abs_shap'] / (total + 1e-12)
        return g.sort_values('contribution_percent', ascending=False)

    def contribution_lookup(sample_id):
        result = {task: {} for task in ENDPOINTS}
        for task in ENDPOINTS:
            d = patient_modality_contribution(sample_id, task)
            for _, r in d.iterrows():
                modal = str(r['modality'])
                if 'meta' in modal.lower() or 'micro' in modal.lower():
                    modal = 'Metagenomics'
                elif 'radio' in modal.lower() or modal.lower() == 'ct':
                    modal = 'Radiomics'
                elif 'urine' in modal.lower():
                    modal = 'Urine'
                result[task][modal] = as_float(r['contribution_percent'])
        return result

    def patient_axis_table(sample_id):
        sid = normalize_sample_id(sample_id)
        d = axis_df[axis_df['sample'] == sid].copy()
        if d.empty:
            return d
        d['axis'] = d['axis'].astype(str)
        d['axis_order'] = d['axis'].map({a: i for i, a in enumerate(AXIS_ORDER)}).fillna(99)
        return d.sort_values('axis_order')

    def patient_axis_percentile(sample_id, axis_name):
        d = patient_axis_table(sample_id)
        sub = d[d['axis'] == axis_name]
        if sub.empty:
            return np.nan
        return as_float(sub.iloc[0].get('axis_percentile', np.nan))

    def _component_position_label(percentile):
        x = as_float(percentile)
        if not np.isfinite(x):
            return 'N/A'
        if x >= 67:
            return 'more responder-like'
        if x <= 33:
            return 'less responder-like'
        return 'intermediate'

    def patient_axis_components(sample_id, axis_name):
        d = patient_axis_table(sample_id)
        sub = d[d['axis'] == axis_name]
        if sub.empty:
            return []
        r = sub.iloc[0]
        specs = [('🧬', 'KO', 'ko_score', 'ko_percentile'), ('🧪', 'Metabolic', 'metabolic_module_score', 'metabolic_percentile'), ('🩻', 'Imaging', 'shared_imaging_score', 'imaging_percentile')]
        out = []
        for icon, label, score_col, pct_col in specs:
            score = as_float(r.get(score_col, np.nan))
            pct = as_float(r.get(pct_col, np.nan))
            out.append({'icon': icon, 'label': label, 'score': score, 'percentile': pct, 'state': _component_position_label(pct)})
        return out

    def patient_top_modules(sample_id, endpoint, n=10):
        d = module_shap.get(endpoint)
        if d is None or sample_id not in d.index:
            return pd.DataFrame()
        vals = pd.to_numeric(d.loc[sample_id], errors='coerce')
        out = pd.DataFrame({'module_id': vals.index.astype(str), 'shap_value': vals.values})
        out['abs_shap'] = out['shap_value'].abs()
        out = out[np.isfinite(out['abs_shap'])].copy()
        if module_meta is not None and (not module_meta.empty):
            mid = find_col(module_meta, ['module_id'], required=True)
            mlabel = find_col(module_meta, ['display_label', 'label', 'module_label'])
            mtype = find_col(module_meta, ['module_type', 'type'])
            maxis = find_col(module_meta, ['axis'])
            keep = [c for c in [mid, mlabel, mtype, maxis] if c is not None]
            mm = module_meta[keep].drop_duplicates(mid).copy()
            mm = mm.rename(columns={mid: 'module_id'})
            if mlabel is not None:
                mm = mm.rename(columns={mlabel: 'display_label'})
            if mtype is not None:
                mm = mm.rename(columns={mtype: 'module_type'})
            if maxis is not None:
                mm = mm.rename(columns={maxis: 'axis'})
            out = out.merge(mm, on='module_id', how='left')
        if 'display_label' not in out.columns:
            out['display_label'] = out['module_id']
        if 'module_type' not in out.columns:
            out['module_type'] = 'Module'
        if 'axis' not in out.columns:
            out['axis'] = ''
        out['direction'] = np.where(out['shap_value'] >= 0, 'Favorable', 'Unfavorable')
        return out.sort_values('abs_shap', ascending=False).head(n).copy()
    st.sidebar.title('Avatar-Agent')
    sample_ids = mbp_df['sample'].astype(str).tolist()
    display_ids = [f'Patient {i + 1:03d}' for i in range(len(sample_ids))]
    selected_display = st.sidebar.selectbox('Select patient avatar', display_ids, index=0)
    selected_sample = sample_ids[display_ids.index(selected_display)]
    endpoint_selected = st.sidebar.selectbox('Evidence endpoint', ENDPOINTS, index=0)
    row = mbp_df[mbp_df['sample'] == selected_sample].iloc[0]
    sample_id = selected_sample
    mbp = canonical_mbp(row)
    benefit_state = benefit_state_from_mbp(mbp)
    accent = mbp_color(mbp)
    mru_score = get_first(row, ['MRU_benefit_score'], np.nan)
    mru_pct = get_first(row, ['MRU_percentile'], np.nan)
    response_pred = get_first(row, ['Response_pred'], np.nan)
    response_pct = get_first(row, ['Response_pred_percentile'], np.nan)
    response_true = infer_observed_response(get_first(row, ['Response_true', 'Response'], np.nan))
    pfs_risk = get_first(row, ['PFS_risk'], np.nan)
    pfs_risk_pct = get_first(row, ['PFS_risk_percentile'], np.nan)
    pfs_time = get_first(row, ['PFS_time'], np.nan)
    pfs_status = status_text(get_first(row, ['PFS_status'], np.nan), 'progression/event', 'censored')
    os_risk = get_first(row, ['OS_risk'], np.nan)
    os_risk_pct = get_first(row, ['OS_risk_percentile'], np.nan)
    os_time = get_first(row, ['OS_time'], np.nan)
    os_status = status_text(get_first(row, ['OS_status'], np.nan), 'death', 'alive/censored')
    contrib = contribution_lookup(sample_id)
    axis_patient = patient_axis_table(sample_id)

    def _axis_pct(axis_name):
        return patient_axis_percentile(sample_id, axis_name)
    axis_amino_pct = _axis_pct('Amino Acids & TME Regulation Axis')
    axis_lipid_pct = _axis_pct('Lipid & 1-Carbon Metabolism Axis')
    axis_scfa_pct = _axis_pct('SCFA & BCAA Energy Metabolism Axis')

    def build_avatar_report():
        top_resp = patient_top_signals(sample_id, 'Response', n=3)
        feature_text = ''
        if not top_resp.empty:
            items = []
            for _, r in top_resp.iterrows():
                sign = 'favorable' if r['shap_value'] >= 0 else 'unfavorable'
                items.append(f"{r['original_feature']} ({sign})")
            feature_text = ' The strongest response-oriented MRU signals include ' + ', '.join(items) + '.'
        if not axis_patient.empty:
            aa = axis_patient.sort_values('axis_percentile', ascending=False).iloc[0]
            axis_text = f" The most responder-like ecosystem-axis position is {AXIS_SHORT.get(str(aa['axis']), str(aa['axis']))} ({as_float(aa['axis_percentile']):.0f}th percentile)."
        else:
            axis_text = ''
        return f'The Avatar-Agent materializes {selected_display} as {mbp}, corresponding to a {benefit_state.lower()}. The early-response projection is {fmt_prob(response_pred)} ({fmt_pct(response_pct, 0)} cohort percentile). PFS and OS risk positions are {fmt_pct(pfs_risk_pct, 0)} and {fmt_pct(os_risk_pct, 0)}, respectively. The integrated MRU benefit score is {fmt_num(mru_score, 3)} ({fmt_pct(mru_pct, 0)} cohort percentile).' + feature_text + axis_text + ' This visualization is a retrospective projection/interpretation prototype and is not intended for independent clinical decision-making.'
    report_text = build_avatar_report()

    def svg_icon(kind):
        icons = {}
        icons['radiomics'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <circle cx="32" cy="32" r="22" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="32" cy="32" r="14" fill="none" stroke="currentColor" stroke-width="2.5" opacity="0.75"/>\n      <path d="M22 35 c6-7 19-7 23 1 c-4 7-17 7-25 2 c-2-1-1-2 2-3z" fill="none" stroke="currentColor" stroke-width="2.5"/>\n      <circle cx="32" cy="36" r="4" fill="currentColor" opacity="0.85"/>\n    </svg>\n    '
        icons['urine'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <rect x="20" y="20" width="24" height="34" rx="3" fill="none" stroke="currentColor" stroke-width="3"/>\n      <rect x="17" y="13" width="30" height="8" rx="2" fill="none" stroke="currentColor" stroke-width="3"/>\n      <path d="M22 38 h20 v12 a3 3 0 0 1-3 3 H25 a3 3 0 0 1-3-3z" fill="currentColor" opacity="0.18"/>\n    </svg>\n    '
        icons['ko'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <path d="M21 12 c22 9 22 31 0 40" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>\n      <path d="M43 12 c-22 9-22 31 0 40" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>\n      <line x1="24" y1="20" x2="40" y2="20" stroke="currentColor" stroke-width="2.5"/>\n      <line x1="21" y1="30" x2="43" y2="30" stroke="currentColor" stroke-width="2.5"/>\n      <line x1="24" y1="40" x2="40" y2="40" stroke="currentColor" stroke-width="2.5"/>\n    </svg>\n    '
        icons['response'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <path d="M13 45 L27 31 L37 39 L51 20" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>\n      <path d="M40 20 h11 v11" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round"/>\n    </svg>\n    '
        icons['pfs'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <path d="M32 8 L50 15 v15 c0 14-8 22-18 28 C22 52 14 44 14 30 V15z" fill="none" stroke="currentColor" stroke-width="3.5"/>\n      <path d="M23 33 l7 7 l13-17" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>\n    </svg>\n    '
        icons['os'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <path d="M21 10 h22 M21 54 h22" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>\n      <path d="M24 12 c0 11 6 15 13 20 c-7 5-13 9-13 20" fill="none" stroke="currentColor" stroke-width="3.5"/>\n      <path d="M40 12 c0 11-6 15-13 20 c7 5 13 9 13 20" fill="none" stroke="currentColor" stroke-width="3.5"/>\n    </svg>\n    '
        icons['amino'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <circle cx="18" cy="34" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="32" cy="22" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="46" cy="35" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <line x1="22" y1="31" x2="28" y2="25" stroke="currentColor" stroke-width="3"/>\n      <line x1="36" y1="25" x2="42" y2="32" stroke="currentColor" stroke-width="3"/>\n      <path d="M20 47 C29 40 38 40 46 47" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>\n    </svg>\n    '
        icons['lipid'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <circle cx="22" cy="18" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="42" cy="18" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <line x1="20" y1="23" x2="15" y2="47" stroke="currentColor" stroke-width="3"/>\n      <line x1="24" y1="23" x2="30" y2="47" stroke="currentColor" stroke-width="3"/>\n      <line x1="40" y1="23" x2="34" y2="47" stroke="currentColor" stroke-width="3"/>\n      <line x1="44" y1="23" x2="49" y2="47" stroke="currentColor" stroke-width="3"/>\n    </svg>\n    '
        icons['scfa'] = '\n    <svg viewBox="0 0 64 64" class="avatar-svg-icon">\n      <circle cx="15" cy="33" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="29" cy="24" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <circle cx="43" cy="33" r="5" fill="none" stroke="currentColor" stroke-width="3"/>\n      <line x1="19" y1="30" x2="25" y2="27" stroke="currentColor" stroke-width="3"/>\n      <line x1="33" y1="27" x2="39" y2="30" stroke="currentColor" stroke-width="3"/>\n      <path d="M37 42 l-7 12 h7 l-3 8 l13-16 h-8 l4-4z" fill="currentColor" opacity="0.85"/>\n    </svg>\n    '
        return icons.get(kind, '')

    def _contrib_pct(modality, endpoint):
        return as_float(contrib.get(endpoint, {}).get(modality, np.nan))

    def source_card(icon_html, title, subtitle, modality, beam_class):
        vals = [_contrib_pct(modality, t) for t in ENDPOINTS]
        labels = [fmt_pct(v, 0) for v in vals]
        widths = [0 if not np.isfinite(v) else clamp(v) for v in vals]
        return f'\n    <div class="source-card">\n      <div class="source-icon {beam_class}">{icon_html}</div>\n      <div class="source-content">\n        <div class="source-title">{esc(title)}</div>\n        <div class="source-subtitle">{esc(subtitle)}</div>\n        <div class="mini-contrib">\n          <span>R {labels[0]}</span><div class="tiny-bar"><div style="width:{widths[0]}%"></div></div>\n          <span>P {labels[1]}</span><div class="tiny-bar blue"><div style="width:{widths[1]}%"></div></div>\n          <span>O {labels[2]}</span><div class="tiny-bar purple"><div style="width:{widths[2]}%"></div></div>\n        </div>\n      </div>\n      <div class="source-tag">MRU CORE</div>\n    </div>\n    '

    def benefit_card(icon_html, title, main_label, main_value, obs_label, obs_value, color_class, meter_value, meter_label):
        meter_w = 50 if not np.isfinite(as_float(meter_value)) else clamp(meter_value)
        meter_txt = 'N/A' if not np.isfinite(as_float(meter_value)) else f'{meter_w:.0f}%'
        return f'\n    <div class="benefit-card {color_class}">\n      <div class="benefit-icon">{icon_html}</div>\n      <div class="benefit-body">\n        <div class="benefit-title">{esc(title)}</div>\n        <div class="benefit-label">{esc(main_label)}</div>\n        <div class="benefit-value">{esc(main_value)}</div>\n        <div class="benefit-observed">{esc(obs_label)} <b>{esc(obs_value)}</b></div>\n      </div>\n      <div class="projection-meter">\n        <div class="meter-label">{esc(meter_label)}</div>\n        <div class="meter-track">\n          <div class="meter-fill" style="width:{meter_w}%"></div>\n          <div class="meter-dot" style="left:{meter_w}%"></div>\n        </div>\n        <div class="meter-scale"><span>low</span><span>{meter_txt}</span><span>high</span></div>\n      </div>\n    </div>\n    '

    def axis_patient_state(percentile):
        x = as_float(percentile)
        if not np.isfinite(x):
            return 'Integrated patient position unavailable'
        if x >= 67:
            return 'Integrated state: more responder-like'
        if x <= 33:
            return 'Integrated state: less responder-like'
        return 'Integrated state: intermediate'

    def _fmt_component_score(x):
        x = as_float(x)
        if not np.isfinite(x):
            return 'score N/A'
        return f'score {x:+.2f}'

    def _fmt_component_percentile(x):
        x = as_float(x)
        if not np.isfinite(x):
            return 'N/A'
        return f'{x:.0f}th pct'

    def axis_bar(axis_name, percentile, color_class, icon_html, sample_id):
        title = AXIS_DISPLAY.get(axis_name, axis_name)
        description = AXIS_DESCRIPTIONS.get(axis_name, '')
        components = patient_axis_components(sample_id, axis_name)
        w = 50 if not np.isfinite(as_float(percentile)) else clamp(percentile)
        val = 'N/A' if not np.isfinite(as_float(percentile)) else f'{w:.0f}%'
        patient_state = axis_patient_state(percentile)
        chip_html = ''.join((f"""\n        <div class="evidence-chip">\n          <span class="chip-emoji">{esc(comp['icon'])}</span>\n          <div class="chip-content">\n            <div class="chip-top">\n              <span class="chip-kind">{esc(comp['label'])}</span>\n              <span class="chip-pct">{esc(_fmt_component_percentile(comp['percentile']))}</span>\n            </div>\n            <div class="chip-bottom">\n              <span class="chip-score">{esc(_fmt_component_score(comp['score']))}</span>\n              <span class="chip-state">{esc(comp['state'])}</span>\n            </div>\n          </div>\n        </div>\n        """ for comp in components))
        return f'\n    <div class="axis-card {color_class}">\n      <div class="axis-card-head">\n        <div class="axis-icon">{icon_html}</div>\n        <div class="axis-heading">\n          <div class="axis-title">{esc(title)}</div>\n          <div class="axis-description">{esc(description)}</div>\n        </div>\n        <div class="axis-value">{val}</div>\n      </div>\n\n      <div class="axis-position">\n        <div class="axis-track">\n          <div class="axis-fill" style="width:{w}%"></div>\n          <div class="axis-knob" style="left:{w}%"></div>\n        </div>\n        <div class="axis-scale">\n          <span>less responder-like</span>\n          <span>more responder-like</span>\n        </div>\n      </div>\n\n      <div class="anchor-label">Current patient — component states</div>\n      <div class="evidence-chips">{chip_html}</div>\n\n      <div class="axis-takeaway">{esc(patient_state)}</div>\n    </div>\n    '

    def render_projection_html():
        body_img_uri = image_to_data_uri(BODY_IMG_PATH)
        radiomics = source_card(svg_icon('radiomics'), 'CT radiomics', 'Tumor imaging features', 'Radiomics', 'beam-blue')
        metagenomics = source_card(svg_icon('ko'), 'Functional metagenomics / KO', 'Microbial functions & pathways', 'Metagenomics', 'beam-purple')
        urine = source_card(svg_icon('urine'), 'Urine metabolomics', 'Excretory metabolome', 'Urine', 'beam-cyan')
        response_card = benefit_card(svg_icon('response'), 'Early response', 'Projected probability', fmt_prob(response_pred), 'Observed:', response_true, 'green', response_pct, 'Response cohort position')
        pfs_card = benefit_card(svg_icon('pfs'), 'Durable benefit / PFS', 'Risk score', fmt_num(pfs_risk, 3), 'Observed PFS:', f'{fmt_months(pfs_time)} • {pfs_status}', 'blue', pfs_risk_pct, 'Risk cohort position')
        os_card = benefit_card(svg_icon('os'), 'Overall survival', 'Risk score', fmt_num(os_risk, 3), 'Observed OS:', f'{fmt_months(os_time)} • {os_status}', 'purple', os_risk_pct, 'Risk cohort position')
        amino_axis = axis_bar(AXIS_ORDER[0], axis_amino_pct, 'blue', svg_icon('amino'), sample_id)
        lipid_axis = axis_bar(AXIS_ORDER[1], axis_lipid_pct, 'purple', svg_icon('lipid'), sample_id)
        scfa_axis = axis_bar(AXIS_ORDER[2], axis_scfa_pct, 'teal', svg_icon('scfa'), sample_id)
        css = '\n    <style>\n    :root{\n      --navy:#071E4D; --blue:#2567C9; --teal:#0C98A4; --cyan:#40D6E8;\n      --purple:#6E4CCB; --soft:#F7FBFF; --line:#C9D8E8; --text:#0B1D44;\n      --muted:#6E7E93; --coral:#E56D57; --gold:#F6C343;\n    }\n    *{box-sizing:border-box;} body{margin:0;}\n    .avatar-wrap{width:100%;min-height:1230px;padding:18px 22px;background:linear-gradient(180deg,#fff 0%,#f5fbff 100%);color:var(--text);font-family:Arial,Helvetica,sans-serif;border:1px solid #dbe7f3;border-radius:18px;overflow:hidden;}\n    .topbar{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:10px;}\n    .title-main{font-size:48px;line-height:1;font-weight:900;color:var(--navy);letter-spacing:-1px;}\n    .title-sub{font-size:26px;font-weight:800;color:#087488;margin-top:4px;}\n    .title-desc{font-size:14px;color:#172A53;margin-top:4px;max-width:900px;}\n    .proto-badge{border:1px solid #178CA0;color:#087488;background:#f7fdff;border-radius:8px;padding:9px 14px;font-weight:700;font-size:14px;}\n    .dashboard-top{position:relative;} .main-grid{display:grid;grid-template-columns:29% 40% 31%;gap:18px;align-items:stretch;position:relative;z-index:2;}\n    .panel-title{text-align:center;color:#062F8F;font-size:22px;font-weight:900;margin:2px 0 10px;}\n    .panel-title:after{content:"";display:block;width:130px;height:2px;background:linear-gradient(90deg,transparent,var(--teal),transparent);margin:5px auto 0;}\n    .sources,.benefits{padding:6px 7px 4px;}\n    .source-card{position:relative;display:flex;align-items:center;height:150px;min-height:150px;padding:14px 68px 14px 12px;margin:12px 0;border:1px solid #bcd2e6;border-radius:17px;background:rgba(255,255,255,.88);box-shadow:0 8px 24px rgba(30,82,136,.08);overflow:hidden;}\n    .source-icon{width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:13px;border:2px solid rgba(255,255,255,.8);color:white;}\n    .source-icon .avatar-svg-icon{width:36px;height:36px;display:block;color:white;}\n    .beam-blue{background:linear-gradient(145deg,#e7f3ff,#2467cb);} .beam-cyan{background:linear-gradient(145deg,#e7fbff,#11a5b9);} .beam-purple{background:linear-gradient(145deg,#f0eaff,#6e4ccb);}\n    .source-title{font-size:20px;font-weight:850;color:#102253;}.source-subtitle{font-size:14px;color:#52677f;margin:3px 0 8px;}\n    .mini-contrib{display:grid;grid-template-columns:auto 52px auto 52px auto 52px;gap:6px;align-items:center;font-size:12px;color:#0d2250;white-space:nowrap;}\n    .tiny-bar{height:6px;border-radius:20px;background:#dce7f1;overflow:hidden;}.tiny-bar div{height:100%;background:#10a4ac;}.tiny-bar.blue div{background:#2567c9;}.tiny-bar.purple div{background:#6e4ccb;}\n    .source-tag{position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:11px;font-weight:900;padding:5px 6px;border-radius:6px;border:1px solid #1d62b6;color:#1d62b6;background:#edf6ff;}\n    .chamber{min-height:540px;border:1px solid #c8daf1;border-radius:18px;position:relative;background:radial-gradient(circle at 50% 42%,rgba(34,114,202,.18),transparent 43%),radial-gradient(circle at 50% 42%,rgba(20,183,201,.14),transparent 31%),rgba(255,255,255,.58);overflow:hidden;box-shadow:inset 0 0 50px rgba(78,151,216,.12);}\n    .orb{position:absolute;left:50%;top:43%;transform:translate(-50%,-50%);width:405px;height:405px;border-radius:50%;background:radial-gradient(circle at 50% 40%,rgba(255,255,255,.3),transparent 22%),radial-gradient(circle at 45% 55%,rgba(9,68,121,.94),rgba(9,108,128,.70) 56%,rgba(202,235,255,.40) 76%,transparent 78%);border:2px solid rgba(90,160,230,.42);box-shadow:0 0 28px rgba(60,176,229,.34),inset 0 0 42px rgba(255,255,255,.34);}\n    .project-body-img{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);width:520px;max-width:94%;height:auto;z-index:2;filter:drop-shadow(0 0 16px rgba(80,190,230,.30));opacity:.98;pointer-events:none;}\n    .avatar-svg{position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);width:230px;height:310px;z-index:2;}\n    .core-badge{position:absolute;left:50%;bottom:24px;transform:translateX(-50%);width:285px;padding:11px 14px;border-radius:12px;background:linear-gradient(180deg,rgba(8,43,78,.96),rgba(7,87,117,.93));border:1px solid rgba(255,255,255,.42);text-align:center;color:white;z-index:4;}\n    .core-badge .small{font-size:14px;font-weight:800;}.core-badge .mbp{font-size:30px;font-weight:950;color:var(--gold);margin:2px 0;}.core-badge .state{font-size:15px;font-weight:700;}.core-badge .mru{font-size:14px;font-weight:800;margin-top:4px;}\n    .benefits{display:flex;flex-direction:column;}\n    .benefit-card{height:150px;min-height:150px;margin:12px 0;display:grid;grid-template-columns:62px minmax(0,1fr) 140px;gap:12px;align-items:center;border:1px solid #bcd2e6;border-radius:17px;background:rgba(255,255,255,.88);box-shadow:0 10px 26px rgba(30,82,136,.10);padding:13px 14px;overflow:hidden;}\n    .benefit-card.green{--card-color:#0b9693;color:var(--card-color);}.benefit-card.blue{--card-color:#2666cf;color:var(--card-color);}.benefit-card.purple{--card-color:#6a4bc4;color:var(--card-color);}\n    .benefit-icon{width:54px;height:54px;min-width:54px;min-height:54px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:var(--card-color);color:white;overflow:hidden;justify-self:center;}.benefit-icon .avatar-svg-icon{width:32px;height:32px;color:white;display:block;}\n    .benefit-title{font-size:22px;font-weight:950;margin-bottom:5px;color:currentColor;line-height:1.08;}.benefit-label{font-size:13px;color:#51657f;line-height:1.15;}.benefit-value{font-size:32px;font-weight:950;color:currentColor;line-height:1.05;margin:2px 0 7px;}.benefit-observed{font-size:13.5px;color:#233752;line-height:1.2;}\n    .projection-meter{width:150px;padding:7px 3px;color:currentColor;}.meter-label{font-size:12px;color:#5f7289;margin-bottom:8px;font-weight:700;}.meter-track{position:relative;height:10px;border-radius:50px;background:#dce7f1;}.meter-fill{height:100%;border-radius:50px;background:linear-gradient(90deg,rgba(255,255,255,.35),currentColor);opacity:.85;}.meter-dot{position:absolute;top:50%;transform:translate(-50%,-50%);width:17px;height:17px;border-radius:50%;background:currentColor;border:3px solid rgba(255,255,255,.90);}.meter-scale{display:flex;justify-content:space-between;margin-top:7px;font-size:10.5px;color:#6e7e93;font-weight:700;}\n    .ecosystem{margin-top:14px;border:1px solid #c7d8ee;border-radius:16px;background:rgba(255,255,255,.84);padding:16px 14px 16px;box-shadow:0 8px 24px rgba(30,82,136,.055);}\n    .eco-title{text-align:center;color:#063597;font-size:24px;font-weight:950;margin-bottom:6px;line-height:1.12;}\n    .eco-subtitle{text-align:center;color:#6e7e93;font-size:13px;margin-bottom:14px;line-height:1.25;}\n    .axis-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}\n    .axis-card{min-width:0;padding:14px 14px 12px;border:1px solid #d5e2ef;border-radius:14px;background:linear-gradient(180deg,#ffffff 0%,#f9fcff 100%);box-shadow:0 6px 18px rgba(30,82,136,.055);}\n    .axis-card.teal{--axis:#0aa6a7;--axis-soft:#eefafa;}\n    .axis-card.blue{--axis:#276dd5;--axis-soft:#f1f6ff;}\n    .axis-card.purple{--axis:#7654c7;--axis-soft:#f6f2ff;}\n    .axis-card-head{display:grid;grid-template-columns:46px minmax(0,1fr) 62px;gap:10px;align-items:center;}\n    .axis-icon{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid var(--axis);color:var(--axis);background:white;}\n    .axis-icon .avatar-svg-icon{width:28px;height:28px;color:var(--axis);}\n    .axis-heading{min-width:0;}\n    .axis-title{font-size:17px;line-height:1.15;color:#0d2250;font-weight:900;margin:0 0 4px;}\n    .axis-description{font-size:12.5px;line-height:1.28;color:#718096;min-height:30px;}\n    .axis-value{font-size:30px;font-weight:950;color:var(--axis);text-align:right;}\n    .axis-position{margin:11px 2px 10px;}\n    .axis-track{position:relative;height:10px;border-radius:50px;background:#d9e4ef;}\n    .axis-fill{height:100%;border-radius:50px;background:linear-gradient(90deg,rgba(255,255,255,.15),var(--axis));}\n    .axis-knob{position:absolute;top:50%;transform:translate(-50%,-50%);width:15px;height:15px;border-radius:50%;background:var(--axis);border:3px solid rgba(255,255,255,.90);box-shadow:0 1px 4px rgba(0,0,0,.12);}\n    .axis-scale{display:flex;justify-content:space-between;font-size:11px;color:#7b8ba0;margin-top:5px;}\n    .anchor-label{font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:#8997aa;font-weight:800;margin:10px 0 7px;}\n    .evidence-chips{display:flex;flex-direction:column;gap:7px;}\n    .evidence-chip{display:flex;align-items:center;gap:9px;min-width:0;padding:10px 11px;border-radius:9px;background:var(--axis-soft);border:1px solid color-mix(in srgb,var(--axis) 18%,#dbe7f1);font-size:12px;line-height:1.18;color:#263b5b;}\n    .chip-emoji{font-size:18px;line-height:1;flex:0 0 auto;}\n    .chip-content{flex:1;min-width:0;}\n    .chip-top,.chip-bottom{display:flex;align-items:center;justify-content:space-between;gap:6px;}\n    .chip-top{margin-bottom:4px;}\n    .chip-kind{font-weight:900;color:#354e72;}\n    .chip-pct{font-weight:950;color:var(--axis);white-space:nowrap;}\n    .chip-score{font-family:Arial,Helvetica,sans-serif;color:#536984;font-size:11px;white-space:nowrap;}\n    .chip-state{font-weight:800;color:#6a7d96;font-size:10.5px;text-align:right;}\n    .axis-takeaway{margin-top:9px;padding-top:8px;border-top:1px dashed #d8e3ee;text-align:center;font-size:12px;font-weight:850;color:var(--axis);}\n    .footer-note{text-align:center;margin-top:12px;font-size:13px;color:#6e7e93;font-style:italic;line-height:1.3;}\n    .flow-overlay{position:absolute;left:0;top:34px;width:100%;height:575px;z-index:1;pointer-events:none;overflow:visible;}\n    .flow-overlay path{fill:none;stroke-linecap:round;stroke-width:3.2;}\n    .flow-overlay .arrow-blue{stroke:#2567C9;}\n    .flow-overlay .arrow-purple{stroke:#6E4CCB;}\n    .flow-overlay .arrow-cyan{stroke:#11A5B9;}\n    .flow-overlay .arrow-green{stroke:#0C9894;}\n    .flow-overlay .arrow-pfs{stroke:#2666CF;}\n    .flow-overlay .arrow-os{stroke:#6A4BC4;}\n    .flow-overlay .ghost-node{fill:white;stroke-width:2.5;}\n    </style>\n    '
        fallback_avatar = '\n    <svg class="avatar-svg" viewBox="0 0 240 320">\n      <defs><linearGradient id="bodyGrad" x1="0" x2="1"><stop offset="0%" stop-color="#BDEFFF" stop-opacity="0.35"/><stop offset="100%" stop-color="#FFFFFF" stop-opacity="0.08"/></linearGradient></defs>\n      <circle cx="120" cy="35" r="26" fill="url(#bodyGrad)" stroke="#D6F7FF" stroke-width="2"/>\n      <path d="M80 78 C92 55 148 55 160 78 C176 112 176 168 160 218 C145 266 95 266 80 218 C64 168 64 112 80 78Z" fill="url(#bodyGrad)" stroke="#D6F7FF" stroke-width="2"/>\n      <path d="M85 130 C110 110 152 116 166 140 C152 164 103 160 78 149 C72 141 74 135 85 130Z" fill="#E56D57" stroke="#FFE9A7" stroke-width="2" opacity="0.92"/>\n      <circle cx="112" cy="142" r="17" fill="#FFB86B" stroke="#FFF1C8" stroke-width="2"/>\n    </svg>\n    '
        body_visual = f'<img class="project-body-img" src="{body_img_uri}" />' if body_img_uri else fallback_avatar
        body = f'\n    <div class="avatar-wrap">\n      <div class="topbar">\n        <div>\n          <div class="title-main">Avatar-Agent</div>\n          <div class="title-sub">Multimodal Avatar Projection Interface</div>\n          \n        </div>\n        \n      </div>\n\n      <div class="dashboard-top">\n        <svg class="flow-overlay" viewBox="0 0 1000 575" preserveAspectRatio="none" aria-hidden="true">\n          <defs>\n            <marker id="arrowBlue" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#2567C9"></path>\n            </marker>\n            <marker id="arrowPurple" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#6E4CCB"></path>\n            </marker>\n            <marker id="arrowCyan" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#11A5B9"></path>\n            </marker>\n            <marker id="arrowGreen" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#0C9894"></path>\n            </marker>\n            <marker id="arrowPfs" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#2666CF"></path>\n            </marker>\n            <marker id="arrowOs" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n              <path d="M0,0 L9,4.5 L0,9 z" fill="#6A4BC4"></path>\n            </marker>\n          </defs>\n\n          <path class="arrow-blue" d="M250,145 C320,145 355,160 412,174" marker-end="url(#arrowBlue)"></path>\n          <circle class="ghost-node" cx="250" cy="145" r="5" stroke="#2567C9"></circle>\n\n          <path class="arrow-purple" d="M250,289 C320,289 355,289 412,289" marker-end="url(#arrowPurple)"></path>\n          <circle class="ghost-node" cx="250" cy="289" r="5" stroke="#6E4CCB"></circle>\n\n          <path class="arrow-cyan" d="M250,432 C320,432 355,416 412,400" marker-end="url(#arrowCyan)"></path>\n          <circle class="ghost-node" cx="250" cy="432" r="5" stroke="#11A5B9"></circle>\n\n          <path class="arrow-green" d="M590,174 C652,160 695,145 748,145" marker-end="url(#arrowGreen)"></path>\n          <circle class="ghost-node" cx="590" cy="174" r="5" stroke="#0C9894"></circle>\n\n          <path class="arrow-pfs" d="M590,289 C652,289 695,289 748,289" marker-end="url(#arrowPfs)"></path>\n          <circle class="ghost-node" cx="590" cy="289" r="5" stroke="#2666CF"></circle>\n\n          <path class="arrow-os" d="M590,400 C652,416 695,432 748,432" marker-end="url(#arrowOs)"></path>\n          <circle class="ghost-node" cx="590" cy="400" r="5" stroke="#6A4BC4"></circle>\n        </svg>\n\n        <div class="main-grid">\n          <div class="sources">\n            <div class="panel-title">Locked MRU Projection Basis</div>\n            {radiomics}\n            {metagenomics}\n            {urine}\n          </div>\n\n          <div>\n            <div class="panel-title">Patient Avatar Projection Chamber</div>\n            <div class="chamber">\n              <div class="orb"></div>\n              {body_visual}\n              <div class="core-badge" style="border-color:{esc(accent)};">\n                <div class="small">Patient Avatar Core</div>\n                <div class="mbp">{esc(mbp)}</div>\n                <div class="state">{esc(benefit_state)}</div>\n                <div class="mru">MRU benefit percentile: {fmt_pct(mru_pct, 0)}</div>\n              </div>\n            </div>\n          </div>\n\n          <div class="benefits">\n            <div class="panel-title">Benefit Projections</div>\n            {response_card}\n            {pfs_card}\n            {os_card}\n          </div>\n        </div>\n      </div>\n\n      <div class="ecosystem">\n        <div class="eco-title">Cross-modal Ecosystem Field — patient position within the responder-oriented axes</div>\n        \n        <div class="axis-grid">{amino_axis}{lipid_axis}{scfa_axis}</div>\n      </div>\n\n      \n    </div>\n    '
        return css + body
    components.html(render_projection_html(), height=1260, scrolling=False)
    st.markdown('## Projection Evidence Layer')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['MRU attribution', 'Ecosystem interpretation', 'Cohort context', 'Observed readout', 'Avatar report'])
    with tab1:
        st.subheader(f'Patient-specific MRU attribution: {endpoint_selected}')
        c1, c2 = st.columns([1.0, 1.2])
        with c1:
            rows = []
            for task in ENDPOINTS:
                d = patient_modality_contribution(sample_id, task)
                for _, r in d.iterrows():
                    modal = str(r['modality'])
                    if 'meta' in modal.lower() or 'micro' in modal.lower():
                        modal = 'Metagenomics'
                    elif 'radio' in modal.lower() or modal.lower() == 'ct':
                        modal = 'Radiomics'
                    elif 'urine' in modal.lower():
                        modal = 'Urine'
                    rows.append({'Endpoint': task, 'Modality': modal, 'Contribution (%)': as_float(r['contribution_percent'])})
            contrib_plot = pd.DataFrame(rows)
            if not contrib_plot.empty:
                fig = px.bar(contrib_plot, x='Endpoint', y='Contribution (%)', color='Modality', barmode='stack', text_auto='.0f', category_orders={'Endpoint': ENDPOINTS, 'Modality': MRU_MODALITIES}, title='Patient-specific modality contribution from |SHAP|')
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10), legend_title_text='MRU modality')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No patient-specific MRU SHAP contribution could be derived.')
        with c2:
            top_sub = patient_top_signals(sample_id, endpoint_selected, n=12)
            if not top_sub.empty:
                p = top_sub.sort_values('shap_value', ascending=True).copy()
                p['feature_short'] = p['original_feature'].astype(str).str.slice(0, 48)
                p['plot_color'] = np.where(p['shap_value'] >= 0, 'Favorable', 'Unfavorable')
                fig = px.bar(p, x='shap_value', y='feature_short', orientation='h', color='plot_color', hover_data=['modality', 'feature', 'abs_shap'], color_discrete_map={'Favorable': '#0C98A4', 'Unfavorable': '#E56D57'}, title=f'Top MRU drivers — {endpoint_selected}')
                fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10), xaxis_title='Benefit-oriented signed SHAP', yaxis_title='', legend_title_text='Direction')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No patient-level MRU SHAP signals were found for this patient.')
        top_table = patient_top_signals(sample_id, endpoint_selected, n=20)
        if not top_table.empty:
            st.dataframe(top_table[['original_feature', 'modality', 'shap_value', 'abs_shap', 'direction']], use_container_width=True, hide_index=True)
    with tab2:
        st.subheader('Cross-modal ecosystem interpretation')
        c1, c2 = st.columns([0.9, 1.3])
        with c1:
            d = patient_axis_table(sample_id)
            if not d.empty:
                p = d[['axis', 'axis_percentile']].copy()
                p['Axis'] = p['axis'].map(AXIS_SHORT).fillna(p['axis'])
                fig = px.bar(p, x='axis_percentile', y='Axis', orientation='h', range_x=[0, 100], text='axis_percentile', title='Responder-oriented ecosystem-axis percentile')
                fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
                fig.update_layout(height=330, margin=dict(l=10, r=35, t=50, b=10), xaxis_title='Cohort percentile', yaxis_title='')
                st.plotly_chart(fig, use_container_width=True)
                cols = [c for c in ['axis', 'ko_score', 'metabolic_module_score', 'shared_imaging_score', 'integrated_score', 'axis_percentile'] if c in d.columns]
                st.dataframe(d[cols], use_container_width=True, hide_index=True)
            else:
                st.info('No ecosystem-axis score was found for this patient.')
        with c2:
            top_mod = patient_top_modules(sample_id, endpoint_selected, n=12)
            if not top_mod.empty:
                pp = top_mod.sort_values('shap_value', ascending=True).copy()
                pp['label'] = pp['display_label'].astype(str).str.slice(0, 52)
                pp['Direction'] = np.where(pp['shap_value'] >= 0, 'Favorable', 'Unfavorable')
                fig = px.bar(pp, x='shap_value', y='label', orientation='h', color='Direction', hover_data=['module_type', 'axis', 'module_id'], color_discrete_map={'Favorable': '#0C98A4', 'Unfavorable': '#E56D57'}, title=f'Top ecosystem modules — {endpoint_selected}')
                fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10), xaxis_title='Benefit-oriented module SHAP', yaxis_title='')
                st.plotly_chart(fig, use_container_width=True)
                st.caption('Ecosystem module SHAP attribution is an interpretation analysis.')
            else:
                st.info('Ecosystem module attribution is unavailable for this patient.')
    with tab3:
        st.subheader('Position in the current internal cohort')
        c1, c2 = st.columns([1, 1])
        with c1:
            x = pd.to_numeric(mbp_df.get('MRU_benefit_score'), errors='coerce').dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=x, nbinsx=24, name='Internal cohort', opacity=0.78))
            if np.isfinite(as_float(mru_score)):
                fig.add_vline(x=as_float(mru_score), line_width=3, line_dash='dash', annotation_text=f'Current: {as_float(mru_score):.3f}', annotation_position='top right')
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10), title='MRU benefit score distribution', xaxis_title='MRU benefit score', yaxis_title='Patients')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            axis_options = [a for a in AXIS_ORDER if a in set(axis_df['axis'].astype(str))]
            if axis_options:
                axis_choice = st.selectbox('Ecosystem axis', axis_options, format_func=lambda x: AXIS_DISPLAY.get(x, x))
                dd = axis_df[axis_df['axis'].astype(str) == axis_choice].copy()
                vals = pd.to_numeric(dd['integrated_score'], errors='coerce').dropna()
                cur = dd.loc[dd['sample'] == sample_id, 'integrated_score']
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=vals, nbinsx=22, name='Internal cohort', opacity=0.78))
                if len(cur) and np.isfinite(as_float(cur.iloc[0])):
                    fig.add_vline(x=as_float(cur.iloc[0]), line_width=3, line_dash='dash', annotation_text='Current patient', annotation_position='top right')
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10), title=AXIS_DISPLAY.get(axis_choice, axis_choice), xaxis_title='Integrated ecosystem score', yaxis_title='Patients')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No axis distribution is available.')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('MBP phenotype', mbp)
        c2.metric('MRU benefit percentile', fmt_pct(mru_pct, 0))
        c3.metric('PFS risk percentile', fmt_pct(pfs_risk_pct, 0))
        c4.metric('OS risk percentile', fmt_pct(os_risk_pct, 0))
    with tab4:
        st.subheader('Observed clinical readout')
        readout = pd.DataFrame([{'Outcome': 'Early response', 'Projection': fmt_prob(response_pred), 'Cohort position': fmt_pct(response_pct, 0), 'Observed': response_true}, {'Outcome': 'PFS', 'Projection': f'risk={fmt_num(pfs_risk, 3)}', 'Cohort position': fmt_pct(pfs_risk_pct, 0), 'Observed': f'{fmt_months(pfs_time)}; {pfs_status}'}, {'Outcome': 'OS', 'Projection': f'risk={fmt_num(os_risk, 3)}', 'Cohort position': fmt_pct(os_risk_pct, 0), 'Observed': f'{fmt_months(os_time)}; {os_status}'}])
        st.dataframe(readout, use_container_width=True, hide_index=True)
        st.caption('Observed outcomes and MBP assignments use the locked patient-level analysis table.')
    with tab5:
        st.subheader('Auto-generated avatar report')
        st.markdown(f'\n        <div style="border:1px solid #c7d8ee;border-radius:16px;padding:22px 24px;background:linear-gradient(180deg,#ffffff,#f6fbff);box-shadow:0 10px 28px rgba(30,82,136,0.08);font-size:16px;line-height:1.65;color:#0B1D44;">\n          <b>Avatar-Agent Avatar Report</b><br><br>\n          <span style="font-size:14px;color:#51657f;">Display ID: {esc(selected_display)}</span><br><br>\n          {esc(report_text)}\n          <br><br>\n          <span style="font-size:13px;color:#6E7E93;">Current-output retrospective projection / interpretation prototype.</span>\n        </div>\n        ', unsafe_allow_html=True)
        report_download = f'Avatar-Agent Avatar Report\nDisplay ID: {selected_display}\nMBP: {mbp}\nProjected benefit state: {benefit_state}\nMRU benefit score: {fmt_num(mru_score, 4)}\nMRU percentile: {fmt_pct(mru_pct, 0)}\n\n{report_text}\n'
        st.download_button('Download avatar report', data=report_download.encode('utf-8'), file_name=f"Avatar_Agent_{selected_display.replace(' ', '_')}_report.txt", mime='text/plain')
except Exception:
    st.error('The application could not validate or display the analysis inputs.')
    st.stop()
