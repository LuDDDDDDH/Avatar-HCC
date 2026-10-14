from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MBP_COLORS, MBP_MARKERS, MODALITY_COLORS, MODALITY_MARKERS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = ROOT / 'inputs'
VERSION = 'v001'
OUTPUT_DIR = ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_323')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 430.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT = {'base': 6.1, 'panel': 8.5, 'header': 6.5, 'axis': 6.0, 'tick': 5.5, 'legend': 5.4, 'value': 5.4, 'small': 5.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
GRID = '#D9E0E6'
LIGHT = NEUTRAL_COLORS['light_background']
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
PAIR_ORDER = ['Response–PFS', 'Response–OS', 'PFS–OS']
MODALITY_ORDER = ['Metagenomics', 'Radiomics', 'Urine']
MBP_ORDER = ['MBP-I', 'MBP-II', 'MBP-III']
ROLE_ORDER = ['Shared backbone', 'Response-preferred', 'PFS-preferred', 'OS-preferred']
BENEFIT_CMAP = LinearSegmentedColormap.from_list('aurora_benefit', DIVERGING_COLORS, N=256)
PERCENTILE_CMAP = LinearSegmentedColormap.from_list('aurora_percentile', DIVERGING_COLORS, N=256)
ALT_CMAP = LinearSegmentedColormap.from_list('aurora_alt', ['#F7FBFF', '#D7E9F3', '#89C4CF', '#617FC4', '#98499F'], N=256)
INPUT_FILES = {'a_full': get_value('figures.config.item_168'), 'a_modality': get_value('figures.config.item_248'), 'a_summary': get_value('figures.config.item_247'), 'b_select': get_value('figures.config.item_242'), 'b_meta_points': get_value('figures.config.item_240'), 'b_meta_trends': get_value('figures.config.item_241'), 'b_radio_points': get_value('figures.config.item_243'), 'b_radio_trends': get_value('figures.config.item_244'), 'b_urine_points': get_value('figures.config.item_245'), 'b_urine_trends': get_value('figures.config.item_246'), 'c_rank_long': get_value('figures.config.item_228'), 'c_rank_wide': get_value('figures.config.item_229'), 'c_importance_wide': get_value('figures.config.item_230'), 'c_beeswarm': get_value('figures.config.item_227'), 'd_Response_features': get_value('figures.config.item_231'), 'd_Response_matrix': get_value('figures.config.item_232'), 'd_Response_order': get_value('figures.config.item_233'), 'd_PFS_features': get_value('figures.config.item_234'), 'd_PFS_matrix': get_value('figures.config.item_235'), 'd_PFS_order': get_value('figures.config.item_236'), 'd_OS_features': get_value('figures.config.item_237'), 'd_OS_matrix': get_value('figures.config.item_238'), 'd_OS_order': get_value('figures.config.item_239'), 'e_select': get_value('figures.config.item_250'), 'e_MBP-I': get_value('figures.config.item_249'), 'e_MBP-II': get_value('figures.config.item_251'), 'e_MBP-III': get_value('figures.config.item_252'), 'f_modality': get_value('figures.config.item_223'), 'f_alt': get_value('figures.config.item_224'), 'f_support': get_value('figures.config.item_225')}

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT['base'], 'axes.labelsize': FONT['axis'], 'axes.titlesize': FONT['header'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'legend.fontsize': FONT['legend'], 'legend.title_fontsize': FONT['legend'], 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.max_open_warning': 0, 'axes.unicode_minus': False})

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')

def _read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for key, filename in INPUT_FILES.items():
        path = INPUT_DIR / filename
        bundle[key] = _read_matrix(path) if key.endswith('_matrix') else pd.read_csv(path)
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8-sig'))
    bundle['contract'] = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> None:
    manifest = {row['snapshot']: row for row in bundle['manifest']['files']}
    for filename in INPUT_FILES.values():
        if filename not in manifest:
            raise ValueError(f'Input manifest does not register {filename}')
        path = INPUT_DIR / filename
        entry = manifest[filename]
        if file_sha256(path) != str(entry['sha256']).upper():
            raise ValueError(f'SHA-256 mismatch for frozen input {filename}')
        shape = pd.read_csv(path).shape
        expected = (int(entry['rows']), int(entry['columns']))
        if shape != expected:
            raise ValueError(f'Unexpected shape for {filename}: {shape}; expected {expected}')
    a_full = bundle['a_full']
    _require_columns(a_full, ['sample', 'pair', 'spearman_rho', 'MBP_raw', 'n_features'], 'S4A full MRU')
    if len(a_full) != 327 or a_full['sample'].nunique() != 109 or set(a_full['pair']) != set(PAIR_ORDER):
        raise ValueError('S4A full-MRU patient concordance membership drifted')
    if not (a_full['n_features'] == 39).all() or not a_full['spearman_rho'].between(-1, 1).all():
        raise ValueError('S4A full-MRU feature count or rho range drifted')
    a_modality = bundle['a_modality']
    _require_columns(a_modality, ['sample', 'pair', 'modal_clean', 'spearman_rho', 'MBP_raw', 'n_features'], 'S4A modality')
    if len(a_modality) != 981 or a_modality['sample'].nunique() != 109:
        raise ValueError('S4A modality concordance shape drifted')
    if set(a_modality['pair']) != set(PAIR_ORDER) or set(a_modality['modal_clean']) != set(MODALITY_ORDER):
        raise ValueError('S4A modality or pair membership drifted')
    expected_feature_counts = {'Metagenomics': 15, 'Radiomics': 9, 'Urine': 15}
    for modality, count in expected_feature_counts.items():
        if not (a_modality.loc[a_modality['modal_clean'] == modality, 'n_features'] == count).all():
            raise ValueError(f'S4A {modality} feature count drifted')
    a_summary = bundle['a_summary']
    _require_columns(a_summary, ['strata_type', 'stratum', 'pair', 'n', 'median_rho', 'q25', 'q75'], 'S4A summary')
    if len(a_summary) != 21 or set(a_summary['pair']) != set(PAIR_ORDER):
        raise ValueError('S4A frozen summary rows drifted')
    if a_summary.groupby('strata_type').size().to_dict() != {'MBP_raw': 9, 'Overall': 3, 'modal_clean': 9}:
        raise ValueError('S4A frozen summary strata drifted')
    selection = bundle['b_select']
    _require_columns(selection, ['modal_clean', 'role', 'feature_clean', 'selection_score'], 'S4B selection')
    if len(selection) != 12 or selection['feature_clean'].nunique() != 12:
        raise ValueError('S4B frozen feature selection must contain 12 unique features')
    if set(selection['modal_clean']) != set(MODALITY_ORDER) or set(selection['role']) != set(ROLE_ORDER):
        raise ValueError('S4B modality or role selection drifted')
    for prefix in ('b_meta', 'b_radio', 'b_urine'):
        points = bundle[f'{prefix}_points']
        trends = bundle[f'{prefix}_trends']
        _require_columns(points, ['modal_clean', 'role', 'feature_clean', 'outcome_clean', 'sample', 'feature_value', 'benefit_oriented_shap', 'spearman_rho'], f'S4B {prefix} points')
        _require_columns(trends, ['x_median', 'y_median', 'n', 'modal_clean', 'role', 'feature_clean', 'outcome_clean'], f'S4B {prefix} trends')
        if len(points) != 1308 or points['sample'].nunique() != 109 or set(points['outcome_clean']) != set(ENDPOINT_ORDER):
            raise ValueError(f'S4B {prefix} frozen point rows drifted')
        if points.groupby(['feature_clean', 'outcome_clean']).size().nunique() != 1 or int(points.groupby(['feature_clean', 'outcome_clean']).size().iloc[0]) != 109:
            raise ValueError(f'S4B {prefix} patient counts per cell drifted')
        if set(points['feature_clean'].unique()) != set(trends['feature_clean'].unique()):
            raise ValueError(f'S4B {prefix} trend feature membership drifted')
    rank_long = bundle['c_rank_long']
    rank_wide = bundle['c_rank_wide']
    importance_wide = bundle['c_importance_wide']
    beeswarm = bundle['c_beeswarm']
    _require_columns(rank_long, ['feature_clean', 'modal_clean', 'outcome_clean', 'relative_importance', 'endpoint_rank'], 'S4C rank long')
    _require_columns(rank_wide, ['feature_clean', 'modal_clean', 'OS', 'PFS', 'Response', 'rank_range', 'overall_mean_abs_shap'], 'S4C rank wide')
    _require_columns(importance_wide, ['feature_clean', 'modal_clean', 'OS', 'PFS', 'Response'], 'S4C importance wide')
    _require_columns(beeswarm, ['sample', 'outcome', 'feature', 'feature_value_percentile', 'benefit_oriented_shap'], 'S4C beeswarm')
    if len(rank_long) != 117 or len(rank_wide) != 39 or len(importance_wide) != 39:
        raise ValueError('S4C rank table dimensions drifted')
    if rank_long['feature_clean'].nunique() != 39 or set(rank_long['outcome_clean']) != set(ENDPOINT_ORDER):
        raise ValueError('S4C rank membership drifted')
    if not rank_long['endpoint_rank'].between(1, 39).all():
        raise ValueError('S4C endpoint ranks left the frozen 1-39 range')
    if len(beeswarm) != 12753 or beeswarm['sample'].nunique() != 109 or beeswarm['feature'].nunique() != 39:
        raise ValueError('S4C beeswarm dimensions drifted')
    if set(beeswarm['outcome']) != set(ENDPOINT_ORDER) or not beeswarm['feature_value_percentile'].between(0, 1).all():
        raise ValueError('S4C beeswarm endpoint or percentile membership drifted')
    for endpoint in ENDPOINT_ORDER:
        features = bundle[f'd_{endpoint}_features']
        matrix = bundle[f'd_{endpoint}_matrix']
        order = bundle[f'd_{endpoint}_order']
        _require_columns(features, ['feature', 'modality', 'row_scale_denominator'], f'S4D {endpoint} features')
        _require_columns(order, ['sample', 'MBP_raw'], f'S4D {endpoint} patient order')
        if matrix.shape != (39, 109) or len(features) != 39 or len(order) != 109:
            raise ValueError(f'S4D {endpoint} frozen dimensions drifted')
        if matrix.index.astype(str).tolist() != features['feature'].astype(str).tolist():
            raise ValueError(f'S4D {endpoint} feature order drifted')
        if matrix.columns.astype(str).tolist() != order['sample'].astype(str).tolist():
            raise ValueError(f'S4D {endpoint} patient order drifted')
        values = matrix.to_numpy(dtype=float)
        if np.nanmin(values) < -1.000001 or np.nanmax(values) > 1.000001:
            raise ValueError(f'S4D {endpoint} row-scaled values left [-1, 1]')
    e_select = bundle['e_select']
    _require_columns(e_select, ['MBP_raw', 'sample', 'robust_distance_to_group_median', 'n_group_patients', 'composite_benefit_rank'], 'S4E selection')
    if len(e_select) != 3 or e_select['sample'].nunique() != 3 or e_select['MBP_raw'].tolist() != MBP_ORDER:
        raise ValueError('S4E representative-patient selection drifted')
    for group in MBP_ORDER:
        frame = bundle[f'e_{group}']
        _require_columns(frame, ['feature', 'display_feature', 'modal_clean', 'benefit_oriented_shap', 'component_type', 'waterfall_start', 'waterfall_end', 'total_local_shap_sum', 'sample', 'MBP_raw', 'outcome_clean'], f'S4E {group}')
        selected_sample = e_select.loc[e_select['MBP_raw'] == group, 'sample'].iloc[0]
        if len(frame) != 30 or frame['sample'].nunique() != 1 or frame['sample'].iloc[0] != selected_sample:
            raise ValueError(f'S4E {group} frozen patient or row count drifted')
        if frame.groupby('outcome_clean').size().reindex(ENDPOINT_ORDER).tolist() != [10, 10, 10]:
            raise ValueError(f'S4E {group} endpoint components drifted')
    f_modality = bundle['f_modality']
    f_alt = bundle['f_alt']
    f_support = bundle['f_support']
    _require_columns(f_modality, ['outcome_clean', 'modal_clean', 'relative_per_feature_contribution', 'relative_total_contribution'], 'S4F modality')
    _require_columns(f_alt, ['task', 'model_family', 'feature_clean', 'modal_clean', 'importance_norm', 'rank', 'topk_support'], 'S4F alternative models')
    _require_columns(f_support, ['feature_clean', 'modal_clean', 'support_count', 'mean_alt_importance', 'median_alt_rank', 'n_model_tasks', 'dnn_norm_importance', 'support_fraction'], 'S4F support')
    if len(f_modality) != 9 or len(f_alt) != 312 or len(f_support) != 15:
        raise ValueError('S4F frozen sensitivity dimensions drifted')
    if f_alt[['task', 'model_family']].drop_duplicates().shape[0] != 8 or not f_support['support_fraction'].between(0, 1).all():
        raise ValueError('S4F frozen model-task or support membership drifted')

def _rect(width_mm: float, height_mm: float, left: float, top: float, width: float, height: float) -> list[float]:
    return [left / width_mm, 1.0 - (top + height) / height_mm, width / width_mm, height / height_mm]

def _ax(fig: mpl.figure.Figure, width_mm: float, height_mm: float, rect: Sequence[float], panel: str) -> mpl.axes.Axes:
    axis = fig.add_axes(_rect(width_mm, height_mm, *rect))
    axis._qa_meta = {'panel': panel}
    return axis

def _tag(registry: ArtistRegistry, artist: Any, *, panel: str, name: str, role: str='direct_label') -> Any:
    return registry.tag(artist, name=name, role=role, panel=panel)

def _fig_text(fig: mpl.figure.Figure, width_mm: float, height_mm: float, x_mm: float, y_mm: float, text: str, registry: ArtistRegistry, *, panel: str, name: str, role: str='direct_label', **kwargs: Any) -> mpl.text.Text:
    defaults = {'fontsize': FONT['small'], 'color': TEXT, 'ha': 'left', 'va': 'top'}
    defaults.update(kwargs)
    artist = fig.text(x_mm / width_mm, 1.0 - y_mm / height_mm, text, transform=fig.transFigure, **defaults)
    return _tag(registry, artist, panel=panel, name=name, role=role)

def _panel_letter(fig: mpl.figure.Figure, width_mm: float, height_mm: float, label: str, top: float, registry: ArtistRegistry) -> None:
    _fig_text(fig, width_mm, height_mm, 3.1, top, label, registry, panel=label, name=f'panel-{label}-letter', role='panel_letter', fontsize=FONT['panel'], fontweight='bold')

def _clean_axis(ax: mpl.axes.Axes, grid_axis: str | None=None) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(width=0.5, length=2.0, pad=1.2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.45, alpha=0.75, zorder=0)

def _stable_jitter(*parts: str, span: float=0.18) -> float:
    raw = '|'.join(map(str, parts)).encode('utf-8')
    value = int(hashlib.sha256(raw).hexdigest()[:12], 16) / float(16 ** 12 - 1)
    return (value - 0.5) * 2.0 * span

def _strip_prefix(feature: str) -> str:
    value = str(feature)
    for prefix in ('Metagenomics__', 'Radiomics__', 'Urine__'):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value

def _display_feature(feature: str, max_len: int=25) -> str:
    value = _strip_prefix(feature)
    aliases = {'wavelet-HLL_glszm_ZoneEntropy': 'Wavelet-HLL zone entropy', 'wavelet-HLL_glszm_SizeZoneNonUniformityNormalized': 'Wavelet-HLL size-zone nonuniformity', 'wavelet-HLL_glszm_SmallAreaEmphasis': 'Wavelet-HLL small-area emphasis', 'log-sigma-4-0-mm-3D_firstorder_Kurtosis': 'LoG-4 mm kurtosis', 'wavelet-HHL_glcm_MCC': 'Wavelet-HHL GLCM MCC'}
    value = aliases.get(value, value)
    value = value.replace('_', ' ')
    if len(value) <= max_len:
        return value
    return value[:max_len - 1].rstrip() + '…'

def _pair_display(value: str) -> str:
    return str(value).replace('–', '-')

def _range_with_pad(values: np.ndarray, fraction: float=0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo, hi = (float(finite.min()), float(finite.max()))
    span = hi - lo
    if span <= 1e-12:
        span = max(abs(lo), 1.0) * 0.2
    return (lo - fraction * span, hi + fraction * span)

def draw_panel_a(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'a'
    positions = [(18.0, 43.0), (75.0, 38.0), (127.0, 47.0)]
    headers = ['', 'MBP strata', 'Modality-specific']
    for index, ((left, width), header) in enumerate(zip(positions, headers)):
        _fig_text(fig, width_mm, height_mm, left + width / 2, top + 1.2, header, registry, panel=panel, name=f'a-header-{index}', role='axis_label', fontsize=FONT['header'], ha='center')
    full = bundle['a_full']
    summary = bundle['a_summary']
    left_ax = _ax(fig, width_mm, height_mm, (18, top + 5, 43, 30.5), panel)
    y_positions = {pair: 2 - index for index, pair in enumerate(PAIR_ORDER)}
    for pair in PAIR_ORDER:
        subset = full.loc[full['pair'] == pair]
        y0 = y_positions[pair]
        for group in MBP_ORDER:
            group_data = subset.loc[subset['MBP_raw'] == group]
            ys = [y0 + _stable_jitter(sample, pair, span=0.18) for sample in group_data['sample'].astype(str)]
            left_ax.scatter(group_data['spearman_rho'], ys, s=7, c=MBP_COLORS[group], marker=MBP_MARKERS[group], alpha=0.52, linewidths=0, rasterized=True, zorder=2)
        row = summary.loc[(summary['strata_type'] == 'Overall') & (summary['pair'] == pair)].iloc[0]
        left_ax.plot([row['q25'], row['q75']], [y0, y0], color=TEXT, lw=1.2, zorder=4)
        left_ax.scatter([row['median_rho']], [y0], s=18, facecolor='white', edgecolor=TEXT, linewidth=0.7, zorder=5)
        _tag(registry, left_ax.text(0.705, y0 + 0.28, f"{float(row['median_rho']):.2f}", fontsize=FONT['value'], ha='left', va='center', color=TEXT), panel=panel, name=f'a-overall-median-{pair}', role='value')
    left_ax.set_xlim(0.7, 1.005)
    left_ax.set_ylim(-0.55, 2.55)
    left_ax.set_yticks([2, 1, 0], [_pair_display(pair) for pair in PAIR_ORDER])
    left_ax.set_xlabel('Patient-level Spearman rho', labelpad=1.5)
    left_ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
    _clean_axis(left_ax, 'x')
    handles = [Line2D([], [], linestyle='', marker=MBP_MARKERS[g], color=MBP_COLORS[g], markersize=3.2, label=g.replace('MBP-', '')) for g in MBP_ORDER]
    left_ax.legend(handles=handles, title='Full MRU (39 features) · MBP', loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.65, handletextpad=0.25, borderaxespad=0.1)
    mbp_ax = _ax(fig, width_mm, height_mm, (75, top + 5, 38, 30.5), panel)
    mbp_rows: list[tuple[str, str]] = [(pair, group) for pair in PAIR_ORDER for group in MBP_ORDER]
    for row_index, (pair, group) in enumerate(mbp_rows):
        y = len(mbp_rows) - 1 - row_index
        row = summary.loc[(summary['strata_type'] == 'MBP_raw') & (summary['pair'] == pair) & (summary['stratum'] == group)].iloc[0]
        mbp_ax.plot([row['q25'], row['q75']], [y, y], color=MBP_COLORS[group], lw=1.4)
        mbp_ax.scatter([row['median_rho']], [y], s=16, c=MBP_COLORS[group], marker=MBP_MARKERS[group], edgecolor='white', linewidth=0.45, zorder=3)
        _tag(registry, mbp_ax.text(0.708, y, f"{float(row['median_rho']):.2f}", fontsize=FONT['value'], ha='left', va='center'), panel=panel, name=f'a-mbp-value-{row_index}', role='value')
    mbp_ax.set_xlim(0.7, 1.005)
    mbp_ax.set_ylim(-0.65, len(mbp_rows) - 0.35)
    pair_abbrev = {'Response–PFS': 'R-P', 'Response–OS': 'R-O', 'PFS–OS': 'P-O'}
    mbp_ax.set_yticks(range(len(mbp_rows)), [f"{pair_abbrev[p]} {g.replace('MBP-', '')}" for p, g in reversed(mbp_rows)])
    mbp_ax.set_xlabel('Frozen median rho and IQR', labelpad=1.5)
    mbp_ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
    _clean_axis(mbp_ax, 'x')
    modality = bundle['a_modality']
    mod_ax = _ax(fig, width_mm, height_mm, (127, top + 5, 47, 30.5), panel)
    mod_rows: list[tuple[str, str]] = [(pair, mod) for pair in PAIR_ORDER for mod in MODALITY_ORDER]
    for row_index, (pair, mod) in enumerate(mod_rows):
        y = len(mod_rows) - 1 - row_index
        pts = modality.loc[(modality['pair'] == pair) & (modality['modal_clean'] == mod)]
        ys = [y + _stable_jitter(sample, pair, mod, span=0.15) for sample in pts['sample'].astype(str)]
        mod_ax.scatter(pts['spearman_rho'], ys, s=4.8, c=MODALITY_COLORS[mod], marker=MODALITY_MARKERS[mod], alpha=0.28, linewidths=0, rasterized=True, zorder=2)
        row = summary.loc[(summary['strata_type'] == 'modal_clean') & (summary['pair'] == pair) & (summary['stratum'] == mod)].iloc[0]
        mod_ax.plot([row['q25'], row['q75']], [y, y], color=MODALITY_COLORS[mod], lw=1.3, zorder=4)
        mod_ax.scatter([row['median_rho']], [y], s=14, c=MODALITY_COLORS[mod], marker=MODALITY_MARKERS[mod], edgecolor='white', linewidth=0.4, zorder=5)
    mod_ax.set_xlim(0.0, 1.005)
    mod_ax.set_ylim(-0.65, len(mod_rows) - 0.35)
    mod_ax.set_yticks(range(len(mod_rows)), [f'{pair_abbrev[p]} {m[:4]}' for p, m in reversed(mod_rows)])
    mod_ax.set_xlabel('Patient-level Spearman rho', labelpad=1.5)
    mod_ax.xaxis.set_major_locator(MaxNLocator(5, prune='both'))
    _clean_axis(mod_ax, 'x')

def draw_panel_b(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'b'
    selection = bundle['b_select']
    points_by_modality = {'Metagenomics': bundle['b_meta_points'], 'Radiomics': bundle['b_radio_points'], 'Urine': bundle['b_urine_points']}
    trends_by_modality = {'Metagenomics': bundle['b_meta_trends'], 'Radiomics': bundle['b_radio_trends'], 'Urine': bundle['b_urine_trends']}
    endpoint_x = {'Response': 48.0, 'PFS': 90.0, 'OS': 132.0}
    axis_width = 39.0
    block_height = 27.0
    role_short = {'Shared backbone': 'Shared', 'Response-preferred': 'Response', 'PFS-preferred': 'PFS', 'OS-preferred': 'OS'}
    for block_index, modality in enumerate(MODALITY_ORDER):
        block_top = top + 1.7 + block_index * block_height
        _fig_text(fig, width_mm, height_mm, 8.0, block_top, modality, registry, panel=panel, name=f'b-modality-{block_index}', role='axis_label', fontsize=FONT['header'], fontweight='bold', color=MODALITY_COLORS[modality])
        for endpoint in ENDPOINT_ORDER:
            _fig_text(fig, width_mm, height_mm, endpoint_x[endpoint] + axis_width / 2, block_top, endpoint, registry, panel=panel, name=f'b-header-{block_index}-{endpoint}', role='axis_label', fontsize=FONT['header'], ha='center', color=ENDPOINT_COLORS[endpoint])
        for role_index, role in enumerate(ROLE_ORDER):
            selected = selection.loc[(selection['modal_clean'] == modality) & (selection['role'] == role)].iloc[0]
            feature = str(selected['feature_clean'])
            row_top = block_top + 3.5 + role_index * 5.75
            label = f'{role_short[role]}  {_display_feature(feature, 29)}'
            _fig_text(fig, width_mm, height_mm, 45.0, row_top + 2.4, label, registry, panel=panel, name=f'b-row-{block_index}-{role_index}', role='tick', fontsize=FONT['small'], ha='right', va='center')
            for endpoint in ENDPOINT_ORDER:
                axis = _ax(fig, width_mm, height_mm, (endpoint_x[endpoint], row_top, axis_width, 5.0), panel)
                pts = points_by_modality[modality]
                pts = pts.loc[(pts['feature_clean'] == feature) & (pts['outcome_clean'] == endpoint)]
                trend = trends_by_modality[modality]
                trend = trend.loc[(trend['feature_clean'] == feature) & (trend['outcome_clean'] == endpoint)]
                axis.scatter(pts['feature_value'], pts['benefit_oriented_shap'], s=3.0, c=ENDPOINT_COLORS[endpoint], alpha=0.3, linewidths=0, rasterized=True, zorder=2)
                axis.plot(trend['x_median'], trend['y_median'], color=TEXT, lw=0.75, marker='o', ms=1.7, mfc='white', mec=TEXT, mew=0.35, zorder=4)
                axis.axhline(0, color=GRID, lw=0.45, zorder=1)
                xlim = _range_with_pad(pts['feature_value'].to_numpy(dtype=float), 0.04)
                ylim = _range_with_pad(pts['benefit_oriented_shap'].to_numpy(dtype=float), 0.12)
                axis.set_xlim(*xlim)
                axis.set_ylim(*ylim)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                rho = float(pts['spearman_rho'].iloc[0])
                _tag(registry, axis.text(0.015, 0.96, f'rho={rho:.2f}', transform=axis.transAxes, fontsize=FONT['small'], ha='left', va='top', color=TEXT), panel=panel, name=f'b-rho-{block_index}-{role_index}-{endpoint}', role='value')
    _fig_text(fig, width_mm, height_mm, 110.0, top + 84.5, 'Feature value', registry, panel=panel, name='b-x-label', role='axis_label', fontsize=FONT['axis'], ha='center', va='bottom')
    _fig_text(fig, width_mm, height_mm, 5.0, top + 43.0, 'Benefit-oriented SHAP', registry, panel=panel, name='b-y-label', role='axis_label', fontsize=FONT['axis'], ha='center', va='center', rotation=90)

def draw_panel_c(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'c'
    rank_long = bundle['c_rank_long']
    rank_wide = bundle['c_rank_wide']
    beeswarm = bundle['c_beeswarm']
    rank_ax = _ax(fig, width_mm, height_mm, (10, top + 5, 52, 50), panel)
    x_positions = {'Response': 0, 'PFS': 1, 'OS': 2}
    for feature, rows in rank_long.groupby('feature_clean', sort=False):
        rows = rows.set_index('outcome_clean').reindex(ENDPOINT_ORDER)
        modality = str(rows['modal_clean'].iloc[0])
        ranks = rows['endpoint_rank'].to_numpy(dtype=float)
        rank_ax.plot([0, 1, 2], ranks, color=MODALITY_COLORS[modality], lw=0.75, alpha=0.55, marker=MODALITY_MARKERS[modality], ms=2.0, mec='white', mew=0.25)
    rank_ax.set_xlim(-0.15, 2.15)
    rank_ax.set_ylim(39.7, 0.3)
    rank_ax.set_xticks([0, 1, 2], ENDPOINT_ORDER)
    rank_ax.set_yticks([1, 10, 20, 30, 39])
    rank_ax.set_ylabel('Frozen endpoint rank')
    _clean_axis(rank_ax, 'y')
    handles = [Line2D([], [], color=MODALITY_COLORS[m], marker=MODALITY_MARKERS[m], markersize=2.7, lw=0.9, label=m) for m in MODALITY_ORDER]
    rank_ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.6, handletextpad=0.25, borderaxespad=0.0)
    ordered_features = rank_wide.sort_values('overall_mean_abs_shap', ascending=False, kind='stable')['feature_clean'].astype(str).tolist()
    feature_y = {feature: len(ordered_features) - 1 - index for index, feature in enumerate(ordered_features)}
    for endpoint, left in zip(ENDPOINT_ORDER, (76.0, 108.0, 140.0)):
        axis = _ax(fig, width_mm, height_mm, (left, top + 5, 29.0, 50.0), panel)
        subset = beeswarm.loc[beeswarm['outcome'] == endpoint]
        ys = [feature_y[str(feature)] + _stable_jitter(sample, endpoint, feature, span=0.25) for sample, feature in zip(subset['sample'].astype(str), subset['feature'].astype(str))]
        axis.scatter(subset['benefit_oriented_shap'], ys, c=subset['feature_value_percentile'], cmap=PERCENTILE_CMAP, norm=Normalize(0, 1), s=2.8, alpha=0.62, linewidths=0, rasterized=True)
        axis.axvline(0, color=REFERENCE, lw=0.55, zorder=0)
        axis.set_ylim(-0.7, 38.7)
        axis.set_yticks([38, 29, 19, 9, 0], ['1', '10', '20', '30', '39'] if endpoint == 'Response' else ['', '', '', '', ''])
        axis.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
        axis.set_title(endpoint, color=ENDPOINT_COLORS[endpoint], fontsize=FONT['header'], pad=2.0)
        if endpoint == 'Response':
            axis.set_ylabel('Frozen feature order')
        if endpoint == 'PFS':
            axis.set_xlabel('Benefit-oriented SHAP', labelpad=1.5)
        _clean_axis(axis, 'x')
    cax = _ax(fig, width_mm, height_mm, (171.5, top + 14, 1.8, 27.0), panel)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=PERCENTILE_CMAP), cax=cax, orientation='vertical')
    colorbar.set_ticks([0, 1], labels=['Low', 'High'])
    colorbar.ax.tick_params(labelsize=FONT['small'], length=1.5, pad=1.0)
    colorbar.outline.set_linewidth(0.45)
    _fig_text(fig, width_mm, height_mm, 172.4, top + 44.0, 'Feature\npercentile', registry, panel=panel, name='c-colorbar-label', role='axis_label', fontsize=FONT['small'], ha='center', va='top')

def _mbp_strip_colors(order: pd.DataFrame) -> np.ndarray:
    mapping = {group: index for index, group in enumerate(MBP_ORDER)}
    return np.array([[mapping[str(value)] for value in order['MBP_raw']]], dtype=float)

def draw_panel_d(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'd'
    mbp_cmap = ListedColormap([MBP_COLORS[group] for group in MBP_ORDER])
    for endpoint, left, panel_width in (('Response', 13.0, 50.0), ('PFS', 68.0, 50.0), ('OS', 123.0, 47.0)):
        matrix = bundle[f'd_{endpoint}_matrix']
        order = bundle[f'd_{endpoint}_order']
        strip = _ax(fig, width_mm, height_mm, (left, top + 4.8, panel_width, 2.1), panel)
        strip.imshow(_mbp_strip_colors(order), cmap=mbp_cmap, vmin=-0.5, vmax=2.5, aspect='auto', interpolation='nearest', rasterized=True)
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)
        heat = _ax(fig, width_mm, height_mm, (left, top + 7.3, panel_width, 48.5), panel)
        heat.imshow(matrix.to_numpy(dtype=float), cmap=BENEFIT_CMAP, vmin=-1, vmax=1, aspect='auto', interpolation='nearest', rasterized=True)
        heat.set_title(endpoint, color=ENDPOINT_COLORS[endpoint], fontsize=FONT['header'], pad=4.2)
        heat.set_yticks([0, 9, 19, 29, 38], ['1', '10', '20', '30', '39'])
        if endpoint == 'Response':
            heat.set_ylabel('Endpoint-specific frozen feature order')
        groups = order['MBP_raw'].astype(str).tolist()
        centers: list[float] = []
        for group in MBP_ORDER:
            indices = [index for index, value in enumerate(groups) if value == group]
            centers.append((indices[0] + indices[-1]) / 2.0)
            if group != MBP_ORDER[-1]:
                heat.axvline(indices[-1] + 0.5, color='white', lw=0.55)
        heat.set_xticks(centers, ['I', 'II', 'III'])
        heat.set_xlabel('MBP', labelpad=1.0)
        heat.tick_params(length=0, pad=1.0)
        for spine in heat.spines.values():
            spine.set_linewidth(0.45)
    cax = _ax(fig, width_mm, height_mm, (172.0, top + 17.0, 1.5, 27.0), panel)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=BENEFIT_CMAP), cax=cax, orientation='vertical')
    colorbar.set_ticks([-1, 0, 1], labels=['-1', '0', '+1'])
    colorbar.ax.tick_params(labelsize=FONT['small'], length=1.5, pad=1.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label('Row-scaled SHAP', fontsize=FONT['small'], labelpad=1.2)

def draw_panel_e(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'e'
    selection = bundle['e_select'].set_index('MBP_raw')
    x_positions = {'Response': 29.0, 'PFS': 86.0, 'OS': 143.0}
    for endpoint in ENDPOINT_ORDER:
        _fig_text(fig, width_mm, height_mm, x_positions[endpoint] + 16.0, top + 1.0, endpoint, registry, panel=panel, name=f'e-endpoint-{endpoint}', role='axis_label', fontsize=FONT['header'], ha='center', color=ENDPOINT_COLORS[endpoint])
    for group_index, group in enumerate(MBP_ORDER):
        patient = selection.loc[group]
        frame = bundle[f'e_{group}']
        row_top = top + 4.0 + group_index * 29.0
        patient_label = f"{group} · {patient['sample']}  d={float(patient['robust_distance_to_group_median']):.2f}  rank={float(patient['composite_benefit_rank']):.2f}"
        _fig_text(fig, width_mm, height_mm, 7.0, row_top + 0.3, patient_label, registry, panel=panel, name=f'e-patient-{group_index}', role='direct_label', fontsize=FONT['small'], fontweight='bold', color=MBP_COLORS[group])
        for endpoint in ENDPOINT_ORDER:
            subset = frame.loc[frame['outcome_clean'] == endpoint].reset_index(drop=True)
            axis = _ax(fig, width_mm, height_mm, (x_positions[endpoint], row_top + 2.5, 32.0, 23.5), panel)
            for row_index, row in subset.iterrows():
                start = float(row['waterfall_start'])
                end = float(row['waterfall_end'])
                left = min(start, end)
                width = abs(end - start)
                contribution = float(row['benefit_oriented_shap'])
                color = NEUTRAL_COLORS['missing'] if str(row['component_type']) == 'other' else ENDPOINT_COLORS['Response'] if contribution >= 0 else ENDPOINT_COLORS['PFS']
                axis.add_patch(Rectangle((left, row_index - 0.34), width, 0.68, facecolor=color, edgecolor='white', linewidth=0.35, alpha=0.92))
                axis.plot([end, end], [row_index + 0.34, row_index + 0.66], color=REFERENCE, lw=0.35, ls=':')
            starts = subset['waterfall_start'].to_numpy(dtype=float)
            ends = subset['waterfall_end'].to_numpy(dtype=float)
            xlim = _range_with_pad(np.concatenate([starts, ends]), 0.09)
            axis.set_xlim(*xlim)
            axis.set_ylim(len(subset) - 0.2, -0.8)
            labels = ['Other 30 features' if value == 'other' else _display_feature(feature, 17) for value, feature in zip(subset['component_type'], subset['display_feature'])]
            axis.set_yticks(range(len(subset)), labels)
            axis.axvline(0, color=REFERENCE, lw=0.5)
            axis.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
            if group_index < 2:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel('Cumulative local SHAP', labelpad=1.0)
            total = float(subset['total_local_shap_sum'].iloc[0])
            _fig_text(fig, width_mm, height_mm, x_positions[endpoint] + 32.0, row_top + 1.0, f'sum={total:+.2f}', registry, panel=panel, name=f'e-total-{group_index}-{endpoint}', role='value', fontsize=FONT['value'], ha='right', va='center', color=ENDPOINT_COLORS[endpoint])
            _clean_axis(axis, 'x')

def draw_panel_f(fig: mpl.figure.Figure, width_mm: float, height_mm: float, top: float, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    panel = 'f'
    modality = bundle['f_modality']
    left_ax = _ax(fig, width_mm, height_mm, (10.0, top + 7.0, 55.0, 35.0), panel)
    x = np.arange(len(ENDPOINT_ORDER), dtype=float)
    offsets = {'Metagenomics': -0.24, 'Radiomics': 0.0, 'Urine': 0.24}
    for mod in MODALITY_ORDER:
        rows = modality.set_index(['outcome_clean', 'modal_clean'])
        values = [float(rows.loc[(endpoint, mod), 'relative_per_feature_contribution']) for endpoint in ENDPOINT_ORDER]
        bars = left_ax.bar(x + offsets[mod], values, width=0.21, color=MODALITY_COLORS[mod], label=mod, zorder=2)
        for index, (bar, value) in enumerate(zip(bars, values)):
            _tag(registry, left_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f'{value:.0%}', ha='center', va='bottom', fontsize=FONT['value'], color=TEXT), panel=panel, name=f'f-modality-value-{mod}-{index}', role='value')
    left_ax.set_xticks(x, ENDPOINT_ORDER)
    left_ax.set_ylim(0, 0.48)
    left_ax.set_yticks([0.0, 0.15, 0.3, 0.45])
    left_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    left_ax.set_ylabel('Relative per-feature contribution')
    _clean_axis(left_ax, 'y')
    left_ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)
    support = bundle['f_support'].reset_index(drop=True)
    alt = bundle['f_alt'].copy()
    task_order = [('Response', 'ExtraTrees'), ('Response', 'GradientBoosting'), ('Response', 'Logistic'), ('Response', 'RandomForest'), ('PFS', 'CoxPH'), ('PFS', 'SurvivalGB'), ('OS', 'CoxPH'), ('OS', 'SurvivalGB')]
    task_labels = ['Resp ET', 'Resp GB', 'Resp LR', 'Resp RF', 'PFS Cox', 'PFS GB', 'OS Cox', 'OS GB']
    feature_order = support['feature_clean'].astype(str).tolist()
    matrix = np.full((len(feature_order), len(task_order)), np.nan)
    for row_index, feature in enumerate(feature_order):
        for col_index, (task, family) in enumerate(task_order):
            cell = alt.loc[(alt['feature_clean'] == feature) & (alt['task'] == task) & (alt['model_family'] == family), 'importance_norm']
            if len(cell) != 1:
                raise ValueError(f'S4F missing frozen alternative-model cell for {feature}, {task}, {family}')
            matrix[row_index, col_index] = float(cell.iloc[0])
    vmax = float(alt['importance_norm'].max())
    heat = _ax(fig, width_mm, height_mm, (101.0, top + 7.0, 58.0, 35.0), panel)
    heat.imshow(matrix, cmap=ALT_CMAP, vmin=0, vmax=vmax, aspect='auto', interpolation='nearest', rasterized=True)
    heat.set_yticks(range(len(feature_order)), [_display_feature(feature, 26) for feature in feature_order])
    heat.set_xticks(range(len(task_order)), task_labels, rotation=55, ha='right')
    heat.tick_params(length=0, pad=1.0)
    for spine in heat.spines.values():
        spine.set_linewidth(0.45)
    support_ax = _ax(fig, width_mm, height_mm, (162.0, top + 7.0, 12.0, 35.0), panel)
    y = np.arange(len(feature_order))
    support_ax.barh(y, support['support_fraction'].to_numpy(dtype=float), color='#7A9CC5', height=0.72)
    support_ax.set_ylim(len(feature_order) - 0.5, -0.5)
    support_ax.set_xlim(0, 1)
    support_ax.set_yticks([])
    support_ax.set_xticks([0, 1], ['0', '100%'])
    support_ax.set_xlabel('Top-k support', labelpad=1.0)
    _clean_axis(support_ax, 'x')
    cax = _ax(fig, width_mm, height_mm, (129.0, top + 2.0, 27.0, 1.8), panel)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=ALT_CMAP), cax=cax, orientation='horizontal')
    colorbar.set_ticks([0, vmax], labels=['0', f'{vmax:.2f}'])
    colorbar.ax.tick_params(labelsize=FONT['small'], length=1.4, pad=0.8)
    colorbar.outline.set_linewidth(0.45)
    _fig_text(fig, width_mm, height_mm, 126.5, top + 2.9, 'Normalized importance', registry, panel=panel, name='f-colorbar-label', role='axis_label', fontsize=FONT['small'], ha='right', va='center')
FULL_PANEL_TOPS = {'a': 3.0, 'b': 47.0, 'c': 137.0, 'd': 202.0, 'e': 269.0, 'f': 366.0}
PANEL_LAYOUTS: Mapping[str, tuple[float, float]] = {'a': (180.0, 44.0), 'b': (180.0, 90.0), 'c': (180.0, 65.0), 'd': (180.0, 67.0), 'e': (180.0, 97.0), 'f': (180.0, 58.0)}
DRAWERS = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c, 'd': draw_panel_d, 'e': draw_panel_e, 'f': draw_panel_f}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    for panel in 'abcdef':
        DRAWERS[panel](fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANEL_TOPS[panel], bundle, registry)
        _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, panel, FULL_PANEL_TOPS[panel], registry)
    return (fig, registry)

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    panel = panel.lower()
    if panel not in PANEL_LAYOUTS:
        raise ValueError(f'Unknown panel {panel!r}')
    configure_style()
    width_mm, height_mm = PANEL_LAYOUTS[panel]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    DRAWERS[panel](fig, width_mm, height_mm, 1.5, bundle, registry)
    return (fig, registry)

def _save_exact(fig: mpl.figure.Figure, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S4 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    paths: list[Path] = []
    for suffix in ('pdf', 'svg', 'png'):
        path = stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, Any] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if suffix == 'png':
            kwargs['dpi'] = EXPORT_DPI
        elif suffix == 'pdf':
            kwargs['metadata'] = metadata
        else:
            kwargs['metadata'] = {'Title': metadata['Title'], 'Creator': metadata['Creator']}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def scientific_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {'render_only': True, 'palette': PALETTE_ID, 'benefit_orientation_locked': True, 'panel_a': {'full_rows': len(bundle['a_full']), 'modality_rows': len(bundle['a_modality']), 'summary_rows': len(bundle['a_summary'])}, 'panel_b': {'selected_features': len(bundle['b_select']), 'point_rows': sum((len(bundle[key]) for key in ('b_meta_points', 'b_radio_points', 'b_urine_points'))), 'trend_rows': sum((len(bundle[key]) for key in ('b_meta_trends', 'b_radio_trends', 'b_urine_trends')))}, 'panel_c': {'features': len(bundle['c_rank_wide']), 'rank_rows': len(bundle['c_rank_long']), 'beeswarm_rows': len(bundle['c_beeswarm'])}, 'panel_d': {endpoint: {'features': bundle[f'd_{endpoint}_matrix'].shape[0], 'patients': bundle[f'd_{endpoint}_matrix'].shape[1]} for endpoint in ENDPOINT_ORDER}, 'panel_e': {'representative_patients': len(bundle['e_select']), 'rows': sum((len(bundle[f'e_{group}']) for group in MBP_ORDER))}, 'panel_f': {'modality_rows': len(bundle['f_modality']), 'alternative_rows': len(bundle['f_alt']), 'support_features': len(bundle['f_support'])}, 'forbidden_operations_executed': []}

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, registry = build_full_figure(bundle)
    layout = audit_figure_layout(fig, f"{get_value('figures.config.item_323')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout, OUTPUT_DIR / f"{get_value('figures.config.item_324')}{VERSION}.json")
    full_paths = _save_exact(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in 'abcdef':
        panel_fig, panel_registry = build_panel_figure(panel, bundle)
        width_mm, height_mm = PANEL_LAYOUTS[panel]
        panel_audits[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_325')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        panel_paths.extend(_save_exact(panel_fig, PANEL_DIR / f"{get_value('figures.config.item_325')}{panel}_Nature_{VERSION}", width_mm=width_mm, height_mm=height_mm))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_327')}{VERSION}.json")
    scientific_path = write_json(scientific_summary(bundle), OUTPUT_DIR / f"{get_value('figures.config.item_328')}{VERSION}.json")
    outputs = full_paths + panel_paths + [layout_path, panel_audit_path, scientific_path]
    manifest = {'figure': 'Supplementary Figure S4', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked frozen tables and matrices', 'panel_letters': 'assembly only', 'layout_summary': layout['summary'], 'panel_layout_summaries': {panel: report['summary'] for panel, report in panel_audits.items()}, 'panel_mapping': bundle['contract']['panel_mapping'], 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in outputs]}
    manifest_path = write_json(manifest, OUTPUT_DIR / f"{get_value('figures.config.item_329')}{VERSION}.json")
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'scientific_guardrail_report': str(scientific_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {panel: report['summary'] for panel, report in panel_audits.items()}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
