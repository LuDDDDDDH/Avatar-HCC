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
import numpy as np
import pandas as pd
PILOT_ROOT = Path(__file__).resolve().parent
STYLE_ROOT = PILOT_ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MBP_COLORS, MBP_MARKERS, MODALITY_COLORS, MODALITY_MARKERS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = PILOT_ROOT / 'inputs'
VERSION = 'v001'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_285')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 210.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT = {'base': 6.3, 'panel': 8.5, 'header': 6.8, 'axis': 6.3, 'tick': 5.7, 'legend': 5.5, 'value': 5.5, 'small': 5.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
GRID = '#D9E0E6'
LIGHT = NEUTRAL_COLORS['light_background']
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
MODALITY_ORDER = ['Metagenomics', 'Radiomics', 'Urine']
MBP_ORDER = ['MBP-I', 'MBP-II', 'MBP-III']
GROUP_ORDER = ['Shared backbone', 'Response-preferred', 'PFS-preferred', 'OS-preferred']
BENEFIT_CMAP = LinearSegmentedColormap.from_list('aurora_benefit', DIVERGING_COLORS, N=256)
FEATURE_VALUE_CMAP = LinearSegmentedColormap.from_list('aurora_feature_value', DIVERGING_COLORS, N=256)
INPUT_FILES = {'a': get_value('figures.config.item_154'), 'b': get_value('figures.config.item_206'), 'c': get_value('figures.config.item_084'), 'd_order': get_value('figures.config.item_173'), 'd_patient_Response': get_value('figures.config.item_176'), 'd_patient_PFS': get_value('figures.config.item_175'), 'd_patient_OS': get_value('figures.config.item_174'), 'd_mean_Response': get_value('figures.config.item_146'), 'd_mean_PFS': get_value('figures.config.item_145'), 'd_mean_OS': get_value('figures.config.item_144'), 'e': get_value('figures.config.item_111'), 'f': get_value('figures.config.item_168')}

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT['base'], 'axes.labelsize': FONT['axis'], 'axes.titlesize': FONT['header'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'legend.fontsize': FONT['legend'], 'legend.title_fontsize': FONT['legend'], 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.max_open_warning': 0})

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
    frame = pd.read_csv(path)
    return frame.set_index(frame.columns[0])

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for key, filename in INPUT_FILES.items():
        path = INPUT_DIR / filename
        bundle[key] = _read_matrix(path) if key.startswith('d_patient_') or key.startswith('d_mean_') else pd.read_csv(path)
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    bundle['contract'] = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> None:
    manifest = {row['snapshot']: row for row in bundle['manifest']['files']}
    for filename in INPUT_FILES.values():
        if filename not in manifest:
            raise ValueError(f'Input manifest does not register {filename}')
        entry = manifest[filename]
        path = INPUT_DIR / filename
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch for frozen input {filename}')
        shape = pd.read_csv(path).shape
        expected = (int(entry['rows']), int(entry['columns']))
        if shape != expected:
            raise ValueError(f'Unexpected shape for {filename}: {shape}; expected {expected}')
    a = bundle['a']
    _require_columns(a, ['outcome_clean', 'modal_clean', 'fraction_clean'], '6A')
    if len(a) != 9 or a.duplicated(['outcome_clean', 'modal_clean']).any():
        raise ValueError('6A must contain one row for every endpoint-modality cell')
    if set(a['outcome_clean']) != set(ENDPOINT_ORDER) or set(a['modal_clean']) != set(MODALITY_ORDER):
        raise ValueError('6A endpoint or modality membership drifted')
    sums = a.groupby('outcome_clean')['fraction_clean'].sum().reindex(ENDPOINT_ORDER)
    if not np.allclose(sums.values, 1.0, atol=2e-06):
        raise ValueError(f'6A modality fractions do not sum to one: {sums.to_dict()}')
    b = bundle['b']
    _require_columns(b, ['feature_clean', 'outcome_clean', 'modal_clean', 'relative_importance', 'direction_rho', 'group'], '6B')
    if len(b) != 42 or b['feature_clean'].nunique() != 14 or b.duplicated(['feature_clean', 'outcome_clean']).any():
        raise ValueError('6B must contain 14 frozen features x 3 endpoints')
    if list(dict.fromkeys(b['group'])) != GROUP_ORDER:
        raise ValueError('6B group order or group membership drifted')
    if not b['relative_importance'].between(0, 1).all() or not b['direction_rho'].between(-1, 1).all():
        raise ValueError('6B importance or direction is outside the locked numeric range')
    c = bundle['c']
    _require_columns(c, ['feature_clean', 'modal_clean', 'overall_relative_importance', 'x', 'y'], '6C')
    if len(c) != 39 or c['feature_clean'].nunique() != 39:
        raise ValueError('6C must contain 39 unique MRU features')
    if not np.isfinite(c[['x', 'y']].to_numpy(dtype=float)).all():
        raise ValueError('6C frozen simplex coordinates contain non-finite values')
    locked_labels = bundle['contract']['panel_mapping']['C']['labelled_features']
    if not set(locked_labels.values()).issubset(set(c['feature_clean'])):
        raise ValueError('6C contract-labelled features are absent')
    order = bundle['d_order']
    _require_columns(order, ['sample', 'MBP_raw', 'composite_benefit_rank'], '6D patient order')
    if len(order) != 109 or order['sample'].nunique() != 109:
        raise ValueError('6D patient order must contain 109 unique patients')
    if order.groupby('MBP_raw', sort=False).size().to_dict() != {'MBP-I': 37, 'MBP-II': 36, 'MBP-III': 36}:
        raise ValueError('6D MBP group counts drifted')
    patients = order['sample'].astype(str).tolist()
    reference_features: list[str] | None = None
    for endpoint in ENDPOINT_ORDER:
        patient_matrix = bundle[f'd_patient_{endpoint}']
        group_mean = bundle[f'd_mean_{endpoint}']
        if patient_matrix.shape != (15, 109) or group_mean.shape != (15, 3):
            raise ValueError(f'6D {endpoint} matrix shape drifted')
        if patient_matrix.columns.astype(str).tolist() != patients:
            raise ValueError(f'6D {endpoint} patient columns do not match frozen order')
        if group_mean.columns.tolist() != MBP_ORDER:
            raise ValueError(f'6D {endpoint} MBP-mean columns drifted')
        features = patient_matrix.index.astype(str).tolist()
        if group_mean.index.astype(str).tolist() != features:
            raise ValueError(f'6D {endpoint} group-mean feature order drifted')
        if reference_features is None:
            reference_features = features
        elif features != reference_features:
            raise ValueError('6D feature order differs across endpoints')
        if np.nanmin(patient_matrix.to_numpy(dtype=float)) < -1.000001 or np.nanmax(patient_matrix.to_numpy(dtype=float)) > 1.000001:
            raise ValueError(f'6D {endpoint} patient matrix is not locked to [-1, 1]')
    e = bundle['e']
    _require_columns(e, ['sample', 'outcome', 'feature', 'feature_value_percentile', 'benefit_oriented_shap', 'y_base', 'y_jitter'], '6E')
    if len(e) != 3270 or e.groupby('outcome').size().reindex(ENDPOINT_ORDER).tolist() != [1090, 1090, 1090]:
        raise ValueError('6E must contain 1090 frozen observations for each endpoint')
    if not e['feature_value_percentile'].between(0, 1).all():
        raise ValueError('6E feature-value percentiles left [0, 1]')
    e_features = e[['feature', 'y_base']].drop_duplicates()
    if len(e_features) != 10 or e_features['y_base'].nunique() != 10:
        raise ValueError('6E shared-y feature order must contain ten positions')
    f = bundle['f']
    _require_columns(f, ['sample', 'endpoint_a', 'endpoint_b', 'spearman_rho', 'MBP_raw', 'n_features'], '6F')
    if len(f) != 327 or f['sample'].nunique() != 109 or (not (f['n_features'] == 39).all()):
        raise ValueError('6F locked row, patient, or feature count drifted')
    expected_pairs = [('Response', 'PFS'), ('Response', 'OS'), ('PFS', 'OS')]
    if set(map(tuple, f[['endpoint_a', 'endpoint_b']].drop_duplicates().to_numpy())) != set(expected_pairs):
        raise ValueError('6F endpoint pairs drifted')
    frozen_medians = bundle['contract']['panel_mapping']['F']['frozen_display_medians']
    for endpoint_a, endpoint_b in expected_pairs:
        key = f'{endpoint_a}-{endpoint_b}'
        observed = float(f.loc[(f['endpoint_a'] == endpoint_a) & (f['endpoint_b'] == endpoint_b), 'spearman_rho'].median())
        if not math.isclose(observed, float(frozen_medians[key]), abs_tol=6e-07):
            raise ValueError(f'6F frozen median mismatch for {key}: {observed}')

def _rect(width_mm: float, height_mm: float, left: float, top: float, width: float, height: float) -> list[float]:
    return [left / width_mm, 1.0 - (top + height) / height_mm, width / width_mm, height / height_mm]

def _new_axis(fig: mpl.figure.Figure, width_mm: float, height_mm: float, rect: Sequence[float], panel: str) -> mpl.axes.Axes:
    ax = fig.add_axes(_rect(width_mm, height_mm, *rect))
    ax._qa_meta = {'panel': panel}
    return ax

def _tag(registry: ArtistRegistry, artist: Any, *, panel: str, name: str, role: str='direct_label') -> Any:
    return registry.tag(artist, name=name, role=role, panel=panel)

def _panel_letter(fig: mpl.figure.Figure, label: str, left: float, top: float, registry: ArtistRegistry) -> None:
    artist = fig.text(left / FIGURE_WIDTH_MM, 1 - top / FIGURE_HEIGHT_MM, label.lower(), ha='left', va='top', fontsize=FONT['panel'], fontweight='bold', color=TEXT)
    _tag(registry, artist, panel=label.lower(), name=f'panel-{label.lower()}-letter', role='panel_label')

def _clean_axis(ax: mpl.axes.Axes, *, grid_axis: str | None=None) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(width=0.55, length=2.2, pad=1.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.35, alpha=0.75, zorder=0)

def _endpoint_header(ax: mpl.axes.Axes, endpoint: str, registry: ArtistRegistry, panel: str, *, y: float=1.06) -> None:
    artist = ax.text(0.5, y, endpoint, transform=ax.transAxes, ha='center', va='bottom', fontsize=FONT['header'], fontweight='bold', color=ENDPOINT_COLORS[endpoint], clip_on=False)
    _tag(registry, artist, panel=panel, name=f'{panel}-{endpoint.lower()}-header', role='panel_title')
    ax.plot([0.43, 0.57], [y - 0.035, y - 0.035], transform=ax.transAxes, color=ENDPOINT_COLORS[endpoint], linewidth=1.2, solid_capstyle='round', clip_on=False)

def _feature_label(feature: str, max_len: int=30) -> str:
    raw = str(feature)
    label = re.sub('^(Metagenomics|Microbiota|Microbiome|Radiomics|Blood|Urine)[_:]{1,2}', '', raw, flags=re.I)
    aliases = {'wavelet-HLL_glszm_ZoneEntropy': 'Wavelet-HLL zone entropy', 'wavelet-HLL_glszm_SizeZoneNonUniformityNormalized': 'Wavelet-HLL size-zone nonuniformity', 'wavelet-HLL_glszm_SmallAreaEmphasis': 'Wavelet-HLL small-area emphasis'}
    label = aliases.get(label, label)
    label = re.sub('\\s+', ' ', label.replace('_', ' ')).strip()
    return label if len(label) <= max_len else label[:max_len - 1].rstrip() + '…'

def _modality_from_feature(feature: str) -> str:
    prefix = str(feature).split('__', 1)[0]
    return prefix if prefix in MODALITY_ORDER else ''

def _text_color(fill: Sequence[float]) -> str:
    r, g, b = fill[:3]
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.58 else TEXT

def _importance_size(values: Any, maximum: float, *, minimum: float=10.0, span: float=62.0) -> Any:
    arr = np.asarray(values, dtype=float)
    scaled = np.clip(arr / max(maximum, 1e-12), 0, None)
    out = minimum + span * np.power(scaled, 0.78)
    return float(out) if out.ndim == 0 else out

def draw_panel_a(fig: mpl.figure.Figure, ax: mpl.axes.Axes, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    data = bundle['a']
    y = np.arange(len(ENDPOINT_ORDER))
    left = np.zeros(len(ENDPOINT_ORDER), dtype=float)
    for modality in MODALITY_ORDER:
        values = np.array([float(data.loc[(data['outcome_clean'] == endpoint) & (data['modal_clean'] == modality), 'fraction_clean'].iloc[0]) for endpoint in ENDPOINT_ORDER])
        color = MODALITY_COLORS[modality]
        bars = ax.barh(y, values, left=left, height=0.56, color=color, edgecolor='white', linewidth=0.55, label=modality, zorder=3)
        for index, (bar, value, offset) in enumerate(zip(bars, values, left)):
            artist = ax.text(offset + value / 2, index, f'{100 * value:.0f}%', ha='center', va='center', fontsize=FONT['value'], fontweight='bold', color=_text_color(mpl.colors.to_rgba(color)), zorder=4)
            _tag(registry, artist, panel='a', name=f'a-{modality.lower()}-{index}-value', role='value')
        left += values
    ax.set_yticks(y, ENDPOINT_ORDER)
    ax.set_xlim(0, 1)
    ax.set_ylim(2.55, -0.55)
    ax.set_xticks([0, 0.5, 1.0], ['0', '50', '100'])
    ax.set_xlabel('Fraction of total mean |SHAP| (%)')
    _clean_axis(ax, grid_axis='x')
    legend = ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=3, handlelength=1.15, columnspacing=0.85, handletextpad=0.35, borderaxespad=0)
    _tag(registry, legend, panel='a', name='a-modality-legend', role='legend')

def draw_panel_b(fig: mpl.figure.Figure, axes: Mapping[str, mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    labels_ax, matrix_ax, size_ax, cbar_ax = (axes['labels'], axes['matrix'], axes['size'], axes['cbar'])
    data = bundle['b'].copy()
    features = list(dict.fromkeys(data['feature_clean'].astype(str)))
    y_map = {feature: index for index, feature in enumerate(features)}
    x_map = {endpoint: index for index, endpoint in enumerate(ENDPOINT_ORDER)}
    maximum = float(data['relative_importance'].max())
    group_fill = {'Shared backbone': '#F3F7FA', 'Response-preferred': '#FCEFF0', 'PFS-preferred': '#EDF4FA', 'OS-preferred': '#F7EEF7'}
    group_color = {'Shared backbone': REFERENCE, 'Response-preferred': ENDPOINT_COLORS['Response'], 'PFS-preferred': ENDPOINT_COLORS['PFS'], 'OS-preferred': ENDPOINT_COLORS['OS']}
    group_short = {'Shared backbone': 'Shared', 'Response-preferred': 'Response', 'PFS-preferred': 'PFS', 'OS-preferred': 'OS'}
    for group in GROUP_ORDER:
        group_features = [feature for feature in features if data.loc[data['feature_clean'] == feature, 'group'].iloc[0] == group]
        positions = [y_map[feature] for feature in group_features]
        lo, hi = (min(positions) - 0.5, max(positions) + 0.5)
        for axis in (labels_ax, matrix_ax):
            axis.axhspan(lo, hi, color=group_fill[group], zorder=-3)
        labels_ax.add_patch(Rectangle((0.015, lo), 0.018, hi - lo, facecolor=group_color[group], edgecolor='none', zorder=2))
        artist = labels_ax.text(0.055, (lo + hi) / 2, group_short[group], ha='left', va='center', fontsize=FONT['small'], fontweight='bold', color=group_color[group])
        _tag(registry, artist, panel='b', name=f'b-{group_short[group].lower()}-group', role='direct_label')
        if lo > -0.5:
            labels_ax.axhline(lo, color='#C9D1D8', linewidth=0.45, zorder=1)
            matrix_ax.axhline(lo, color='#C9D1D8', linewidth=0.45, zorder=1)
    for feature in features:
        row = data.loc[data['feature_clean'] == feature].iloc[0]
        y = y_map[feature]
        modality = str(row['modal_clean'])
        labels_ax.scatter(0.305, y, s=18, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.35, zorder=4)
        artist = labels_ax.text(0.345, y, _feature_label(feature, 29), ha='left', va='center', fontsize=FONT['small'], color=TEXT)
        _tag(registry, artist, panel='b', name=f'b-feature-{y}', role='tick')
    matrix_ax.scatter(data['outcome_clean'].map(x_map), data['feature_clean'].map(y_map), s=_importance_size(data['relative_importance'], maximum), c=data['direction_rho'], cmap=BENEFIT_CMAP, norm=Normalize(-1, 1), edgecolor='white', linewidth=0.45, zorder=4)
    labels_ax.set_xlim(0, 1)
    labels_ax.set_ylim(len(features) - 0.5, -0.5)
    labels_ax.axis('off')
    matrix_ax.set_xlim(-0.5, 2.5)
    matrix_ax.set_ylim(len(features) - 0.5, -0.5)
    matrix_ax.set_xticks(range(3), ENDPOINT_ORDER)
    matrix_ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=2)
    for label, endpoint in zip(matrix_ax.get_xticklabels(), ENDPOINT_ORDER):
        label.set_color(ENDPOINT_COLORS[endpoint])
        label.set_fontweight('bold')
    matrix_ax.set_yticks(range(len(features)), [''] * len(features))
    matrix_ax.tick_params(axis='y', length=0)
    matrix_ax.grid(True, color=GRID, linewidth=0.38, zorder=0)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.axis('off')
    artist = size_ax.text(0.0, 0.98, 'Relative importance', ha='left', va='top', fontsize=FONT['legend'], fontweight='bold', color=TEXT)
    _tag(registry, artist, panel='b', name='b-size-title', role='legend')
    for idx, value in enumerate([0.01, 0.03, 0.06]):
        y = 0.7 - idx * 0.26
        size_ax.scatter(0.18, y, s=_importance_size(value, maximum), facecolor='#AAB5BE', edgecolor='white', linewidth=0.4)
        artist = size_ax.text(0.39, y, f'{100 * value:.0f}%', ha='left', va='center', fontsize=FONT['small'], color=TEXT)
        _tag(registry, artist, panel='b', name=f'b-size-{idx}', role='value')
    sm = mpl.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=BENEFIT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=FONT['small'], length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label('Higher-value direction (ρ)', fontsize=FONT['small'], labelpad=1.2)

def draw_panel_c(fig: mpl.figure.Figure, axes: Mapping[str, mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    ax, key_ax = (axes['simplex'], axes['key'])
    data = bundle['c']
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color=REFERENCE, linewidth=0.7, zorder=1)

    def bary(response: float, pfs: float, os_value: float) -> tuple[float, float]:
        return (pfs + 0.5 * os_value, np.sqrt(3) / 2 * os_value)
    for fraction in (0.25, 0.5, 0.75):
        families = [(bary(fraction, 1 - fraction, 0), bary(fraction, 0, 1 - fraction)), (bary(1 - fraction, fraction, 0), bary(0, fraction, 1 - fraction)), (bary(1 - fraction, 0, fraction), bary(0, 1 - fraction, fraction))]
        for start, stop in families:
            ax.plot([start[0], stop[0]], [start[1], stop[1]], color=GRID, linewidth=0.35, zorder=0)
    maximum = float(data['overall_relative_importance'].max())
    for modality in MODALITY_ORDER:
        subset = data.loc[data['modal_clean'] == modality]
        if subset.empty:
            continue
        ax.scatter(subset['x'], subset['y'], s=_importance_size(subset['overall_relative_importance'], maximum, minimum=9, span=52), marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.45, alpha=0.94, zorder=3)
    locked = bundle['contract']['panel_mapping']['C']['labelled_features']
    offsets = {'Response': (-23, -9), 'PFS': (25, -7), 'OS': (17, 18)}
    align = {'Response': ('right', 'top'), 'PFS': ('left', 'top'), 'OS': ('left', 'bottom')}
    for endpoint in ENDPOINT_ORDER:
        row = data.loc[data['feature_clean'] == locked[endpoint]].iloc[0]
        artist = ax.annotate(_feature_label(row['feature_clean'], 21), xy=(float(row['x']), float(row['y'])), xytext=offsets[endpoint], textcoords='offset points', ha=align[endpoint][0], va=align[endpoint][1], fontsize=FONT['small'], color=TEXT, bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.9), arrowprops=dict(arrowstyle='-', color='#9AA5AE', linewidth=0.45, shrinkA=1, shrinkB=2), zorder=5)
        _tag(registry, artist, panel='c', name=f'c-{endpoint.lower()}-feature', role='direct_label')
    vertex_specs = [(0.0, -0.025, 'Response', 'left', 'top'), (1.0, -0.025, 'PFS', 'right', 'top'), (0.5, np.sqrt(3) / 2 + 0.035, 'OS', 'center', 'bottom')]
    for x, y, endpoint, ha, va in vertex_specs:
        artist = ax.text(x, y, endpoint, ha=ha, va=va, fontsize=FONT['header'], fontweight='bold', color=ENDPOINT_COLORS[endpoint])
        _tag(registry, artist, panel='c', name=f'c-{endpoint.lower()}-vertex', role='axis_label')
    center_x, center_y = bary(1 / 3, 1 / 3, 1 / 3)
    artist = ax.text(center_x, center_y - 0.04, 'Equal', ha='center', va='top', fontsize=FONT['small'], color=MUTED, bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.86))
    _tag(registry, artist, panel='c', name='c-equal', role='direct_label')
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.1, 0.96)
    ax.set_aspect('equal')
    ax.axis('off')
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.axis('off')
    artist = key_ax.text(0.0, 0.98, 'Modality', ha='left', va='top', fontsize=FONT['legend'], fontweight='bold', color=TEXT)
    _tag(registry, artist, panel='c', name='c-modality-title', role='legend')
    for idx, modality in enumerate(MODALITY_ORDER):
        y = 0.81 - idx * 0.16
        key_ax.scatter(0.1, y, s=20, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.35)
        artist = key_ax.text(0.25, y, modality, ha='left', va='center', fontsize=FONT['small'], color=TEXT)
        _tag(registry, artist, panel='c', name=f'c-modality-{idx}', role='legend')
    artist = key_ax.text(0.0, 0.29, 'Overall importance', ha='left', va='top', fontsize=FONT['legend'], fontweight='bold', color=TEXT)
    _tag(registry, artist, panel='c', name='c-importance-title', role='legend')
    for idx, value in enumerate([0.02, 0.05]):
        y = 0.13
        x = 0.13 + idx * 0.45
        key_ax.scatter(x, y, s=_importance_size(value, maximum, minimum=9, span=52), facecolor='#AAB5BE', edgecolor='white', linewidth=0.35)
        artist = key_ax.text(x + 0.1, y, f'{100 * value:.0f}%', ha='left', va='center', fontsize=FONT['small'], color=TEXT)
        _tag(registry, artist, panel='c', name=f'c-importance-{idx}', role='value')

def draw_panel_d(fig: mpl.figure.Figure, axes: Mapping[str, Any], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    label_ax = axes['labels']
    order = bundle['d_order']
    features = bundle['d_patient_Response'].index.astype(str).tolist()
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(len(features) - 0.5, -0.5)
    label_ax.axis('off')
    for index, feature in enumerate(features):
        modality = _modality_from_feature(feature)
        label_ax.scatter(0.025, index, s=16, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.3)
        artist = label_ax.text(0.075, index, _feature_label(feature, 30), ha='left', va='center', fontsize=FONT['small'], color=TEXT)
        _tag(registry, artist, panel='d', name=f'd-feature-{index}', role='tick')
    handles = [Line2D([0], [0], marker=MBP_MARKERS[group], linestyle='', markerfacecolor=MBP_COLORS[group], markeredgecolor='none', markersize=4.2, label=group) for group in MBP_ORDER]
    legend = label_ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.035), ncol=3, handletextpad=0.3, columnspacing=0.65, borderaxespad=0)
    _tag(registry, legend, panel='d', name='d-mbp-legend', role='legend')
    group_codes = np.array([[MBP_ORDER.index(group) for group in order['MBP_raw']]], dtype=float)
    group_cmap = ListedColormap([MBP_COLORS[group] for group in MBP_ORDER])
    last_image = None
    for endpoint in ENDPOINT_ORDER:
        strip_ax = axes[endpoint]['strip']
        heat_ax = axes[endpoint]['heat']
        mean_ax = axes[endpoint]['mean']
        patient_matrix = bundle[f'd_patient_{endpoint}'].to_numpy(dtype=float)
        group_mean = bundle[f'd_mean_{endpoint}'].to_numpy(dtype=float)
        strip_ax.imshow(group_codes, aspect='auto', interpolation='nearest', cmap=group_cmap, vmin=0, vmax=2)
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
        for spine in strip_ax.spines.values():
            spine.set_visible(False)
        _endpoint_header(strip_ax, endpoint, registry, 'd', y=1.52)
        last_image = heat_ax.imshow(patient_matrix, aspect='auto', interpolation='nearest', cmap=BENEFIT_CMAP, vmin=-1, vmax=1, rasterized=True)
        heat_ax.set_xticks([])
        heat_ax.set_yticks([])
        for boundary in (36.5, 72.5):
            heat_ax.axvline(boundary, color='white', linewidth=0.55)
            strip_ax.axvline(boundary, color='white', linewidth=0.7)
        for spine in heat_ax.spines.values():
            spine.set_visible(False)
        mean_ax.imshow(group_mean, aspect='auto', interpolation='nearest', cmap=BENEFIT_CMAP, vmin=-1, vmax=1, rasterized=True)
        mean_ax.set_yticks([])
        mean_ax.set_xticks(range(3), ['I', 'II', 'III'])
        mean_ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=1, labelsize=FONT['small'])
        for spine in mean_ax.spines.values():
            spine.set_visible(False)
    if last_image is None:
        raise RuntimeError('6D did not create a heatmap')
    cbar = fig.colorbar(last_image, cax=axes['cbar'], orientation='horizontal')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '+1'])
    cbar.ax.tick_params(labelsize=FONT['small'], length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label('Row-scaled local SHAP (benefit-oriented)', fontsize=FONT['small'], labelpad=1.0)

def draw_panel_e(fig: mpl.figure.Figure, axes: Mapping[str, Any], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    data = bundle['e']
    order = data[['feature', 'y_base']].drop_duplicates().sort_values('y_base')
    features = order['feature'].astype(str).tolist()
    y_values = order['y_base'].astype(float).tolist()
    for index, endpoint in enumerate(ENDPOINT_ORDER):
        ax = axes[endpoint]
        subset = data.loc[data['outcome'] == endpoint]
        ax.scatter(subset['benefit_oriented_shap'], subset['y_base'] + subset['y_jitter'], c=subset['feature_value_percentile'], cmap=FEATURE_VALUE_CMAP, norm=Normalize(0, 1), s=5.2, alpha=0.82, edgecolors='none', rasterized=True, zorder=3)
        ax.axvline(0, color=REFERENCE, linewidth=0.55, linestyle=(0, (2.5, 2.5)), zorder=1)
        extent = float(np.nanmax(np.abs(subset['benefit_oriented_shap'].to_numpy(dtype=float))))
        limit = math.ceil(extent * 20) / 20
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.55, 9.55)
        ax.set_yticks(y_values)
        if index == 0:
            ax.set_yticklabels([_feature_label(feature, 31) for feature in features], fontsize=FONT['small'])
            ax.tick_params(axis='y', length=0, pad=1.8)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        if index == 1:
            ax.set_xlabel('Benefit-oriented SHAP')
        _clean_axis(ax, grid_axis='x')
        _endpoint_header(ax, endpoint, registry, 'e', y=1.05)
    sm = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_VALUE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axes['cbar'], orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=FONT['small'], length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label('Feature value', fontsize=FONT['small'], labelpad=1.2)

def _display_jitter(sample: str, endpoint_a: str, endpoint_b: str) -> float:
    raw = hashlib.sha256(f'{sample}|{endpoint_a}|{endpoint_b}'.encode('utf-8')).digest()
    value = int.from_bytes(raw[:4], 'big') / (2 ** 32 - 1)
    return (value - 0.5) * 0.045

def draw_panel_f(fig: mpl.figure.Figure, ax: mpl.axes.Axes, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    data = bundle['f']
    pair_order = [('Response', 'PFS'), ('Response', 'OS'), ('PFS', 'OS')]
    group_offset = {'MBP-I': -0.115, 'MBP-II': 0.0, 'MBP-III': 0.115}
    medians = bundle['contract']['panel_mapping']['F']['frozen_display_medians']
    for pair_index, (endpoint_a, endpoint_b) in enumerate(pair_order):
        subset = data.loc[(data['endpoint_a'] == endpoint_a) & (data['endpoint_b'] == endpoint_b)]
        for group in MBP_ORDER:
            group_data = subset.loc[subset['MBP_raw'] == group]
            x = [pair_index + group_offset[group] + _display_jitter(row.sample, endpoint_a, endpoint_b) for row in group_data.itertuples()]
            ax.scatter(x, group_data['spearman_rho'], s=7.0, marker=MBP_MARKERS[group], facecolor=MBP_COLORS[group], edgecolor='white', linewidth=0.25, alpha=0.85, rasterized=True, zorder=3)
        key = f'{endpoint_a}-{endpoint_b}'
        median = float(medians[key])
        ax.plot([pair_index - 0.2, pair_index + 0.2], [median, median], color=TEXT, linewidth=1.15, zorder=5)
        artist = ax.text(pair_index, 0.708, f'median ρ = {median:.2f}', ha='center', va='bottom', fontsize=FONT['value'], color=TEXT, bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.92), zorder=6)
        _tag(registry, artist, panel='f', name=f'f-median-{pair_index}', role='value')
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(0.7, 1.01)
    ax.set_xticks(range(3), ['Response–PFS', 'Response–OS', 'PFS–OS'])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel('Patient-level Spearman ρ')
    _clean_axis(ax, grid_axis='y')
    handles = [Line2D([0], [0], marker=MBP_MARKERS[group], linestyle='', markerfacecolor=MBP_COLORS[group], markeredgecolor='none', markersize=4.2, label=group) for group in MBP_ORDER]
    legend = ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3, handletextpad=0.3, columnspacing=0.65, borderaxespad=0)
    _tag(registry, legend, panel='f', name='f-mbp-legend', role='legend')
FULL_AXES_MM: Mapping[str, Any] = {'a': (18, 9, 70, 26), 'b': {'labels': (13, 53, 80, 34), 'matrix': (99, 53, 49, 34), 'size': (152, 53, 23, 18), 'cbar': (152, 77, 23, 2.5)}, 'c': {'simplex': (98, 5, 56, 38), 'key': (157, 8, 20, 30)}, 'd': {'labels': (6, 104, 47, 31), 'Response': {'strip': (57, 102, 29, 2), 'heat': (57, 105, 29, 30), 'mean': (88, 105, 6, 30)}, 'PFS': {'strip': (97, 102, 29, 2), 'heat': (97, 105, 29, 30), 'mean': (128, 105, 6, 30)}, 'OS': {'strip': (137, 102, 29, 2), 'heat': (137, 105, 29, 30), 'mean': (168, 105, 6, 30)}, 'cbar': (57, 140, 109, 2.4)}, 'e': {'Response': (54, 151, 35, 26), 'PFS': (95, 151, 35, 26), 'OS': (136, 151, 29, 26), 'cbar': (168, 155, 2.5, 18)}, 'f': (27, 184, 147, 20)}

def _axes_from_spec(fig: mpl.figure.Figure, width_mm: float, height_mm: float, spec: Any, panel: str) -> Any:
    if isinstance(spec, tuple):
        return _new_axis(fig, width_mm, height_mm, spec, panel)
    return {key: _axes_from_spec(fig, width_mm, height_mm, value, panel) for key, value in spec.items()}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = {panel: _axes_from_spec(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_AXES_MM[panel], panel) for panel in 'abcdef'}
    draw_panel_a(fig, axes['a'], bundle, registry)
    draw_panel_b(fig, axes['b'], bundle, registry)
    draw_panel_c(fig, axes['c'], bundle, registry)
    draw_panel_d(fig, axes['d'], bundle, registry)
    draw_panel_e(fig, axes['e'], bundle, registry)
    draw_panel_f(fig, axes['f'], bundle, registry)
    for panel, position in {'a': (2, 4), 'b': (2, 49), 'c': (93, 4), 'd': (2, 98), 'e': (2, 148), 'f': (2, 181)}.items():
        _panel_letter(fig, panel, *position, registry)
    return (fig, registry)
PANEL_LAYOUTS: Mapping[str, Mapping[str, Any]] = {'a': {'size': (95.0, 48.0), 'spec': (18, 11, 68, 25)}, 'b': {'size': (180.0, 62.0), 'spec': {'labels': (12, 9, 83, 42), 'matrix': (101, 9, 50, 42), 'size': (155, 9, 21, 22), 'cbar': (155, 42, 21, 3)}}, 'c': {'size': (90.0, 65.0), 'spec': {'simplex': (6, 8, 58, 48), 'key': (67, 10, 20, 38)}}, 'd': {'size': (180.0, 63.0), 'spec': {'labels': (6, 14, 47, 32), 'Response': {'strip': (57, 11, 29, 2), 'heat': (57, 14, 29, 32), 'mean': (88, 14, 6, 32)}, 'PFS': {'strip': (97, 11, 29, 2), 'heat': (97, 14, 29, 32), 'mean': (128, 14, 6, 32)}, 'OS': {'strip': (137, 11, 29, 2), 'heat': (137, 14, 29, 32), 'mean': (168, 14, 6, 32)}, 'cbar': (57, 53, 109, 2.5)}}, 'e': {'size': (180.0, 57.0), 'spec': {'Response': (54, 9, 35, 34), 'PFS': (95, 9, 35, 34), 'OS': (136, 9, 29, 34), 'cbar': (168, 14, 2.5, 24)}}, 'f': {'size': (145.0, 50.0), 'spec': (23, 9, 117, 30)}}

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    panel = panel.lower()
    if panel not in PANEL_LAYOUTS:
        raise ValueError(f'Unknown panel {panel!r}')
    configure_style()
    layout = PANEL_LAYOUTS[panel]
    width_mm, height_mm = layout['size']
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = _axes_from_spec(fig, width_mm, height_mm, layout['spec'], panel)
    if panel == 'a':
        draw_panel_a(fig, axes, bundle, registry)
    elif panel == 'b':
        draw_panel_b(fig, axes, bundle, registry)
    elif panel == 'c':
        draw_panel_c(fig, axes, bundle, registry)
    elif panel == 'd':
        draw_panel_d(fig, axes, bundle, registry)
    elif panel == 'e':
        draw_panel_e(fig, axes, bundle, registry)
    else:
        draw_panel_f(fig, axes, bundle, registry)
    return (fig, registry)

def _save_exact(fig: mpl.figure.Figure, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Figure 6 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
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
    f = bundle['f']
    return {'render_only': True, 'palette': PALETTE_ID, 'benefit_orientation_locked': True, 'panel_a_fraction_sums': bundle['a'].groupby('outcome_clean')['fraction_clean'].sum().reindex(ENDPOINT_ORDER).round(8).to_dict(), 'panel_b_rows_features': [len(bundle['b']), int(bundle['b']['feature_clean'].nunique())], 'panel_c_features': int(bundle['c']['feature_clean'].nunique()), 'panel_d_patients_features': [len(bundle['d_order']), len(bundle['d_patient_Response'])], 'panel_e_rows': len(bundle['e']), 'panel_f_rows_patients_features': [len(f), int(f['sample'].nunique()), int(f['n_features'].iloc[0])], 'forbidden_operations_executed': []}

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, registry = build_full_figure(bundle)
    layout = audit_figure_layout(fig, f"{get_value('figures.config.item_285')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout, OUTPUT_DIR / f"{get_value('figures.config.item_286')}{VERSION}.json")
    full_paths = _save_exact(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in 'abcdef':
        panel_fig, panel_registry = build_panel_figure(panel, bundle)
        width_mm, height_mm = PANEL_LAYOUTS[panel]['size']
        panel_audits[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_287')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        panel_paths.extend(_save_exact(panel_fig, PANEL_DIR / f"{get_value('figures.config.item_287')}{panel}_Nature_{VERSION}", width_mm=width_mm, height_mm=height_mm))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_288')}{VERSION}.json")
    scientific_path = write_json(scientific_summary(bundle), OUTPUT_DIR / f"{get_value('figures.config.item_289')}{VERSION}.json")
    outputs = full_paths + panel_paths + [layout_path, panel_audit_path, scientific_path]
    manifest = {'figure': 'Figure 6', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked panel-specific CSV snapshots', 'panel_letters': 'assembly only', 'layout_summary': layout['summary'], 'panel_layout_summaries': {panel: report['summary'] for panel, report in panel_audits.items()}, 'panel_mapping': bundle['contract']['panel_mapping'], 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in outputs]}
    manifest_path = write_json(manifest, OUTPUT_DIR / f"{get_value('figures.config.item_290')}{VERSION}.json")
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'scientific_guardrail_report': str(scientific_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {panel: report['summary'] for panel, report in panel_audits.items()}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
