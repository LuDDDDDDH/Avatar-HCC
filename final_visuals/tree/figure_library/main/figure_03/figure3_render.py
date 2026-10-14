from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
PILOT_ROOT = Path(__file__).resolve().parent
STYLE_ROOT = PILOT_ROOT.parents[1]
INPUT_DIR = PILOT_ROOT / 'inputs'
VERSION = 'v004'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_353')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_269')}{VERSION}"
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MODALITY_COLORS, NEUTRAL_COLORS, PALETTE_ID, add_panel_label, apply_theme, style_axis
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 210.0
MM_PER_INCH = 25.4
FONT_SIZES = {'base': 6.4, 'panel_label': 8.5, 'panel_title': 7.2, 'axis_label': 6.8, 'tick': 5.8, 'legend': 5.6, 'small': 5.4, 'tiny': 5.4}
PANEL_C_VALUE_LABEL_THRESHOLD = 0.1
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
LIGHT = NEUTRAL_COLORS['light_background']
RESPONSE_GROUP_COLORS = {'Responder': '#2F6FB0', 'Non-responder': ENDPOINT_COLORS['Response']}
SPECIES_COLOR = '#62AA9C'
FUSED_COLOR = '#617FC4'
MODALITY_NODE_COLORS = {'Metagenomics': MODALITY_COLORS['Metagenomics'], 'Species': SPECIES_COLOR, 'Radiomics': MODALITY_COLORS['Radiomics'], 'Blood': MODALITY_COLORS['Blood'], 'Urine': MODALITY_COLORS['Urine']}
MODALITY_ORDER = ['Metagenomics', 'Species', 'Radiomics', 'Blood', 'Urine']
MODALITY_DISPLAY = {'Metagenomics': 'KO functions', 'Species': 'Species', 'Radiomics': 'Radiomics', 'Blood': 'Blood', 'Urine': 'Urine', 'Fused': 'Fused'}
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
METRIC_ORDER = ['Response_AUC', 'PFS_Cindex', 'OS_Cindex']
METRIC_TO_ENDPOINT = {'Response_AUC': 'Response', 'PFS_Cindex': 'PFS', 'OS_Cindex': 'OS'}
METRIC_TITLES = {'Response_AUC': 'Response AUC', 'PFS_Cindex': 'PFS C-index', 'OS_Cindex': 'OS C-index'}
CLINICAL_DOMAINS = [('Host background', 0, 3), ('Tumor burden', 3, 8), ('Liver reserve / injury', 8, 13), ('Inflammation / nutrition', 13, 17)]
CLINICAL_LABELS = {'Age': 'Age', 'Hepatitis B': 'HBV', 'ECOG': 'ECOG', 'AFP': 'AFP', 'PVIKA': 'PIVKA-II', 'TBS': 'TBS', 'PVTT': 'PVTT', 'Metastasis': 'Metast.', 'Child-Pugh': 'Child-Pugh', 'ALBI_grade': 'ALBI', 'ALT': 'ALT', 'PT': 'PT', 'LDH': 'LDH', 'PNI': 'PNI', 'NLR': 'NLR', 'SII': 'SII', 'FIB-4': 'FIB-4'}

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _read_matrix(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / name, index_col=0)

def load_inputs() -> dict[str, object]:
    metadata = json.loads((INPUT_DIR / get_value('figures.config.item_109')).read_text(encoding='utf-8'))
    bundle: dict[str, object] = {'coordinates': pd.read_csv(INPUT_DIR / get_value('figures.config.item_108')), 'coordinate_metadata': metadata, 'permanova': pd.read_csv(INPUT_DIR / get_value('figures.config.item_178')), 'clinical_r': _read_matrix(get_value('figures.config.item_062')), 'clinical_q': _read_matrix(get_value('figures.config.item_061')), 'outcome_effect': _read_matrix(get_value('figures.config.item_045')), 'outcome_q': _read_matrix(get_value('figures.config.item_046')), 'contribution': pd.read_csv(INPUT_DIR / get_value('figures.config.item_133')), 'directional': pd.read_csv(INPUT_DIR / get_value('figures.config.item_049')), 'lomo': pd.read_csv(INPUT_DIR / get_value('figures.config.item_132')), 'outcomes': pd.read_csv(INPUT_DIR / get_value('figures.config.item_159'))}
    validate_inputs(bundle)
    return bundle

def _expected_axis_order(index: pd.Index) -> list[str]:
    groups = [[x for x in index if str(x).startswith('Metagenomics_')], [x for x in index if str(x).startswith('Species_')], [x for x in index if str(x).startswith('Radiomics_')], [x for x in index if str(x).startswith('Blood_')], [x for x in index if str(x).startswith('Urine_')]]
    return [item for group in groups for item in group]

def validate_inputs(bundle: Mapping[str, object]) -> None:
    coords = bundle['coordinates']
    metadata = bundle['coordinate_metadata']
    permanova = bundle['permanova']
    clinical_r = bundle['clinical_r']
    clinical_q = bundle['clinical_q']
    outcome_effect = bundle['outcome_effect']
    outcome_q = bundle['outcome_q']
    contribution = bundle['contribution']
    directional = bundle['directional']
    lomo = bundle['lomo']
    outcomes = bundle['outcomes']
    required_coordinate_columns = {'sample_id', 'fused_pcoa1', 'fused_pcoa2', 'response_label', 'pfs_landmark_group', 'os_landmark_group', 'os_early_event'}
    missing = required_coordinate_columns - set(coords.columns)
    if missing:
        raise ValueError(f'3A coordinate columns missing: {sorted(missing)}')
    if len(coords) != 109 or coords['sample_id'].nunique() != 109:
        raise ValueError('3A coordinates must contain 109 unique patients')
    expected_coord_hash = metadata['coordinate_csv']['sha256'].upper()
    observed_coord_hash = file_sha256(INPUT_DIR / get_value('figures.config.item_108'))
    if observed_coord_hash != expected_coord_hash:
        raise ValueError('3A coordinate snapshot SHA-256 mismatch')
    if len(outcomes) != 109 or get_column('figures.column.response') not in outcomes.columns:
        raise ValueError(get_value('figures.annotation.figure_03'))
    if permanova.shape[0] != 18:
        raise ValueError(f'3B expected 18 result rows, found {permanova.shape[0]}')
    if clinical_r.shape != (15, 17) or clinical_q.shape != (15, 17):
        raise ValueError('3C Spearman matrices must both be 15 x 17')
    if outcome_effect.shape != (15, 3) or outcome_q.shape != (15, 3):
        raise ValueError('3C adjusted-outcome matrices must both be 15 x 3')
    if not clinical_r.index.equals(clinical_q.index):
        raise ValueError('3C Spearman effect and FDR row orders differ')
    if not outcome_effect.index.equals(outcome_q.index):
        raise ValueError('3C adjusted effect and FDR row orders differ')
    if set(clinical_r.index) != set(outcome_effect.index):
        raise ValueError('3C left and right axis sets differ')
    if contribution.shape[0] != 15:
        raise ValueError('3D contribution summary must contain 5 blocks x 3 endpoints')
    if directional.shape[0] != 60:
        raise ValueError('3D directional summary must contain 5 x 4 x 3 rows')
    if lomo.shape[0] != 30:
        raise ValueError('3E LOMO summary must contain 2 x 5 x 3 rows')
    if not (lomo['Repeat_N'] == 5).all():
        raise ValueError('3E ranges are expected to summarize exactly five repeats')

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT_SIZES['base'], 'axes.labelsize': FONT_SIZES['axis_label'], 'axes.titlesize': FONT_SIZES['panel_title'], 'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'], 'legend.fontsize': FONT_SIZES['legend'], 'legend.title_fontsize': FONT_SIZES['small'], 'axes.linewidth': 0.55, 'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def clean_style_axis(ax: mpl.axes.Axes, *, grid: bool=False) -> mpl.axes.Axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(direction='out', width=0.55, length=2.3, pad=2)
    if grid:
        ax.grid(True, which='major', color='#D9E0E6', linewidth=0.35, alpha=0.65, zorder=0)
    else:
        ax.grid(False)
    return ax

def _lighten(color: str, amount: float=0.82) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)

def _format_signed_3(value: float) -> str:
    if abs(float(value)) < 0.0005:
        return '0.000'
    formatted = f'{float(value):+.3f}'
    return formatted.replace('-', '−')

def _heatmap_text_color(value: float, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> str:
    red, green, blue, _ = cmap(norm(float(value)))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.5 else TEXT

def _bbox_overlap_area(a: mpl.transforms.Bbox, b: mpl.transforms.Bbox) -> float:
    width = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    height = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return float(width * height)

def _assert_unique_keys(keys: list[tuple[str, ...]], expected: int, label: str) -> None:
    if len(keys) != expected or len(set(keys)) != expected:
        raise RuntimeError(f'{label} expected {expected} unique plotted records; found {len(keys)} records and {len(set(keys))} unique keys')

def _covariance_ellipse(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, color: str, *, n_std: float=1.5) -> None:
    if len(x) < 3:
        return
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))
    patch = Ellipse((float(np.mean(x)), float(np.mean(y))), width=float(width), height=float(height), angle=float(angle), facecolor=color, edgecolor=color, linewidth=0.8, linestyle=(0, (4, 2)), alpha=0.1, zorder=0)
    ax.add_patch(patch)

def _fused_stats(permanova: pd.DataFrame) -> str:
    rows = permanova.loc[permanova['Modality'].eq('Fused')].set_index('Endpoint')
    parts = []
    for endpoint, short in [('Response', 'Response'), ('Early PFS', 'PFS 6 mo'), ('Early OS', 'OS 12 mo')]:
        row = rows.loc[endpoint]
        effect = 100 * float(row['Adjusted_incremental_R2'])
        q_value = float(row['Adjusted_partial_PERMANOVA_FDR'])
        parts.append(f'{short}: {effect:.1f}% (q={q_value:.3g})')
    return '   |   '.join(parts)

def draw_panel_3a(fig: mpl.figure.Figure, spec: mpl.gridspec.SubplotSpec, coordinates: pd.DataFrame, permanova: pd.DataFrame, coordinate_metadata: Mapping[str, object], *, panel_label: str='a') -> list[mpl.axes.Axes]:
    inner = spec.subgridspec(1, 2, width_ratios=[4.6, 1.45], wspace=0.08)
    ax = fig.add_subplot(inner[0, 0])
    legend_ax = fig.add_subplot(inner[0, 1])
    legend_ax.axis('off')
    for group in ['Responder', 'Non-responder']:
        subset = coordinates.loc[coordinates['response_label'].eq(group)]
        _covariance_ellipse(ax, subset['fused_pcoa1'].to_numpy(), subset['fused_pcoa2'].to_numpy(), RESPONSE_GROUP_COLORS[group])
    marker_map = {'No early event': 'o', 'Early event': '^', 'Censored before landmark': 'X'}
    for row in coordinates.itertuples(index=False):
        face = RESPONSE_GROUP_COLORS[str(row.response_label)]
        marker = marker_map.get(str(row.pfs_landmark_group), 'o')
        early_os = bool(row.os_early_event)
        ax.scatter(float(row.fused_pcoa1), float(row.fused_pcoa2), s=15.5, marker=marker, facecolor=face, edgecolor=ENDPOINT_COLORS['OS'] if early_os else TEXT, linewidth=1.0 if early_os else 0.35, alpha=0.88, zorder=3)
    variance = coordinate_metadata['validation']['fused_pcoa_explained_variance_percent']
    ax.axhline(0, color='#D6DDE3', linewidth=0.45, linestyle=(0, (3, 2)), zorder=0)
    ax.axvline(0, color='#D6DDE3', linewidth=0.45, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlabel(f'Fused PCoA1 ({variance[0]:.1f}%)', labelpad=1.5)
    ax.set_ylabel(f'Fused PCoA2 ({variance[1]:.1f}%)', labelpad=1.5)
    ax.text(0.0, 1.015, _fused_stats(permanova), transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['small'], color=MUTED, clip_on=False)
    clean_style_axis(ax)
    add_panel_label(ax, panel_label, x_pt=-28, y_pt=0)
    legend_ax.text(0.0, 0.97, 'Response', fontsize=FONT_SIZES['small'], fontweight='semibold', va='top')
    for label, y in [('Responder', 0.86), ('Non-responder', 0.75)]:
        legend_ax.scatter(0.08, y, s=17, marker='o', facecolor=RESPONSE_GROUP_COLORS[label], edgecolor=TEXT, linewidth=0.35)
        legend_ax.text(0.22, y, label, fontsize=FONT_SIZES['small'], va='center')
    legend_ax.text(0.0, 0.63, 'PFS at 6 mo', fontsize=FONT_SIZES['small'], fontweight='semibold', va='top')
    for marker, label, y in [('o', 'No event', 0.52), ('^', 'Early event', 0.41), ('X', 'Censored', 0.3)]:
        legend_ax.scatter(0.08, y, s=17, marker=marker, facecolor='white', edgecolor=TEXT, linewidth=0.55)
        legend_ax.text(0.22, y, label, fontsize=FONT_SIZES['small'], va='center')
    legend_ax.text(0.0, 0.18, 'OS at 12 mo', fontsize=FONT_SIZES['small'], fontweight='semibold', va='top')
    legend_ax.scatter(0.08, 0.07, s=20, marker='o', facecolor='white', edgecolor=ENDPOINT_COLORS['OS'], linewidth=1.1)
    legend_ax.text(0.22, 0.07, 'Early event', fontsize=FONT_SIZES['small'], va='center')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    return [ax, legend_ax]

def draw_panel_3b(fig: mpl.figure.Figure, spec: mpl.gridspec.SubplotSpec, permanova: pd.DataFrame, *, panel_label: str='b') -> list[mpl.axes.Axes]:
    inner = spec.subgridspec(1, 2, width_ratios=[18, 1], wspace=0.18)
    ax = fig.add_subplot(inner[0, 0])
    cax = fig.add_subplot(inner[0, 1])
    row_order = MODALITY_ORDER + ['Fused']
    endpoint_source = ['Response', 'Early PFS', 'Early OS']
    matrix = np.full((len(row_order), len(endpoint_source)), np.nan)
    q_matrix = np.full_like(matrix, np.nan)
    dispersion_q = np.full_like(matrix, np.nan)
    for i, modality in enumerate(row_order):
        rows = permanova.loc[permanova['Modality'].eq(modality)].set_index('Endpoint')
        for j, endpoint in enumerate(endpoint_source):
            row = rows.loc[endpoint]
            matrix[i, j] = 100 * float(row['Adjusted_incremental_R2'])
            q_matrix[i, j] = float(row['Adjusted_partial_PERMANOVA_FDR'])
            dispersion_q[i, j] = float(row['PERMDISP_FDR'])
    cmap = LinearSegmentedColormap.from_list(get_value('figures.config.item_381'), ['#F7FBFF', '#DCEFEF', '#AFD7D1', '#72B7AB', '#2A8F78', '#285F9E'])
    vmax = max(6.0, math.ceil(float(np.nanmax(matrix))))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect='auto', interpolation='none')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', fontsize=FONT_SIZES['small'], color=TEXT)
            if q_matrix[i, j] < 0.05 and matrix[i, j] >= 3.0 and (dispersion_q[i, j] >= 0.05):
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=TEXT, linewidth=0.75))
            if dispersion_q[i, j] < 0.05:
                ax.text(j + 0.34, i - 0.29, '†', ha='center', va='center', fontsize=6.0, fontweight='bold', color=TEXT)
    ax.set_xticks(range(3), ['Response', 'PFS 6 mo', 'OS 12 mo'])
    ax.set_yticks(range(len(row_order)), [MODALITY_DISPLAY[x] for x in row_order])
    for label in ax.get_yticklabels():
        if label.get_text() == 'Fused':
            label.set_fontweight('semibold')
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, panel_label, x_pt=-11, y_pt=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('ΔR² (%)', fontsize=FONT_SIZES['small'], labelpad=2)
    cbar.ax.tick_params(labelsize=FONT_SIZES['tiny'], length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    return [ax, cax]

def _axis_label(name: str) -> str:
    prefix, axis = name.split('_', 1)
    display = {'Metagenomics': 'KO', 'Species': 'Species', 'Radiomics': 'Radiomics', 'Blood': 'Blood', 'Urine': 'Urine'}[prefix]
    if axis.startswith('PCoA'):
        axis = 'PCoA ' + axis.removeprefix('PCoA')
    elif axis.startswith('PC'):
        axis = 'PC ' + axis.removeprefix('PC')
    return f'{display} {axis.strip()}'

def draw_panel_3c(fig: mpl.figure.Figure, spec: mpl.gridspec.SubplotSpec, clinical_r: pd.DataFrame, clinical_q: pd.DataFrame, outcome_effect: pd.DataFrame, outcome_q: pd.DataFrame, *, panel_label: str='c') -> list[mpl.axes.Axes]:
    row_order = _expected_axis_order(clinical_r.index)
    clinical_r = clinical_r.loc[row_order]
    clinical_q = clinical_q.loc[row_order]
    outcome_effect = outcome_effect.loc[row_order, ENDPOINT_ORDER]
    outcome_q = outcome_q.loc[row_order, ENDPOINT_ORDER]
    inner = spec.subgridspec(1, 5, width_ratios=[16, 0.42, 0.7, 4, 0.42], wspace=0.16)
    ax_left = fig.add_subplot(inner[0, 0])
    cax_left = fig.add_subplot(inner[0, 1])
    ax_right = fig.add_subplot(inner[0, 3], sharey=ax_left)
    cax_right = fig.add_subplot(inner[0, 4])
    cmap = LinearSegmentedColormap.from_list(get_value('figures.config.item_363'), DIVERGING_COLORS)
    left_norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
    right_norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=1.6)
    left_im = ax_left.imshow(clinical_r.to_numpy(), cmap=cmap, norm=left_norm, aspect='auto', interpolation='none')
    right_im = ax_right.imshow(outcome_effect.to_numpy(), cmap=cmap, norm=right_norm, aspect='auto', interpolation='none')
    left_value_count = 0
    left_star_count = 0
    left_dot_count = 0
    for i in range(clinical_r.shape[0]):
        for j in range(clinical_r.shape[1]):
            value = float(clinical_r.iloc[i, j])
            q_value = float(clinical_q.iloc[i, j])
            reaches_threshold = abs(value) >= 0.2
            symbol = '*' if q_value < 0.05 and reaches_threshold else ''
            if not symbol and q_value < 0.1 and reaches_threshold:
                symbol = '·'
            display_value = round(value, 2)
            if abs(display_value) <= PANEL_C_VALUE_LABEL_THRESHOLD:
                continue
            left_star_count += int(symbol == '*')
            left_dot_count += int(symbol == '·')
            label = f'{display_value:.2f}{symbol}'
            artist = ax_left.text(j, i, label, ha='center', va='center', fontsize=FONT_SIZES['tiny'], fontweight='normal', color=_heatmap_text_color(value, cmap, left_norm), zorder=4)
            artist.set_gid(f'panel-c-clinical-value-{i}-{j}')
            left_value_count += 1
    right_value_count = 0
    right_star_count = 0
    right_dot_count = 0
    for i in range(outcome_effect.shape[0]):
        for j in range(outcome_effect.shape[1]):
            value = float(outcome_effect.iloc[i, j])
            q_value = float(outcome_q.iloc[i, j])
            reaches_threshold = abs(value) >= 0.25
            symbol = '*' if q_value < 0.05 and reaches_threshold else ''
            if not symbol and q_value < 0.1 and reaches_threshold:
                symbol = '·'
            display_value = round(value, 2)
            if abs(display_value) <= PANEL_C_VALUE_LABEL_THRESHOLD:
                continue
            right_star_count += int(symbol == '*')
            right_dot_count += int(symbol == '·')
            label = f'{display_value:.2f}{symbol}'
            artist = ax_right.text(j, i, label, ha='center', va='center', fontsize=FONT_SIZES['tiny'], fontweight='normal', color=_heatmap_text_color(value, cmap, right_norm), zorder=4)
            artist.set_gid(f'panel-c-outcome-value-{i}-{j}')
            right_value_count += 1
    if left_value_count != 86 or right_value_count != 31:
        raise RuntimeError(f'Panel c value-label contract failed: {left_value_count} + {right_value_count}')
    if (left_star_count, left_dot_count, right_star_count, right_dot_count) != (6, 0, 16, 0):
        raise RuntimeError(f'Panel c significance-symbol contract failed: left */·={left_star_count}/{left_dot_count}, right */·={right_star_count}/{right_dot_count}')
    row_labels = [_axis_label(x) for x in row_order]
    ax_left.set_yticks(range(len(row_order)), row_labels)
    ax_left.set_xticks(range(len(clinical_r.columns)), [CLINICAL_LABELS.get(x, x) for x in clinical_r.columns], rotation=55, ha='right', rotation_mode='anchor')
    ax_right.set_xticks(range(3), ['Response', 'PFS', 'OS'])
    ax_right.tick_params(axis='y', left=False, labelleft=False)
    ax_left.tick_params(length=0, pad=1.2, labelsize=FONT_SIZES['small'])
    ax_right.tick_params(length=0, pad=1.2, labelsize=FONT_SIZES['small'])
    for boundary in [2.5, 5.5, 8.5, 11.5]:
        ax_left.axhline(boundary, color=TEXT, linewidth=0.45)
        ax_right.axhline(boundary, color=TEXT, linewidth=0.45)
    for _, start, _ in CLINICAL_DOMAINS[1:]:
        ax_left.axvline(start - 0.5, color=TEXT, linewidth=0.45)
    header_labels = {'Host background': 'Host\nbackground', 'Tumor burden': 'Tumor\nburden', 'Liver reserve / injury': 'Liver reserve\n/ injury', 'Inflammation / nutrition': 'Inflammation\n/ nutrition'}
    for domain, start, end in CLINICAL_DOMAINS:
        ax_left.add_patch(Rectangle((start - 0.5, -1.6), end - start, 1.02, facecolor='#EDF3F6', edgecolor='white', linewidth=0.8, clip_on=False, zorder=2))
        ax_left.text((start + end - 1) / 2, -1.09, header_labels[domain], ha='center', va='center', fontsize=FONT_SIZES['tiny'], fontweight='semibold', color=TEXT, linespacing=0.88, clip_on=False, zorder=3)
    ax_right.add_patch(Rectangle((-0.5, -1.6), 3.0, 1.02, facecolor='#EDF3F6', edgecolor='white', linewidth=0.8, clip_on=False, zorder=2))
    ax_right.text(1.0, -1.09, 'Adjusted outcomes', ha='center', va='center', fontsize=FONT_SIZES['tiny'], fontweight='semibold', color=TEXT, clip_on=False, zorder=3)
    ax_left.set_ylim(len(row_order) - 0.5, -1.65)
    ax_right.set_ylim(len(row_order) - 0.5, -1.65)
    for axis in [ax_left, ax_right]:
        for spine in axis.spines.values():
            spine.set_visible(False)
    add_panel_label(ax_left, panel_label, x_pt=-11, y_pt=2)
    left_bar = fig.colorbar(left_im, cax=cax_left)
    left_bar.set_label('Spearman ρ', fontsize=FONT_SIZES['tiny'], labelpad=1.5)
    left_bar.ax.tick_params(labelsize=FONT_SIZES['tiny'], length=1.8, pad=1)
    left_bar.outline.set_linewidth(0.4)
    right_bar = fig.colorbar(right_im, cax=cax_right)
    right_bar.set_label('Adjusted effect', fontsize=FONT_SIZES['tiny'], labelpad=1.5)
    right_bar.ax.tick_params(labelsize=FONT_SIZES['tiny'], length=1.8, pad=1)
    right_bar.outline.set_linewidth(0.4)
    return [ax_left, cax_left, ax_right, cax_right]

def draw_contribution_summary(ax: mpl.axes.Axes, contribution: pd.DataFrame) -> list[mpl.text.Text]:
    offsets = {'Response_AUC': -0.28, 'PFS_Cindex': 0.0, 'OS_Cindex': 0.28}
    y_positions = np.arange(len(MODALITY_ORDER))
    value_artists: list[mpl.text.Text] = []
    plotted_keys: list[tuple[str, ...]] = []
    all_values: list[float] = []
    for metric in METRIC_ORDER:
        endpoint = METRIC_TO_ENDPOINT[metric]
        subset = contribution.loc[contribution['Metric'].eq(metric)].set_index('Added_modality')
        values = [float(subset.loc[m, 'Mean_increment']) for m in MODALITY_ORDER]
        all_values.extend(values)
        ax.scatter(values, y_positions + offsets[metric], s=16, marker={'Response_AUC': 'o', 'PFS_Cindex': 's', 'OS_Cindex': '^'}[metric], color=ENDPOINT_COLORS[endpoint], edgecolor='white', linewidth=0.35, label=endpoint, zorder=3)
        for modality, y, value in zip(MODALITY_ORDER, y_positions + offsets[metric], values):
            ax.plot([0, value], [y, y], color=ENDPOINT_COLORS[endpoint], linewidth=0.7, alpha=0.65, zorder=2)
            artist = ax.annotate(_format_signed_3(value), xy=(value, y), xytext=(2.8 if value >= 0 else -2.8, 0), textcoords='offset points', ha='left' if value >= 0 else 'right', va='center', fontsize=FONT_SIZES['tiny'], color=MUTED, bbox={'boxstyle': 'round,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88}, zorder=4)
            artist.set_gid(f'panel-d-contribution-{metric}-{modality}')
            value_artists.append(artist)
            plotted_keys.append((metric, modality))
    _assert_unique_keys(plotted_keys, 15, 'Panel d contribution labels')
    ax.axvline(0, color=REFERENCE, linewidth=0.55, zorder=1)
    ax.set_yticks(y_positions, [MODALITY_DISPLAY[m] for m in MODALITY_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel('Mean pairwise Δ', labelpad=0.5)
    span = max(all_values) - min(all_values)
    padding = max(0.035, 0.23 * span)
    ax.set_xlim(min(-0.035, min(all_values) - padding), max(all_values) + padding)
    ax.set_ylim(len(MODALITY_ORDER) - 0.48, -1.08)
    ax.legend(loc='upper center', bbox_to_anchor=(0.54, 0.995), ncol=3, handletextpad=0.3, columnspacing=0.65, borderaxespad=0.0)
    clean_style_axis(ax)
    return value_artists
NETWORK_POSITIONS = {'Metagenomics': (0.5, 0.92), 'Species': (0.91, 0.63), 'Radiomics': (0.76, 0.1), 'Blood': (0.24, 0.1), 'Urine': (0.09, 0.63)}

def _edge_rad(added: str, base: str) -> float:
    del added, base
    return 0.2

def _quadratic_edge_point(start: tuple[float, float], end: tuple[float, float], rad: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = b - a
    control = (a + b) / 2.0 + float(rad) * np.array([delta[1], -delta[0]])
    point = (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t ** 2 * b
    tangent = 2 * (1 - t) * (control - a) + 2 * t * (b - control)
    return (point, tangent)

def draw_directional_network(ax: mpl.axes.Axes, directional: pd.DataFrame, metric: str) -> list[mpl.text.Text]:
    subset = directional.loc[directional['Metric'].eq(metric)].copy()
    stable_positive = subset['Mean_repeat_delta'].gt(0) & subset['Positive_repeat_fraction'].ge(0.8)
    stable_negative = subset['Mean_repeat_delta'].lt(0) & subset['Positive_repeat_fraction'].le(0.2)
    subset = subset.loc[subset['Mean_repeat_delta'].abs().ge(0.02) & (stable_positive | stable_negative)]
    max_abs = max(0.02, float(subset['Mean_repeat_delta'].abs().max())) if len(subset) else 0.02
    plotted_keys = [(str(metric), str(row.Added_modality), str(row.Base_modality)) for row in subset.itertuples(index=False)]
    _assert_unique_keys(plotted_keys, len(subset), f'Panel d {metric} edge labels')
    for row in subset.itertuples(index=False):
        added = str(row.Added_modality)
        base = str(row.Base_modality)
        delta = float(row.Mean_repeat_delta)
        color = '#C44A58' if delta > 0 else '#2F6FB0'
        linestyle = '-' if delta > 0 else (0, (3, 2))
        width = 0.45 + 1.75 * abs(delta) / max_abs
        arrow = FancyArrowPatch(NETWORK_POSITIONS[added], NETWORK_POSITIONS[base], arrowstyle='-|>', mutation_scale=5.8, connectionstyle=f'arc3,rad={_edge_rad(added, base)}', linewidth=width, linestyle=linestyle, color=color, alpha=0.92, shrinkA=12, shrinkB=12, zorder=1)
        ax.add_patch(arrow)
    node_artists: list[mpl.artist.Artist] = []
    for modality, (x, y) in NETWORK_POSITIONS.items():
        color = MODALITY_NODE_COLORS[modality]
        circle = Circle((x, y), radius=0.082, facecolor=_lighten(color, 0.83), edgecolor=color, linewidth=0.9, zorder=3)
        ax.add_patch(circle)
        node_artists.append(circle)
        label = {'Metagenomics': 'KO', 'Radiomics': 'Radio.'}.get(modality, modality)
        node_artists.append(ax.text(x, y, label, ha='center', va='center', fontsize=FONT_SIZES['small'], fontweight='semibold', zorder=4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    header = ax.text(0.5, 1.015, METRIC_TO_ENDPOINT[metric], transform=ax.transAxes, ha='center', va='bottom', fontsize=FONT_SIZES['panel_title'], color=ENDPOINT_COLORS[METRIC_TO_ENDPOINT[metric]], fontweight='semibold', clip_on=False)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = [artist.get_window_extent(renderer).expanded(1.12, 1.18) for artist in node_artists]
    occupied.append(header.get_window_extent(renderer).expanded(1.05, 1.1))
    edge_labels: list[mpl.text.Text] = []
    edge_samples = []
    for row in subset.itertuples(index=False):
        start = NETWORK_POSITIONS[str(row.Added_modality)]
        end = NETWORK_POSITIONS[str(row.Base_modality)]
        rad = _edge_rad(str(row.Added_modality), str(row.Base_modality))
        points = [_quadratic_edge_point(start, end, rad, float(t))[0] for t in np.linspace(0.1, 0.9, 49)]
        edge_samples.extend(ax.transData.transform(np.asarray(points)))
    edge_samples_px = np.asarray(edge_samples)
    candidate_t = [0.5, 0.38, 0.62, 0.27, 0.73, 0.18, 0.82]
    candidate_offset = [0.035, -0.035, 0.065, -0.065, 0.095, -0.095, 0.13, -0.13, 0.17, -0.17]
    for row in subset.reindex(subset['Mean_repeat_delta'].abs().sort_values(ascending=False).index).itertuples(index=False):
        added = str(row.Added_modality)
        base = str(row.Base_modality)
        delta = float(row.Mean_repeat_delta)
        rad = _edge_rad(added, base)
        label = ax.text(0.5, 0.5, _format_signed_3(delta), ha='center', va='center', fontsize=FONT_SIZES['tiny'], color=MUTED, bbox={'boxstyle': 'round,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.92}, zorder=6)
        label.set_gid(f'panel-d-edge-{metric}-{added}-{base}')
        best: tuple[float, tuple[float, float], mpl.transforms.Bbox] | None = None
        for t in candidate_t:
            point, tangent = _quadratic_edge_point(NETWORK_POSITIONS[added], NETWORK_POSITIONS[base], rad, t)
            norm = float(np.linalg.norm(tangent))
            normal = np.array([-tangent[1], tangent[0]]) / max(norm, 1e-09)
            for offset in candidate_offset:
                candidate = point + offset * normal
                label.set_position((float(candidate[0]), float(candidate[1])))
                bbox = label.get_window_extent(renderer).expanded(1.06, 1.12)
                outside = max(0.0, ax.bbox.x0 - bbox.x0) + max(0.0, bbox.x1 - ax.bbox.x1) + max(0.0, ax.bbox.y0 - bbox.y0) + max(0.0, bbox.y1 - ax.bbox.y1)
                overlap = sum((_bbox_overlap_area(bbox, other) for other in occupied))
                path_hits = int(np.count_nonzero((edge_samples_px[:, 0] >= bbox.x0 - 1.0) & (edge_samples_px[:, 0] <= bbox.x1 + 1.0) & (edge_samples_px[:, 1] >= bbox.y0 - 1.0) & (edge_samples_px[:, 1] <= bbox.y1 + 1.0)))
                distance_penalty = 2.0 * abs(t - 0.5) + 0.8 * abs(offset)
                score = 100000.0 * outside + 10000.0 * overlap + 2000.0 * path_hits + distance_penalty
                if best is None or score < best[0]:
                    best = (score, (float(candidate[0]), float(candidate[1])), bbox)
        if best is None:
            raise RuntimeError('Could not place network label')
        label.set_position(best[1])
        occupied.append(best[2])
        edge_labels.append(label)
    if len(edge_labels) != len(subset):
        raise RuntimeError(f'Panel d {metric} lost an edge label')
    return edge_labels

def draw_panel_3d(fig: mpl.figure.Figure, spec: mpl.gridspec.SubplotSpec, contribution: pd.DataFrame, directional: pd.DataFrame, *, panel_label: str='d') -> list[mpl.axes.Axes]:
    inner = spec.subgridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.12)
    axes = [fig.add_subplot(inner[0, i]) for i in range(4)]
    contribution_labels = draw_contribution_summary(axes[0], contribution)
    edge_labels: list[mpl.text.Text] = []
    for ax, metric in zip(axes[1:], METRIC_ORDER):
        edge_labels.extend(draw_directional_network(ax, directional, metric))
    if len(contribution_labels) != 15 or len(edge_labels) != 38:
        raise RuntimeError(f'Panel d numeric-label contract failed: {len(contribution_labels)} + {len(edge_labels)}')
    add_panel_label(axes[0], panel_label, x_pt=-11, y_pt=3)
    return axes

def draw_panel_3e(fig: mpl.figure.Figure, spec: mpl.gridspec.SubplotSpec, lomo: pd.DataFrame, *, panel_label: str='e') -> list[mpl.axes.Axes]:
    inner = spec.subgridspec(1, 3, wspace=0.18)
    axes = [fig.add_subplot(inner[0, i]) for i in range(3)]
    strategy_styles = {'Block-PCA early fusion': {'marker': 'o', 'filled': True, 'offset': -0.18}, 'Equal-weight late fusion': {'marker': '^', 'filled': False, 'offset': 0.18}}
    global_min = float(lomo['Min_repeat_loss'].min())
    global_max = float(lomo['Max_repeat_loss'].max())
    lower = math.floor((global_min - 0.022) / 0.025) * 0.025
    upper = math.ceil((global_max + 0.022) / 0.025) * 0.025
    y_positions = np.arange(len(MODALITY_ORDER))
    for ax_index, (ax, metric) in enumerate(zip(axes, METRIC_ORDER)):
        endpoint = METRIC_TO_ENDPOINT[metric]
        color = ENDPOINT_COLORS[endpoint]
        label_records: list[dict[str, object]] = []
        for strategy, style in strategy_styles.items():
            subset = lomo.loc[lomo['Strategy'].eq(strategy) & lomo['Metric'].eq(metric)].set_index('Removed_modality')
            for base_y, modality in zip(y_positions, MODALITY_ORDER):
                row = subset.loc[modality]
                mean = float(row['Mean_repeat_loss'])
                lo = float(row['Min_repeat_loss'])
                hi = float(row['Max_repeat_loss'])
                y = base_y + float(style['offset'])
                ax.plot([lo, hi], [y, y], color=color, linewidth=0.9, alpha=0.78, zorder=2)
                ax.plot([lo, lo], [y - 0.045, y + 0.045], color=color, linewidth=0.7, alpha=0.78)
                ax.plot([hi, hi], [y - 0.045, y + 0.045], color=color, linewidth=0.7, alpha=0.78)
                ax.scatter(mean, y, s=17, marker=style['marker'], facecolor=color if style['filled'] else 'white', edgecolor=color, linewidth=0.75, zorder=3)
                label_records.append({'strategy': strategy, 'modality': modality, 'mean': mean, 'y': y})
        label_keys: list[tuple[str, ...]] = []
        for modality in MODALITY_ORDER:
            pair = [record for record in label_records if record['modality'] == modality]
            pair.sort(key=lambda record: (float(record['mean']), str(record['strategy'])))
            for pair_index, record in enumerate(pair):
                side = -1 if pair_index == 0 else 1
                artist = ax.annotate(_format_signed_3(float(record['mean'])), xy=(float(record['mean']), float(record['y'])), xytext=(3.0 * side, 0), textcoords='offset points', ha='right' if side < 0 else 'left', va='center', fontsize=FONT_SIZES['tiny'], color=MUTED, bbox={'boxstyle': 'round,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88}, zorder=4)
                artist.set_gid(f"panel-e-value-{metric}-{str(record['strategy']).replace(' ', '_')}-{modality}")
                label_keys.append((metric, str(record['strategy']), modality))
        _assert_unique_keys(label_keys, 10, f'Panel e {metric} labels')
        ax.axvline(0, color=REFERENCE, linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(len(MODALITY_ORDER) - 0.48, -1.28)
        ax.set_title(METRIC_TITLES[metric], pad=2.0, fontsize=FONT_SIZES['panel_title'], color=color, fontweight='semibold')
        if ax_index == 0:
            ax.set_yticks(y_positions, [MODALITY_DISPLAY[m] for m in MODALITY_ORDER])
        else:
            ax.set_yticks(y_positions, [])
        clean_style_axis(ax)
    axes[1].set_xlabel('Performance loss (full − leave-one-out)', labelpad=1.5)
    axes[1].legend(handles=[Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=4, label='Block-PCA early fusion'), Line2D([0], [0], marker='^', linestyle='none', markerfacecolor='white', markeredgecolor=TEXT, markersize=4, label='Equal-weight late fusion')], loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, handletextpad=0.35, columnspacing=0.9, frameon=False)
    add_panel_label(axes[0], panel_label, x_pt=-11, y_pt=3)
    return axes

def build_full_figure(bundle: Mapping[str, object]) -> tuple[mpl.figure.Figure, dict[str, list[mpl.axes.Axes]]]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    outer = fig.add_gridspec(4, 1, height_ratios=[0.2, 0.3, 0.26, 0.24], left=0.098, right=0.96, bottom=0.035, top=0.982, hspace=0.23)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    axes: dict[str, list[mpl.axes.Axes]] = {}
    axes['a'] = draw_panel_3a(fig, top[0, 0], bundle['coordinates'], bundle['permanova'], bundle['coordinate_metadata'])
    axes['b'] = draw_panel_3b(fig, top[0, 1], bundle['permanova'])
    axes['c'] = draw_panel_3c(fig, outer[1], bundle['clinical_r'], bundle['clinical_q'], bundle['outcome_effect'], bundle['outcome_q'])
    axes['d'] = draw_panel_3d(fig, outer[2], bundle['contribution'], bundle['directional'])
    axes['e'] = draw_panel_3e(fig, outer[3], bundle['lomo'])
    for panel, panel_axes in axes.items():
        for axis in panel_axes:
            axis._qa_meta = {'panel': panel}
    return (fig, axes)
PANEL_SIZE_MM = {'a': (118.0, 58.0), 'b': (92.0, 58.0), 'c': (180.0, 72.0), 'd': (180.0, 70.0), 'e': (180.0, 62.0)}

def build_panel_figure(panel: str, bundle: Mapping[str, object]) -> mpl.figure.Figure:
    panel = panel.lower()
    if panel not in PANEL_SIZE_MM:
        raise ValueError(f'Unknown panel {panel!r}')
    configure_style()
    width_mm, height_mm = PANEL_SIZE_MM[panel]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    left_margin = {'a': 0.12, 'b': 0.17, 'c': 0.11, 'd': 0.1, 'e': 0.1}[panel]
    right_margin = {'a': 0.98, 'b': 0.92, 'c': 0.95, 'd': 0.98, 'e': 0.98}[panel]
    bottom_margin = {'a': 0.17, 'b': 0.2, 'c': 0.19, 'd': 0.12, 'e': 0.15}[panel]
    top_margin = {'a': 0.88, 'b': 0.94, 'c': 0.96, 'd': 0.9, 'e': 0.94}[panel]
    spec = fig.add_gridspec(1, 1, left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)[0]
    if panel == 'a':
        draw_panel_3a(fig, spec, bundle['coordinates'], bundle['permanova'], bundle['coordinate_metadata'])
    elif panel == 'b':
        draw_panel_3b(fig, spec, bundle['permanova'])
    elif panel == 'c':
        draw_panel_3c(fig, spec, bundle['clinical_r'], bundle['clinical_q'], bundle['outcome_effect'], bundle['outcome_q'])
    elif panel == 'd':
        draw_panel_3d(fig, spec, bundle['contribution'], bundle['directional'])
    elif panel == 'e':
        draw_panel_3e(fig, spec, bundle['lomo'])
    return fig

def _save_working_composite(fig: mpl.figure.Figure, output_stem: Path, *, dpi: int=600, metadata: Mapping[str, str] | None=None) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    export_metadata = dict(metadata or {})
    saved: list[Path] = []
    for suffix in ('pdf', 'svg', 'png'):
        path = output_stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, object] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if suffix == 'png':
            kwargs['dpi'] = int(dpi)
        elif suffix == 'pdf':
            kwargs['metadata'] = {key: value for key, value in export_metadata.items() if key in {'Title', 'Author', 'Subject', 'Keywords', 'Creator'}}
        else:
            kwargs['metadata'] = {key: value for key, value in export_metadata.items() if key in {'Title', 'Description', 'Creator', 'Keywords'}}
        fig.savefig(path, **kwargs)
        saved.append(path)
    return saved

def export_all(bundle: Mapping[str, object]) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = build_full_figure(bundle)
    full_paths = _save_working_composite(fig, FIGURE_STEM, dpi=600, metadata={'Title': f'Avatar-HCC Figure 3 a-e - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'})
    plt.close(fig)
    panel_paths: list[Path] = []
    for panel in 'abcde':
        panel_fig = build_panel_figure(panel, bundle)
        stem = PANEL_DIR / f"{get_value('figures.config.item_270')}{panel}_Nature_{VERSION}"
        for suffix in ['svg', 'png', 'pdf']:
            path = stem.with_suffix(f'.{suffix}')
            kwargs = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
            if suffix == 'png':
                kwargs['dpi'] = 600
            panel_fig.savefig(path, **kwargs)
            panel_paths.append(path)
        plt.close(panel_fig)
    manifest = {'version': VERSION, 'palette': PALETTE_ID, 'figure_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only frozen CSVs; no model fitting', 'scientific_decisions': {'outcome_table': get_value('figures.annotation.figure_03_2'), 'response_column': get_column('figures.column.response'), 'figure3c_spearman_fdr': 'separate FDR control within four prespecified clinical domains, matching executed code', 'figure3c_numeric_annotations': 'shown only when the two-decimal absolute effect is >0.10; all cells remain color encoded', 'terminology': 'five data blocks across four assay modalities', 'figure3e_interval': 'min-max across five repeat means, not confidence intervals'}, 'outputs': [{'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size} for path in full_paths + panel_paths]}
    manifest_path = OUTPUT_DIR / f"{get_value('figures.config.item_272')}{VERSION}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {'full_paths': full_paths, 'panel_paths': panel_paths, 'manifest': manifest_path}

def summarize_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    coords = bundle['coordinates']
    return {'patients': int(len(coords)), 'response_counts': coords['response_label'].value_counts().to_dict(), 'pfs_landmark_counts': coords['pfs_landmark_group'].value_counts().to_dict(), 'os_landmark_counts': coords['os_landmark_group'].value_counts().to_dict(), 'figure3c_spearman_fdr': 'within four clinical domains', 'permutation_or_model_fitting_performed': False}
configure_style()
