from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
PILOT_ROOT = Path(__file__).resolve().parent
STYLE_ROOT = PILOT_ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, ENDPOINT_MARKERS, MODALITY_COLORS, MODALITY_MARKERS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = PILOT_ROOT / 'inputs'
VERSION = 'v004'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_353')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_273')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 210.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT_SIZES: Mapping[str, float] = {'base': 6.4, 'panel_label': 8.5, 'header': 7.0, 'axis_label': 6.8, 'tick': 5.8, 'legend': 5.6, 'annotation': 5.6, 'small': 5.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
LIGHT = NEUTRAL_COLORS['light_background']
GRID = '#D9E0E6'
FULL_BAND = '#E9ECEF'
MRU_BAND = '#E1F2EE'
MRU_EDGE = '#14665B'
MODALITY_ORDER = ['Metagenomics', 'Radiomics', 'Blood', 'Urine']
MODALITY_SHORT = {'Metagenomics': 'M', 'Radiomics': 'R', 'Blood': 'B', 'Urine': 'U'}
SHORT_TO_MODALITY = {value: key for key, value in MODALITY_SHORT.items()}
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
METRICS = {'Response': {'mean': 'Response_AUC_mean', 'sd': 'Response_AUC_sd', 'external': 'Response_AUROC', 'low': 'Response_AUROC_CI_lower', 'high': 'Response_AUROC_CI_upper', 'xlabel': 'AUROC'}, 'PFS': {'mean': 'PFS_Cindex_mean', 'sd': 'PFS_Cindex_sd', 'external': 'PFS_Cindex', 'low': 'PFS_Cindex_CI_lower', 'high': 'PFS_Cindex_CI_upper', 'xlabel': 'C-index'}, 'OS': {'mean': 'OS_Cindex_mean', 'sd': 'OS_Cindex_sd', 'external': 'OS_Cindex', 'low': 'OS_Cindex_CI_lower', 'high': 'OS_Cindex_CI_upper', 'xlabel': 'C-index'}}
INPUT_FILES = {'compression': get_value('figures.config.item_096'), 'overlap': get_value('figures.config.item_083'), 'bootstrap': get_value('figures.config.item_064'), 'landscape': get_value('figures.config.item_177'), 'ablation': get_value('figures.config.item_043'), 'paired_delta': get_value('figures.config.item_165'), 'depth_points': get_value('figures.config.item_050'), 'depth_summary': get_value('figures.config.item_110'), 'external': get_value('figures.config.item_095')}

def file_sha256(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {key: pd.read_csv(INPUT_DIR / filename) for key, filename in INPUT_FILES.items()}
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    bundle['contract'] = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    validate_inputs(bundle)
    return bundle

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')

def _require_unique(frame: pd.DataFrame, keys: Sequence[str], expected: int, label: str) -> None:
    if len(frame) != expected:
        raise ValueError(f'{label} expected {expected} rows; found {len(frame)}')
    if frame.duplicated(list(keys)).any():
        duplicated = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
        raise ValueError(f'{label} contains duplicate keys:\n{duplicated}')

def validate_inputs(bundle: Mapping[str, Any]) -> None:
    manifest = bundle['manifest']
    manifest_files = {entry['snapshot']: entry for entry in manifest['files']}
    for filename in INPUT_FILES.values():
        if filename not in manifest_files:
            raise ValueError(f'Input manifest does not register {filename}')
        entry = manifest_files[filename]
        path = INPUT_DIR / filename
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch for frozen input {filename}')
        frame = pd.read_csv(path)
        if frame.shape != (int(entry['rows']), int(entry['columns'])):
            raise ValueError(f"Unexpected shape for {filename}: {frame.shape}; expected {(entry['rows'], entry['columns'])}")
    compression = bundle['compression']
    overlap = bundle['overlap']
    bootstrap = bundle['bootstrap']
    _require_columns(compression, ['modal_clean', 'input_features', 'retained_features'], '4A compression')
    _require_columns(overlap, ['modal_clean', 'pattern', 'n_features'], '4A overlap')
    _require_columns(bootstrap, ['modal_clean', 'feature_clean', 'mean_topk_frequency', 'min_topk_frequency', 'median_bootstrap_rank', 'n_selected_endpoints', 'pattern'], '4A bootstrap')
    _require_unique(compression, ['modal_clean'], 4, '4A compression')
    _require_unique(overlap, ['modal_clean', 'pattern'], 20, '4A overlap')
    _require_unique(bootstrap, ['modal_clean', 'feature_clean'], 54, '4A bootstrap')
    if set(compression['modal_clean']) != set(MODALITY_ORDER):
        raise ValueError('4A modalities do not match the locked four-modality order')
    retained = compression.set_index('modal_clean')['retained_features'].astype(int)
    overlap_totals = overlap.groupby('modal_clean')['n_features'].sum().astype(int)
    bootstrap_totals = bootstrap.groupby('modal_clean').size().astype(int)
    if not retained.sort_index().equals(overlap_totals.sort_index()):
        raise ValueError('4A endpoint-overlap counts do not sum to retained feature counts')
    if not retained.sort_index().equals(bootstrap_totals.sort_index()):
        raise ValueError('4A bootstrap feature counts do not match retained feature counts')
    if not ((bootstrap['mean_topk_frequency'] >= bootstrap['min_topk_frequency']) & bootstrap['mean_topk_frequency'].between(0, 1) & bootstrap['min_topk_frequency'].between(0, 1)).all():
        raise ValueError('4A bootstrap frequency bounds are invalid')
    if not bootstrap['n_selected_endpoints'].isin([1, 2, 3]).all():
        raise ValueError('4A selected-endpoint counts must be 1, 2, or 3')
    landscape = bundle['landscape']
    _require_columns(landscape, ['short', 'modalities', 'n_modalities', 'n_samples', 'n_features', 'Response_AUC_mean', 'Response_AUC_sd', 'PFS_Cindex_mean', 'PFS_Cindex_sd', 'OS_Cindex_mean', 'OS_Cindex_sd'], '4B landscape')
    _require_unique(landscape, ['short'], 15, '4B landscape')
    if landscape['n_modalities'].value_counts().sort_index().to_dict() != {1: 4, 2: 6, 3: 4, 4: 1}:
        raise ValueError('4B does not contain the expected 4/6/4/1 modality combinations')
    if not (landscape['n_samples'] == 109).all():
        raise ValueError('4B expected n=109 for every internal model')
    ablation = bundle['ablation']
    paired = bundle['paired_delta']
    _require_columns(ablation, ['removed_modality', 'leave_one_out_model', 'endpoint', 'mean_delta', 'sd_delta', 'median_delta', 'min_delta', 'max_delta', 'n_paired_folds'], '4C ablation summary')
    _require_columns(paired, ['removed_modality', 'leave_one_out_model', 'fold_id', 'endpoint', 'LOO_performance', 'Full_performance', 'delta'], '4C paired deltas')
    _require_unique(ablation, ['removed_modality', 'endpoint'], 12, '4C ablation summary')
    _require_unique(paired, ['removed_modality', 'fold_id', 'endpoint'], 300, '4C paired deltas')
    if not np.allclose(paired['delta'].to_numpy(float), paired['LOO_performance'].to_numpy(float) - paired['Full_performance'].to_numpy(float), atol=1e-12):
        raise ValueError('4C paired delta is not LOO minus Full')
    reconstructed = paired.groupby(['removed_modality', 'leave_one_out_model', 'endpoint'], as_index=False).agg(mean_check=('delta', 'mean'), sd_check=('delta', 'std'), median_check=('delta', 'median'), min_check=('delta', 'min'), max_check=('delta', 'max'), n_check=('delta', 'size'))
    merged = ablation.merge(reconstructed, on=['removed_modality', 'leave_one_out_model', 'endpoint'], validate='one_to_one')
    for observed, expected in [('mean_delta', 'mean_check'), ('sd_delta', 'sd_check'), ('median_delta', 'median_check'), ('min_delta', 'min_check'), ('max_delta', 'max_check')]:
        if not np.allclose(merged[observed], merged[expected], atol=1e-12):
            raise ValueError(f'4C summary mismatch for {observed}')
    if not (merged['n_paired_folds'] == merged['n_check']).all() or not (merged['n_check'] == 25).all():
        raise ValueError('4C expected 25 paired folds in every cell')
    depth_points = bundle['depth_points']
    depth_summary = bundle['depth_summary']
    _require_unique(depth_points, ['short'], 15, '4D model points')
    _require_unique(depth_summary, ['endpoint', 'n_modalities'], 12, '4D depth summary')
    linked = landscape.merge(depth_points, on=['short', 'modalities', 'n_modalities'], validate='one_to_one')
    for endpoint in ENDPOINT_ORDER:
        metric = METRICS[endpoint]['mean']
        if not np.allclose(linked[f'{metric}_x'], linked[f'{metric}_y'], atol=1e-12):
            raise ValueError(f'4D points differ from 4B for {endpoint}')
        endpoint_points = depth_points[['n_modalities', metric]].rename(columns={metric: 'value'})
        calculated = endpoint_points.groupby('n_modalities')['value'].agg(n_models='size', mean_performance='mean', median_performance='median', min_performance='min', max_performance='max', sd_across_models='std').reset_index()
        recorded = depth_summary.loc[depth_summary['endpoint'].eq(endpoint)].merge(calculated, on='n_modalities', suffixes=('_recorded', '_check'), validate='one_to_one')
        for column in ['mean_performance', 'median_performance', 'min_performance', 'max_performance']:
            if not np.allclose(recorded[f'{column}_recorded'], recorded[f'{column}_check'], atol=1e-12):
                raise ValueError(f'4D summary mismatch for {endpoint} {column}')
        if not (recorded['n_models_recorded'] == recorded['n_models_check']).all():
            raise ValueError(f'4D n_models mismatch for {endpoint}')
    external = bundle['external']
    _require_unique(external, ['short'], 4, '4E external transferability')
    expected_n = {'M': 133, 'R': 200, 'B': 72, 'U': 33}
    if external.set_index('short')['n_external'].astype(int).to_dict() != expected_n:
        raise ValueError('4E external cohort sizes differ from the frozen contract')
    for endpoint in ENDPOINT_ORDER:
        spec = METRICS[endpoint]
        _require_columns(external, [spec['mean'], spec['sd'], spec['external'], spec['low'], spec['high']], f'4E {endpoint}')
        if not ((external[spec['low']] <= external[spec['external']]) & (external[spec['external']] <= external[spec['high']])).all():
            raise ValueError(f'4E invalid external interval for {endpoint}')

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT_SIZES['base'], 'axes.labelsize': FONT_SIZES['axis_label'], 'axes.titlesize': FONT_SIZES['header'], 'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'], 'legend.fontsize': FONT_SIZES['legend'], 'legend.title_fontsize': FONT_SIZES['legend'], 'axes.linewidth': 0.55, 'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _lighten(color: str, amount: float=0.82) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * float(amount))

def _clean_axis(ax: mpl.axes.Axes, *, grid: bool=False) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(direction='out', width=0.55, length=2.3, pad=2)
    ax.grid(False)
    if grid:
        ax.grid(True, axis='x', color=GRID, linewidth=0.35, alpha=0.7, zorder=0)

def _format_signed(value: float) -> str:
    if abs(float(value)) < 0.0005:
        return '0.000'
    return f'{float(value):+.3f}'.replace('-', '−')

def _text_color_for_fill(color: Sequence[float]) -> str:
    red, green, blue = color[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.52 else TEXT

def _axis_header(ax: mpl.axes.Axes, title: str, *, accent: str | None=None, y: float=1.08, fontsize: float | None=None, fontweight: str='bold') -> mpl.text.Text:
    artist = ax.text(0.5, y, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=FONT_SIZES['header'] if fontsize is None else float(fontsize), fontweight=fontweight, color=TEXT, clip_on=False)
    if accent is not None:
        ax.plot([0.43, 0.57], [y - 0.035, y - 0.035], transform=ax.transAxes, color=accent, linewidth=1.4, solid_capstyle='round', clip_on=False)
    return artist

def _rect_fraction(width_mm: float, height_mm: float, left_mm: float, top_mm: float, rect_width_mm: float, rect_height_mm: float) -> list[float]:
    return [left_mm / width_mm, 1.0 - (top_mm + rect_height_mm) / height_mm, rect_width_mm / width_mm, rect_height_mm / height_mm]

def _new_axis(fig: mpl.figure.Figure, rect: Sequence[float], panel: str) -> mpl.axes.Axes:
    ax = fig.add_axes(rect)
    ax._qa_meta = {'panel': panel}
    return ax

def _add_panel_letter(fig: mpl.figure.Figure, label: str, *, left_mm: float, top_mm: float, width_mm: float, height_mm: float, registry: ArtistRegistry) -> mpl.text.Text:
    artist = fig.text(left_mm / width_mm, 1.0 - top_mm / height_mm, label.lower(), ha='left', va='top', fontsize=FONT_SIZES['panel_label'], fontweight='bold', color=TEXT)
    return registry.tag(artist, name=f'panel-{label.lower()}-label', role='panel_label', panel=label)

def _add_group_bands_and_separators(ax: mpl.axes.Axes, y_positions: Mapping[str, float]) -> None:
    ax.axhspan(y_positions['M+R+U'] - 0.42, y_positions['M+R+U'] + 0.42, color=MRU_BAND, zorder=-5)
    ax.axhspan(y_positions['M+R+B+U'] - 0.42, y_positions['M+R+B+U'] + 0.42, color=FULL_BAND, zorder=-5)
    for y in (10.5, 4.5, 0.5):
        ax.axhline(y, color='#AEB8C1', linewidth=0.45, zorder=-2)

def draw_panel_4a(fig: mpl.figure.Figure, axes: Sequence[mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    ax_compression, ax_overlap, ax_bootstrap = axes
    compression = bundle['compression'].set_index('modal_clean').loc[MODALITY_ORDER].reset_index()
    overlap = bundle['overlap']
    bootstrap = bundle['bootstrap']
    y_map = {modality: 3 - index for index, modality in enumerate(MODALITY_ORDER)}
    for _, row in compression.iterrows():
        modality = row['modal_clean']
        y = y_map[modality]
        color = MODALITY_COLORS[modality]
        marker = MODALITY_MARKERS[modality]
        retained = float(row['retained_features'])
        raw = float(row['input_features'])
        ax_compression.plot([retained, raw], [y, y], color='#B8C1C9', linewidth=0.9, zorder=1)
        ax_compression.scatter(raw, y, s=22, marker=marker, facecolor='white', edgecolor=color, linewidth=0.8, zorder=3)
        ax_compression.scatter(retained, y, s=24, marker=marker, facecolor=color, edgecolor=TEXT, linewidth=0.35, zorder=4)
        retained_text = ax_compression.annotate(f'{int(retained):,}', (retained, y), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=FONT_SIZES['annotation'], fontweight='bold', color=color)
        raw_text = ax_compression.annotate(f'{int(raw):,}', (raw, y), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=FONT_SIZES['annotation'], color=MUTED)
        registry.tag(retained_text, name=f'compression-retained-{modality}', role='value', panel='a')
        registry.tag(raw_text, name=f'compression-input-{modality}', role='value', panel='a')
    ax_compression.set_xscale('log')
    ax_compression.set_xlim(7, 9000)
    ax_compression.set_xticks([10, 100, 1000])
    ax_compression.set_ylim(-0.55, 3.55)
    ax_compression.set_yticks([y_map[value] for value in MODALITY_ORDER], labels=MODALITY_ORDER)
    ax_compression.set_xlabel('Number of features (log scale)')
    _clean_axis(ax_compression, grid=True)
    legend = ax_compression.legend(handles=[Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor=REFERENCE, markersize=4, label='Input'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=4, label='SHAP-retained')], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    registry.tag(legend, name='compression-legend', role='legend', panel='a')
    pattern_order = ['Response-specific', 'PFS-specific', 'OS-specific', 'Shared by 2', 'Shared by 3']
    pattern_labels = ['Response\nonly', 'PFS\nonly', 'OS\nonly', '2\nendpoints', '3\nendpoints']
    for modality in MODALITY_ORDER:
        frame = overlap.loc[overlap['modal_clean'].eq(modality)].set_index('pattern').loc[pattern_order]
        y = y_map[modality]
        for x, (_, row) in enumerate(frame.iterrows()):
            count = int(row['n_features'])
            ax_overlap.scatter(x, y, s=18 + 10 * count, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.55, zorder=3)
            text = ax_overlap.text(x, y, str(count), ha='center', va='center', fontsize=FONT_SIZES['small'], fontweight='bold', color='white', zorder=4)
            registry.tag(text, name=f'overlap-{modality}-{x}', role='value', panel='a')
    ax_overlap.set_xlim(-0.55, 4.55)
    ax_overlap.set_ylim(-0.55, 3.55)
    ax_overlap.set_xticks(range(5), labels=pattern_labels)
    ax_overlap.set_yticks([])
    ax_overlap.tick_params(axis='x', length=0, pad=2)
    for y in y_map.values():
        ax_overlap.axhline(y, color=GRID, linewidth=0.35, zorder=0)
    for x in range(5):
        ax_overlap.axvline(x, color=GRID, linewidth=0.35, zorder=0)
    for spine in ax_overlap.spines.values():
        spine.set_visible(False)
    for modality in MODALITY_ORDER:
        frame = bootstrap.loc[bootstrap['modal_clean'].eq(modality)].sort_values(['median_bootstrap_rank', 'feature_clean'], ascending=[False, True]).reset_index(drop=True)
        offsets = np.linspace(-0.28, 0.28, len(frame))
        for index, row in frame.iterrows():
            y = y_map[modality] + offsets[index]
            minimum = float(row['min_topk_frequency'])
            mean = float(row['mean_topk_frequency'])
            selected = int(row['n_selected_endpoints'])
            ax_bootstrap.plot([minimum, mean], [y, y], color='#8E99A3', linewidth=0.55, alpha=0.9, zorder=1)
            ax_bootstrap.scatter(minimum, y, s=7, marker='o', facecolor='white', edgecolor=REFERENCE, linewidth=0.5, zorder=2)
            ax_bootstrap.scatter(mean, y, s=12 + 9 * selected, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor='white', linewidth=0.45, alpha=0.95, zorder=3)
    ax_bootstrap.axvline(0.5, color=REFERENCE, linestyle=(0, (2.2, 2.2)), linewidth=0.65, zorder=0)
    ax_bootstrap.set_xlim(0, 1.02)
    ax_bootstrap.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bootstrap.set_ylim(-0.55, 3.55)
    ax_bootstrap.set_yticks([])
    ax_bootstrap.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{int(round(value * 100))}%'))
    ax_bootstrap.set_xlabel('Mean Top-k retention')
    _clean_axis(ax_bootstrap, grid=True)

def draw_panel_4b(fig: mpl.figure.Figure, axes: Sequence[mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    membership_ax, *performance_axes = axes
    frame = bundle['landscape'].sort_values('n_modalities', kind='stable').reset_index(drop=True)
    y_values = {row['short']: 14 - index for index, row in frame.iterrows()}
    modality_x = {short: index for index, short in enumerate(['M', 'R', 'B', 'U'])}
    for axis in axes:
        _add_group_bands_and_separators(axis, y_values)
        axis.set_ylim(-0.7, 14.7)
    for _, row in frame.iterrows():
        y = y_values[row['short']]
        selected = set(str(row['short']).split('+'))
        selected_x = [modality_x[short] for short in selected]
        if len(selected_x) > 1:
            membership_ax.plot([min(selected_x), max(selected_x)], [y, y], color=TEXT, linewidth=0.8, zorder=2)
        for short, x in modality_x.items():
            modality = SHORT_TO_MODALITY[short]
            if short in selected:
                membership_ax.scatter(x, y, s=17, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor=TEXT, linewidth=0.35, zorder=3)
            else:
                membership_ax.scatter(x, y, s=13, marker='o', facecolor='white', edgecolor='#9CA6AF', linewidth=0.55, zorder=2)
    labels = []
    for value in frame['short']:
        if value == 'M+R+U':
            labels.append('M+R+U  (MRU)')
        elif value == 'M+R+B+U':
            labels.append('M+R+B+U  (Full)')
        else:
            labels.append(value)
    membership_ax.set_xlim(-0.55, 3.55)
    membership_ax.set_xticks(range(4), labels=['M', 'R', 'B', 'U'])
    membership_ax.xaxis.tick_top()
    membership_ax.tick_params(axis='x', length=0, pad=3)
    membership_ax.set_yticks([y_values[value] for value in frame['short']], labels=labels)
    membership_ax.tick_params(axis='y', length=0, pad=4)
    for tick, short in zip(membership_ax.get_yticklabels(), frame['short']):
        if short in {'M+R+U', 'M+R+B+U'}:
            tick.set_fontweight('bold')
    for spine in membership_ax.spines.values():
        spine.set_visible(False)
    for ax, endpoint in zip(performance_axes, ENDPOINT_ORDER):
        spec = METRICS[endpoint]
        color = ENDPOINT_COLORS[endpoint]
        for _, row in frame.iterrows():
            short = row['short']
            y = y_values[short]
            marker = 'D' if short == 'M+R+U' else 's' if short == 'M+R+B+U' else 'o'
            size = 4.4 if short in {'M+R+U', 'M+R+B+U'} else 3.0
            edge = MRU_EDGE if short == 'M+R+U' else TEXT if short == 'M+R+B+U' else 'white'
            ax.errorbar(float(row[spec['mean']]), y, xerr=float(row[spec['sd']]), fmt=marker, color=color, ecolor=_lighten(color, 0.3), elinewidth=0.75, capsize=1.3, capthick=0.65, markersize=size, markerfacecolor=color, markeredgecolor=edge, markeredgewidth=0.55, alpha=1.0 if short in {'M+R+U', 'M+R+B+U'} else 0.82, zorder=3)
            if short in {'M+R+U', 'M+R+B+U'}:
                text = ax.text(0.465, y, f"{float(row[spec['mean']]):.3f}", ha='left', va='center', fontsize=FONT_SIZES['annotation'], fontweight='bold' if short == 'M+R+U' else 'normal', color=MRU_EDGE if short == 'M+R+U' else TEXT, zorder=4)
                registry.tag(text, name=f'landscape-{endpoint}-{short}', role='value', panel='b')
        ax.set_xlim(0.45, 0.98)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_yticks([])
        ax.set_xlabel(spec['xlabel'])
        _clean_axis(ax, grid=True)
        _axis_header(ax, endpoint, accent=color, y=1.06)

def draw_panel_4c(fig: mpl.figure.Figure, heat_ax: mpl.axes.Axes, colorbar_ax: mpl.axes.Axes, bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    frame = bundle['ablation']
    removed_order = ['M', 'R', 'B', 'U']
    mean_matrix = frame.pivot(index='removed_modality', columns='endpoint', values='mean_delta').loc[removed_order, ENDPOINT_ORDER]
    sd_matrix = frame.pivot(index='removed_modality', columns='endpoint', values='sd_delta').loc[removed_order, ENDPOINT_ORDER]
    loo_map = frame.drop_duplicates('removed_modality').set_index('removed_modality')['leave_one_out_model'].to_dict()
    cmap = LinearSegmentedColormap.from_list(get_value('figures.config.item_344'), DIVERGING_COLORS)
    norm = TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.08)
    image = heat_ax.imshow(mean_matrix.to_numpy(float), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    heat_ax.set_xticks(range(3), labels=ENDPOINT_ORDER)
    heat_ax.xaxis.tick_top()
    heat_ax.tick_params(axis='x', length=0, pad=3)
    row_labels = []
    for removed in removed_order:
        suffix = '  (MRU)' if removed == 'B' else ''
        row_labels.append(f'−{removed} → {loo_map[removed]}{suffix}')
    heat_ax.set_yticks(range(4), labels=row_labels)
    heat_ax.tick_params(axis='y', length=0, pad=5)
    for row_index, removed in enumerate(removed_order):
        modality = SHORT_TO_MODALITY[removed]
        heat_ax.plot([-0.54, -0.54], [row_index - 0.34, row_index + 0.34], color=MODALITY_COLORS[modality], linewidth=2.1, clip_on=False)
        for column_index, endpoint in enumerate(ENDPOINT_ORDER):
            mean = float(mean_matrix.loc[removed, endpoint])
            sd = float(sd_matrix.loc[removed, endpoint])
            fill = cmap(norm(mean))
            color = _text_color_for_fill(fill)
            mean_text = heat_ax.text(column_index, row_index - 0.09, _format_signed(mean), ha='center', va='center', fontsize=FONT_SIZES['annotation'], fontweight='bold' if removed == 'B' else 'normal', color=color)
            sd_color = '#EEF3F6' if color == 'white' else MUTED
            sd_text = heat_ax.text(column_index, row_index + 0.22, f'SD {sd:.3f}', ha='center', va='center', fontsize=FONT_SIZES['small'], color=sd_color)
            registry.tag(mean_text, name=f'ablation-mean-{removed}-{endpoint}', role='value', panel='c')
            registry.tag(sd_text, name=f'ablation-sd-{removed}-{endpoint}', role='small_value', panel='c')
    heat_ax.set_xlim(-0.5, 2.5)
    heat_ax.set_ylim(3.5, -0.5)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    heat_ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    heat_ax.grid(which='minor', color='white', linewidth=0.9)
    heat_ax.tick_params(which='minor', bottom=False, left=False)
    heat_ax.add_patch(Rectangle((-0.5, 1.5), 3.0, 1.0, fill=False, edgecolor=MRU_EDGE, linewidth=0.8, clip_on=False, zorder=6))
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation='horizontal')
    colorbar.set_ticks([-0.08, -0.04, 0.0, 0.04, 0.08])
    colorbar.ax.tick_params(labelsize=FONT_SIZES['small'], width=0.45, length=2, pad=1)
    colorbar.outline.set_visible(False)
    colorbar.set_label('Δ performance (LOO − Full)', fontsize=FONT_SIZES['legend'], labelpad=1.5)
    colorbar_ax._qa_meta = {'panel': 'c'}

def draw_panel_4d(fig: mpl.figure.Figure, axes: Sequence[mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    points = bundle['depth_points']
    summary = bundle['depth_summary']
    for axis_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINT_ORDER)):
        spec = METRICS[endpoint]
        color = ENDPOINT_COLORS[endpoint]
        metric = spec['mean']
        endpoint_summary = summary.loc[summary['endpoint'].eq(endpoint)].sort_values('n_modalities')
        for depth in [1, 2, 3, 4]:
            subset = points.loc[points['n_modalities'].eq(depth)].sort_values('short')
            offsets = np.linspace(-0.11, 0.11, len(subset)) if len(subset) > 1 else np.array([0.0])
            ax.scatter(depth + offsets, subset[metric].to_numpy(float), s=12, marker='o', facecolor=_lighten(color, 0.3), edgecolor='white', linewidth=0.35, alpha=0.9, zorder=2)
        ax.vlines(endpoint_summary['n_modalities'], endpoint_summary['min_performance'], endpoint_summary['max_performance'], color=_lighten(color, 0.64), linewidth=4.5, zorder=1)
        ax.plot(endpoint_summary['n_modalities'], endpoint_summary['mean_performance'], color=color, linewidth=1.0, marker='o', markersize=3.8, markerfacecolor='white', markeredgecolor=color, markeredgewidth=0.65, zorder=3)
        mru = points.loc[points['short'].eq('M+R+U')].iloc[0]
        full = points.loc[points['short'].eq('M+R+B+U')].iloc[0]
        mru_value = float(mru[metric])
        full_value = float(full[metric])
        ax.scatter(3, mru_value, s=33, marker='D', facecolor='#2A8F78', edgecolor=MRU_EDGE, linewidth=0.65, zorder=5)
        ax.scatter(4, full_value, s=31, marker='s', facecolor=REFERENCE, edgecolor=TEXT, linewidth=0.6, zorder=5)
        mru_text = ax.annotate(f'{mru_value:.3f}', (3, mru_value), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=FONT_SIZES['small'], fontweight='bold', color=MRU_EDGE)
        full_text = ax.annotate(f'{full_value:.3f}', (4, full_value), xytext=(0, -7), textcoords='offset points', ha='center', va='top', fontsize=FONT_SIZES['small'], color=TEXT)
        registry.tag(mru_text, name=f'depth-{endpoint}-MRU', role='value', panel='d')
        registry.tag(full_text, name=f'depth-{endpoint}-Full', role='value', panel='d')
        ax.set_xlim(0.75, 4.25)
        ax.set_ylim(0.54, 0.96)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_yticks([0.6, 0.7, 0.8, 0.9] if axis_index == 0 else [])
        _clean_axis(ax, grid=True)
        _axis_header(ax, endpoint, accent=color, y=1.06)
        if axis_index == 0:
            ax.set_ylabel('Performance')
        if axis_index == 1:
            ax.set_xlabel('No. of integrated modalities')
    legend = axes[1].legend(handles=[Line2D([0], [0], marker='D', linestyle='none', markerfacecolor='#2A8F78', markeredgecolor=MRU_EDGE, markersize=4.2, label='MRU'), Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=REFERENCE, markeredgecolor=TEXT, markersize=4.2, label='Full')], loc='lower center', bbox_to_anchor=(0.5, 1.18), ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    registry.tag(legend, name='depth-model-legend', role='legend', panel='d')

def draw_panel_4e(fig: mpl.figure.Figure, axes: Sequence[mpl.axes.Axes], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    frame = bundle['external'].set_index('short').loc[['M', 'R', 'B', 'U']].reset_index()
    y_map = {short: 3 - index for index, short in enumerate(['M', 'R', 'B', 'U'])}
    for axis_index, (ax, endpoint) in enumerate(zip(axes, ENDPOINT_ORDER)):
        spec = METRICS[endpoint]
        color = ENDPOINT_COLORS[endpoint]
        for _, row in frame.iterrows():
            short = row['short']
            y = y_map[short]
            internal_mean = float(row[spec['mean']])
            internal_sd = float(row[spec['sd']])
            external = float(row[spec['external']])
            low = float(row[spec['low']])
            high = float(row[spec['high']])
            left = max(0.45, internal_mean - internal_sd)
            right = min(0.98, internal_mean + internal_sd)
            ax.add_patch(Rectangle((left, y - 0.22), right - left, 0.44, facecolor='#E4E9ED', edgecolor='none', zorder=0))
            ax.vlines(internal_mean, y - 0.24, y + 0.24, color=REFERENCE, linewidth=1.05, zorder=1)
            ax.errorbar(external, y, xerr=np.array([[external - low], [high - external]]), fmt=ENDPOINT_MARKERS[endpoint], color=color, ecolor=color, elinewidth=0.8, capsize=1.6, capthick=0.7, markersize=4.1, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.55, zorder=3)
            x_offset = -4 if external > 0.89 else 4
            ha = 'right' if external > 0.89 else 'left'
            value_text = ax.annotate(f'{external:.3f}', (external, y), xytext=(x_offset, 5), textcoords='offset points', ha=ha, va='bottom', fontsize=FONT_SIZES['annotation'], fontweight='bold', color=TEXT)
            registry.tag(value_text, name=f'external-{endpoint}-{short}', role='value', panel='e')
            if axis_index == 0:
                modality = SHORT_TO_MODALITY[short]
                ax.scatter([-0.018], [y], transform=ax.get_yaxis_transform(), clip_on=False, s=18, marker=MODALITY_MARKERS[modality], facecolor=MODALITY_COLORS[modality], edgecolor=TEXT, linewidth=0.35, zorder=4)
        ax.axvline(0.5, color='#AAB3BB', linestyle=(0, (2, 2)), linewidth=0.55, zorder=-1)
        ax.set_xlim(0.45, 0.98)
        ax.set_ylim(-0.55, 3.55)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        if axis_index == 0:
            labels = [f"{row['display_name_4d']}\nn = {int(row['n_external'])}" for _, row in frame.iterrows()]
            ax.set_yticks([3, 2, 1, 0], labels=labels)
            ax.tick_params(axis='y', length=0, pad=6)
        else:
            ax.set_yticks([])
        ax.set_xlabel(spec['xlabel'])
        _clean_axis(ax, grid=True)
        _axis_header(ax, endpoint, accent=color, y=1.06)
    legend = axes[1].legend(handles=[Line2D([0], [0], color=TEXT, marker='o', markerfacecolor=TEXT, markeredgecolor='white', linewidth=0.8, markersize=4, label='External estimate (95% CI)'), Patch(facecolor='#E4E9ED', edgecolor=REFERENCE, linewidth=0.7, label='Internal CV mean ± SD')], loc='lower center', bbox_to_anchor=(0.5, 1.22), ncol=2, handlelength=1.4, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    registry.tag(legend, name='external-legend', role='legend', panel='e')
FULL_AXES_MM: Mapping[str, tuple[float, float, float, float]] = {'a_compression': (20, 8, 38, 24), 'a_overlap': (65, 8, 48, 24), 'a_bootstrap': (119, 8, 57, 24), 'b_membership': (27, 48, 31, 43), 'b_response': (63, 48, 34, 43), 'b_pfs': (103, 48, 34, 43), 'b_os': (143, 48, 33, 43), 'c_heatmap': (28, 110, 44, 29), 'c_colorbar': (28, 145, 44, 2.5), 'd_response': (82, 111, 28.5, 31), 'd_pfs': (114.5, 111, 28.5, 31), 'd_os': (147, 111, 28.5, 31), 'e_response': (40, 166, 42, 29), 'e_pfs': (87, 166, 42, 29), 'e_os': (134, 166, 42, 29)}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry, dict[str, list[mpl.axes.Axes]]]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()

    def axes_for(names: Sequence[str], panel: str) -> list[mpl.axes.Axes]:
        return [_new_axis(fig, _rect_fraction(FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, *FULL_AXES_MM[name]), panel) for name in names]
    axes: dict[str, list[mpl.axes.Axes]] = {}
    axes['a'] = axes_for(['a_compression', 'a_overlap', 'a_bootstrap'], 'a')
    axes['b'] = axes_for(['b_membership', 'b_response', 'b_pfs', 'b_os'], 'b')
    axes['c'] = axes_for(['c_heatmap', 'c_colorbar'], 'c')
    axes['d'] = axes_for(['d_response', 'd_pfs', 'd_os'], 'd')
    axes['e'] = axes_for(['e_response', 'e_pfs', 'e_os'], 'e')
    draw_panel_4a(fig, axes['a'], bundle, registry)
    draw_panel_4b(fig, axes['b'], bundle, registry)
    draw_panel_4c(fig, axes['c'][0], axes['c'][1], bundle, registry)
    draw_panel_4d(fig, axes['d'], bundle, registry)
    draw_panel_4e(fig, axes['e'], bundle, registry)
    label_positions = {'a': (2, 4), 'b': (2, 42), 'c': (2, 106), 'd': (76, 106), 'e': (2, 157)}
    for label, (left, top) in label_positions.items():
        _add_panel_letter(fig, label, left_mm=left, top_mm=top, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM, registry=registry)
    return (fig, registry, axes)
PANEL_LAYOUTS: Mapping[str, Mapping[str, Any]] = {'a': {'size': (180.0, 42.0), 'rects': [(22, 7, 36, 24), (65, 7, 48, 24), (121, 7, 54, 24)], 'label': (2, 3)}, 'b': {'size': (180.0, 72.0), 'rects': [(28, 8, 31, 52), (64, 8, 33, 52), (104, 8, 33, 52), (144, 8, 32, 52)], 'label': (2, 3)}, 'c': {'size': (95.0, 68.0), 'rects': [(31, 8, 58, 38), (31, 54, 58, 3)], 'label': (2, 3)}, 'd': {'size': (125.0, 58.0), 'rects': [(15, 11, 31, 34), (52, 11, 31, 34), (89, 11, 31, 34)], 'label': (2, 3)}, 'e': {'size': (180.0, 62.0), 'rects': [(40, 16, 42, 34), (87, 16, 42, 34), (134, 16, 42, 34)], 'label': (2, 3)}}

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry, list[mpl.axes.Axes]]:
    panel = panel.lower()
    if panel not in PANEL_LAYOUTS:
        raise ValueError(f'Unknown panel {panel!r}')
    configure_style()
    layout = PANEL_LAYOUTS[panel]
    width_mm, height_mm = layout['size']
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = [_new_axis(fig, _rect_fraction(width_mm, height_mm, *rect), panel) for rect in layout['rects']]
    if panel == 'a':
        draw_panel_4a(fig, axes, bundle, registry)
    elif panel == 'b':
        draw_panel_4b(fig, axes, bundle, registry)
    elif panel == 'c':
        draw_panel_4c(fig, axes[0], axes[1], bundle, registry)
    elif panel == 'd':
        draw_panel_4d(fig, axes, bundle, registry)
    elif panel == 'e':
        draw_panel_4e(fig, axes, bundle, registry)
    left, top = layout['label']
    _add_panel_letter(fig, panel, left_mm=left, top_mm=top, width_mm=width_mm, height_mm=height_mm, registry=registry)
    return (fig, registry, axes)

def _save_exact_figure(fig: mpl.figure.Figure, output_stem: Path, *, width_mm: float, height_mm: float, formats: Sequence[str], dpi: int=EXPORT_DPI) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metadata = {'Title': f'Avatar-HCC Figure 4 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    for suffix in formats:
        suffix = suffix.lower().lstrip('.')
        path = output_stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, Any] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if suffix == 'png':
            kwargs['dpi'] = int(dpi)
        elif suffix == 'pdf':
            kwargs['metadata'] = metadata
        elif suffix == 'svg':
            kwargs['metadata'] = {'Title': metadata['Title'], 'Creator': metadata['Creator']}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, registry, _ = build_full_figure(bundle)
    layout_report = audit_figure_layout(fig, f"{get_value('figures.config.item_273')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout_report, OUTPUT_DIR / f"{get_value('figures.config.item_274')}{VERSION}.json")
    full_paths = _save_exact_figure(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM, formats=('pdf', 'svg', 'png'))
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in 'abcde':
        panel_fig, panel_registry, _ = build_panel_figure(panel, bundle)
        panel_width, panel_height = PANEL_LAYOUTS[panel]['size']
        panel_audits[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_275')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        stem = PANEL_DIR / f"{get_value('figures.config.item_275')}{panel}_Nature_{VERSION}"
        panel_paths.extend(_save_exact_figure(panel_fig, stem, width_mm=panel_width, height_mm=panel_height, formats=('pdf', 'svg', 'png')))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_276')}{VERSION}.json")
    outputs = full_paths + panel_paths + [layout_path, panel_audit_path]
    export_manifest = {'figure': 'Figure 4', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only frozen CSV snapshots; no model fitting, SHAP, bootstrap, or statistical recalculation', 'panel_mapping': bundle['contract']['panel_mapping'], 'layout_summary': layout_report['summary'], 'file_qa': 'Run verify_figure4_files.py with the bundled pypdf runtime.', 'outputs': [{'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size} for path in outputs]}
    manifest_path = write_json(export_manifest, OUTPUT_DIR / f"{get_value('figures.config.item_277')}{VERSION}.json")
    return {'full_paths': full_paths, 'panel_paths': panel_paths, 'layout_audit': layout_path, 'panel_audit': panel_audit_path, 'manifest': manifest_path, 'layout_summary': layout_report['summary']}

def summarize_bundle(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    return {'palette': PALETTE_ID, 'render_only': True, 'figure_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'panel_4a': {'modalities': len(bundle['compression']), 'retained_features': int(bundle['compression']['retained_features'].sum()), 'bootstrap_feature_rows': len(bundle['bootstrap'])}, 'panel_4b': {'models': len(bundle['landscape']), 'internal_n': int(bundle['landscape']['n_samples'].iloc[0]), 'cv_fold_estimates_per_model': 25}, 'panel_4c': {'cells': len(bundle['ablation']), 'paired_fold_rows': len(bundle['paired_delta'])}, 'panel_4d': {'models': len(bundle['depth_points']), 'depth_cells': len(bundle['depth_summary'])}, 'panel_4e': {'external_cohort_sizes': bundle['external'].set_index('short')['n_external'].astype(int).to_dict(), 'bootstrap_resamples': 1000}}
configure_style()
if __name__ == '__main__':
    result = export_all(load_inputs())
