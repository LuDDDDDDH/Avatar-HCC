from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
PILOT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PILOT_ROOT / 'inputs'
STYLE_ROOT = PILOT_ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_qa import audit_figure_layout, write_json
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MODALITY_COLORS, NEUTRAL_COLORS, PALETTE_ID, SEQUENTIAL_COLORS, apply_theme, save_figure, style_axis
VERSION = 'v001'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f'Supplementary_Figure_S1_full_a_to_e_Nature_{VERSION}'
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 240.0
MM_PER_INCH = 25.4
PANEL_SIZE_MM = {'a': (180.0, 82.0), 'b': (180.0, 64.0), 'c': (104.0, 58.0), 'd': (180.0, 72.0), 'e': (180.0, 62.0)}
FONT = {'base': 6.3, 'panel': 8.5, 'axis': 6.3, 'tick': 5.6, 'legend': 5.5, 'small': 5.4, 'value': 5.4, 'header': 6.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
GRID = '#D8DEE3'
LIGHT = NEUTRAL_COLORS['light_background']
MISSING = NEUTRAL_COLORS['missing']
POSITIVE = '#C44A58'
NEGATIVE = '#2F6FB0'
SPECIES_COLOR = '#62AA9C'
MODALITY_ORDER = ['Metagenomics', 'Species', 'Radiomics', 'Blood', 'Urine']
MODALITY_LABEL = {'Metagenomics': 'KO functions', 'Species': 'Species', 'Radiomics': 'Radiomics', 'Blood': 'Blood', 'Urine': 'Urine'}
MODALITY_COLOR = {'Metagenomics': MODALITY_COLORS['Metagenomics'], 'Species': SPECIES_COLOR, 'Radiomics': MODALITY_COLORS['Radiomics'], 'Blood': MODALITY_COLORS['Blood'], 'Urine': MODALITY_COLORS['Urine']}
METRIC_ORDER = ['Response_AUC', 'PFS_Cindex', 'OS_Cindex']
METRIC_LABEL = {'Response_AUC': 'Response AUC', 'PFS_Cindex': 'PFS C-index', 'OS_Cindex': 'OS C-index'}
ENDPOINT_FROM_METRIC = {'Response_AUC': 'Response', 'PFS_Cindex': 'PFS', 'OS_Cindex': 'OS'}
ENDPOINT_ORDER = ['Response', 'Early PFS', 'Early OS']
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(get_value('figures.config.item_370'), SEQUENTIAL_COLORS)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(get_value('figures.config.item_369'), DIVERGING_COLORS)

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT['base'], 'axes.labelsize': FONT['axis'], 'axes.titlesize': FONT['header'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'legend.fontsize': FONT['legend'], 'legend.title_fontsize': FONT['legend'], 'axes.linewidth': 0.55, 'lines.linewidth': 0.85, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / name, **kwargs)

def load_inputs() -> dict[str, object]:
    manifest = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION or manifest.get('file_count') != 31:
        raise RuntimeError('Unexpected Supplementary Figure S1 input manifest')
    for record in manifest['files']:
        path = INPUT_DIR / record['name']
        actual = sha256(path)
        if actual != record['sha256']:
            raise RuntimeError(f"Frozen input hash mismatch: {record['name']}")
    contract = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    if contract.get('version') != VERSION or contract.get('palette') != PALETTE_ID:
        raise RuntimeError('Unexpected Supplementary Figure S1 scientific contract')
    bundle: dict[str, object] = {'manifest': manifest, 'contract': contract, 'batch_scores': _read(get_value('figures.config.item_189')), 'batch_effect': _read(get_value('figures.config.item_188')), 'retention': _read(get_value('figures.config.item_190')), 'prevalence': _read(get_value('figures.config.item_193')), 'radiomics_manifest': _read(get_value('figures.config.item_191')), 'radiomics_family': _read(get_value('figures.config.item_192')), 'similarity_matrix': _read(get_value('figures.config.item_079'), index_col=0), 'similarity_fdr': _read(get_value('figures.config.item_078'), index_col=0), 'similarity_pairs': _read(get_value('figures.config.item_166')), 'distance_audit': _read(get_value('figures.config.item_076')), 'similarity_metadata': _read(get_value('figures.config.item_207')), 'benchmark': _read(get_value('figures.config.item_209')), 'benchmark_fold': _read(get_value('figures.config.item_208')), 'best_single': _read(get_value('figures.config.item_063')), 'permanova': _read(get_value('figures.config.item_178')), 'permanova_distance': _read(get_value('figures.config.item_077')), 'landmark_audit': _read(get_value('figures.config.item_117')), 'directional': _read(get_value('figures.config.item_049')), 'directional_repeat': _read(get_value('figures.config.item_048')), 'directional_fold': _read(get_value('figures.config.item_047')), 'asymmetry': _read(get_value('figures.config.item_075')), 'coalitions': _read(get_value('figures.config.item_217')), 'shapley_efficiency': _read(get_value('figures.config.item_218')), 'shapley_fold': _read(get_value('figures.config.item_219')), 'shapley_repeat': _read(get_value('figures.config.item_220')), 'shapley': _read(get_value('figures.config.item_221')), 'lomo': _read(get_value('figures.config.item_131')), 'lomo_repeat': _read(get_value('figures.config.item_129')), 'lomo_fold': _read(get_value('figures.config.item_128')), 'lomo_strategy': _read(get_value('figures.config.item_130')), 'lomo_range': _read(get_value('figures.config.item_216'))}
    validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, object]) -> None:
    checks = {'batch effect': (bundle['batch_effect'].shape[0], 8), 'batch scores': (bundle['batch_scores'].shape[0], 872), 'retention': (bundle['retention'].shape[0], 5), 'prevalence': (bundle['prevalence'].shape[0], 6438), 'similarity pairs': (bundle['similarity_pairs'].shape[0], 10), 'benchmark': (bundle['benchmark'].shape[0], 24), 'PERMANOVA/PERMDISP': (bundle['permanova'].shape[0], 18), 'directional': (bundle['directional'].shape[0], 60), 'Shapley': (bundle['shapley'].shape[0], 15), 'LOMO': (bundle['lomo_range'].shape[0], 30)}
    failed = {name: actual for name, (actual, expected) in checks.items() if actual != expected}
    if failed:
        raise RuntimeError(f'Unexpected frozen input topology: {failed}')
    prevalence = bundle['prevalence']
    pass_count = int(prevalence['Pass p10'].astype(str).str.lower().eq('true').sum())
    if pass_count != 1961:
        raise RuntimeError(f'Unexpected frozen species p10 pass count: {pass_count}')
    if set(bundle['shapley']['Strategy']) != {'Block-PCA early fusion'}:
        raise RuntimeError('Unexpected Shapley strategy')
    if set(bundle['lomo_range']['Strategy']) != {'Block-PCA early fusion', 'Equal-weight late fusion'}:
        raise RuntimeError('Unexpected LOMO strategies')

def _tag(artist, gid: str, panel: str, role: str='numeric'):
    artist.set_gid(gid)
    artist._qa_meta = {'name': gid, 'panel': panel, 'role': role}
    return artist

def _panel_letter(fig: mpl.figure.Figure, spec, panel: str):
    bbox = spec.get_position(fig)
    artist = fig.text(max(0.012, bbox.x0 - 0.028), min(0.97, bbox.y1 + 0.004), panel, ha='left', va='bottom', fontsize=FONT['panel'], fontweight='bold', color=TEXT)
    return _tag(artist, f'panel-{panel}-label', panel, 'panel_label')

def _clean_heatmap_axis(ax, panel: str) -> None:
    ax._qa_meta = {'panel': panel}
    ax.tick_params(length=0, pad=1.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_panel_a(fig: mpl.figure.Figure, spec, bundle: Mapping[str, object]):
    _panel_letter(fig, spec, 'a')
    grid = spec.subgridspec(2, 4, height_ratios=[0.53, 0.47], width_ratios=[1.0, 1.0, 1.08, 1.22], hspace=0.76, wspace=0.55)
    axes = {}
    scores = bundle['batch_scores']
    effects = bundle['batch_effect']
    batch_colors = {'1': ENDPOINT_COLORS['PFS'], '2': '#8ED5E4'}
    for column, stage in enumerate(['Pre-ComBat', 'Post-ComBat']):
        ax = fig.add_subplot(grid[0, column])
        axes[f'batch-{stage}'] = ax
        subset = scores.loc[scores['Modality'].eq('Blood') & scores['Ion mode'].eq('POS') & scores['Stage'].eq(stage)]
        for batch, data in subset.groupby(subset['Batch ID'].astype(str)):
            ax.scatter(data['PC1'], data['PC2'], s=8.5, color=batch_colors[str(batch)], edgecolor='white', linewidth=0.25, alpha=0.84, label=f'Batch {batch}')
        r2 = float(effects.loc[effects['Modality'].eq('Blood') & effects['Ion mode'].eq('POS') & effects['Stage'].eq(stage), 'Descriptive batch R2 on first PCs'].iloc[0])
        text = ax.text(0.02, 0.98, f'{stage}\n$R^2$ {r2:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=FONT['small'], color=TEXT)
        _tag(text, f'panel-a-blood-pos-{stage}-r2', 'a')
        xvar = float(subset['PC1 explained variance (%)'].iloc[0])
        yvar = float(subset['PC2 explained variance (%)'].iloc[0])
        ax.set_xlabel(f'PC1 ({xvar:.1f}%)')
        ax.set_ylabel(f'PC2 ({yvar:.1f}%)')
        style_axis(ax)
        ax._qa_meta = {'panel': 'a'}
        if column == 0:
            legend = ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, borderaxespad=0, handletextpad=0.25, columnspacing=0.7)
            legend._qa_meta = {'name': 'panel-a-batch-legend', 'panel': 'a', 'role': 'legend'}
    ax_r2 = fig.add_subplot(grid[0, 2])
    axes['batch-r2'] = ax_r2
    modes = [('Blood', 'POS'), ('Blood', 'NEG'), ('Urine', 'POS'), ('Urine', 'NEG')]
    y = np.arange(len(modes))
    for yi, (modality, mode) in enumerate(modes):
        values = effects.loc[effects['Modality'].eq(modality) & effects['Ion mode'].eq(mode)].set_index('Stage')['Descriptive batch R2 on first PCs']
        pre, post = (float(values['Pre-ComBat']), float(values['Post-ComBat']))
        ax_r2.plot([post, pre], [yi, yi], color=GRID, linewidth=1.0, zorder=1)
        ax_r2.scatter(pre, yi, s=17, color=POSITIVE, marker='o', zorder=2)
        ax_r2.scatter(post, yi, s=20, facecolor='white', edgecolor=MODALITY_COLOR[modality], marker='s', linewidth=0.8, zorder=3)
        value_text = ax_r2.text(pre + 0.008, yi, f'{pre:.3f}', ha='left', va='center', fontsize=FONT['value'], color=MUTED)
        _tag(value_text, f'panel-a-batch-r2-{modality}-{mode}', 'a')
    ax_r2.set_yticks(y)
    ax_r2.set_yticklabels([f'{m} {mode}' for m, mode in modes])
    ax_r2.invert_yaxis()
    ax_r2.set_xlim(-0.012, 0.325)
    ax_r2.set_xlabel('Descriptive batch $R^2$')
    style_axis(ax_r2)
    ax_r2._qa_meta = {'panel': 'a'}
    ax_r2.legend(handles=[Line2D([0], [0], marker='o', linestyle='none', color=POSITIVE, label='Pre'), Line2D([0], [0], marker='s', linestyle='none', markerfacecolor='white', markeredgecolor=MODALITY_COLOR['Blood'], label='Post')], loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, borderaxespad=0, handletextpad=0.25, columnspacing=0.7)
    ax_rad = fig.add_subplot(grid[0, 3])
    axes['radiomics'] = ax_rad
    rad = bundle['radiomics_family'].groupby('Feature class', as_index=False)['Feature count'].sum().sort_values('Feature count', ascending=True)
    labels = rad['Feature class'].replace({'SHAPE': 'Shape'}).tolist()
    values = rad['Feature count'].to_numpy(float)
    ax_rad.barh(np.arange(len(rad)), values, color=MODALITY_COLOR['Radiomics'], alpha=0.83, height=0.68)
    ax_rad.set_yticks(np.arange(len(rad)))
    ax_rad.set_yticklabels(labels)
    for yi, value in enumerate(values):
        artist = ax_rad.text(value + 3, yi, f'{int(value)}', ha='left', va='center', fontsize=FONT['value'], color=TEXT)
        _tag(artist, f'panel-a-radiomics-{labels[yi]}', 'a')
    ax_rad.set_xlim(0, max(values) * 1.18)
    ax_rad.set_xlabel('Radiomics features')
    style_axis(ax_rad)
    ax_rad._qa_meta = {'panel': 'a'}
    ax_ret = fig.add_subplot(grid[1, :3])
    axes['retention'] = ax_ret
    retention = bundle['retention'].set_index('Modality').reindex(['KO', 'Species', 'Radiomics', 'Blood', 'Urine'])
    columns = ['Raw/input features', 'After filter', 'Final model-ready']
    numeric = retention[columns].apply(pd.to_numeric, errors='coerce')
    display = np.log10(numeric.to_numpy(float) + 1.0)
    masked = np.ma.masked_invalid(display)
    cmap = SEQUENTIAL_CMAP.copy()
    cmap.set_bad('#ECEEEF')
    ax_ret.imshow(masked, aspect='auto', cmap=cmap, vmin=2.8, vmax=4.0)
    ax_ret.set_xticks(np.arange(3))
    ax_ret.set_xticklabels(['Raw/input', 'After filter', 'Model-ready'])
    ax_ret.set_yticks(np.arange(5))
    ax_ret.set_yticklabels(['KO functions', 'Species', 'Radiomics', 'Blood', 'Urine'])
    for row in range(5):
        for column in range(3):
            value = numeric.iloc[row, column]
            text_value = 'NA' if pd.isna(value) else f'{int(value):,}'
            artist = ax_ret.text(column, row, text_value, ha='center', va='center', fontsize=FONT['value'], color=TEXT)
            _tag(artist, f'panel-a-retention-{row}-{column}', 'a')
    _clean_heatmap_axis(ax_ret, 'a')
    ax_prev = fig.add_subplot(grid[1, 3])
    axes['prevalence'] = ax_prev
    prevalence = pd.to_numeric(bundle['prevalence']['Internal prevalence'], errors='coerce').dropna()
    ax_prev.hist(prevalence, bins=np.linspace(0, 1, 21), color=SPECIES_COLOR, edgecolor='white', linewidth=0.3)
    ax_prev.axvline(0.1, color=TEXT, linestyle='--', linewidth=0.8)
    count_text = ax_prev.text(0.98, 0.96, '1,961 / 6,438', transform=ax_prev.transAxes, ha='right', va='top', fontsize=FONT['value'], color=MUTED)
    _tag(count_text, 'panel-a-prevalence-count', 'a')
    ax_prev.set_xlabel('Internal prevalence')
    ax_prev.set_ylabel('Species')
    style_axis(ax_prev)
    ax_prev._qa_meta = {'panel': 'a'}
    return axes

def draw_panel_b(fig: mpl.figure.Figure, spec, bundle: Mapping[str, object]):
    _panel_letter(fig, spec, 'b')
    grid = spec.subgridspec(1, 2, width_ratios=[0.4, 0.6], wspace=0.48)
    axes = {}
    pairs = bundle['similarity_pairs'].sort_values('Distance_similarity_rho', ascending=True)
    ax_similarity = fig.add_subplot(grid[0, 0])
    axes['similarity'] = ax_similarity
    values = pairs['Distance_similarity_rho'].to_numpy(float)
    y = np.arange(len(pairs))
    colors = [POSITIVE if value > 0 else NEGATIVE for value in values]
    ax_similarity.barh(y, values, color=colors, alpha=0.72, height=0.68)
    ax_similarity.axvline(0, color=TEXT, linewidth=0.65)
    ax_similarity.set_yticks(y)
    ax_similarity.set_yticklabels([pair.replace('Metagenomics', 'KO').replace('Radiomics', 'Radio.') for pair in pairs['Pair']])
    ax_similarity.set_xlim(-0.11, 0.31)
    ax_similarity.set_xlabel('Distance-matrix Spearman $\\rho$')
    for yi, (_, row) in enumerate(pairs.iterrows()):
        value = float(row['Distance_similarity_rho'])
        symbol = '*' if float(row['FDR']) < 0.05 else ''
        artist = ax_similarity.text(value + (0.008 if value >= 0 else -0.008), yi, f'{value:+.2f}{symbol}', ha='left' if value >= 0 else 'right', va='center', fontsize=FONT['value'], color=TEXT)
        _tag(artist, f'panel-b-similarity-{yi}', 'b')
    style_axis(ax_similarity)
    ax_similarity._qa_meta = {'panel': 'b'}
    ax_benchmark = fig.add_subplot(grid[0, 1])
    axes['benchmark'] = ax_benchmark
    method_order = ['Metagenomics', 'Species', 'Radiomics', 'Blood', 'Urine', 'Pooled PCA fusion', 'Block-PCA early fusion', 'Equal-weight late fusion']
    benchmark = bundle['benchmark'].pivot(index='Method', columns='Metric', values='Mean').reindex(index=method_order, columns=METRIC_ORDER)
    best = bundle['best_single'].set_index('Metric')['Best_single_mean'].reindex(METRIC_ORDER)
    matrix = benchmark.to_numpy(float)
    ax_benchmark.imshow(matrix, cmap=SEQUENTIAL_CMAP, norm=Normalize(vmin=0.52, vmax=0.93), aspect='auto')
    ax_benchmark.set_xticks(np.arange(3))
    ax_benchmark.set_xticklabels([METRIC_LABEL[metric] for metric in METRIC_ORDER])
    ax_benchmark.set_yticks(np.arange(len(method_order)))
    ax_benchmark.set_yticklabels([MODALITY_LABEL.get(method, method) for method in method_order])
    ax_benchmark.axhline(4.5, color=TEXT, linewidth=0.65)
    for row_index, method in enumerate(method_order):
        for column_index, metric in enumerate(METRIC_ORDER):
            value = float(benchmark.loc[method, metric])
            text = f'{value:.3f}'
            if 'fusion' in method.lower():
                text += f'\n$\\Delta${value - float(best.loc[metric]):+.3f}'
            artist = ax_benchmark.text(column_index, row_index, text, ha='center', va='center', fontsize=FONT['value'], color=TEXT, linespacing=1.05)
            _tag(artist, f'panel-b-benchmark-{row_index}-{column_index}', 'b')
    _clean_heatmap_axis(ax_benchmark, 'b')
    return axes

def draw_panel_c(fig: mpl.figure.Figure, spec, bundle: Mapping[str, object]):
    _panel_letter(fig, spec, 'c')
    ax = fig.add_subplot(spec)
    data = bundle['permanova'].copy()
    modality_order = MODALITY_ORDER + ['Fused']
    matrix = np.full((len(modality_order), len(ENDPOINT_ORDER)), np.nan)
    q_matrix = np.full_like(matrix, np.nan)
    dispersion_matrix = np.full_like(matrix, np.nan)
    for row_index, modality in enumerate(modality_order):
        for column_index, endpoint in enumerate(ENDPOINT_ORDER):
            row = data.loc[data['Modality'].eq(modality) & data['Endpoint'].eq(endpoint)]
            if len(row) != 1:
                continue
            matrix[row_index, column_index] = float(row['Adjusted_incremental_R2'].iloc[0])
            q_matrix[row_index, column_index] = float(row['Adjusted_partial_PERMANOVA_FDR'].iloc[0])
            dispersion_matrix[row_index, column_index] = float(row['PERMDISP_FDR'].iloc[0])
    ax.imshow(matrix * 100.0, cmap=SEQUENTIAL_CMAP, norm=Normalize(vmin=0.0, vmax=5.0), aspect='auto')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Response', 'PFS 6 mo', 'OS 12 mo'])
    ax.set_yticks(np.arange(len(modality_order)))
    ax.set_yticklabels([MODALITY_LABEL.get(modality, modality) for modality in modality_order])
    ax.axhline(len(modality_order) - 1.5, color=TEXT, linewidth=0.65)
    for row_index in range(len(modality_order)):
        for column_index in range(3):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            artist = ax.text(column_index, row_index, f'{value * 100:.1f}', ha='center', va='center', fontsize=FONT['value'], color=TEXT)
            _tag(artist, f'panel-c-r2-{row_index}-{column_index}', 'c')
            robust = q_matrix[row_index, column_index] < 0.05 and value >= 0.03 and (dispersion_matrix[row_index, column_index] >= 0.05)
            if robust:
                ax.add_patch(Rectangle((column_index - 0.47, row_index - 0.47), 0.94, 0.94, fill=False, edgecolor=TEXT, linewidth=0.75, zorder=4))
            if dispersion_matrix[row_index, column_index] < 0.05:
                dagger = ax.text(column_index + 0.33, row_index - 0.3, '†', ha='center', va='center', fontsize=FONT['small'], color=TEXT)
                _tag(dagger, f'panel-c-dispersion-{row_index}-{column_index}', 'c')
    _clean_heatmap_axis(ax, 'c')
    ax.set_xlabel('Adjusted incremental $R^2$ (%)')
    return {'audit': ax}

def _directional_matrix(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((5, 5), np.nan)
    positive = np.full((5, 5), np.nan)
    stable = np.zeros((5, 5), dtype=bool)
    for row_index, added in enumerate(MODALITY_ORDER):
        for column_index, base in enumerate(MODALITY_ORDER):
            if added == base:
                continue
            row = data.loc[data['Added_modality'].eq(added) & data['Base_modality'].eq(base) & data['Metric'].eq(metric)]
            if len(row) != 1:
                continue
            values[row_index, column_index] = float(row['Mean_repeat_delta'].iloc[0])
            positive[row_index, column_index] = float(row['Positive_fold_fraction'].iloc[0])
            stable[row_index, column_index] = float(row['Lower_95'].iloc[0]) > 0 and float(row['Positive_fold_fraction'].iloc[0]) >= 0.8
    return (values, positive, stable)

def draw_panel_d(fig: mpl.figure.Figure, spec, bundle: Mapping[str, object]):
    _panel_letter(fig, spec, 'd')
    grid = spec.subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.88], wspace=0.42)
    axes = {}
    directional = bundle['directional']
    for column, metric in enumerate(METRIC_ORDER):
        ax = fig.add_subplot(grid[0, column])
        axes[f'directional-{metric}'] = ax
        values, positive_fraction, stable = _directional_matrix(directional, metric)
        ax.imshow(values, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.22), aspect='auto')
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['KO', 'Sp.', 'Rad.', 'Bl.', 'Ur.'], rotation=0)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels([MODALITY_LABEL[m] for m in MODALITY_ORDER] if column == 0 else [])
        ax.set_title(METRIC_LABEL[metric], pad=3, fontweight='bold')
        for row_index in range(5):
            for column_index in range(5):
                value = values[row_index, column_index]
                if not np.isfinite(value):
                    continue
                artist = ax.text(column_index, row_index - 0.11, f'{value:.3f}', ha='center', va='center', fontsize=FONT['value'], color=TEXT)
                _tag(artist, f'panel-d-directional-{metric}-{row_index}-{column_index}', 'd')
                fraction = ax.text(column_index, row_index + 0.24, f'{positive_fraction[row_index, column_index]:.0%}', ha='center', va='center', fontsize=FONT['small'], color=MUTED)
                _tag(fraction, f'panel-d-positive-fold-{metric}-{row_index}-{column_index}', 'd')
                if stable[row_index, column_index]:
                    dagger = ax.text(column_index + 0.35, row_index - 0.32, '†', ha='center', va='center', fontsize=FONT['small'], color=TEXT)
                    _tag(dagger, f'panel-d-stable-{metric}-{row_index}-{column_index}', 'd')
        _clean_heatmap_axis(ax, 'd')
        if column == 0:
            ax.set_ylabel('Added block')
        if column == 1:
            ax.set_xlabel('Base block', labelpad=2)
    ax_shapley = fig.add_subplot(grid[0, 3])
    axes['shapley'] = ax_shapley
    shapley = bundle['shapley'].pivot(index='Modality', columns='Metric', values='Mean_Shapley').reindex(index=MODALITY_ORDER, columns=METRIC_ORDER)
    matrix = shapley.to_numpy(float)
    ax_shapley.imshow(matrix, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-0.13, vcenter=0.0, vmax=0.13), aspect='auto')
    ax_shapley.set_xticks(np.arange(3))
    ax_shapley.set_xticklabels(['Resp.', 'PFS', 'OS'])
    ax_shapley.set_yticks(np.arange(5))
    ax_shapley.set_yticklabels([MODALITY_LABEL[m] for m in MODALITY_ORDER])
    ax_shapley.set_title('Shapley', pad=3, fontweight='bold')
    for row_index in range(5):
        for column_index in range(3):
            value = matrix[row_index, column_index]
            artist = ax_shapley.text(column_index, row_index, f'{value:.3f}', ha='center', va='center', fontsize=FONT['value'], color=TEXT)
            _tag(artist, f'panel-d-shapley-{row_index}-{column_index}', 'd')
    _clean_heatmap_axis(ax_shapley, 'd')
    ax_shapley.set_xlabel('Mean contribution')
    return axes

def draw_panel_e(fig: mpl.figure.Figure, spec, bundle: Mapping[str, object]):
    _panel_letter(fig, spec, 'e')
    grid = spec.subgridspec(1, 3, wspace=0.3)
    axes = {}
    data = bundle['lomo_range']
    strategies = ['Block-PCA early fusion', 'Equal-weight late fusion']
    strategy_style = {'Block-PCA early fusion': {'marker': 'o', 'color': TEXT, 'fill': TEXT}, 'Equal-weight late fusion': {'marker': '^', 'color': ENDPOINT_COLORS['PFS'], 'fill': 'white'}}
    y = np.arange(5)
    x_limits = {}
    for metric in METRIC_ORDER:
        subset = data.loc[data['Metric'].eq(metric)]
        lower = float(subset['Min_repeat_loss'].min())
        upper = float(subset['Max_repeat_loss'].max())
        pad = max(0.032, (upper - lower) * 0.25)
        x_limits[metric] = (min(-0.06, lower - pad), max(0.095, upper + pad))
    for column, metric in enumerate(METRIC_ORDER):
        ax = fig.add_subplot(grid[0, column])
        axes[metric] = ax
        ax.axvline(0, color=MUTED, linestyle='--', linewidth=0.7)
        for strategy_index, strategy in enumerate(strategies):
            subset = data.loc[data['Metric'].eq(metric) & data['Strategy'].eq(strategy)].set_index('Removed_modality').reindex(MODALITY_ORDER)
            means = subset['Mean_repeat_loss'].to_numpy(float)
            lows = subset['Min_repeat_loss'].to_numpy(float)
            highs = subset['Max_repeat_loss'].to_numpy(float)
            offset = -0.11 if strategy_index == 0 else 0.11
            label_offset = -0.29 if strategy_index == 0 else 0.29
            style = strategy_style[strategy]
            ax.errorbar(means, y + offset, xerr=np.vstack([means - lows, highs - means]), fmt=style['marker'], color=style['color'], markerfacecolor=style['fill'], markeredgewidth=0.8, markersize=3.3, capsize=1.4, elinewidth=0.8, linewidth=0, label=strategy, zorder=3)
            for row_index, value in enumerate(means):
                label_x = highs[row_index] + 0.004 if value >= 0 else lows[row_index] - 0.004
                artist = ax.text(label_x, y[row_index] + label_offset, f'{value:+.3f}', ha='left' if value >= 0 else 'right', va='center', fontsize=FONT['value'], color=style['color'])
                _tag(artist, f'panel-e-{metric}-{strategy_index}-{row_index}', 'e')
        ax.set_yticks(y)
        ax.set_yticklabels([MODALITY_LABEL[m] for m in MODALITY_ORDER] if column == 0 else [])
        ax.invert_yaxis()
        ax.set_ylim(4.62, -0.62)
        ax.set_xlim(*x_limits[metric])
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, prune='both'))
        ax.set_title(METRIC_LABEL[metric], pad=3, fontweight='bold')
        style_axis(ax)
        ax._qa_meta = {'panel': 'e'}
        if column == 0:
            legend = ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.14), ncol=2, frameon=False, borderaxespad=0, columnspacing=0.8, handletextpad=0.3)
            legend._qa_meta = {'name': 'panel-e-strategy-legend', 'panel': 'e', 'role': 'legend'}
    axes[METRIC_ORDER[1]].set_xlabel('Performance loss after removal')
    return axes

def build_full_figure(bundle: Mapping[str, object]):
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    outer = fig.add_gridspec(4, 1, height_ratios=[0.295, 0.205, 0.285, 0.215], left=0.1, right=0.95, top=0.97, bottom=0.052, hspace=0.42)
    axes = {'a': draw_panel_a(fig, outer[0], bundle)}
    middle = outer[1].subgridspec(1, 2, width_ratios=[0.65, 0.35], wspace=0.28)
    axes['b'] = draw_panel_b(fig, middle[0], bundle)
    axes['c'] = draw_panel_c(fig, middle[1], bundle)
    axes['d'] = draw_panel_d(fig, outer[2], bundle)
    axes['e'] = draw_panel_e(fig, outer[3], bundle)
    return (fig, axes)

def build_panel_figure(panel: str, bundle: Mapping[str, object]):
    panel = panel.lower()
    if panel not in PANEL_SIZE_MM:
        raise ValueError(f'Unknown panel: {panel}')
    configure_style()
    width_mm, height_mm = PANEL_SIZE_MM[panel]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    margins = {'a': (0.1, 0.95, 0.88, 0.18), 'b': (0.14, 0.97, 0.94, 0.2), 'c': (0.15, 0.965, 0.94, 0.15), 'd': (0.11, 0.97, 0.94, 0.13), 'e': (0.1, 0.95, 0.8, 0.18)}
    left, right, top, bottom = margins[panel]
    spec = fig.add_gridspec(1, 1, left=left, right=right, top=top, bottom=bottom)[0]
    drawer = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c, 'd': draw_panel_d, 'e': draw_panel_e}[panel]
    axes = drawer(fig, spec, bundle)
    for artist in list(fig.texts):
        if artist.get_gid() == f'panel-{panel}-label':
            artist.remove()
    return (fig, axes)

def audit_panel_e_annotation_clearance(fig: mpl.figure.Figure, axes: Mapping[str, mpl.axes.Axes], bundle: Mapping[str, object]) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_px = fig.dpi / 72.0
    labels = {artist.get_gid(): artist for artist in fig.findobj(match=mpl.text.Text) if isinstance(artist.get_gid(), str) and artist.get_gid().startswith('panel-e-')}
    strategies = ['Block-PCA early fusion', 'Equal-weight late fusion']
    data = bundle['lomo_range']
    graphics: dict[str, list[dict[str, object]]] = {metric: [] for metric in METRIC_ORDER}
    for metric in METRIC_ORDER:
        ax = axes[metric]
        zero_x = float(ax.transData.transform((0.0, 0.0))[0])
        graphics[metric].append({'name': f'{metric}-zero-line', 'bbox': Bbox.from_extents(zero_x - 0.65 * point_px, ax.bbox.y0, zero_x + 0.65 * point_px, ax.bbox.y1)})
        for strategy_index, strategy in enumerate(strategies):
            subset = data.loc[data['Metric'].eq(metric) & data['Strategy'].eq(strategy)].set_index('Removed_modality').reindex(MODALITY_ORDER)
            offset = -0.11 if strategy_index == 0 else 0.11
            for row_index, (_, row) in enumerate(subset.iterrows()):
                low = float(row['Min_repeat_loss'])
                high = float(row['Max_repeat_loss'])
                mean = float(row['Mean_repeat_loss'])
                left = ax.transData.transform((low, row_index + offset))
                right = ax.transData.transform((high, row_index + offset))
                centre = ax.transData.transform((mean, row_index + offset))
                graphics[metric].append({'name': f'{metric}-{strategy_index}-{row_index}-errorbar', 'bbox': Bbox.from_extents(min(left[0], right[0]) - 0.8 * point_px, left[1] - 2.2 * point_px, max(left[0], right[0]) + 0.8 * point_px, left[1] + 2.2 * point_px)})
                marker_radius = 2.5 * point_px
                graphics[metric].append({'name': f'{metric}-{strategy_index}-{row_index}-marker', 'bbox': Bbox.from_extents(centre[0] - marker_radius, centre[1] - marker_radius, centre[0] + marker_radius, centre[1] + marker_radius)})
    intersections: list[dict[str, str]] = []
    label_records = []
    for gid, artist in labels.items():
        metric = next((candidate for candidate in METRIC_ORDER if f'panel-e-{candidate}-' in gid), None)
        if metric is None:
            intersections.append({'label': gid, 'graphic': 'unresolved-metric'})
            continue
        bbox = artist.get_window_extent(renderer).padded(0.2 * point_px)
        label_records.append({'gid': gid, 'metric': metric, 'bbox_pt': [round(float(value) / point_px, 3) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)]})
        for graphic in graphics[metric]:
            if bbox.overlaps(graphic['bbox']):
                intersections.append({'label': gid, 'graphic': str(graphic['name'])})
    label_items = list(labels.items())
    label_label_intersections = []
    for index, (gid_a, artist_a) in enumerate(label_items):
        bbox_a = artist_a.get_window_extent(renderer).padded(0.2 * point_px)
        for gid_b, artist_b in label_items[index + 1:]:
            bbox_b = artist_b.get_window_extent(renderer).padded(0.2 * point_px)
            if bbox_a.overlaps(bbox_b):
                label_label_intersections.append({'label_a': gid_a, 'label_b': gid_b})
    if len(labels) != 30 or intersections or label_label_intersections:
        raise RuntimeError(f'S1e annotation-to-graphic clearance failed: labels={len(labels)}, graphics={len(intersections)}, text={len(label_label_intersections)}')
    return {'figure': 'Supplementary Figure S1 panel e', 'version': VERSION, 'method': 'display-coordinate bounding-box check against every marker, errorbar, zero line and value label', 'labels_checked': len(labels), 'graphic_primitives_checked': sum((len(rows) for rows in graphics.values())), 'label_graphic_intersections': intersections, 'label_label_intersections': label_label_intersections, 'design_offsets': {'early_data_y': -0.11, 'early_label_y': -0.29, 'late_data_y': 0.11, 'late_label_y': 0.29, 'horizontal_gap_beyond_interval': 0.004}, 'labels': label_records, 'pass': True}

def _audit_and_require(fig: mpl.figure.Figure, name: str, path: Path) -> dict[str, object]:
    report = audit_figure_layout(fig, name, minimum_font_pt=5.4, safe_pad_pt=0.8, check_collisions=True)
    write_json(report, path)
    summary = report['summary']
    if any((summary[key] for key in ('critical', 'warning', 'low_clearance'))):
        raise RuntimeError(f'Strict layout QA failed for {name}: {summary}')
    return report

def _save_supplement_figure(fig: mpl.figure.Figure, output_stem: Path, *, metadata: Mapping[str, str]) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in ('pdf', 'svg', 'png'):
        path = output_stem.with_suffix(f'.{fmt}')
        kwargs: dict[str, object] = {'format': fmt, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if fmt == 'png':
            kwargs['dpi'] = 600
        if fmt in {'pdf', 'svg'}:
            kwargs['metadata'] = dict(metadata)
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def export_all(bundle: Mapping[str, object]) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    full_fig, _ = build_full_figure(bundle)
    full_audit_path = OUTPUT_DIR / f'Supplementary_Figure_S1_layout_audit_{VERSION}.json'
    _audit_and_require(full_fig, f'Supplementary_Figure_S1_full_{VERSION}', full_audit_path)
    full_paths = _save_supplement_figure(full_fig, FIGURE_STEM, metadata={'Title': 'Avatar-HCC Supplementary Figure S1 a-e', 'Creator': 'Avatar-HCC Figure Harmonization'})
    plt.close(full_fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, object] = {}
    panel_e_clearance_path: Path | None = None
    for panel in 'abcde':
        panel_fig, panel_axes = build_panel_figure(panel, bundle)
        report = audit_figure_layout(panel_fig, f'Supplementary_Figure_S1_panel_{panel}_{VERSION}', minimum_font_pt=5.4, safe_pad_pt=0.8, check_collisions=True)
        panel_audits[panel] = report
        summary = report['summary']
        if any((summary[key] for key in ('critical', 'warning', 'low_clearance'))):
            raise RuntimeError(f'Standalone S1{panel} layout QA failed: {summary}')
        paths = save_figure(panel_fig, PANEL_DIR / f'Supplementary_Figure_S1_panel_{panel}_Nature_{VERSION}', formats=('pdf', 'svg', 'png'), dpi=600, tight=False, metadata={'Title': f'Avatar-HCC Supplementary Figure S1 panel {panel}', 'Creator': 'Avatar-HCC Figure Harmonization'})
        panel_paths.extend(paths)
        if panel == 'e':
            clearance = audit_panel_e_annotation_clearance(panel_fig, panel_axes, bundle)
            clearance['panel_e_svg_sha256'] = sha256(paths[1])
            clearance['panel_e_png_sha256'] = sha256(paths[2])
            panel_e_clearance_path = OUTPUT_DIR / f'Supplementary_Figure_S1_panel_e_annotation_clearance_{VERSION}.json'
            write_json(clearance, panel_e_clearance_path)
        plt.close(panel_fig)
    combined = {'figure': {'name': f'Supplementary_Figure_S1_standalone_panels_{VERSION}'}, 'summary': {'critical': sum((report['summary']['critical'] for report in panel_audits.values())), 'warning': sum((report['summary']['warning'] for report in panel_audits.values())), 'low_clearance': sum((report['summary']['low_clearance'] for report in panel_audits.values())), 'minimum_font_pt': min((report['summary']['minimum_font_pt'] for report in panel_audits.values())), 'required_minimum_font_pt': 5.4, 'pass': all((report['summary']['pass'] for report in panel_audits.values())), 'strict_pass': all((report['summary']['strict_pass'] for report in panel_audits.values()))}, 'panels': panel_audits}
    panel_audit_path = OUTPUT_DIR / f'Supplementary_Figure_S1_panel_layout_audit_{VERSION}.json'
    write_json(combined, panel_audit_path)
    base_manifest = {'version': VERSION, 'palette': PALETTE_ID, 'figure_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only frozen copied tables; no statistical or model recomputation', 'input_manifest_sha256': sha256(INPUT_DIR / get_value('figures.config.item_116')), 'scientific_contract_sha256': sha256(INPUT_DIR / get_value('figures.config.item_201')), 'outputs': [{'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size} for path in full_paths + panel_paths + [full_audit_path, panel_audit_path, panel_e_clearance_path] if path is not None]}
    manifest_path = OUTPUT_DIR / f'Supplementary_Figure_S1_export_manifest_{VERSION}.json'
    manifest_path.write_text(json.dumps(base_manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {'full_paths': full_paths, 'panel_paths': panel_paths, 'full_audit': full_audit_path, 'panel_audit': panel_audit_path, 'panel_e_clearance': panel_e_clearance_path, 'manifest': manifest_path}

def summarize_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    prevalence = bundle['prevalence']
    efficiency = pd.to_numeric(bundle['shapley_efficiency']['Efficiency_error'], errors='coerce').abs()
    return {'frozen_input_files': int(bundle['manifest']['file_count']), 'batch_score_rows': int(len(bundle['batch_scores'])), 'species_prevalence_features': int(len(prevalence)), 'species_p10_pass': int(prevalence['Pass p10'].astype(str).str.lower().eq('true').sum()), 'cross_modal_pairs': int(len(bundle['similarity_pairs'])), 'directional_increment_rows': int(len(bundle['directional'])), 'shapley_efficiency_max_abs_error': float(efficiency.max()), 'lomo_repeat_rows': int(len(bundle['lomo_range'])), 'analysis_recomputed': False}
configure_style()
