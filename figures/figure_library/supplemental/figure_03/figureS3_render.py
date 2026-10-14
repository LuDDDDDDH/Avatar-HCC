from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MBP_COLORS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = ROOT / 'inputs'
VERSION = 'v001'
OUTPUT_DIR = ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FULL_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_317')}{VERSION}"
WIDTH_MM = 180.0
HEIGHT_MM = 240.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT: Mapping[str, float] = {'base': 5.8, 'panel': 8.5, 'facet': 6.0, 'axis': 5.8, 'tick': 5.4, 'annotation': 5.4, 'cell': 5.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
GRID = '#D9E0E6'
LIGHT = NEUTRAL_COLORS['light_background']
MBP_ORDER = ['MBP-I', 'MBP-II', 'MBP-III']
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
CONTEXT_ORDER = ['Pretreatment', 'Post-mRECIST']
INPUT_FILES = {'landscape': get_value('figures.config.item_171'), 'boundaries': get_value('figures.config.item_143'), 'mrecist_cells': get_value('figures.config.item_148'), 'mrecist_assoc': get_value('figures.config.item_147'), 'within_mrecist': get_value('figures.config.item_256'), 'phenotype': get_value('figures.config.item_068'), 'phenotype_tests': get_value('figures.config.item_069'), 'multivariable': get_value('figures.config.item_107'), 'outer_summary': get_value('figures.config.item_161'), 'outer_folds': get_value('figures.config.item_160'), 'repeat_folds': get_value('figures.config.item_185'), 'repeat_summary': get_value('figures.config.item_184'), 'cutoff_values': get_value('figures.config.item_163'), 'cutoff_summary': get_value('figures.config.item_162'), 'excluded_effects': get_value('figures.config.item_080'), 'excluded_agreement': get_value('figures.config.item_081'), 'excluded_counts': get_value('figures.config.item_082'), 'weight_agreement': get_value('figures.config.item_149'), 'weight_effects': get_value('figures.config.item_150'), 'cutoff_effects_extreme': get_value('figures.config.item_138'), 'cutoff_effects_category': get_value('figures.config.item_139')}
FULL_LAYOUT: Mapping[str, tuple[float, float, float, float]] = {'a': (7.0, 6.0, 166.0, 32.0), 'b': (7.0, 41.0, 166.0, 33.0), 'c': (7.0, 77.0, 166.0, 54.0), 'd': (7.0, 134.0, 166.0, 43.0), 'e': (7.0, 180.0, 80.0, 54.0), 'f': (93.0, 180.0, 80.0, 54.0)}
PANEL_SIZES_MM: Mapping[str, tuple[float, float]] = {'a': (180.0, 45.0), 'b': (135.0, 52.0), 'c': (180.0, 72.0), 'd': (180.0, 62.0), 'e': (112.0, 78.0), 'f': (112.0, 80.0)}

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'], 'font.size': FONT['base'], 'axes.labelsize': FONT['axis'], 'axes.titlesize': FONT['facet'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'legend.fontsize': FONT['tick'], 'legend.title_fontsize': FONT['tick'], 'axes.linewidth': 0.55, 'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing columns: {missing}')

def _require_rows(frame: pd.DataFrame, count: int, label: str) -> None:
    if len(frame) != int(count):
        raise ValueError(f'{label}: expected {count} rows, found {len(frame)}')

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {key: pd.read_csv(INPUT_DIR / filename) for key, filename in INPUT_FILES.items()}
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    bundle['contract'] = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    bundle['numeric_qa'] = validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle['manifest']
    if manifest['palette_id'] != PALETTE_ID:
        raise ValueError('Palette identifier differs from the shared style')
    records = {entry['snapshot']: entry for entry in manifest['files']}
    file_checks: list[dict[str, Any]] = []
    for filename in INPUT_FILES.values():
        path = INPUT_DIR / filename
        if filename not in records:
            raise ValueError(f'Manifest does not register {filename}')
        record = records[filename]
        actual_hash = _sha256(path)
        frame = pd.read_csv(path)
        if actual_hash != record['sha256']:
            raise ValueError(f'SHA-256 mismatch for {filename}')
        if frame.shape != (int(record['rows']), int(record['columns'])):
            raise ValueError(f'Shape mismatch for {filename}')
        file_checks.append({'file': filename, 'shape': [int(frame.shape[0]), int(frame.shape[1])], 'sha256': actual_hash, 'pass': True})
    landscape = bundle['landscape']
    _require_rows(landscape, 109, 'patient landscape')
    _require_columns(landscape, ['sample', 'MBP_raw', 'patient_order', 'Response_benefit', 'PFS_benefit', 'OS_benefit', 'Response_observed', 'AFP_favorability', 'TBS_favorability', 'PVTT_favorability', 'ALBI_favorability'], 'patient landscape')
    if sorted(landscape['patient_order'].astype(int).tolist()) != list(range(1, 110)):
        raise ValueError('Patient order is not the locked 1..109 sequence')
    group_n = landscape['MBP_raw'].value_counts().reindex(MBP_ORDER).astype(int)
    if group_n.tolist() != [37, 36, 36]:
        raise ValueError('MBP group sizes differ from 37/36/36')
    boundaries = bundle['boundaries'].set_index('group').reindex(MBP_ORDER)
    if boundaries['n'].astype(int).tolist() != group_n.tolist():
        raise ValueError('Landscape and boundary group counts disagree')
    assoc = bundle['mrecist_assoc']
    cells = bundle['mrecist_cells']
    _require_rows(assoc, 1, 'mRECIST association')
    _require_rows(cells, 6, 'mRECIST cells')
    if int(cells['Observed_n'].sum()) != 109 or int(assoc.loc[0, 'n']) != 109:
        raise ValueError('mRECIST association totals differ from 109')
    within = bundle['within_mrecist']
    _require_rows(within, 4, 'within-mRECIST summary')
    if set(within['Endpoint']) != {'PFS', 'OS'} or set(within['mRECIST']) != {'Responder', 'Nonresponder'}:
        raise ValueError('Within-mRECIST endpoint/stratum cells are incomplete')
    expected_ne = {('PFS', 'Responder'): (53, 44), ('PFS', 'Nonresponder'): (56, 54), ('OS', 'Responder'): (53, 22), ('OS', 'Nonresponder'): (56, 45)}
    for row in within.itertuples(index=False):
        if (int(row.n), int(row.events)) != expected_ne[row.Endpoint, row.mRECIST]:
            raise ValueError('Within-mRECIST n/event contract changed')
    phenotype = bundle['phenotype']
    phenotype_tests = bundle['phenotype_tests']
    _require_rows(phenotype, 69, 'phenotype effect matrix')
    _require_rows(phenotype_tests, 23, 'phenotype global tests')
    if phenotype['Variable'].nunique() != 23 or set(phenotype['MBP_group']) != set(MBP_ORDER):
        raise ValueError('Phenotype matrix is not 23 variables x 3 MBP groups')
    if set(phenotype['Variable']) != set(phenotype_tests['Variable']):
        raise ValueError('Phenotype rows and global tests differ')
    multivariable = bundle['multivariable']
    _require_rows(multivariable, 12, 'full multivariable effects')
    if set(multivariable['Endpoint']) != {'PFS', 'OS'}:
        raise ValueError('Full multivariable table must contain PFS and OS')
    if set(multivariable['n'].astype(int)) != {109} or set(multivariable['events'].astype(int)) != {98, 67}:
        raise ValueError('Multivariable n/events changed')
    expected_rows = {'outer_summary': 4, 'outer_folds': 200, 'repeat_folds': 100, 'repeat_summary': 4, 'cutoff_values': 200, 'cutoff_summary': 4, 'excluded_effects': 6, 'excluded_agreement': 3, 'excluded_counts': 27, 'weight_agreement': 4, 'weight_effects': 12, 'cutoff_effects_extreme': 9, 'cutoff_effects_category': 9}
    for key, count in expected_rows.items():
        _require_rows(bundle[key], count, key)
    if not (bundle['outer_summary']['n_valid_outer_folds'].astype(int) == 50).all():
        raise ValueError('Outer-CV fold count differs from 50')
    if not (bundle['repeat_summary']['n_folds'].astype(int) == 25).all():
        raise ValueError('Repeated-CV fold count differs from 25')
    if not (bundle['cutoff_summary']['n_outer_training_folds'].astype(int) == 50).all():
        raise ValueError('Cutoff summary fold count differs from 50')
    if not (bundle['excluded_counts'].groupby('Endpoint_excluded')['n'].sum().astype(int) == 109).all():
        raise ValueError('Endpoint-excluded confusion tables do not total 109')
    if not (bundle['excluded_agreement']['n'].astype(int) == 109).all():
        raise ValueError('Endpoint-excluded agreement n differs from 109')
    if not (bundle['weight_agreement']['n'].astype(int) == 109).all():
        raise ValueError('Weight-perturbation agreement n differs from 109')
    if not (bundle['excluded_effects']['n'].astype(int) == 109).all():
        raise ValueError('Endpoint-excluded effects n differs from 109')
    return {'figure': 'Supplementary Figure S3', 'version': VERSION, 'source_mode': 'frozen snapshot validation only; no statistical estimation', 'files': file_checks, 'locked': {'cohort_n': 109, 'MBP_group_n': group_n.to_dict(), 'within_mRECIST_rows': 4, 'phenotype_variables': 23, 'outerCV_fold_rows': 200, 'repeatedCV_fold_rows': 100, 'cutoff_rows': 200, 'endpoint_excluded_confusion_total_per_endpoint': 109}, 'pass': True}

def _axis(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], rel: tuple[float, float, float, float], panel: str) -> mpl.axes.Axes:
    page_w, page_h = page_size
    left, top, width, height = rect
    rel_left, rel_bottom, rel_w, rel_h = rel
    abs_left = left + rel_left * width
    abs_bottom = page_h - top - height + rel_bottom * height
    ax = fig.add_axes([abs_left / page_w, abs_bottom / page_h, rel_w * width / page_w, rel_h * height / page_h])
    ax._qa_meta = {'panel': panel}
    return ax

def _clean(ax: mpl.axes.Axes, *, grid_axis: str | None=None) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(direction='out', width=0.55, length=2.0, pad=1.5)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.35, alpha=0.75, zorder=0)

def _format_p(value: float) -> str:
    value = float(value)
    return 'P<0.001' if value < 0.001 else f'P={value:.3f}'

def _contrast_text(rgba: Sequence[float]) -> str:
    r, g, b = rgba[:3]
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.54 else TEXT

def _offsets(n: int, amplitude: float=0.22) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    pattern = np.array([0.0, -0.5, 0.5, -1.0, 1.0])
    return np.resize(pattern, n) * float(amplitude)

def _add_panel_letter(fig: mpl.figure.Figure, panel: str, rect: tuple[float, float, float, float], registry: ArtistRegistry) -> None:
    left, top, _, _ = rect
    artist = fig.text(max(3.8, left - 1.8) / WIDTH_MM, 1.0 - max(2.6, top - 1.2) / HEIGHT_MM, panel, ha='left', va='top', fontsize=FONT['panel'], fontweight='bold', color=TEXT)
    registry.tag(artist, name=f'panel-{panel}-label', role='panel_label', panel=panel)

def draw_panel_a(fig, page_size, rect, bundle, registry):
    header = _axis(fig, page_size, rect, (0.145, 0.79, 0.76, 0.14), 'a')
    ax = _axis(fig, page_size, rect, (0.145, 0.15, 0.76, 0.6), 'a')
    cax = _axis(fig, page_size, rect, (0.925, 0.18, 0.012, 0.54), 'a')
    frame = bundle['landscape'].sort_values('patient_order')
    rows = [('Response_benefit', 'Response benefit'), ('PFS_benefit', 'PFS benefit'), ('OS_benefit', 'OS benefit'), ('Response_observed', 'mRECIST response'), ('AFP_favorability', 'AFP favourability'), ('TBS_favorability', 'TBS favourability'), ('PVTT_favorability', 'No PVTT'), ('ALBI_favorability', 'ALBI favourability')]
    matrix = frame[[name for name, _ in rows]].to_numpy(float).T
    cmap = LinearSegmentedColormap.from_list('favourable', list(DIVERGING_COLORS)[::-1])
    image = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    for boundary in (36.5, 72.5):
        ax.axvline(boundary, color=TEXT, linewidth=0.45, alpha=0.62)
    ax.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=FONT['tick'])
    ax.set_xticks([0, 36, 72, 108], ['1', '37', '73', '109'], fontsize=FONT['tick'])
    ax.set_xlabel('Patients ordered by decreasing MRU benefit', fontsize=FONT['axis'], labelpad=1.5)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.tick_params(axis='x', length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    header.set_xlim(0, 109)
    header.set_ylim(0, 1)
    for group in MBP_ORDER:
        row = bundle['boundaries'].set_index('group').loc[group]
        start, end = (float(row['start']), float(row['end']))
        header.barh(0.5, end - start - 0.4, left=start + 0.2, height=0.72, color=MBP_COLORS[group], edgecolor='white', linewidth=0.45)
        header.text((start + end) / 2, 0.5, f"{group}  n={int(row['n'])}", ha='center', va='center', fontsize=FONT['annotation'], color='white', fontweight='bold')
    summary = bundle['mrecist_assoc'].iloc[0]
    header.text(108.5, 0.98, f"mRECIST  χ²={summary['chi_square']:.2f}; V={summary['Cramers_V']:.2f}; {_format_p(summary['chi_square_P'])}", ha='right', va='bottom', fontsize=FONT['annotation'], color=TEXT, clip_on=False)
    header.axis('off')
    colorbar = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
    colorbar.ax.set_yticklabels(['Less', '0.5', 'More'], fontsize=FONT['tick'])
    colorbar.outline.set_linewidth(0.45)
    return [header, ax, cax]

def draw_panel_b(fig, page_size, rect, bundle, registry):
    ax = _axis(fig, page_size, rect, (0.29, 0.16, 0.66, 0.73), 'b')
    frame = bundle['within_mrecist'].copy()
    order = [('PFS', 'Responder'), ('PFS', 'Nonresponder'), ('OS', 'Responder'), ('OS', 'Nonresponder')]
    labels = []
    y = np.arange(3, -1, -1)
    for yi, key in zip(y, order):
        row = frame.loc[frame['Endpoint'].eq(key[0]) & frame['mRECIST'].eq(key[1])].iloc[0]
        colour = ENDPOINT_COLORS[key[0]]
        estimate = float(row['HR_per_one_level_MBP_worsening'])
        low, high = (float(row['CI_lower']), float(row['CI_upper']))
        ax.errorbar(estimate, yi, xerr=[[estimate - low], [high - estimate]], fmt='o', ms=4.4, capsize=2.0, color=colour, ecolor=colour, elinewidth=1.1, markeredgecolor='white', markeredgewidth=0.45, zorder=3)
        ax.text(8.2, yi, f"{estimate:.2f} ({low:.2f}–{high:.2f}); {_format_p(row['P'])}", ha='left', va='center', fontsize=FONT['annotation'], color=TEXT)
        stratum = 'responder' if key[1] == 'Responder' else 'non-responder'
        labels.append(f"{key[0]} · {stratum}\nn={int(row['n'])}; events={int(row['events'])}")
    ax.axvline(1, color=REFERENCE, linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(0.7, 22)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.set_xticklabels(['1', '2', '5', '10', '20'])
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(-0.65, 3.65)
    ax.set_yticks(y, labels, fontsize=FONT['tick'])
    ax.set_xlabel('HR per one-level MBP worsening (95% CI)', fontsize=FONT['axis'], labelpad=1.5)
    ax.tick_params(axis='y', length=0, pad=3)
    _clean(ax, grid_axis='x')
    return [ax]

def draw_panel_c(fig, page_size, rect, bundle, registry):
    heat = _axis(fig, page_size, rect, (0.145, 0.14, 0.285, 0.79), 'c')
    fdr_ax = _axis(fig, page_size, rect, (0.455, 0.14, 0.08, 0.79), 'c')
    forest = _axis(fig, page_size, rect, (0.66, 0.16, 0.325, 0.75), 'c')
    pheno = bundle['phenotype']
    tests = bundle['phenotype_tests']
    variable_order = tests['Variable'].tolist()
    matrix = pheno.pivot(index='Variable', columns='MBP_group', values='Standardized_deviation').reindex(index=variable_order, columns=MBP_ORDER)
    cmap = LinearSegmentedColormap.from_list('phenotype', DIVERGING_COLORS)
    vmax = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy(float)))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = heat.imshow(matrix.to_numpy(float), aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    heat.set_yticks(range(len(variable_order)), variable_order, fontsize=FONT['tick'])
    heat.set_xticks(range(3), MBP_ORDER, fontsize=FONT['tick'])
    heat.tick_params(length=0, pad=1.4)
    for spine in heat.spines.values():
        spine.set_visible(False)
    domains = tests['Domain'].tolist()
    for idx in range(1, len(domains)):
        if domains[idx] != domains[idx - 1]:
            heat.axhline(idx - 0.5, color=REFERENCE, linewidth=0.45, alpha=0.75)
    heat.text(0.5, 1.02, 'Standardized deviation', transform=heat.transAxes, ha='center', va='bottom', fontsize=FONT['facet'], color=TEXT)
    fdr_values = -np.log10(np.clip(tests['P_FDR'].to_numpy(float), 1e-300, 1.0))
    y = np.arange(len(variable_order))
    fdr_ax.scatter(fdr_values, y, s=10, color=np.where(tests['P_FDR'].to_numpy(float) < 0.05, ENDPOINT_COLORS['Response'], REFERENCE), zorder=3)
    fdr_ax.axvline(-np.log10(0.05), color=REFERENCE, linestyle=(0, (3, 2)), linewidth=0.6)
    fdr_ax.set_ylim(len(variable_order) - 0.5, -0.5)
    fdr_ax.set_yticks([])
    fdr_ax.set_xlabel('−log10 FDR', fontsize=FONT['axis'], labelpad=1.2)
    fdr_ax.set_xlim(0, max(4.2, float(fdr_values.max()) + 0.35))
    _clean(fdr_ax, grid_axis='x')
    for yi, q, x in zip(y, tests['P_FDR'].to_numpy(float), fdr_values):
        if q < 0.05:
            fdr_ax.text(x + 0.08, yi, '<0.001' if q < 0.001 else f'{q:.3f}', ha='left', va='center', fontsize=FONT['annotation'], color=TEXT)
    mv = bundle['multivariable']
    covariate_order = mv.loc[mv['Endpoint'].eq('PFS'), 'Label'].tolist()
    yy = np.arange(len(covariate_order) - 1, -1, -1)
    offsets = {'PFS': 0.12, 'OS': -0.12}
    for endpoint in ['PFS', 'OS']:
        colour = ENDPOINT_COLORS[endpoint]
        for base_y, label in zip(yy, covariate_order):
            row = mv.loc[mv['Endpoint'].eq(endpoint) & mv['Label'].eq(label)].iloc[0]
            est, low, high = map(float, (row['HR'], row['CI_lower'], row['CI_upper']))
            py = base_y + offsets[endpoint]
            forest.errorbar(est, py, xerr=[[est - low], [high - est]], fmt='o', ms=3.5, capsize=1.5, color=colour, ecolor=colour, elinewidth=0.9, markeredgecolor='white', markeredgewidth=0.35, zorder=3)
            forest.text(6.1, py, f'{est:.2f} ({low:.2f}–{high:.2f})', ha='left', va='center', fontsize=FONT['annotation'], color=TEXT)
    forest.axvline(1, color=REFERENCE, linestyle=(0, (3, 2)), linewidth=0.6)
    forest.set_xscale('log')
    forest.set_xlim(0.55, 13.5)
    forest.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
    forest.set_xticklabels(['1', '2', '5', '10'])
    forest.xaxis.set_minor_locator(FixedLocator([]))
    forest.set_ylim(-0.55, len(covariate_order) - 0.45)
    forest.set_yticks(yy, covariate_order, fontsize=FONT['tick'])
    forest.set_xlabel('Adjusted HR (95% CI)', fontsize=FONT['axis'], labelpad=1.2)
    forest.tick_params(axis='y', length=0, pad=2)
    _clean(forest, grid_axis='x')
    legend = forest.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color=ENDPOINT_COLORS[e], label=e, markersize=3.6) for e in ['PFS', 'OS']], loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2, borderaxespad=0, columnspacing=0.6, handletextpad=0.25)
    registry.tag(legend, name='c-endpoint-legend', role='legend', panel='c')
    return [heat, fdr_ax, forest]

def _four_row_labels() -> list[tuple[str, str, str]]:
    return [('PFS', 'Pretreatment', 'PFS · baseline'), ('PFS', 'Post-mRECIST', 'PFS · +mRECIST'), ('OS', 'Pretreatment', 'OS · baseline'), ('OS', 'Post-mRECIST', 'OS · +mRECIST')]

def draw_panel_d(fig, page_size, rect, bundle, registry):
    outer = _axis(fig, page_size, rect, (0.1, 0.18, 0.25, 0.69), 'd')
    repeat = _axis(fig, page_size, rect, (0.405, 0.18, 0.25, 0.69), 'd')
    cutoff = _axis(fig, page_size, rect, (0.755, 0.18, 0.23, 0.69), 'd')
    rows = _four_row_labels()
    yy = np.arange(3, -1, -1)
    outer_folds = bundle['outer_folds']
    outer_summary = bundle['outer_summary']
    for yi, (endpoint, context, _) in zip(yy, rows):
        vals = outer_folds.loc[outer_folds['Endpoint'].eq(endpoint) & outer_folds['Context'].eq(context), 'Delta_Cindex'].sort_values().to_numpy(float)
        colour = ENDPOINT_COLORS[endpoint]
        outer.scatter(vals, yi + _offsets(len(vals), 0.15), s=7, color=colour, alpha=0.28, edgecolors='none', zorder=2)
        row = outer_summary.loc[outer_summary['Endpoint'].eq(endpoint) & outer_summary['Context'].eq(context)].iloc[0]
        est = float(row['patient_level_outer_Delta_Cindex'])
        low = float(row['Delta_Cindex_bootstrap_CI_lower'])
        high = float(row['Delta_Cindex_bootstrap_CI_upper'])
        outer.errorbar(est, yi, xerr=[[est - low], [high - est]], fmt='D', ms=3.8, capsize=1.7, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor='white', markeredgewidth=0.4, zorder=4)
        outer.text(0.535, yi, f'{est:.3f}\n[{low:.3f}, {high:.3f}]', ha='right', va='center', fontsize=FONT['annotation'], color=TEXT)
    outer.axvline(0, color=REFERENCE, linestyle=(0, (3, 2)), linewidth=0.6)
    outer.set_xlim(-0.1, 0.55)
    outer.xaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.2, 0.4]))
    outer.set_xticklabels(['−0.1', '0', '0.2', '0.4'])
    outer.set_ylim(-0.6, 3.6)
    outer.set_yticks(yy, [label for _, _, label in rows], fontsize=FONT['tick'])
    outer.set_xlabel('Outer-CV ΔC-index', fontsize=FONT['axis'], labelpad=1.2)
    outer.tick_params(axis='y', length=0, pad=2)
    outer.set_title('Outer CV', loc='left', fontsize=FONT['facet'], pad=2)
    _clean(outer, grid_axis='x')
    repeat_folds = bundle['repeat_folds']
    repeat_summary = bundle['repeat_summary']
    for yi, (endpoint, context, _) in zip(yy, rows):
        vals = repeat_folds.loc[repeat_folds['Endpoint'].eq(endpoint) & repeat_folds['Context'].eq(context), 'Delta_Cindex'].sort_values().to_numpy(float)
        colour = ENDPOINT_COLORS[endpoint]
        repeat.scatter(vals, yi + _offsets(len(vals), 0.16), s=8, color=colour, alpha=0.34, edgecolors='none')
        row = repeat_summary.loc[repeat_summary['Endpoint'].eq(endpoint) & repeat_summary['Context'].eq(context)].iloc[0]
        median = float(row['median_Delta_Cindex'])
        repeat.plot(median, yi, marker='D', ms=3.6, color=colour, markeredgecolor='white', markeredgewidth=0.35)
        repeat.text(0.335, yi, f"{median:.3f}; +{100 * float(row['fraction_Delta_gt_0']):.0f}%", ha='right', va='center', fontsize=FONT['annotation'], color=TEXT)
    repeat.axvline(0, color=REFERENCE, linestyle=(0, (3, 2)), linewidth=0.6)
    repeat.set_xlim(-0.12, 0.35)
    repeat.xaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.2]))
    repeat.set_xticklabels(['−0.1', '0', '0.2'])
    repeat.set_ylim(-0.6, 3.6)
    repeat.set_yticks(yy, ['', '', '', ''], fontsize=FONT['tick'])
    repeat.set_xlabel('Repeated-CV ΔC-index', fontsize=FONT['axis'], labelpad=1.2)
    repeat.tick_params(axis='y', length=0, pad=2)
    repeat.set_title('Repeated CV', loc='left', fontsize=FONT['facet'], pad=2)
    _clean(repeat, grid_axis='x')
    cut_values = bundle['cutoff_values']
    cut_summary = bundle['cutoff_summary']
    cut_rows = [('PFS', 'Lower tertile boundary', 'PFS · lower'), ('PFS', 'Upper tertile boundary', 'PFS · upper'), ('OS', 'Lower tertile boundary', 'OS · lower'), ('OS', 'Upper tertile boundary', 'OS · upper')]
    for yi, (endpoint, boundary, _) in zip(yy, cut_rows):
        vals = cut_values.loc[cut_values['Endpoint'].eq(endpoint) & cut_values['Tertile_boundary'].eq(boundary), 'Cutoff'].sort_values().to_numpy(float)
        colour = ENDPOINT_COLORS[endpoint]
        cutoff.scatter(vals, yi + _offsets(len(vals), 0.15), s=7, color=colour, alpha=0.28, edgecolors='none')
        row = cut_summary.loc[cut_summary['Endpoint'].eq(endpoint) & cut_summary['Boundary'].eq(boundary)].iloc[0]
        mean, sd = (float(row['Mean_cutoff']), float(row['SD_cutoff']))
        cutoff.errorbar(mean, yi, xerr=sd, fmt='D', ms=3.6, capsize=1.7, color=colour, ecolor=colour, elinewidth=1.0, markeredgecolor='white', markeredgewidth=0.35)
        cutoff.text(0.515, yi, f'{mean:.3f} ± {sd:.3f}', ha='center', va='center', fontsize=FONT['annotation'], color=TEXT)
    cutoff.set_xlim(0.31, 0.7)
    cutoff.xaxis.set_major_locator(FixedLocator([0.35, 0.5, 0.65]))
    cutoff.set_xticklabels(['0.35', '0.50', '0.65'])
    cutoff.set_ylim(-0.6, 3.6)
    cutoff.set_yticks(yy, ['PFS · L', 'PFS · U', 'OS · L', 'OS · U'], fontsize=FONT['tick'])
    cutoff.set_xlabel('Training MBP cutoff', fontsize=FONT['axis'], labelpad=1.2)
    cutoff.tick_params(axis='y', length=0, pad=2)
    cutoff.set_title('Cutoff stability', loc='left', fontsize=FONT['facet'], pad=2)
    _clean(cutoff, grid_axis='x')
    return [outer, repeat, cutoff]

def draw_panel_e(fig, page_size, rect, bundle, registry):
    forest = _axis(fig, page_size, rect, (0.22, 0.52, 0.41, 0.41), 'e')
    numbers = _axis(fig, page_size, rect, (0.68, 0.52, 0.3, 0.41), 'e')
    effects = bundle['excluded_effects']
    rows = [('Response', 'Primary MBP', 'Response · primary'), ('Response', 'Response-excluded MBP', 'Response · excluded'), ('PFS', 'Primary MBP', 'PFS · primary'), ('PFS', 'PFS-excluded MBP', 'PFS · excluded'), ('OS', 'Primary MBP', 'OS · primary'), ('OS', 'OS-excluded MBP', 'OS · excluded')]
    yy = np.arange(5, -1, -1)
    for yi, (endpoint, definition, _) in zip(yy, rows):
        row = effects.loc[effects['Endpoint'].eq(endpoint) & effects['Definition'].eq(definition)].iloc[0]
        est, low, high = map(float, (row['Estimate'], row['CI_lower'], row['CI_upper']))
        colour = ENDPOINT_COLORS[endpoint]
        forest.errorbar(est, yi, xerr=[[est - low], [high - est]], fmt='o', ms=3.4, capsize=1.4, color=colour, ecolor=colour, elinewidth=0.85, markeredgecolor='white', markeredgewidth=0.35)
        numbers.text(0.0, yi, f'{est:.2f} [{low:.2f}–{high:.2f}]', ha='left', va='center', fontsize=FONT['annotation'], color=TEXT)
    forest.axvline(1, color=REFERENCE, linestyle=(0, (3, 2)), linewidth=0.6)
    forest.set_xscale('log')
    forest.set_xlim(0.6, 32)
    forest.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    forest.set_xticklabels(['1', '2', '5', '10', '20'])
    forest.xaxis.set_minor_locator(FixedLocator([]))
    forest.set_ylim(-0.6, 5.6)
    forest.set_yticks(yy, [label for _, _, label in rows], fontsize=FONT['tick'])
    forest.set_xlabel('Adjusted OR / HR (95% CI)', fontsize=FONT['axis'], labelpad=1.1)
    forest.tick_params(axis='y', length=0, pad=2)
    _clean(forest, grid_axis='x')
    numbers.set_xlim(0, 1)
    numbers.set_ylim(-0.6, 5.6)
    numbers.axis('off')
    counts = bundle['excluded_counts']
    summary = bundle['excluded_agreement'].set_index('Endpoint_excluded')
    mini_axes = []
    for index, endpoint in enumerate(ENDPOINT_ORDER):
        ax = _axis(fig, page_size, rect, (0.1 + index * 0.3, 0.06, 0.24, 0.24), 'e')
        sub = counts.loc[counts['Endpoint_excluded'].eq(endpoint)]
        matrix = sub.pivot(index='Primary_MBP', columns='Endpoint_excluded_MBP', values='n').reindex(index=MBP_ORDER, columns=MBP_ORDER).to_numpy(int)
        im = ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list('agreement', ['#F7FBFF', '#98C8E1', '#617FC4']), vmin=0, vmax=37, aspect='equal')
        for i in range(3):
            for j in range(3):
                rgba = im.cmap(im.norm(matrix[i, j]))
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=FONT['cell'], color=_contrast_text(rgba), fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(3), ['I', 'II', 'III'], fontsize=FONT['tick'])
        ax.set_yticks(range(3), ['I', 'II', 'III'] if index == 0 else ['', '', ''], fontsize=FONT['tick'])
        ax.tick_params(length=0, pad=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        s = summary.loc[endpoint]
        ax.set_title(f"{endpoint}\n{100 * float(s['Exact_agreement']):.1f}% · κ={float(s['Quadratic_weighted_kappa']):.3f}", fontsize=FONT['facet'], pad=1.5)
        mini_axes.append(ax)
    return [forest, numbers, *mini_axes]

def _effect_heatmap(ax, frame, row_order, *, title, estimate_column='Estimate'):
    matrix = frame.pivot(index='Definition', columns='Endpoint', values=estimate_column).reindex(index=row_order, columns=ENDPOINT_ORDER)
    values = np.log2(matrix.to_numpy(float))
    cmap = LinearSegmentedColormap.from_list('positive_effect', ['#F7FBFF', '#79AAD1', '#7163C6', '#B74496'])
    norm = Normalize(vmin=0, vmax=max(7.0, float(np.nanmax(values))))
    im = ax.imshow(values, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = cmap(norm(values[i, j]))
            ax.text(j, i, f'{matrix.iloc[i, j]:.1f}', ha='center', va='center', fontsize=FONT['cell'], color=_contrast_text(rgba), fontweight='bold')
    ax.set_xticks(range(3), ENDPOINT_ORDER, fontsize=FONT['tick'])
    ax.set_yticks(range(len(row_order)), row_order, fontsize=FONT['tick'])
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc='left', fontsize=FONT['facet'], pad=1.5)
    return im

def draw_panel_f(fig, page_size, rect, bundle, registry):
    agreement = _axis(fig, page_size, rect, (0.31, 0.75, 0.65, 0.18), 'f')
    frame = bundle['weight_agreement']
    scheme_order = ['Equal weight (primary)', 'Response emphasis', 'PFS emphasis', 'OS emphasis']
    labels = ['Equal', 'Response', 'PFS', 'OS']
    yy = np.arange(3, -1, -1)
    for yi, scheme in zip(yy, scheme_order):
        row = frame.loc[frame['Weight_scheme'].eq(scheme)].iloc[0]
        agreement.plot([0, float(row['Exact_agreement'])], [yi, yi], color=GRID, linewidth=2.6, solid_capstyle='round')
        agreement.plot(float(row['Exact_agreement']), yi, marker='o', ms=3.6, color=ENDPOINT_COLORS['PFS'], markeredgecolor='white', markeredgewidth=0.35)
        agreement.text(1.075, yi, f"{100 * float(row['Exact_agreement']):.1f}% · κ={float(row['Quadratic_weighted_kappa']):.3f}", ha='right', va='center', fontsize=FONT['annotation'], color=TEXT)
    agreement.set_xlim(0.92, 1.08)
    agreement.set_ylim(-0.5, 3.5)
    agreement.set_yticks(yy, labels, fontsize=FONT['tick'])
    agreement.set_xticks([0.94, 0.97, 1.0], ['94', '97', '100'])
    agreement.set_xlabel('')
    agreement.tick_params(axis='y', length=0, pad=2)
    agreement.set_title('Weight perturbation', loc='left', fontsize=FONT['facet'], pad=1.5)
    _clean(agreement, grid_axis='x')
    weight_ax = _axis(fig, page_size, rect, (0.31, 0.39, 0.65, 0.25), 'f')
    cutoff_ax = _axis(fig, page_size, rect, (0.31, 0.06, 0.65, 0.24), 'f')
    _effect_heatmap(weight_ax, bundle['weight_effects'], ['Equal weight', 'Response emphasis', 'PFS emphasis', 'OS emphasis'], title='Extreme-group OR / HR')
    _effect_heatmap(cutoff_ax, bundle['cutoff_effects_category'], ['Median split', 'Tertiles (primary)', 'Quartiles'], title='Per-category OR / HR')
    return [agreement, weight_ax, cutoff_ax]

def _draw_panel(panel, fig, page_size, rect, bundle, registry):
    functions = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c, 'd': draw_panel_d, 'e': draw_panel_e, 'f': draw_panel_f}
    return functions[panel](fig, page_size, rect, bundle, registry)

def build_full_figure(bundle: Mapping[str, Any] | None=None):
    if bundle is None:
        bundle = load_inputs()
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = []
    for panel in 'abcdef':
        axes.extend(_draw_panel(panel, fig, (WIDTH_MM, HEIGHT_MM), FULL_LAYOUT[panel], bundle, registry))
        _add_panel_letter(fig, panel, FULL_LAYOUT[panel], registry)
    return (fig, registry, axes)

def build_panel_figure(panel: str, bundle: Mapping[str, Any] | None=None):
    panel = panel.lower().strip()
    if panel not in PANEL_SIZES_MM:
        raise ValueError(f'Unknown panel: {panel}')
    if bundle is None:
        bundle = load_inputs()
    width, height = PANEL_SIZES_MM[panel]
    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = _draw_panel(panel, fig, (width, height), (0, 0, width, height), bundle, registry)
    return (fig, registry, axes)

def _save(fig, stem: Path, *, width_mm: float, height_mm: float, formats=('pdf', 'svg', 'png')) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S3 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    for suffix in formats:
        path = stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, Any] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if suffix == 'png':
            kwargs['dpi'] = EXPORT_DPI
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
    numeric_path = write_json(bundle['numeric_qa'], OUTPUT_DIR / f"{get_value('figures.config.item_318')}{VERSION}.json")
    fig, registry, _ = build_full_figure(bundle)
    layout = audit_figure_layout(fig, f"{get_value('figures.config.item_317')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.0, check_collisions=True)
    layout_path = write_json(layout, OUTPUT_DIR / f"{get_value('figures.config.item_319')}{VERSION}.json")
    full_paths = _save(fig, FULL_STEM, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    plt.close(fig)
    panel_paths = []
    panel_audits = {}
    for panel in 'abcdef':
        panel_fig, panel_registry, _ = build_panel_figure(panel, bundle)
        width, height = PANEL_SIZES_MM[panel]
        panel_audits[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_320')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.0, check_collisions=True)
        stem = PANEL_DIR / f"{get_value('figures.config.item_320')}{panel}_Nature_{VERSION}"
        panel_paths.extend(_save(panel_fig, stem, width_mm=width, height_mm=height))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_321')}{VERSION}.json")
    outputs = full_paths + panel_paths + [numeric_path, layout_path, panel_audit_path]
    manifest = {'figure': 'Supplementary Figure S3', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [WIDTH_MM, HEIGHT_MM], 'source_mode': 'render-only frozen CSV snapshots; no KM/Cox/CV/bootstrap/ECDF/cutoff recomputation', 'panel_mapping': bundle['contract']['panel_mapping'], 'layout_summary': layout['summary'], 'numeric_qa': {'status': 'passed', 'report': str(numeric_path)}, 'file_qa': 'Run figureS3_file_qa.py with the bundled verifier Python.', 'outputs': [{'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size} for path in outputs]}
    manifest_path = write_json(manifest, OUTPUT_DIR / f"{get_value('figures.config.item_322')}{VERSION}.json")
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'numeric_qa': str(numeric_path), 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {p: r['summary'] for p, r in panel_audits.items()}}

def summarize_bundle(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    return {'figure': 'Supplementary Figure S3', 'version': VERSION, 'palette': PALETTE_ID, 'render_only': True, 'figure_size_mm': [WIDTH_MM, HEIGHT_MM], 'panels': {'a': {'patients': len(bundle['landscape']), 'MBP_groups': 3}, 'b': {'stored_HR_rows': len(bundle['within_mrecist']), 'KM_refit': False}, 'c': {'phenotype_variables': bundle['phenotype']['Variable'].nunique(), 'MV_effect_rows': len(bundle['multivariable'])}, 'd': {'outer_fold_rows': len(bundle['outer_folds']), 'repeated_fold_rows': len(bundle['repeat_folds']), 'cutoff_rows': len(bundle['cutoff_values'])}, 'e': {'stored_effect_rows': len(bundle['excluded_effects']), 'stored_confusion_cells': len(bundle['excluded_counts'])}, 'f': {'weight_schemes': len(bundle['weight_agreement']), 'weight_effect_rows': len(bundle['weight_effects']), 'cutoff_effect_rows': len(bundle['cutoff_effects_category'])}}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
