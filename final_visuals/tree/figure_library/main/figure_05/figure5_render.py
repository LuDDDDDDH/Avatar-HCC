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
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
PILOT_ROOT = Path(__file__).resolve().parent
STYLE_ROOT = PILOT_ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MBP_COLORS, MBP_LINESTYLES, MBP_MARKERS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = PILOT_ROOT / 'inputs'
VERSION = 'v001'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_278')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 210.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT_SIZES: Mapping[str, float] = {'base': 6.2, 'panel_label': 8.5, 'facet': 6.6, 'axis_label': 6.2, 'tick': 5.6, 'legend': 5.4, 'annotation': 5.5, 'small': 5.4, 'cell_primary': 5.9}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
GRID = '#D9E0E6'
LIGHT = NEUTRAL_COLORS['light_background']
MBP_ORDER = ['MBP-I', 'MBP-II', 'MBP-III']
MRECIST_ORDER = ['R', 'NR']
ENDPOINT_ORDER = ['Response', 'PFS', 'OS']
INPUT_FILES = {'group_boundaries': get_value('figures.config.item_143'), 'association_cells': get_value('figures.config.item_148'), 'association_summary': get_value('figures.config.item_147'), 'pfs_curve': get_value('figures.config.item_179'), 'pfs_patients': get_value('figures.config.item_180'), 'pfs_summary': get_value('figures.config.item_181'), 'os_curve': get_value('figures.config.item_156'), 'os_patients': get_value('figures.config.item_157'), 'os_summary': get_value('figures.config.item_158'), 'adjusted_association': get_value('figures.config.item_044'), 'within_mrecist': get_value('figures.config.item_255'), 'incremental_summary': get_value('figures.config.item_113'), 'incremental_folds': get_value('figures.config.item_112'), 'spline_curve': get_value('figures.config.item_072'), 'spline_stats': get_value('figures.config.item_073'), 'phenotype': get_value('figures.config.item_070')}
FULL_LAYOUT: Mapping[str, tuple[float, float, float, float]] = {'a': (7.0, 8.0, 63.0, 40.0), 'd': (76.0, 8.0, 98.0, 40.0), 'b': (7.0, 54.0, 80.5, 54.0), 'c': (93.5, 54.0, 80.5, 54.0), 'e': (7.0, 114.0, 66.0, 40.0), 'f': (79.0, 114.0, 95.0, 40.0), 'g': (7.0, 160.0, 91.0, 43.0), 'h': (104.0, 160.0, 70.0, 43.0)}
PANEL_SIZES_MM: Mapping[str, tuple[float, float]] = {'a': (72.0, 48.0), 'b': (88.0, 65.0), 'c': (88.0, 65.0), 'd': (115.0, 48.0), 'e': (82.0, 46.0), 'f': (116.0, 48.0), 'g': (120.0, 50.0), 'h': (95.0, 50.0)}

def file_sha256(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')

def _require_unique(frame: pd.DataFrame, keys: Sequence[str], expected: int, label: str) -> None:
    if len(frame) != int(expected):
        raise ValueError(f'{label} expected {expected} rows; found {len(frame)}')
    if frame.duplicated(list(keys)).any():
        duplicated = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
        raise ValueError(f'{label} contains duplicate keys:\n{duplicated}')

def _close(left: Sequence[float], right: Sequence[float], *, atol: float=1e-10) -> bool:
    return bool(np.allclose(np.asarray(left, float), np.asarray(right, float), atol=atol, rtol=1e-09))

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {key: pd.read_csv(INPUT_DIR / filename) for key, filename in INPUT_FILES.items()}
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    bundle['contract'] = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    bundle['numeric_qa'] = validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle['manifest']
    if manifest['palette_id'] != PALETTE_ID:
        raise ValueError('Input manifest palette does not match the locked shared palette')
    records = {entry['snapshot']: entry for entry in manifest['files']}
    file_checks: list[dict[str, Any]] = []
    for filename in INPUT_FILES.values():
        if filename not in records:
            raise ValueError(f'Input manifest does not register {filename}')
        path = INPUT_DIR / filename
        entry = records[filename]
        actual_hash = file_sha256(path)
        if actual_hash != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch for frozen input {filename}')
        frame = pd.read_csv(path)
        expected_shape = (int(entry['rows']), int(entry['columns']))
        if frame.shape != expected_shape:
            raise ValueError(f'Unexpected shape for {filename}: {frame.shape}; expected {expected_shape}')
        file_checks.append({'file': filename, 'shape': [int(frame.shape[0]), int(frame.shape[1])], 'sha256': actual_hash, 'pass': True})
    boundaries = bundle['group_boundaries']
    _require_unique(boundaries, ['group'], 3, 'MBP group boundaries')
    boundaries = boundaries.set_index('group').loc[MBP_ORDER]
    if boundaries['n'].astype(int).tolist() != [37, 36, 36]:
        raise ValueError('MBP group sizes differ from the locked 37/36/36 contract')
    if int(boundaries['n'].sum()) != 109:
        raise ValueError('MBP group sizes do not sum to 109')
    cells = bundle['association_cells']
    assoc = bundle['association_summary']
    _require_unique(cells, ['mRECIST', 'MBP'], 6, '5A cell summary')
    _require_unique(assoc.assign(_row=0), ['_row'], 1, '5A association summary')
    _require_columns(cells, ['mRECIST', 'MBP', 'Observed_n', 'Expected_n_under_independence', 'Row_percent', 'Column_percent', 'Pearson_residual', 'Adjusted_standardized_residual'], '5A cell summary')
    if set(cells['mRECIST']) != set(MRECIST_ORDER) or set(cells['MBP']) != set(MBP_ORDER):
        raise ValueError('5A does not contain the locked mRECIST by MBP cells')
    row_sums = cells.groupby('mRECIST')['Row_percent'].sum().reindex(MRECIST_ORDER)
    if not _close(row_sums, [100.0, 100.0], atol=1e-08):
        raise ValueError('5A row percentages do not sum to 100')
    group_counts = cells.groupby('MBP')['Observed_n'].sum().reindex(MBP_ORDER).astype(int)
    if group_counts.tolist() != [37, 36, 36]:
        raise ValueError('5A cell counts do not reproduce MBP group sizes')
    if int(cells['Observed_n'].sum()) != int(assoc.loc[0, 'n']) or int(assoc.loc[0, 'n']) != 109:
        raise ValueError('5A association total is not 109')
    chi_from_cells = float(np.square(cells['Pearson_residual'].to_numpy(float)).sum())
    if not np.isclose(chi_from_cells, float(assoc.loc[0, 'chi_square']), atol=1e-10):
        raise ValueError('5A Pearson residuals do not reproduce the stored chi-square')
    v_from_stored_chi = float(np.sqrt(float(assoc.loc[0, 'chi_square']) / 109.0))
    if not np.isclose(v_from_stored_chi, float(assoc.loc[0, 'Cramers_V']), atol=1e-10):
        raise ValueError("5A stored chi-square and Cramer's V are inconsistent")
    survival_checks: dict[str, Any] = {}
    for endpoint, curve_key, patient_key, summary_key, event_total in [('PFS', 'pfs_curve', 'pfs_patients', 'pfs_summary', 98), ('OS', 'os_curve', 'os_patients', 'os_summary', 67)]:
        curve = bundle[curve_key]
        patients = bundle[patient_key]
        summary = bundle[summary_key]
        _require_columns(curve, ['time_months', 'survival_probability', 'CI_lower', 'CI_upper', 'MBP'], f'5{endpoint} curve')
        _require_columns(patients, ['sample', 'MBP_raw', f'{endpoint}_time', f'{endpoint}_status', 'time_plot'], f'5{endpoint} patients')
        _require_unique(patients, ['sample'], 109, f'5{endpoint} patients')
        _require_unique(summary, ['MBP'], 3, f'5{endpoint} summary')
        if set(curve['MBP']) != set(MBP_ORDER) or set(summary['MBP']) != set(MBP_ORDER):
            raise ValueError(f'{endpoint} survival groups differ from the locked MBP order')
        if not ((curve['CI_lower'] <= curve['survival_probability']) & (curve['survival_probability'] <= curve['CI_upper'])).all():
            raise ValueError(f'{endpoint} stored curve lies outside its confidence limits')
        if not curve[['survival_probability', 'CI_lower', 'CI_upper']].apply(lambda col: col.between(0, 1)).all().all():
            raise ValueError(f'{endpoint} survival probabilities are outside [0, 1]')
        for group in MBP_ORDER:
            values = curve.loc[curve['MBP'].eq(group)].sort_values('time_months')
            if not values['time_months'].is_monotonic_increasing:
                raise ValueError(f'{endpoint} {group} curve is not time ordered')
            if not np.isclose(float(values.iloc[0]['time_months']), 0.0) or not np.isclose(float(values.iloc[0]['survival_probability']), 1.0):
                raise ValueError(f'{endpoint} {group} curve does not begin at (0, 1)')
            if (np.diff(values['survival_probability'].to_numpy(float)) > 1e-10).any():
                raise ValueError(f'{endpoint} {group} curve increases')
        observed_n = patients['MBP_raw'].value_counts().reindex(MBP_ORDER).astype(int)
        summary_n = summary.set_index('MBP')['n'].reindex(MBP_ORDER).astype(int)
        if not observed_n.equals(summary_n) or summary_n.tolist() != [37, 36, 36]:
            raise ValueError(f'{endpoint} patient and summary group sizes differ')
        event_column = f'{endpoint}_status'
        summary_event_column = f'{endpoint}_events'
        observed_events = patients.groupby('MBP_raw')[event_column].sum().reindex(MBP_ORDER).astype(int)
        summary_events = summary.set_index('MBP')[summary_event_column].reindex(MBP_ORDER).astype(int)
        if not observed_events.equals(summary_events) or int(summary_events.sum()) != event_total:
            raise ValueError(f'{endpoint} event counts differ from the frozen summary')
        if summary['global_logrank_P'].nunique() != 1:
            raise ValueError(f'{endpoint} global log-rank P is not constant across summary rows')
        survival_checks[endpoint] = {'n': int(summary_n.sum()), 'events': int(summary_events.sum()), 'group_n': summary_n.to_dict(), 'global_logrank_P': float(summary['global_logrank_P'].iloc[0]), 'pass': True}
    associations = bundle['adjusted_association']
    _require_unique(associations, ['Endpoint', 'Effect_type'], 6, '5D associations')
    if set(associations['Endpoint']) != {'Nonresponse', 'PFS', 'OS'}:
        raise ValueError('5D endpoints differ from the frozen contract')
    if set(associations['Effect_type']) != {'Unadjusted', 'Adjusted'}:
        raise ValueError('5D effect types differ from the frozen contract')
    if not ((associations['CI_lower'] <= associations['Estimate']) & (associations['Estimate'] <= associations['CI_upper'])).all():
        raise ValueError('5D estimates lie outside their intervals')
    expected_measure = {'Nonresponse': 'OR', 'PFS': 'HR', 'OS': 'HR'}
    if any((str(row.Measure) != expected_measure[str(row.Endpoint)] for row in associations.itertuples())):
        raise ValueError('5D effect measures do not match endpoint meanings')
    within = bundle['within_mrecist']
    _require_unique(within, ['mRECIST', 'Endpoint'], 4, '5E within-mRECIST effects')
    if set(within['mRECIST']) != {'Responder', 'Nonresponder'} or set(within['Endpoint']) != {'PFS', 'OS'}:
        raise ValueError('5E strata or endpoints differ from the frozen contract')
    if not ((within['CI_lower'] <= within['HR_per_MBP_level_worsening']) & (within['HR_per_MBP_level_worsening'] <= within['CI_upper'])).all():
        raise ValueError('5E HRs lie outside their intervals')
    summary = bundle['incremental_summary']
    folds = bundle['incremental_folds']
    _require_unique(summary, ['Endpoint', 'Context'], 4, '5F incremental summary')
    _require_unique(folds, ['Endpoint', 'Context', 'fold_index'], 100, '5F stored folds')
    if set(summary['Endpoint']) != {'PFS', 'OS'} or set(summary['Context']) != {'Pretreatment', 'Post-mRECIST'}:
        raise ValueError('5F endpoint/context cells differ from the frozen contract')
    if not _close(folds['Delta_Cindex'], folds['Cindex_with_MBP'] - folds['Cindex_without_MBP'], atol=1e-12):
        raise ValueError('5F stored paired delta is not with-MBP minus without-MBP')
    reconstructed = folds.groupby(['Endpoint', 'Context'], as_index=False).agg(n_check=('Delta_Cindex', 'size'), base_mean_check=('Cindex_without_MBP', 'mean'), base_sd_check=('Cindex_without_MBP', 'std'), extended_mean_check=('Cindex_with_MBP', 'mean'), extended_sd_check=('Cindex_with_MBP', 'std'), delta_mean_check=('Delta_Cindex', 'mean'), delta_sd_check=('Delta_Cindex', 'std'))
    linked = summary.merge(reconstructed, on=['Endpoint', 'Context'], validate='one_to_one')
    checks = [('Cindex_without_MBP_mean', 'base_mean_check'), ('Cindex_without_MBP_SD', 'base_sd_check'), ('Cindex_with_MBP_mean', 'extended_mean_check'), ('Cindex_with_MBP_SD', 'extended_sd_check'), ('Delta_Cindex_mean', 'delta_mean_check'), ('Delta_Cindex_SD', 'delta_sd_check')]
    for stored, check in checks:
        if not _close(linked[stored], linked[check], atol=1e-12):
            raise ValueError(f'5F stored summary differs from frozen folds for {stored}')
    if not (linked['n_valid_CV_folds'].astype(int) == linked['n_check'].astype(int)).all() or not (linked['n_check'] == 25).all():
        raise ValueError('5F expected 25 valid repeated-CV folds in each cell')
    curve = bundle['spline_curve']
    stats = bundle['spline_stats']
    _require_unique(stats, ['Endpoint'], 2, '5G spline stats')
    if set(curve['Endpoint']) != {'PFS', 'OS'} or set(stats['Endpoint']) != {'PFS', 'OS'}:
        raise ValueError('5G endpoints differ from the frozen contract')
    if curve.groupby('Endpoint').size().to_dict() != {'OS': 160, 'PFS': 160}:
        raise ValueError('5G expected 160 stored curve points per endpoint')
    if not ((curve['HR_lower'] <= curve['HR']) & (curve['HR'] <= curve['HR_upper']) & (curve['HR_lower'] > 0)).all():
        raise ValueError('5G HR curve or intervals are invalid')
    if not (stats['spline_df'].astype(int) == 3).all() or not (stats['fit_mode'] == 'unpenalized_centered_natural_cubic_spline').all():
        raise ValueError('5G spline specification differs from the locked contract')
    for endpoint in ['PFS', 'OS']:
        source_ref = curve.loc[curve['Endpoint'].eq(endpoint), 'score_reference'].unique()
        stat_ref = float(stats.loc[stats['Endpoint'].eq(endpoint), 'score_reference'].iloc[0])
        if len(source_ref) != 1 or not np.isclose(float(source_ref[0]), stat_ref, atol=1e-12):
            raise ValueError(f'5G {endpoint} reference score differs between frozen tables')
    phenotype = bundle['phenotype']
    _require_unique(phenotype, ['Variable', 'MBP_group'], 21, '5H phenotype profile')
    if phenotype['Variable'].nunique() != 7 or set(phenotype['MBP_group']) != set(MBP_ORDER):
        raise ValueError('5H expected seven variables across all three MBP groups')
    if not ((phenotype['Lower'] <= phenotype['Effect']) & (phenotype['Effect'] <= phenotype['Upper'])).all():
        raise ValueError('5H effects lie outside their intervals')
    if phenotype.groupby('MBP_group')['n'].first().reindex(MBP_ORDER).astype(int).tolist() != [37, 36, 36]:
        raise ValueError('5H group sizes differ from the locked contract')
    return {'figure': 'Figure 5', 'version': VERSION, 'source_mode': 'frozen-table integrity checks only; no statistical estimation', 'files': file_checks, 'scientific_checks': {'mbp_group_n': {group: int(boundaries.loc[group, 'n']) for group in MBP_ORDER}, '5A': {'n': 109, 'chi_square_from_stored_residual_identity': chi_from_cells, 'stored_chi_square': float(assoc.loc[0, 'chi_square']), 'stored_cramers_v': float(assoc.loc[0, 'Cramers_V']), 'pass': True}, '5B_5C': survival_checks, '5D': {'rows': 6, 'pass': True}, '5E': {'cells': 4, 'pass': True}, '5F': {'cells': 4, 'stored_folds_per_cell': 25, 'paired_arithmetic_pass': True}, '5G': {'curve_points_per_endpoint': 160, 'spline_df': 3, 'pass': True}, '5H': {'variables': 7, 'groups': 3, 'pass': True}}, 'pass': True}

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT_SIZES['base'], 'axes.labelsize': FONT_SIZES['axis_label'], 'axes.titlesize': FONT_SIZES['facet'], 'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'], 'legend.fontsize': FONT_SIZES['legend'], 'legend.title_fontsize': FONT_SIZES['legend'], 'axes.linewidth': 0.55, 'lines.linewidth': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def _clean_axis(ax: mpl.axes.Axes, *, grid_axis: str | None=None) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.55)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(direction='out', width=0.55, length=2.2, pad=1.8)
    ax.grid(False)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.35, alpha=0.72, zorder=0)

def _format_p(value: float, *, prefix: str='P') -> str:
    p = float(value)
    if p < 0.001:
        return f'{prefix}<0.001'
    return f'{prefix}={p:.3f}'

def _text_colour_for_fill(fill: Sequence[float]) -> str:
    red, green, blue = fill[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return 'white' if luminance < 0.52 else TEXT

def _add_axis(fig: mpl.figure.Figure, page_size: tuple[float, float], panel_rect: tuple[float, float, float, float], rel_rect: tuple[float, float, float, float], panel: str) -> mpl.axes.Axes:
    page_width, page_height = page_size
    left, top, width, height = panel_rect
    rel_left, rel_bottom, rel_width, rel_height = rel_rect
    abs_left = left + rel_left * width
    abs_bottom = page_height - top - height + rel_bottom * height
    ax = fig.add_axes([abs_left / page_width, abs_bottom / page_height, rel_width * width / page_width, rel_height * height / page_height])
    ax._qa_meta = {'panel': panel}
    return ax

def _tag(registry: ArtistRegistry, artist: Any, *, name: str, role: str, panel: str) -> Any:
    return registry.tag(artist, name=name, role=role, panel=panel)

def _add_panel_letter(fig: mpl.figure.Figure, panel: str, rect: tuple[float, float, float, float], registry: ArtistRegistry) -> None:
    left, top, _, _ = rect
    artist = fig.text(max(4.5, left - 1.5) / FIGURE_WIDTH_MM, 1.0 - max(3.5, top - 1.2) / FIGURE_HEIGHT_MM, panel, ha='left', va='top', fontsize=FONT_SIZES['panel_label'], fontweight='bold', color=TEXT)
    _tag(registry, artist, name=f'panel-{panel}-label', role='panel_label', panel=panel)

def draw_panel_5a(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.21, 0.19, 0.64, 0.62), 'a')
    cax = _add_axis(fig, page_size, rect, (0.88, 0.21, 0.035, 0.58), 'a')
    cells = bundle['association_cells']
    summary = bundle['association_summary'].iloc[0]
    pivot = cells.pivot(index='mRECIST', columns='MBP', values='Adjusted_standardized_residual').loc[MRECIST_ORDER, MBP_ORDER]
    limit = max(2.0, float(np.ceil(np.nanmax(np.abs(pivot.to_numpy(float))) * 2.0) / 2.0))
    cmap = LinearSegmentedColormap.from_list('association_residual', list(reversed(DIVERGING_COLORS)), N=256)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(pivot.to_numpy(float), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(3), ['MBP-I\nHigh benefit', 'MBP-II\nIntermediate', 'MBP-III\nLow benefit'])
    ax.set_yticks([0, 1], ['Responder\n(n=53)', 'Nonresponder\n(n=56)'])
    for tick, group in zip(ax.get_xticklabels(), MBP_ORDER):
        tick.set_color(MBP_COLORS[group])
        tick.set_fontweight('bold')
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='both', which='major', length=0, pad=2.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, mrecist in enumerate(MRECIST_ORDER):
        for j, group in enumerate(MBP_ORDER):
            row = cells.loc[cells['mRECIST'].eq(mrecist) & cells['MBP'].eq(group)].iloc[0]
            fill = cmap(norm(float(row['Adjusted_standardized_residual'])))
            color = _text_colour_for_fill(fill)
            primary = ax.text(j, i - 0.09, f"{int(row['Observed_n'])} ({float(row['Row_percent']):.1f}%)", ha='center', va='center', fontsize=FONT_SIZES['cell_primary'], fontweight='bold', color=color)
            secondary = ax.text(j, i + 0.2, f"ASR {float(row['Adjusted_standardized_residual']):+.2f}", ha='center', va='center', fontsize=FONT_SIZES['small'], color=color)
            _tag(registry, primary, name=f'a-{mrecist}-{group}-count', role='value', panel='a')
            _tag(registry, secondary, name=f'a-{mrecist}-{group}-asr', role='small_value', panel='a')
    annotation = ax.text(0.0, 1.08, f"n={int(summary['n'])}; V={float(summary['Cramers_V']):.2f}; {_format_p(float(summary['chi_square_P']))}", transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['annotation'], color=TEXT)
    _tag(registry, annotation, name='a-association-summary', role='value', panel='a')
    colorbar = fig.colorbar(image, cax=cax, orientation='vertical')
    colorbar.set_label('Adjusted standardized residual', fontsize=FONT_SIZES['small'], labelpad=2.2)
    colorbar.ax.tick_params(labelsize=FONT_SIZES['small'], width=0.45, length=1.8, pad=1.5)
    colorbar.outline.set_linewidth(0.4)
    return [ax, cax]

def _step_value_at(curve: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    ordered = curve.sort_values('time_months')
    curve_times = ordered['time_months'].to_numpy(float)
    values = ordered['survival_probability'].to_numpy(float)
    indices = np.searchsorted(curve_times, np.asarray(times, float), side='right') - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]

def draw_survival_panel(panel: str, endpoint: str, fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], curve: pd.DataFrame, patients: pd.DataFrame, summary: pd.DataFrame, registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.14, 0.31, 0.83, 0.52), panel)
    axrisk = _add_axis(fig, page_size, rect, (0.14, 0.11, 0.83, 0.15), panel)
    xmax = 48.0
    ticks = np.array([0.0, 12.0, 24.0, 36.0, 48.0])
    handles: list[Line2D] = []
    labels: list[str] = []
    summary_indexed = summary.set_index('MBP').loc[MBP_ORDER]
    for group in MBP_ORDER:
        values = curve.loc[curve['MBP'].eq(group)].sort_values('time_months')
        x = values['time_months'].to_numpy(float)
        survival = values['survival_probability'].to_numpy(float)
        lower = values['CI_lower'].to_numpy(float)
        upper = values['CI_upper'].to_numpy(float)
        color = MBP_COLORS[group]
        ax.fill_between(x, lower, upper, step='post', color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.step(x, survival, where='post', color=color, linewidth=1.15, linestyle=MBP_LINESTYLES[group], zorder=3)
        censored = patients.loc[patients['MBP_raw'].eq(group) & patients[f'{endpoint}_status'].eq(0), 'time_plot'].to_numpy(float)
        if len(censored):
            y_censor = _step_value_at(values, censored)
            ax.plot(censored, y_censor, linestyle='none', marker='|', markersize=3.2, markeredgewidth=0.65, color=color, zorder=4)
        row = summary_indexed.loc[group]
        handles.append(Line2D([0], [0], color=color, linewidth=1.15, linestyle=MBP_LINESTYLES[group]))
        labels.append(f"{group} ({int(row['n'])}; {float(row[f'median_{endpoint}_months']):.1f})")
    legend = ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.045), ncol=3, frameon=False, borderaxespad=0.0, handlelength=1.25, handletextpad=0.25, columnspacing=0.55, fontsize=FONT_SIZES['legend'], title='n; median (months)')
    _tag(registry, legend, name=f'{panel}-mbp-legend', role='legend', panel=panel)
    p_value = float(summary['global_logrank_P'].iloc[0])
    p_text = ax.text(0.02, 0.05, f'Global log-rank {_format_p(p_value)}', transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['annotation'], color=TEXT, bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88})
    _tag(registry, p_text, name=f'{panel}-logrank', role='value', panel=panel)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ticks)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel('PFS probability' if endpoint == 'PFS' else 'OS probability')
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _clean_axis(ax)
    axrisk.set_xlim(0, xmax)
    axrisk.set_ylim(-0.65, 2.65)
    axrisk.set_xticks(ticks)
    axrisk.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{int(value)}'))
    axrisk.set_xlabel('Time (months)', labelpad=1.4)
    axrisk.set_yticks([])
    for spine in axrisk.spines.values():
        spine.set_visible(False)
    axrisk.tick_params(axis='x', width=0.5, length=2.0, pad=1.4)
    header = axrisk.text(-0.035, 1.04, 'No. at risk', transform=axrisk.transAxes, ha='right', va='bottom', fontsize=FONT_SIZES['small'], fontweight='bold', color=TEXT)
    _tag(registry, header, name=f'{panel}-risk-header', role='direct_label', panel=panel)
    row_y = {'MBP-I': 2.0, 'MBP-II': 1.0, 'MBP-III': 0.0}
    for group in MBP_ORDER:
        y = row_y[group]
        group_label = axrisk.text(-0.035, (y + 0.65) / 3.3, group, transform=axrisk.transAxes, ha='right', va='center', fontsize=FONT_SIZES['small'], fontweight='bold', color=MBP_COLORS[group])
        _tag(registry, group_label, name=f'{panel}-risk-label-{group}', role='direct_label', panel=panel)
        group_patients = patients.loc[patients['MBP_raw'].eq(group)]
        for index, time in enumerate(ticks):
            count = int((group_patients['time_plot'] >= time).sum())
            value_text = axrisk.text(time, y, str(count), ha='center', va='center', fontsize=FONT_SIZES['small'], color=TEXT)
            _tag(registry, value_text, name=f'{panel}-risk-{group}-{index}', role='small_value', panel=panel)
    return [ax, axrisk]

def draw_panel_5d(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.22, 0.18, 0.39, 0.65), 'd')
    table = _add_axis(fig, page_size, rect, (0.64, 0.18, 0.35, 0.65), 'd')
    frame = bundle['adjusted_association']
    endpoint_labels = {'Nonresponse': 'mRECIST nonresponse', 'PFS': 'PFS', 'OS': 'OS'}
    endpoint_colors = {'Nonresponse': ENDPOINT_COLORS['Response'], 'PFS': ENDPOINT_COLORS['PFS'], 'OS': ENDPOINT_COLORS['OS']}
    y_base = {'Nonresponse': 2.0, 'PFS': 1.0, 'OS': 0.0}
    offsets = {'Unadjusted': 0.13, 'Adjusted': -0.13}
    ax.axvline(1.0, color=REFERENCE, linestyle='--', linewidth=0.65, zorder=0)
    for y in [0.0, 1.0, 2.0]:
        ax.axhline(y, color=GRID, linewidth=0.35, zorder=0)
    for row in frame.itertuples(index=False):
        endpoint = str(row.Endpoint)
        effect_type = str(row.Effect_type)
        y = y_base[endpoint] + offsets[effect_type]
        estimate, lower, upper = (float(row.Estimate), float(row.CI_lower), float(row.CI_upper))
        if effect_type == 'Adjusted':
            color = endpoint_colors[endpoint]
            marker = 'D'
            face = color
            edge = color
            width = 1.05
            size = 18
            zorder = 4
        else:
            color = '#98A2AB'
            marker = 'o'
            face = 'white'
            edge = REFERENCE
            width = 0.85
            size = 15
            zorder = 2
        ax.plot([lower, upper], [y, y], color=color, linewidth=width, zorder=zorder)
        ax.plot([lower, lower], [y - 0.04, y + 0.04], color=color, linewidth=0.65, zorder=zorder)
        ax.plot([upper, upper], [y - 0.04, y + 0.04], color=color, linewidth=0.65, zorder=zorder)
        ax.scatter(estimate, y, s=size, marker=marker, facecolor=face, edgecolor=edge, linewidth=0.7, zorder=5)
    ax.set_xscale('log')
    ax.set_xlim(0.4, 34.0)
    ax.set_ylim(-0.48, 2.48)
    ax.set_yticks([2, 1, 0], [endpoint_labels['Nonresponse'], endpoint_labels['PFS'], endpoint_labels['OS']])
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.set_xlabel('OR / HR per MBP level (log scale)')
    ax.tick_params(axis='y', length=0, pad=2.5)
    _clean_axis(ax)
    handles = [Line2D([0], [0], marker='o', color='#98A2AB', markerfacecolor='white', markeredgecolor=REFERENCE, linewidth=0.85, markersize=3.4, label='Unadjusted'), Line2D([0], [0], marker='D', color=TEXT, markerfacecolor=TEXT, linewidth=0.85, markersize=3.2, label='Adjusted')]
    legend = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=2, borderaxespad=0, handlelength=1.4, handletextpad=0.3, columnspacing=0.8)
    _tag(registry, legend, name='d-effect-legend', role='legend', panel='d')
    table.set_axis_off()
    table.set_xlim(0, 1)
    table.set_ylim(-0.48, 2.48)
    heading = table.text(0.0, 2.56, 'Adjusted effect (95% CI); P', ha='left', va='bottom', fontsize=FONT_SIZES['small'], fontweight='bold', color=TEXT)
    _tag(registry, heading, name='d-table-heading', role='direct_label', panel='d')
    for endpoint in ['Nonresponse', 'PFS', 'OS']:
        row = frame.loc[frame['Endpoint'].eq(endpoint) & frame['Effect_type'].eq('Adjusted')].iloc[0]
        text = table.text(0.0, y_base[endpoint], f"{float(row['Estimate']):.2f} ({float(row['CI_lower']):.2f}-{float(row['CI_upper']):.2f}); {_format_p(float(row['P']))}", ha='left', va='center', fontsize=FONT_SIZES['small'], color=endpoint_colors[endpoint])
        _tag(registry, text, name=f'd-adjusted-text-{endpoint}', role='value', panel='d')
    return [ax, table]

def draw_panel_5e(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.17, 0.19, 0.68, 0.62), 'e')
    cax = _add_axis(fig, page_size, rect, (0.89, 0.21, 0.035, 0.58), 'e')
    frame = bundle['within_mrecist']
    endpoints = ['PFS', 'OS']
    strata = ['Responder', 'Nonresponder']
    matrix = np.full((2, 2), np.nan)
    for i, endpoint in enumerate(endpoints):
        for j, stratum in enumerate(strata):
            row = frame.loc[frame['Endpoint'].eq(endpoint) & frame['mRECIST'].eq(stratum)].iloc[0]
            matrix[i, j] = np.log2(float(row['HR_per_MBP_level_worsening']))
    limit = max(1.0, float(np.ceil(np.nanmax(np.abs(matrix)) * 2.0) / 2.0))
    cmap = LinearSegmentedColormap.from_list('within_mrecist_effect', DIVERGING_COLORS, N=256)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    ax.set_xticks([0, 1], ['Responder', 'Nonresponder'])
    ax.set_yticks([0, 1], endpoints)
    for tick, endpoint in zip(ax.get_yticklabels(), endpoints):
        tick.set_color(ENDPOINT_COLORS[endpoint])
        tick.set_fontweight('bold')
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.25)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='both', which='major', length=0, pad=2.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, endpoint in enumerate(endpoints):
        for j, stratum in enumerate(strata):
            row = frame.loc[frame['Endpoint'].eq(endpoint) & frame['mRECIST'].eq(stratum)].iloc[0]
            fill = cmap(norm(matrix[i, j]))
            color = _text_colour_for_fill(fill)
            lines = [(i - 0.19, f"HR {float(row['HR_per_MBP_level_worsening']):.2f}", FONT_SIZES['cell_primary'], 'bold', 'primary'), (i + 0.02, f"95% CI {float(row['CI_lower']):.2f}-{float(row['CI_upper']):.2f}", FONT_SIZES['small'], 'normal', 'ci'), (i + 0.22, f"{_format_p(float(row['P']))}; n={int(row['n'])}; e={int(row['events'])}", FONT_SIZES['small'], 'normal', 'detail')]
            for y, label, fontsize, weight, suffix in lines:
                artist = ax.text(j, y, label, ha='center', va='center', fontsize=fontsize, fontweight=weight, color=color)
                _tag(registry, artist, name=f'e-{endpoint}-{stratum}-{suffix}', role='small_value', panel='e')
    colorbar = fig.colorbar(image, cax=cax, orientation='vertical')
    colorbar.set_label('log2 HR per MBP level', fontsize=FONT_SIZES['small'], labelpad=2.0)
    colorbar.ax.tick_params(labelsize=FONT_SIZES['small'], width=0.45, length=1.8, pad=1.5)
    colorbar.outline.set_linewidth(0.4)
    return [ax, cax]

def draw_panel_5f(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.25, 0.19, 0.49, 0.64), 'f')
    table = _add_axis(fig, page_size, rect, (0.76, 0.19, 0.23, 0.64), 'f')
    frame = bundle['incremental_summary']
    row_order = [('PFS', 'Pretreatment'), ('PFS', 'Post-mRECIST'), ('OS', 'Pretreatment'), ('OS', 'Post-mRECIST')]
    row_labels = {('PFS', 'Pretreatment'): 'PFS | baseline', ('PFS', 'Post-mRECIST'): 'PFS | +mRECIST', ('OS', 'Pretreatment'): 'OS | baseline', ('OS', 'Post-mRECIST'): 'OS | +mRECIST'}
    y_positions = {key: 3 - index for index, key in enumerate(row_order)}
    ax.axvline(0.5, color=REFERENCE, linestyle='--', linewidth=0.65, zorder=0)
    for endpoint, context in row_order:
        y = float(y_positions[endpoint, context])
        row = frame.loc[frame['Endpoint'].eq(endpoint) & frame['Context'].eq(context)].iloc[0]
        base = float(row['Cindex_without_MBP_mean'])
        extended = float(row['Cindex_with_MBP_mean'])
        color = ENDPOINT_COLORS[endpoint]
        ax.axhline(y, color=GRID, linewidth=0.35, zorder=0)
        ax.plot([base, extended], [y, y], color=color, linewidth=1.45, alpha=0.78, zorder=2)
        ax.scatter(base, y, s=22, facecolor='white', edgecolor=color, linewidth=0.9, zorder=4)
        ax.scatter(extended, y, s=25, facecolor=color, edgecolor='white', linewidth=0.5, zorder=5)
        base_label = ax.text(base, y + 0.17, f'{base:.3f}', ha='center', va='bottom', fontsize=FONT_SIZES['small'], color=MUTED)
        ext_label = ax.text(extended, y + 0.17, f'{extended:.3f}', ha='center', va='bottom', fontsize=FONT_SIZES['small'], fontweight='bold', color=color)
        _tag(registry, base_label, name=f'f-{endpoint}-{context}-base', role='small_value', panel='f')
        _tag(registry, ext_label, name=f'f-{endpoint}-{context}-extended', role='small_value', panel='f')
    ax.set_xlim(0.47, 0.79)
    ax.set_ylim(-0.48, 3.48)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_yticks([y_positions[key] for key in row_order], [row_labels[key] for key in row_order])
    for tick, key in zip(ax.get_yticklabels(), row_order):
        tick.set_color(ENDPOINT_COLORS[key[0]])
    ax.set_xlabel('Repeated-CV Harrell C-index')
    ax.tick_params(axis='y', length=0, pad=2.5)
    _clean_axis(ax)
    legend = ax.legend(handles=[Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor=TEXT, markersize=3.6, label='Without MBP'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=3.6, label='+ MBP')], loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=2, borderaxespad=0, handletextpad=0.35, columnspacing=0.8)
    _tag(registry, legend, name='f-model-legend', role='legend', panel='f')
    table.set_axis_off()
    table.set_xlim(0, 1)
    table.set_ylim(-0.48, 3.48)
    heading = table.text(0.0, 3.56, 'delta C; LRT P', ha='left', va='bottom', fontsize=FONT_SIZES['small'], fontweight='bold', color=TEXT)
    _tag(registry, heading, name='f-table-heading', role='direct_label', panel='f')
    for endpoint, context in row_order:
        row = frame.loc[frame['Endpoint'].eq(endpoint) & frame['Context'].eq(context)].iloc[0]
        p_value = float(row['LRT_P'])
        p_label = '<0.001' if p_value < 0.001 else f'={p_value:.3f}'
        detail = table.text(0.0, y_positions[endpoint, context], f"{float(row['Delta_Cindex_mean']):+.3f}; {p_label}", ha='left', va='center', fontsize=FONT_SIZES['small'], color=ENDPOINT_COLORS[endpoint])
        _tag(registry, detail, name=f'f-{endpoint}-{context}-detail', role='value', panel='f')
    return [ax, table]

def draw_panel_5g(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    axes = [_add_axis(fig, page_size, rect, (0.11, 0.19, 0.38, 0.64), 'g'), _add_axis(fig, page_size, rect, (0.59, 0.19, 0.38, 0.64), 'g')]
    curve = bundle['spline_curve']
    stats = bundle['spline_stats'].set_index('Endpoint')
    for index, (ax, endpoint) in enumerate(zip(axes, ['PFS', 'OS'])):
        values = curve.loc[curve['Endpoint'].eq(endpoint)].sort_values('score')
        row = stats.loc[endpoint]
        color = ENDPOINT_COLORS[endpoint]
        x = values['score'].to_numpy(float)
        ax.fill_between(x, values['HR_lower'].to_numpy(float), values['HR_upper'].to_numpy(float), color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(x, values['HR'].to_numpy(float), color=color, linewidth=1.2, zorder=3)
        ax.axhline(1.0, color=REFERENCE, linestyle='--', linewidth=0.65, zorder=0)
        ax.axvline(float(row['score_reference']), color='#ABB4BC', linestyle=':', linewidth=0.65, zorder=0)
        ax.set_yscale('log')
        ax.set_xlim(0, 1)
        ax.set_ylim(0.025, 30)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.set_xlabel('MRU benefit score')
        if index == 0:
            ax.set_ylabel('Hazard ratio vs median')
        else:
            ax.tick_params(axis='y', labelleft=False)
        _clean_axis(ax)
        facet = ax.text(0.5, 1.06, endpoint, transform=ax.transAxes, ha='center', va='bottom', fontsize=FONT_SIZES['facet'], fontweight='bold', color=color)
        stats_text = ax.text(0.98, 0.97, f"Overall {_format_p(float(row['P_overall']))}\nNonlinear {_format_p(float(row['P_nonlinear']))}", transform=ax.transAxes, ha='right', va='top', fontsize=FONT_SIZES['small'], color=TEXT, bbox={'boxstyle': 'round,pad=0.10', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.84})
        _tag(registry, facet, name=f'g-{endpoint}-facet', role='direct_label', panel='g')
        _tag(registry, stats_text, name=f'g-{endpoint}-stats', role='value', panel='g')
    return axes

def draw_panel_5h(fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    ax = _add_axis(fig, page_size, rect, (0.25, 0.18, 0.72, 0.66), 'h')
    frame = bundle['phenotype']
    variables = list(dict.fromkeys(frame['Variable'].tolist()))
    y_positions = {variable: len(variables) - 1 - index for index, variable in enumerate(variables)}
    offsets = {'MBP-I': 0.2, 'MBP-II': 0.0, 'MBP-III': -0.2}
    ax.axvline(0.0, color=REFERENCE, linestyle='--', linewidth=0.65, zorder=0)
    for variable in variables:
        ax.axhline(y_positions[variable], color=GRID, linewidth=0.35, zorder=0)
    for row in frame.itertuples(index=False):
        group = str(row.MBP_group)
        y = y_positions[str(row.Variable)] + offsets[group]
        color = MBP_COLORS[group]
        ax.plot([float(row.Lower), float(row.Upper)], [y, y], color=color, linewidth=0.95, zorder=2)
        ax.scatter(float(row.Effect), y, s=18, marker=MBP_MARKERS[group], facecolor=color, edgecolor='white', linewidth=0.45, zorder=3)
    ax.set_xlim(-0.72, 0.78)
    ax.set_ylim(-0.52, len(variables) - 0.48)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([y_positions[value] for value in variables], variables)
    ax.set_xlabel('Standardized deviation from cohort mean')
    ax.tick_params(axis='y', length=0, pad=2.5)
    _clean_axis(ax)
    legend = ax.legend(handles=[Line2D([0], [0], marker=MBP_MARKERS[group], color=MBP_COLORS[group], markerfacecolor=MBP_COLORS[group], markeredgecolor='white', linewidth=0.9, markersize=3.5, label=group) for group in MBP_ORDER], loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=3, borderaxespad=0, handlelength=1.1, handletextpad=0.25, columnspacing=0.55)
    _tag(registry, legend, name='h-mbp-legend', role='legend', panel='h')
    return [ax]

def _draw_panel(panel: str, fig: mpl.figure.Figure, page_size: tuple[float, float], rect: tuple[float, float, float, float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> list[mpl.axes.Axes]:
    if panel == 'a':
        return draw_panel_5a(fig, page_size, rect, bundle, registry)
    if panel == 'b':
        return draw_survival_panel('b', 'PFS', fig, page_size, rect, bundle['pfs_curve'], bundle['pfs_patients'], bundle['pfs_summary'], registry)
    if panel == 'c':
        return draw_survival_panel('c', 'OS', fig, page_size, rect, bundle['os_curve'], bundle['os_patients'], bundle['os_summary'], registry)
    if panel == 'd':
        return draw_panel_5d(fig, page_size, rect, bundle, registry)
    if panel == 'e':
        return draw_panel_5e(fig, page_size, rect, bundle, registry)
    if panel == 'f':
        return draw_panel_5f(fig, page_size, rect, bundle, registry)
    if panel == 'g':
        return draw_panel_5g(fig, page_size, rect, bundle, registry)
    if panel == 'h':
        return draw_panel_5h(fig, page_size, rect, bundle, registry)
    raise ValueError(f'Unknown panel: {panel}')

def build_full_figure(bundle: Mapping[str, Any] | None=None) -> tuple[mpl.figure.Figure, ArtistRegistry, list[mpl.axes.Axes]]:
    if bundle is None:
        bundle = load_inputs()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes: list[mpl.axes.Axes] = []
    for panel in 'abcdefgh':
        rect = FULL_LAYOUT[panel]
        axes.extend(_draw_panel(panel, fig, (FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM), rect, bundle, registry))
        _add_panel_letter(fig, panel, rect, registry)
    return (fig, registry, axes)

def build_panel_figure(panel: str, bundle: Mapping[str, Any] | None=None) -> tuple[mpl.figure.Figure, ArtistRegistry, list[mpl.axes.Axes]]:
    normalized = str(panel).lower().strip()
    if normalized not in PANEL_SIZES_MM:
        raise ValueError(f'Unknown panel: {panel}')
    if bundle is None:
        bundle = load_inputs()
    width_mm, height_mm = PANEL_SIZES_MM[normalized]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    axes = _draw_panel(normalized, fig, (width_mm, height_mm), (0.0, 0.0, width_mm, height_mm), bundle, registry)
    return (fig, registry, axes)

def _save_exact_figure(fig: mpl.figure.Figure, output_stem: Path, *, width_mm: float, height_mm: float, formats: Sequence[str], dpi: int=EXPORT_DPI) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metadata = {'Title': f'Avatar-HCC Figure 5 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    for suffix in formats:
        normalized = suffix.lower().lstrip('.')
        path = output_stem.with_suffix(f'.{normalized}')
        kwargs: dict[str, Any] = {'format': normalized, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if normalized == 'png':
            kwargs['dpi'] = int(dpi)
        elif normalized == 'pdf':
            kwargs['metadata'] = metadata
        elif normalized == 'svg':
            kwargs['metadata'] = {'Title': metadata['Title'], 'Creator': metadata['Creator']}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    numeric_qa_path = write_json(bundle['numeric_qa'], OUTPUT_DIR / f"{get_value('figures.config.item_279')}{VERSION}.json")
    fig, registry, _ = build_full_figure(bundle)
    layout_report = audit_figure_layout(fig, f"{get_value('figures.config.item_278')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout_report, OUTPUT_DIR / f"{get_value('figures.config.item_280')}{VERSION}.json")
    full_paths = _save_exact_figure(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM, formats=('pdf', 'svg', 'png'))
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in 'abcdefgh':
        panel_fig, panel_registry, _ = build_panel_figure(panel, bundle)
        width_mm, height_mm = PANEL_SIZES_MM[panel]
        panel_audits[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_281')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        stem = PANEL_DIR / f"{get_value('figures.config.item_281')}{panel}_Nature_{VERSION}"
        panel_paths.extend(_save_exact_figure(panel_fig, stem, width_mm=width_mm, height_mm=height_mm, formats=('pdf', 'svg', 'png')))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_283')}{VERSION}.json")
    outputs = full_paths + panel_paths + [numeric_qa_path, layout_path, panel_audit_path]
    export_manifest = {'figure': 'Figure 5', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only frozen CSV snapshots; no model, KM, Cox, CV, bootstrap or spline refitting', 'panel_mapping': bundle['contract']['panel_mapping'], 'layout_summary': layout_report['summary'], 'numeric_qa': {'status': 'passed', 'report': str(numeric_qa_path)}, 'file_qa': 'Run figure5_file_qa.py with the bundled pypdf runtime.', 'outputs': [{'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size} for path in outputs]}
    manifest_path = write_json(export_manifest, OUTPUT_DIR / f"{get_value('figures.config.item_284')}{VERSION}.json")
    return {'full_paths': full_paths, 'panel_paths': panel_paths, 'numeric_qa': numeric_qa_path, 'layout_audit': layout_path, 'panel_audit': panel_audit_path, 'manifest': manifest_path, 'layout_summary': layout_report['summary'], 'panel_layout_summaries': {panel: report['summary'] for panel, report in panel_audits.items()}}

def summarize_bundle(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    return {'palette': PALETTE_ID, 'render_only': True, 'figure_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'panel_5a': {'n': int(bundle['association_summary']['n'].iloc[0]), 'cells': len(bundle['association_cells'])}, 'panel_5b': {'n': len(bundle['pfs_patients']), 'events': int(bundle['pfs_summary']['PFS_events'].sum())}, 'panel_5c': {'n': len(bundle['os_patients']), 'events': int(bundle['os_summary']['OS_events'].sum())}, 'panel_5d': {'stored_effect_rows': len(bundle['adjusted_association'])}, 'panel_5e': {'stored_effect_rows': len(bundle['within_mrecist'])}, 'panel_5f': {'stored_summary_rows': len(bundle['incremental_summary']), 'stored_fold_rows': len(bundle['incremental_folds'])}, 'panel_5g': {'stored_curve_rows': len(bundle['spline_curve']), 'stored_stat_rows': len(bundle['spline_stats'])}, 'panel_5h': {'variables': int(bundle['phenotype']['Variable'].nunique()), 'rows': len(bundle['phenotype'])}}
configure_style()
if __name__ == '__main__':
    result = export_all(load_inputs())
