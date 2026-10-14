from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
PILOT_ROOT = Path(__file__).resolve().parent
HARMONIZATION_ROOT = PILOT_ROOT.parents[1]
if str(HARMONIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(HARMONIZATION_ROOT))
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
from figure_style import DIVERGING_COLORS, MBP_COLORS, MBP_MARKERS, NEUTRAL_COLORS, PALETTE_ID, rc_params
VERSION = 'v001'
DATE_TAG = get_value('figures.config.item_350')
INPUT_DIR = PILOT_ROOT / 'inputs'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_300')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 210.0
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
PANEL_A_TEXT_GRAPHIC_CLEARANCE_MM = 4.0
PANEL_A_TEXT_GRAPHIC_CLEARANCE_PT = PANEL_A_TEXT_GRAPHIC_CLEARANCE_MM * 72.0 / 25.4
MM_PER_INCH = 25.4
AXIS_ORDER = ('Amino Acids & TME Regulation Axis', 'Lipid & 1-Carbon Metabolism Axis', 'SCFA & BCAA Energy Metabolism Axis')
AXIS_LABEL = {AXIS_ORDER[0]: 'Amino acids & TME', AXIS_ORDER[1]: 'Lipid & 1-carbon', AXIS_ORDER[2]: 'SCFA & BCAA'}
AXIS_SLUG = {AXIS_ORDER[0]: 'amino', AXIS_ORDER[1]: 'lipid', AXIS_ORDER[2]: 'scfa'}
IMAGING_ORDER = ('Low-gray-weighted phenotype', 'Coarse structural organization', 'Fine-scale heterogeneity')
IMAGING_SHORT = ('Low-gray', 'Coarse', 'Fine')
ENDPOINT_ORDER = ('Response', 'PFS', 'OS')
MBP_ORDER = ('MBP-I', 'MBP-II', 'MBP-III')
INPUT_FILES = {'paired': INPUT_DIR / get_value('figures.config.item_164'), 'correlation': INPUT_DIR / get_value('figures.config.item_115'), 'correlation_family': INPUT_DIR / get_value('figures.config.item_114'), 'legend': INPUT_DIR / get_value('figures.config.item_152'), 'shap': INPUT_DIR / get_value('figures.config.item_153'), 'scores': INPUT_DIR / get_value('figures.config.item_136'), 'stats': INPUT_DIR / get_value('figures.config.item_137'), 'pairwise': INPUT_DIR / get_value('figures.config.item_135'), 'selected_ko': INPUT_DIR / get_value('figures.config.item_202'), 'selected_metabolite': INPUT_DIR / get_value('figures.config.item_204')}
for _slug in AXIS_SLUG.values():
    INPUT_FILES[f'{_slug}_nodes'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_155')}"
    INPUT_FILES[f'{_slug}_nr'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_103')}"
    INPUT_FILES[f'{_slug}_r'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_104')}"
SOURCE_ROOT = Path(__file__).resolve().parent / 'inputs' / 'source_inputs'
BIOMAP_ROOT = SOURCE_ROOT / get_value('figures.config.item_257')
RADIOMICS_V2_ROOT = SOURCE_ROOT / get_value('figures.config.item_258')
ORIGINAL_SOURCES = {'paired': BIOMAP_ROOT / get_value('figures.path.part_006') / get_value('figures.path.part_003') / get_value('figures.config.item_089'), 'correlation': RADIOMICS_V2_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_038'), 'correlation_family': RADIOMICS_V2_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_037'), 'legend': BIOMAP_ROOT / get_value('figures.config.item_259') / get_value('figures.path.part_003') / get_value('figures.config.item_041'), 'shap': SOURCE_ROOT / get_value('figures.config.item_301') / get_value('figures.path.part_009') / get_value('figures.config.item_151'), 'scores': BIOMAP_ROOT / get_value('figures.path.part_007') / get_value('figures.path.part_004') / get_value('figures.config.item_141'), 'stats': BIOMAP_ROOT / get_value('figures.path.part_007') / get_value('figures.path.part_004') / get_value('figures.config.item_142'), 'pairwise': BIOMAP_ROOT / get_value('figures.path.part_007') / get_value('figures.path.part_004') / get_value('figures.config.item_140'), 'selected_ko': BIOMAP_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_203'), 'selected_metabolite': BIOMAP_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_035')}
_SANKY_SOURCE_STEM = {'amino': get_value('figures.path.part_010'), 'lipid': get_value('figures.path.part_011'), 'scfa': get_value('figures.path.part_012')}
for _slug, _stem in _SANKY_SOURCE_STEM.items():
    _sankey_root = BIOMAP_ROOT / get_value('figures.path.part_008') / get_value('figures.path.part_005')
    ORIGINAL_SOURCES[f'{_slug}_nodes'] = _sankey_root / f"{_stem}{get_value('figures.config.item_100')}"
    ORIGINAL_SOURCES[f'{_slug}_nr'] = _sankey_root / f"{_stem}{get_value('figures.config.item_098')}"
    ORIGINAL_SOURCES[f'{_slug}_r'] = _sankey_root / f"{_stem}{get_value('figures.config.item_099')}"
HTML_REFERENCE_LOCAL = PILOT_ROOT / get_value('figures.config.item_018')
HTML_REFERENCE_SOURCE = HARMONIZATION_ROOT.parent / 'tmp' / 'html_reference' / get_value('figures.config.item_019')
COLOR_REMAP = {'#00A087': '#2A8F78', '#20854E': '#4F9D73', '#3C5488': '#3E5F91', '#4DBBD5': '#2F91C3', '#5B9BD5': '#6CA8D1', '#6A6599': '#7163A6', '#6F6FAE': '#7D70D1', '#7E6148': '#95735E', '#7F95B8': '#7892B6', '#8491B4': '#617FC4', '#8F7700': '#A68A38', '#91D1C2': '#78B8AB', '#DC0000': '#C44A58', '#E64B35': '#DE646C', '#F39B7F': '#E49592', '#FFDC91': '#D9B866'}
EFFECT_CMAP = LinearSegmentedColormap.from_list(get_value('figures.config.item_363'), DIVERGING_COLORS)
EFFECT_NORM = Normalize(vmin=-1.0, vmax=1.0)
RHO_NORM = Normalize(vmin=-0.4, vmax=0.4)
RIBBON_COLORS = {'Positive': '#DE646C', 'Negative': '#2F6FB0'}
DISPLAY_LABEL_ALIASES = {'Allysine (not validated) (U)': 'Allysine (U)', 'Coumaric acid (isomer of 192, 194) (U)': 'Coumaric acid (U)'}

def configure_style() -> None:
    params = rc_params()
    params.update({'font.size': 6.0, 'axes.labelsize': 6.0, 'axes.titlesize': 6.2, 'xtick.labelsize': MIN_FONT_PT, 'ytick.labelsize': MIN_FONT_PT, 'legend.fontsize': MIN_FONT_PT, 'savefig.dpi': EXPORT_DPI, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    mpl.rcParams.update(params)

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{label} missing columns: {missing}')

def _read_csv(key: str) -> pd.DataFrame:
    path = INPUT_FILES[key]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {'paired': _read_csv('paired'), 'correlation': _read_csv('correlation'), 'correlation_family': _read_csv('correlation_family'), 'legend': _read_csv('legend'), 'shap': _read_csv('shap'), 'scores': _read_csv('scores'), 'stats': _read_csv('stats'), 'pairwise': _read_csv('pairwise'), 'selected_ko': _read_csv('selected_ko'), 'selected_metabolite': _read_csv('selected_metabolite'), 'nodes': {}, 'flows': {}}
    for axis in AXIS_ORDER:
        slug = AXIS_SLUG[axis]
        bundle['nodes'][axis] = _read_csv(f'{slug}_nodes')
        bundle['flows'][axis] = {'NR': _read_csv(f'{slug}_nr'), 'R': _read_csv(f'{slug}_r')}
    validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> None:
    _require_columns(bundle['paired'], ['feature', 'internal_effect', 'internal_p', 'internal_q_display', 'external_effect', 'external_p', 'external_q_display', 'axis', 'modality'], 'paired effects')
    _require_columns(bundle['correlation'], ['axis', 'metabolite', 'source', 'in_main', 'module', 'rho', 'p', 'n', 'q_BH_within_axis', 'star_nominal'], 'individual-metabolite imaging correlation (main)')
    _require_columns(bundle['correlation_family'], ['axis', 'metabolite', 'source', 'in_main', 'module', 'rho', 'p', 'n', 'q_BH_within_axis', 'star_nominal'], 'individual-metabolite imaging correlation (full BH family)')
    _require_columns(bundle['legend'], ['metabolic_module', 'legend_label', 'plot_color', 'plot_order'], 'module legend')
    _require_columns(bundle['shap'], ['axis', 'display_label', 'row_color', 'row_order', 'outcome', 'plot_value', 'abs_value'], 'module SHAP contribution')
    _require_columns(bundle['scores'], ['sample', 'axis', 'MBP', 'integrated_score', 'score_direction'], 'MBP scores')
    _require_columns(bundle['stats'], ['axis', 'n', 'trend_rho', 'trend_p', 'trend_q_BH', 'kw_q_BH'], 'MBP stats')
    _require_columns(bundle['pairwise'], ['axis', 'group1', 'group2', 'q_value'], 'MBP pairwise stats')
    _require_columns(bundle['selected_ko'], ['axis', 'ko_id', 'gene_symbol', 'subpathway_en', 'main_rank', 'in_main'], 'selected KO master')
    _require_columns(bundle['selected_metabolite'], ['axis', 'metabolite', 'metabolic_module_plot', 'main_rank', 'in_main'], 'selected metabolite master')
    if tuple(bundle['stats']['axis'].tolist()) != AXIS_ORDER:
        raise ValueError('Frozen axis order differs from the authoritative HTML order')
    if len(bundle['shap']) != 99 or bundle['shap']['plot_value'].isna().any():
        raise ValueError('Expected the final 99-row NEW_module_model SHAP plotting table')
    correlation = bundle['correlation']
    expected_axis_counts = {AXIS_ORDER[0]: 33, AXIS_ORDER[1]: 27, AXIS_ORDER[2]: 18}
    axis_metabolite_n = correlation[['axis', 'metabolite']].drop_duplicates().shape[0]
    if len(correlation) != 78 or axis_metabolite_n != 26:
        raise ValueError('Expected frozen 11_MAIN table with 78 rows = 26 axis-metabolite entries x 3 phenotypes')
    if correlation.groupby('axis').size().to_dict() != expected_axis_counts:
        raise ValueError('Frozen 11_MAIN axis counts must be 33/27/18 in HTML order')
    if len(bundle['correlation_family']) != 138:
        raise ValueError('Expected the full 138-row within-axis BH family table')
    main_from_family = bundle['correlation_family'][bundle['correlation_family']['in_main'].astype(str).str.upper().eq('TRUE')].reset_index(drop=True)
    compare_columns = ['axis', 'metabolite', 'source', 'module', 'rho', 'p', 'n', 'q_BH_within_axis', 'star_nominal']
    if not correlation[compare_columns].reset_index(drop=True).equals(main_from_family[compare_columns]):
        raise ValueError('11_MAIN is not identical to the in_main subset of the 138-row frozen family')
    ornithine = correlation[correlation['axis'].eq(AXIS_ORDER[0]) & correlation['metabolite'].eq('Ornithine (B)')].set_index('module').reindex(IMAGING_ORDER)
    expected_ornithine = np.array([-0.281086741467582, 0.12528061644139, -0.174089922366148])
    if len(ornithine) != 3 or not np.allclose(ornithine['rho'].to_numpy(float), expected_ornithine, rtol=0, atol=1e-12):
        raise ValueError('Ornithine HTML anchor values do not match the frozen 11_MAIN table')
    if len(bundle['scores']) != 327 or bundle['scores']['sample'].nunique() != 109:
        raise ValueError('Expected 109 patients across three frozen axis-score blocks')
    if not bundle['scores']['score_direction'].astype(str).str.contains('Responder-like', regex=False).all():
        raise ValueError('Frozen response orientation is not explicit for all ecosystem scores')
    paired = bundle['paired']
    external_nominal_only = pd.to_numeric(paired['external_p'], errors='raise').le(0.05) & pd.to_numeric(paired['external_q_display'], errors='raise').gt(0.05)
    if int(external_nominal_only.sum()) != 6:
        raise ValueError('Expected six external bubbles with nominal p<=0.05 but q_display>0.05')
    for axis in AXIS_ORDER:
        nodes = bundle['nodes'][axis]
        _require_columns(nodes, ['node_id', 'display_name', 'side', 'module', 'color', 'plot_order', 'raw_feature'], f'{axis} nodes')
        for group in ('NR', 'R'):
            flow = bundle['flows'][axis][group]
            _require_columns(flow, ['source_node', 'target_node', 'thickness', 'rho', 'direction'], f'{axis} {group} flow')
            if set(flow['source_node']).difference(nodes['node_id']) or set(flow['target_node']).difference(nodes['node_id']):
                raise ValueError(f'{axis} {group} flow references undefined nodes')

def _hex_display(source: Any) -> str:
    key = str(source).strip().upper()
    return COLOR_REMAP.get(key, str(source))

def _stars(value: Any) -> str:
    try:
        significance_value = float(value)
    except (TypeError, ValueError):
        return ''
    if significance_value <= 0.001:
        return '***'
    if significance_value <= 0.01:
        return '**'
    if significance_value <= 0.05:
        return '*'
    return ''

def _short(text: Any, limit: int=23) -> str:
    source = str(text)
    value = DISPLAY_LABEL_ALIASES.get(source, source).replace(' (not validated by standard)', '').replace(' (not validated)', '').replace('.1 (', ' (').replace('.2 (', ' (')
    return value if len(value) <= limit else value[:limit - 1] + '…'

def _rect(width_mm: float, height_mm: float, left: float, top: float, width: float, height: float) -> list[float]:
    return [left / width_mm, 1.0 - (top + height) / height_mm, width / width_mm, height / height_mm]

def _subrect(panel: Sequence[float], rel: Sequence[float]) -> tuple[float, float, float, float]:
    left, top, width, height = panel
    x, y, w, h = rel
    return (left + x * width, top + y * height, w * width, h * height)

def _axis(fig: mpl.figure.Figure, width_mm: float, height_mm: float, rect: Sequence[float], panel: str) -> mpl.axes.Axes:
    ax = fig.add_axes(_rect(width_mm, height_mm, *rect))
    ax._qa_meta = {'panel': panel}
    return ax

def _panel_letter(fig: mpl.figure.Figure, width_mm: float, height_mm: float, label: str, left_mm: float, top_mm: float, registry: ArtistRegistry) -> None:
    artist = fig.text(left_mm / width_mm, 1.0 - top_mm / height_mm, label.lower(), ha='left', va='top', fontsize=8.0, fontweight='bold', color=NEUTRAL_COLORS['text'])
    registry.tag(artist, name=f'panel-{label}-letter', role='panel_letter', panel=label)

def _bubble_rows(bundle: Mapping[str, Any], axis: str, side: str) -> pd.DataFrame:
    nodes = bundle['nodes'][axis].copy()
    module_colors = nodes.drop_duplicates('module').set_index('module')['color'].map(_hex_display).to_dict()
    if side == 'source':
        metadata = bundle['selected_ko']
        metadata = metadata[metadata['axis'].eq(axis) & metadata['in_main'].astype(str).str.upper().eq('TRUE')][['ko_id', 'gene_symbol', 'subpathway_en', 'main_rank']].rename(columns={'ko_id': 'raw_feature', 'gene_symbol': 'display_name', 'subpathway_en': 'module', 'main_rank': 'plot_order'})
    else:
        metadata = bundle['selected_metabolite']
        metadata = metadata[metadata['axis'].eq(axis) & metadata['in_main'].astype(str).str.upper().eq('TRUE')][['metabolite', 'metabolic_module_plot', 'main_rank']].rename(columns={'metabolite': 'raw_feature', 'metabolic_module_plot': 'module', 'main_rank': 'plot_order'})
        metadata['display_name'] = metadata['raw_feature']
    metadata['plot_order'] = pd.to_numeric(metadata['plot_order'], errors='raise')
    metadata['color'] = metadata['module'].map(module_colors).fillna(NEUTRAL_COLORS['missing'])
    effects = bundle['paired']
    effects = effects[effects['axis'].eq(axis)].copy()
    effects = effects[effects['modality'].eq('KO') if side == 'source' else ~effects['modality'].eq('KO')]
    merged = metadata.merge(effects, left_on='raw_feature', right_on='feature', how='inner', validate='one_to_one')
    if len(merged) != len(effects):
        missing = sorted(set(effects['feature']).difference(merged['feature']))
        raise ValueError(f'{axis} {side} paired-effect features lack frozen node metadata: {missing}')
    return merged.sort_values('plot_order', kind='stable').reset_index(drop=True)

def _draw_bubbles(ax: mpl.axes.Axes, rows: pd.DataFrame, side: str) -> None:
    n = len(rows)
    y = np.arange(n)[::-1]
    for xi, (effect_col, p_col) in enumerate((('internal_effect', 'internal_p'), ('external_effect', 'external_p'))):
        vals = pd.to_numeric(rows[effect_col], errors='raise').to_numpy(float)
        ax.scatter(np.full(n, xi), y, s=30, c=vals, cmap=EFFECT_CMAP, norm=EFFECT_NORM, edgecolors='white', linewidths=0.35, zorder=2)
        for yi, p_value, value in zip(y, rows[p_col], vals):
            stars = _stars(p_value)
            if stars:
                ax.text(xi, yi, stars, ha='center', va='center', fontsize=MIN_FONT_PT, fontweight='bold', color='white' if abs(value) >= 0.4 else NEUTRAL_COLORS['text'], zorder=3)
    labels = rows['display_name'].map(lambda value: _short(value, 22 if side == 'source' else 24)).tolist()
    ax.set_yticks(y, labels=labels)
    ax.tick_params(axis='y', length=0, pad=1.1, labelsize=MIN_FONT_PT)
    ax.set_xticks([])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.6, n - 0.4)
    for yi, color in zip(y, rows['color']):
        ax.add_patch(Rectangle((-0.5, yi - 0.42), 0.11, 0.84, facecolor=_hex_display(color), edgecolor='none', clip_on=False))
    for spine in ax.spines.values():
        spine.set_visible(False)

def _node_intervals(nodes: pd.DataFrame, flows: Sequence[pd.DataFrame], side: str) -> dict[str, tuple[float, float]]:
    node_ids = set()
    weight_by_flow: list[dict[str, float]] = []
    for flow in flows:
        weights: dict[str, float] = {}
        column = 'source_node' if side == 'source' else 'target_node'
        for row in flow.itertuples(index=False):
            node = str(getattr(row, column))
            weights[node] = weights.get(node, 0.0) + float(getattr(row, 'thickness'))
            node_ids.add(node)
        weight_by_flow.append(weights)
    d = nodes[nodes['node_id'].isin(node_ids) & nodes['side'].str.lower().eq(side)].copy()
    d['plot_order'] = pd.to_numeric(d['plot_order'], errors='raise')
    d = d.sort_values('plot_order', kind='stable')
    d['weight'] = [max((weights.get(node, 0.0) for weights in weight_by_flow)) for node in d['node_id']]
    modules = d['module'].drop_duplicates().tolist()
    gap = 0.022
    usable = 0.94 - gap * max(0, len(modules) - 1)
    mass = np.maximum(d['weight'].to_numpy(float), 0.035) ** 0.6
    heights = usable * mass / mass.sum()
    intervals: dict[str, tuple[float, float]] = {}
    cursor = 0.97
    last_module: str | None = None
    for (_, row), height in zip(d.iterrows(), heights):
        module = str(row['module'])
        if last_module is not None and module != last_module:
            cursor -= gap
        intervals[str(row['node_id'])] = (cursor - height, cursor)
        cursor -= height
        last_module = module
    return intervals

def _link_patch(x0: float, x1: float, source: tuple[float, float], target: tuple[float, float], color: str, alpha: float) -> PathPatch:
    sy0, sy1 = source
    ty0, ty1 = target
    xm = (x0 + x1) * 0.5
    vertices = [(x0, sy0), (xm, sy0), (xm, ty0), (x1, ty0), (x1, ty1), (xm, ty1), (xm, sy1), (x0, sy1), (x0, sy0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor='none', alpha=alpha, zorder=1)

def _draw_sankey_pair(ax: mpl.axes.Axes, nodes: pd.DataFrame, flow_nr: pd.DataFrame, flow_r: pd.DataFrame) -> None:
    flows = [flow_nr.copy(), flow_r.copy()]
    for flow in flows:
        flow['thickness'] = pd.to_numeric(flow['thickness'], errors='raise')
        flow['rho'] = pd.to_numeric(flow['rho'], errors='raise')
    source_intervals = _node_intervals(nodes, flows, 'source')
    target_intervals = _node_intervals(nodes, flows, 'target')
    node_lookup = nodes.drop_duplicates('node_id').set_index('node_id')
    for flow, (sx, tx), group in zip(flows, ((0.02, 0.47), (0.53, 0.98)), ('NR', 'R')):
        totals: dict[str, float] = {}
        for column in ('source_node', 'target_node'):
            for node, values in flow.groupby(column, sort=False)['thickness']:
                totals[str(node)] = float(values.sum())
        scale_candidates: list[float] = []
        for node, total in totals.items():
            interval = source_intervals.get(node, target_intervals.get(node))
            if interval and total > 0:
                scale_candidates.append((interval[1] - interval[0]) / total)
        scale = min(scale_candidates) * 0.92
        source_cursor = {node: interval[1] for node, interval in source_intervals.items()}
        target_cursor = {node: interval[1] for node, interval in target_intervals.items()}
        ordered = flow.sort_values(['source_node', 'target_node'], kind='stable')
        for row in ordered.itertuples(index=False):
            source_node = str(row.source_node)
            target_node = str(row.target_node)
            height = float(row.thickness) * scale
            sy1 = source_cursor[source_node]
            sy0 = sy1 - height
            source_cursor[source_node] = sy0
            ty1 = target_cursor[target_node]
            ty0 = ty1 - height
            target_cursor[target_node] = ty0
            direction = 'Positive' if float(row.rho) >= 0 else 'Negative'
            alpha = min(0.82, 0.25 + 1.35 * abs(float(row.rho)))
            ax.add_patch(_link_patch(sx + 0.014, tx - 0.014, (sy0, sy1), (ty0, ty1), RIBBON_COLORS[direction], alpha))
        for node, interval in source_intervals.items():
            color = _hex_display(node_lookup.loc[node, 'color'])
            ax.add_patch(Rectangle((sx, interval[0]), 0.014, interval[1] - interval[0], facecolor=color, edgecolor='none', zorder=3))
        for node, interval in target_intervals.items():
            color = _hex_display(node_lookup.loc[node, 'color'])
            ax.add_patch(Rectangle((tx - 0.014, interval[0]), 0.014, interval[1] - interval[0], facecolor=color, edgecolor='none', zorder=3))
        ax.text((sx + tx) / 2, -0.015, group, ha='center', va='top', fontsize=MIN_FONT_PT, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.01)
    ax.axis('off')

def _correlation_rows(bundle: Mapping[str, Any], axis: str) -> pd.DataFrame:
    data = bundle['correlation']
    data = data[data['axis'].eq(axis)].copy()
    data = data[data['in_main'].astype(str).str.upper().eq('TRUE')]
    nodes = bundle['nodes'][axis]
    module_colors = nodes[nodes['side'].str.lower().eq('target')].drop_duplicates('module').set_index('module')['color'].map(_hex_display).to_dict()
    targets = bundle['selected_metabolite']
    targets = targets[targets['axis'].eq(axis) & targets['in_main'].astype(str).str.upper().eq('TRUE')][['metabolite', 'main_rank', 'metabolic_module_plot']].rename(columns={'main_rank': 'plot_order', 'metabolic_module_plot': 'biological_module'})
    targets['node_color'] = targets['biological_module'].map(module_colors)
    data = data.merge(targets, on='metabolite', how='left', validate='many_to_one')
    if data[['plot_order', 'node_color']].isna().any().any():
        raise ValueError(f'{axis} individual correlation rows lack frozen node metadata')
    data['plot_order'] = pd.to_numeric(data['plot_order'], errors='raise')
    data['module_rank'] = data['module'].map({name: idx for idx, name in enumerate(IMAGING_ORDER)})
    return data.sort_values(['plot_order', 'module_rank'], kind='stable')

def _draw_correlation(ax: mpl.axes.Axes, rows: pd.DataFrame) -> None:
    metabolite_order = rows.sort_values('plot_order', kind='stable')['metabolite'].drop_duplicates().tolist()
    pivot = rows.pivot(index='metabolite', columns='module', values='rho').reindex(index=metabolite_order, columns=IMAGING_ORDER)
    matrix = pivot.to_numpy(float)
    ax.imshow(matrix, cmap=EFFECT_CMAP, norm=RHO_NORM, aspect='auto', interpolation='none')
    star_pivot = rows.pivot(index='metabolite', columns='module', values='star_nominal').reindex(index=metabolite_order, columns=IMAGING_ORDER).fillna('')
    for yi, metabolite in enumerate(metabolite_order):
        node_color = _hex_display(rows.loc[rows['metabolite'].eq(metabolite), 'node_color'].iloc[0])
        ax.add_patch(Rectangle((-0.78, yi - 0.48), 0.18, 0.96, facecolor=node_color, edgecolor='none', clip_on=False))
        for xi, value in enumerate(matrix[yi]):
            color = 'white' if abs(value) >= 0.24 else NEUTRAL_COLORS['text']
            star = str(star_pivot.iloc[yi, xi])
            ax.text(xi, yi, f'{value:.2f}{star}', ha='center', va='center', fontsize=MIN_FONT_PT, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.85, 2.5)
    ax.set_ylim(len(metabolite_order) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _add_figure_text(fig: mpl.figure.Figure, width_mm: float, height_mm: float, x_mm: float, y_mm: float, text: str, *, ha: str='center', va: str='top', fontsize: float=6.0, fontweight: str='normal') -> mpl.text.Text:
    return fig.text(x_mm / width_mm, 1.0 - y_mm / height_mm, text, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight, color=NEUTRAL_COLORS['text'])

def draw_panel_a(fig: mpl.figure.Figure, width_mm: float, height_mm: float, panel_rect: Sequence[float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    left, top, width, height = panel_rect
    legend = bundle['legend'].copy()
    legend['plot_order'] = pd.to_numeric(legend['plot_order'], errors='raise')
    legend = legend.sort_values('plot_order', kind='stable').reset_index(drop=True)
    module_colors = dict(zip(legend['metabolic_module'], legend['plot_color'].map(_hex_display)))
    headers = ((0.13, 'KO response effect\nInternal | External'), (0.405, 'KO-metabolite association\nNR | R'), (0.728, 'Metabolite response effect\nInternal | External'), (0.905, 'Imaging phenotype\nLow-gray | Coarse | Fine'))
    for x_rel, text in headers:
        artist = _add_figure_text(fig, width_mm, height_mm, left + x_rel * width, top + 0.2, text, fontsize=5.8, fontweight='bold')
        registry.tag(artist, name=f'a-header-{x_rel}', role='axis_label', panel='a')
    row_tops = (0.075, 0.345, 0.675)
    row_heights = (0.245, 0.3, 0.175)
    for axis, row_top, row_height in zip(AXIS_ORDER, row_tops, row_heights):
        title = _add_figure_text(fig, width_mm, height_mm, left + 0.002 * width, top + (row_top - 0.012) * height, AXIS_LABEL[axis], ha='left', fontsize=5.8, fontweight='bold')
        registry.tag(title, name=f'a-axis-{AXIS_SLUG[axis]}', role='direct_label', panel='a')
        ko_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.075, row_top, 0.145, row_height)), 'a')
        sankey_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.235, row_top, 0.255, row_height)), 'a')
        met_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.663, row_top, 0.13, row_height)), 'a')
        corr_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.835, row_top, 0.15, row_height)), 'a')
        sankey_ax._figure7_axis_name = axis
        sankey_ax._figure7_axis_kind = 'sankey'
        met_ax._figure7_axis_name = axis
        met_ax._figure7_axis_kind = 'metabolite_effect'
        _draw_bubbles(ko_ax, _bubble_rows(bundle, axis, 'source'), 'source')
        _draw_sankey_pair(sankey_ax, bundle['nodes'][axis], bundle['flows'][axis]['NR'], bundle['flows'][axis]['R'])
        _draw_bubbles(met_ax, _bubble_rows(bundle, axis, 'target'), 'target')
        _draw_correlation(corr_ax, _correlation_rows(bundle, axis))
    legend_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.015, 0.875, 0.82, 0.12)), 'a')
    legend_ax.axis('off')
    ncols = 4
    nrows = int(math.ceil(len(legend) / ncols))
    for idx, row in legend.iterrows():
        col = idx // nrows
        row_idx = idx % nrows
        x = col / ncols + 0.012
        y = 1.0 - (row_idx + 0.45) / nrows
        color = _hex_display(row['plot_color'])
        legend_ax.add_patch(Rectangle((x, y - 0.035), 0.014, 0.07, transform=legend_ax.transAxes, facecolor=color, edgecolor='none'))
        legend_ax.text(x + 0.02, y, str(row['legend_label']).replace('\r', ''), transform=legend_ax.transAxes, ha='left', va='center', fontsize=MIN_FONT_PT, linespacing=0.92)
    effect_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.865, 0.885, 0.11, 0.02)), 'a')
    corrbar_ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.865, 0.95, 0.11, 0.02)), 'a')
    gradient = np.linspace(-1, 1, 256)[None, :]
    effect_ax.imshow(gradient, cmap=EFFECT_CMAP, aspect='auto', extent=(-1, 1, 0, 1))
    effect_ax.set_xticks([-1, 0, 1], labels=['−1', '0', '1'])
    effect_ax.set_yticks([])
    effect_ax.set_title('Response effect z', fontsize=MIN_FONT_PT, pad=1.5)
    effect_ax.tick_params(axis='x', length=1.5, pad=1)
    corrbar_ax.imshow(gradient, cmap=EFFECT_CMAP, aspect='auto', extent=(-0.4, 0.4, 0, 1))
    corrbar_ax.set_xticks([-0.4, 0, 0.4], labels=['−0.4', '0', '0.4'])
    corrbar_ax.set_yticks([])
    corrbar_ax.set_title('Spearman ρ', fontsize=MIN_FONT_PT, pad=1.5)
    corrbar_ax.tick_params(axis='x', length=1.5, pad=1)
    for ax in (effect_ax, corrbar_ax):
        for spine in ax.spines.values():
            spine.set_visible(False)

def audit_panel_a_text_graphic_clearance(fig: mpl.figure.Figure, *, required_pt: float=PANEL_A_TEXT_GRAPHIC_CLEARANCE_PT) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    rows: list[dict[str, Any]] = []
    for axis_name in AXIS_ORDER:
        sankey_axes = [ax for ax in fig.axes if getattr(ax, '_figure7_axis_kind', None) == 'sankey' and getattr(ax, '_figure7_axis_name', None) == axis_name]
        metabolite_axes = [ax for ax in fig.axes if getattr(ax, '_figure7_axis_kind', None) == 'metabolite_effect' and getattr(ax, '_figure7_axis_name', None) == axis_name]
        if len(sankey_axes) != 1 or len(metabolite_axes) != 1:
            raise ValueError(f'Cannot identify one Sankey/metabolite axis pair for {axis_name}')
        sankey_right = sankey_axes[0].get_window_extent(renderer).x1
        for label in metabolite_axes[0].get_yticklabels():
            if not label.get_visible() or not label.get_text().strip():
                continue
            bbox = label.get_window_extent(renderer)
            clearance_pt = (bbox.x0 - sankey_right) / px_per_pt
            rows.append({'axis': axis_name, 'label': label.get_text(), 'clearance_pt': round(float(clearance_pt), 3), 'pass': bool(clearance_pt >= required_pt)})
    minimum = min((row['clearance_pt'] for row in rows))
    report = {'required_clearance_pt': required_pt, 'required_clearance_mm': required_pt * 25.4 / 72.0, 'minimum_clearance_pt': minimum, 'minimum_clearance_mm': minimum * 25.4 / 72.0, 'pass': all((row['pass'] for row in rows)), 'rows': rows}
    return report

def _draw_shap_axis(ax: mpl.axes.Axes, data: pd.DataFrame, axis_name: str) -> None:
    axis_data = data[data['axis'].eq(axis_name)].copy()
    axis_data['row_order'] = pd.to_numeric(axis_data['row_order'], errors='raise')
    axis_data['plot_value'] = pd.to_numeric(axis_data['plot_value'], errors='raise')
    module_order = axis_data.sort_values('row_order', kind='stable')['display_label'].drop_duplicates().tolist()
    color_map = axis_data.drop_duplicates('display_label').set_index('display_label')['row_color'].map(_hex_display).to_dict()
    y_map = {outcome: value for outcome, value in zip(ENDPOINT_ORDER, (2, 1, 0))}
    for outcome in ENDPOINT_ORDER:
        rows = axis_data[axis_data['outcome'].eq(outcome)].set_index('display_label').reindex(module_order)
        neg = 0.0
        pos = 0.0
        for module, row in rows.iterrows():
            value = float(row['plot_value'])
            if value < 0:
                ax.barh(y_map[outcome], value, left=neg, height=0.55, color=color_map[module], edgecolor='white', linewidth=0.25)
                neg += value
            else:
                ax.barh(y_map[outcome], value, left=pos, height=0.55, color=color_map[module], edgecolor='white', linewidth=0.25)
                pos += value
    ax.axvline(0, color=NEUTRAL_COLORS['axis'], linewidth=0.55, zorder=3)
    ax.set_yticks((2, 1, 0), labels=ENDPOINT_ORDER)
    ax.set_ylim(-0.7, 2.7)
    ax.set_title(AXIS_LABEL[axis_name], loc='left', pad=1.2, fontsize=5.8, fontweight='bold')
    ax.grid(axis='x', color='#D9E0E6', linewidth=0.35, alpha=0.65, zorder=0)
    ax.tick_params(axis='both', labelsize=MIN_FONT_PT, pad=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

def draw_panel_b(fig: mpl.figure.Figure, width_mm: float, height_mm: float, panel_rect: Sequence[float], bundle: Mapping[str, Any], registry: ArtistRegistry, *, standalone: bool=False) -> None:
    data = bundle['shap']
    values = pd.to_numeric(data['plot_value'], errors='raise')
    totals = data.assign(value=values).groupby(['axis', 'outcome'])['value'].agg(neg=lambda x: x[x < 0].sum(), pos=lambda x: x[x > 0].sum())
    x_min = min(-0.95, float(totals['neg'].min()) - 0.04)
    x_max = max(0.45, float(totals['pos'].max()) + 0.04)
    for index, axis_name in enumerate(AXIS_ORDER):
        ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.115, 0.055 + index * 0.305, 0.86, 0.255)), 'b')
        _draw_shap_axis(ax, data, axis_name)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.arange(-0.8, 0.401, 0.2))
        if index < 2:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Signed module-level SHAP contribution', labelpad=1.5, fontsize=MIN_FONT_PT)
    if standalone:
        legend_data = data.copy()
        legend_data['row_order'] = pd.to_numeric(legend_data['row_order'], errors='raise')
        legend_data = legend_data.sort_values('row_order', kind='stable').drop_duplicates('display_label')[['display_label', 'row_color']].reset_index(drop=True)
        legend_ax = _axis(fig, width_mm, height_mm, (8.0, 77.0, 135.0, 20.0), 'b')
        legend_ax.axis('off')
        ncols = 4
        nrows = int(math.ceil(len(legend_data) / ncols))
        for idx, row in legend_data.iterrows():
            col = idx // nrows
            row_idx = idx % nrows
            x = col / ncols + 0.008
            y = 1.0 - (row_idx + 0.5) / nrows
            legend_ax.add_patch(Rectangle((x, y - 0.034), 0.014, 0.068, transform=legend_ax.transAxes, facecolor=_hex_display(row['row_color']), edgecolor='none'))
            legend_ax.text(x + 0.02, y, _short(row['display_label'], 31), transform=legend_ax.transAxes, ha='left', va='center', fontsize=MIN_FONT_PT)

def _deterministic_jitter(sample: str, group: str, width: float=0.085) -> float:
    digest = hashlib.sha256(f'{sample}|{group}'.encode('utf-8')).digest()
    integer = int.from_bytes(digest[:8], 'little')
    return (integer / (2 ** 64 - 1) * 2.0 - 1.0) * width

def _format_q(q: float) -> str:
    if q < 0.001:
        return '<0.001'
    return f'={q:.3f}'.lstrip('0')

def _draw_violin_axis(ax: mpl.axes.Axes, scores: pd.DataFrame, stat: pd.Series, pairwise: pd.DataFrame, axis_name: str, *, show_ylabel: bool) -> None:
    arrays = [pd.to_numeric(scores.loc[scores['MBP'].eq(group), 'integrated_score'], errors='raise').to_numpy(float) for group in MBP_ORDER]
    positions = np.arange(3)
    violins = ax.violinplot(arrays, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False, bw_method=0.35)
    for body, group in zip(violins['bodies'], MBP_ORDER):
        body.set_facecolor(MBP_COLORS[group])
        body.set_edgecolor(MBP_COLORS[group])
        body.set_alpha(0.72)
        body.set_linewidth(0.45)
    for xpos, group, values in zip(positions, MBP_ORDER, arrays):
        subset = scores[scores['MBP'].eq(group)]
        marker = MBP_MARKERS[group]
        jitter = np.array([_deterministic_jitter(str(sample), group) for sample in subset['sample']])
        ax.scatter(np.full(len(values), xpos) + jitter, values, s=5.0, marker=marker, color=MBP_COLORS[group], alpha=0.82, edgecolors='white', linewidths=0.22, zorder=3)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([xpos, xpos], [q1, q3], color='white', linewidth=2.2, solid_capstyle='butt', zorder=4)
        ax.plot([xpos - 0.12, xpos + 0.12], [median, median], color=NEUTRAL_COLORS['text'], linewidth=0.75, zorder=5)
    low = min((float(values.min()) for values in arrays))
    high = max((float(values.max()) for values in arrays))
    span = max(high - low, 1.0)
    ax.set_ylim(low - 0.08 * span, high + 0.31 * span)
    i_iii = pairwise[pairwise['group1'].eq('MBP-I') & pairwise['group2'].eq('MBP-III')]
    if len(i_iii) != 1:
        raise ValueError(f'Expected one frozen MBP-I vs MBP-III row for {axis_name}')
    pair_q = float(i_iii.iloc[0]['q_value'])
    trend_rho = float(stat['trend_rho'])
    trend_q = float(stat['trend_q_BH'])
    ax.text(0.5, 0.98, f'ρ={trend_rho:.2f}, q{_format_q(trend_q)}\nI-III q{_format_q(pair_q)}', transform=ax.transAxes, ha='center', va='top', fontsize=MIN_FONT_PT, linespacing=0.95)
    ax.set_title(AXIS_LABEL[axis_name], fontsize=5.8, pad=1.5)
    ax.set_xticks(positions, labels=('I', 'II', 'III'))
    if show_ylabel:
        ax.set_ylabel('Response score', fontsize=MIN_FONT_PT, labelpad=1.5)
    else:
        ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(axis='both', labelsize=MIN_FONT_PT, pad=1.2)
    ax.grid(axis='y', color='#D9E0E6', linewidth=0.35, alpha=0.65, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

def draw_panel_c(fig: mpl.figure.Figure, width_mm: float, height_mm: float, panel_rect: Sequence[float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    scores = bundle['scores']
    stats = bundle['stats'].set_index('axis')
    pairwise = bundle['pairwise']
    for index, axis_name in enumerate(AXIS_ORDER):
        ax = _axis(fig, width_mm, height_mm, _subrect(panel_rect, (0.08 + index * 0.305, 0.095, 0.27, 0.82)), 'c')
        _draw_violin_axis(ax, scores[scores['axis'].eq(axis_name)], stats.loc[axis_name], pairwise[pairwise['axis'].eq(axis_name)], axis_name, show_ylabel=index == 0)
    label = _add_figure_text(fig, width_mm, height_mm, panel_rect[0] + 0.53 * panel_rect[2], panel_rect[1] + 0.985 * panel_rect[3], 'MBP', fontsize=MIN_FONT_PT)
    registry.tag(label, name='c-mbp-label', role='axis_label', panel='c')
FULL_PANELS_MM = {'a': (5.0, 4.0, 171.0, 139.0), 'b': (5.0, 149.0, 101.0, 56.0), 'c': (111.0, 149.0, 65.0, 56.0)}
PANEL_LAYOUTS = {'a': {'size': (180.0, 150.0), 'rect': (5.0, 4.0, 171.0, 142.0)}, 'b': {'size': (150.0, 102.0), 'rect': (8.0, 5.0, 135.0, 68.0)}, 'c': {'size': (125.0, 75.0), 'rect': (5.0, 5.0, 115.0, 65.0)}}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    draw_panel_a(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANELS_MM['a'], bundle, registry)
    draw_panel_b(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANELS_MM['b'], bundle, registry)
    draw_panel_c(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANELS_MM['c'], bundle, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'a', 1.5, 2.0, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'b', 1.5, 147.0, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'c', 107.5, 147.0, registry)
    return (fig, registry)

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    panel = panel.lower()
    if panel not in PANEL_LAYOUTS:
        raise ValueError(panel)
    configure_style()
    width_mm, height_mm = PANEL_LAYOUTS[panel]['size']
    rect = PANEL_LAYOUTS[panel]['rect']
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    if panel == 'a':
        draw_panel_a(fig, width_mm, height_mm, rect, bundle, registry)
    elif panel == 'b':
        draw_panel_b(fig, width_mm, height_mm, rect, bundle, registry, standalone=True)
    else:
        draw_panel_c(fig, width_mm, height_mm, rect, bundle, registry)
    _panel_letter(fig, width_mm, height_mm, panel, 1.5, 2.0, registry)
    return (fig, registry)

def _save_exact(fig: mpl.figure.Figure, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Figure 7 {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    outputs: list[Path] = []
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
        outputs.append(path)
    return outputs

def input_manifest() -> dict[str, Any]:
    if set(INPUT_FILES) != set(ORIGINAL_SOURCES):
        raise ValueError('Formal input/source mapping keys differ')
    records: list[dict[str, Any]] = []
    for key, snapshot in INPUT_FILES.items():
        source = ORIGINAL_SOURCES[key]
        if not snapshot.exists():
            raise FileNotFoundError(snapshot)
        if not source.exists():
            raise FileNotFoundError(source)
        snapshot_hash = sha256_file(snapshot).upper()
        source_hash = sha256_file(source).upper()
        if snapshot_hash != source_hash or snapshot.stat().st_size != source.stat().st_size:
            raise ValueError(f'Frozen snapshot does not match original source: {key}')
        frame = pd.read_csv(snapshot)
        records.append({'key': key, 'snapshot': snapshot.name, 'source': source.as_posix(), 'rows': int(len(frame)), 'columns': int(len(frame.columns)), 'bytes': int(snapshot.stat().st_size), 'sha256': snapshot_hash, 'source_sha256': source_hash, 'byte_for_byte_match': True})
    if not HTML_REFERENCE_LOCAL.exists() or not HTML_REFERENCE_SOURCE.exists():
        raise FileNotFoundError('Figure 7 HTML reference image is missing')
    reference_hash = sha256_file(HTML_REFERENCE_LOCAL).upper()
    reference_source_hash = sha256_file(HTML_REFERENCE_SOURCE).upper()
    if reference_hash != reference_source_hash:
        raise ValueError('Local Figure 7 HTML reference does not match authoritative source')
    return {'figure': 'Figure 7', 'snapshot_date': '2026-09-02', 'snapshot_root': INPUT_DIR.as_posix(), 'copy_policy': 'Byte-for-byte copies of the minimum frozen tables required for render-only reconstruction; no source file was edited.', 'source_version_policy': get_value('figures.provenance.source_policy'), 'authoritative_html_reference': {'snapshot': HTML_REFERENCE_LOCAL.name, 'source': HTML_REFERENCE_SOURCE.as_posix(), 'bytes': int(HTML_REFERENCE_LOCAL.stat().st_size), 'sha256': reference_hash, 'source_sha256': reference_source_hash, 'byte_for_byte_match': True}, 'files': records}

def scientific_contract() -> dict[str, Any]:
    return {'figure': 'Figure 7', 'version': VERSION, 'render_mode': 'frozen-source render-only', 'authoritative_content_and_order': HTML_REFERENCE_LOCAL.name, 'working_master_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'ecosystem_axis_order': list(AXIS_ORDER), 'endpoint_order': list(ENDPOINT_ORDER), 'mbp_order': list(MBP_ORDER), 'panel_mapping': {'A': {'inputs': [get_value('figures.config.item_164'), get_value('figures.config.item_202'), get_value('figures.config.item_204'), get_value('figures.config.item_056'), get_value('figures.config.item_054'), get_value('figures.config.item_055'), get_value('figures.config.item_127'), get_value('figures.config.item_125'), get_value('figures.config.item_126'), get_value('figures.config.item_199'), get_value('figures.config.item_197'), get_value('figures.config.item_198'), get_value('figures.config.item_115'), get_value('figures.config.item_114'), get_value('figures.config.item_152')], 'meaning': 'Three ecosystem axes with frozen internal/external response effects, NR/R KO-to-metabolite links, and individual-metabolite correlations with three imaging phenotypes.', 'locked_counts': {'paired_effect_rows': 54, 'main_correlation_rows': 78, 'main_axis_row_counts': [33, 27, 18], 'full_fdr_family_rows': 138, 'main_axis_metabolite_entries': 26, 'metabolic_legend_items': 13}, 'correlation_guardrail': 'Plot rho from 11_MAIN directly. 11_MAIN must exactly equal in_main=TRUE from 10_ALL. q_BH_within_axis belongs to the full 138-test, per-axis BH families.', 'star_guardrail': 'Display star_nominal copied from the frozen table; these stars encode nominal p, not BH q.', 'bubble_star_guardrail': "For both Internal and External bubbles, derive display stars only from the already frozen internal_p/external_p values using the original notebook's NOMINAL mode; internal_q_display/external_q_display are provenance only.", 'anchor_guardrail': 'Ornithine must be the first Amino row with rho -0.281086741467582, 0.12528061644139, -0.174089922366148 in Low-gray, Coarse, Fine order (displayed as -0.28, 0.13, -0.17).', 'geometry_guardrail': 'Reconstruct ribbon geometry only from frozen node order and flow thickness; do not calculate correlations or select edges.'}, 'B': {'inputs': [get_value('figures.config.item_153')], 'meaning': 'Frozen signed module-level SHAP contributions by ecosystem axis and endpoint.', 'locked_counts': {'rows': 99, 'non_null_plot_value': 99}, 'numeric_guardrail': 'Use plot_value from NEW_module_model directly; do not use the older 81-row signed_contribution_benefit table and do not recalculate SHAP.'}, 'C': {'inputs': [get_value('figures.config.item_136'), get_value('figures.config.item_137'), get_value('figures.config.item_135')], 'meaning': 'Frozen response-oriented ecosystem scores across MBP-I, MBP-II, and MBP-III with frozen global trend and pairwise tests.', 'locked_counts': {'score_rows': 327, 'patients': 109, 'axes': 3, 'global_stats_rows': 3, 'pairwise_rows': 9}, 'numeric_guardrail': 'Use integrated_score, trend_rho, trend_q_BH, and pairwise q_value directly; do not assign MBP groups or reconstruct scores/statistics.', 'benefit_orientation': 'positive = more Responder-like'}}, 'display_label_aliases': {key: {'display': value, 'reason': 'HTML-authoritative display alias; underlying frozen identifier and all values remain unchanged.'} for key, value in DISPLAY_LABEL_ALIASES.items()}, 'allowed_render_operations': ['One-to-one palette remapping while preserving all categories.', 'Sankey ribbon path geometry from frozen node order, sign, and thickness.', 'Violin density geometry and deterministic point jitter from frozen scores.', 'Numeric formatting; bubble significance symbols from frozen nominal p fields in both cohorts; correlation stars copied from star_nominal.', 'HTML-authoritative display aliases that do not alter source identifiers.'], 'forbidden_operations': ['Model fitting or refitting', 'SHAP calculation or recalculation', 'Correlation calculation or recalculation', 'FDR calculation or recalculation', 'Feature or edge selection', 'MBP assignment or patient grouping', 'Ecosystem score calculation, normalization, or orientation changes', 'Changes to frozen effects, p values, q values, or sample sizes']}

def scientific_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    paired = bundle['paired']
    bubble_nominal_only = {cohort: int((pd.to_numeric(paired[f'{cohort}_p'], errors='raise').le(0.05) & pd.to_numeric(paired[f'{cohort}_q_display'], errors='raise').gt(0.05)).sum()) for cohort in ('internal', 'external')}
    return {'figure': 'Figure 7', 'version': VERSION, 'render_only': True, 'authoritative_reference': get_value('figures.config.item_018'), 'palette': PALETTE_ID, 'panel_mapping': {'a': 'Three HTML-ordered ecosystem axes: paired internal/external response effects, paired NR/R KO-to-metabolite Sankey geometry, and frozen main individual-metabolite-to-imaging-phenotype correlations.', 'b': 'Final 99-row NEW_module_model signed module-level SHAP plotting table; displayed plot_value used directly.', 'c': 'Frozen response-oriented ecosystem scores by MBP with frozen trend and pairwise q values.'}, 'scientific_locks': {'radiomics_content': '78-row 11_MAIN frozen individual-metabolite x imaging-phenotype table; exactly the in_main subset of the 138-row within-axis BH family', 'individual_metabolite_radiomics_table_used': True, 'module_aggregate_14E_used': False, 'correlation_stars': 'star_nominal copied verbatim from the frozen table; q_BH_within_axis retained for provenance and not used to generate stars', 'bubble_stars': 'internal_p and external_p in NOMINAL mode for both cohorts; q_display retained for provenance only', 'bubble_nominal_p_le_0.05_but_q_gt_0.05_counts': bubble_nominal_only, 'paired_effect_rows': int(len(bundle['paired'])), 'shap_rows': int(len(bundle['shap'])), 'score_rows': int(len(bundle['scores'])), 'unique_patients': int(bundle['scores']['sample'].nunique()), 'benefit_orientation': sorted(bundle['scores']['score_direction'].astype(str).unique().tolist())}, 'display_only_operations': ['one-to-one palette remapping', 'Sankey ribbon geometry from frozen node and subgroup flow tables', 'violin density geometry from frozen integrated scores', 'deterministic point jitter', 'numeric formatting; bubble significance symbols from frozen nominal p values in both cohorts; correlation stars copied from frozen nominal-p star_nominal'], 'display_label_aliases': DISPLAY_LABEL_ALIASES, 'forbidden_operations_executed': []}

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    input_manifest_path = write_json(input_manifest(), INPUT_DIR / get_value('figures.config.item_116'))
    scientific_contract_path = write_json(scientific_contract(), INPUT_DIR / get_value('figures.config.item_201'))
    fig, registry = build_full_figure(bundle)
    full_clearance = audit_panel_a_text_graphic_clearance(fig)
    if not full_clearance['pass']:
        raise RuntimeError(f"Figure 7A full-figure text/graphic clearance failed: minimum={full_clearance['minimum_clearance_pt']} pt")
    layout = audit_figure_layout(fig, f"{get_value('figures.config.item_300')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout, OUTPUT_DIR / f"{get_value('figures.config.item_295')}{VERSION}.json")
    full_paths = _save_exact(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    standalone_a_clearance: dict[str, Any] | None = None
    for panel in 'abc':
        panel_fig, panel_registry = build_panel_figure(panel, bundle)
        if panel == 'a':
            standalone_a_clearance = audit_panel_a_text_graphic_clearance(panel_fig)
            if not standalone_a_clearance['pass']:
                raise RuntimeError(f"Figure 7A standalone text/graphic clearance failed: minimum={standalone_a_clearance['minimum_clearance_pt']} pt")
        panel_width, panel_height = PANEL_LAYOUTS[panel]['size']
        report = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_294')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        panel_audits[panel] = report
        panel_paths.extend(_save_exact(panel_fig, PANEL_DIR / f"{get_value('figures.config.item_294')}{panel}_Nature_{VERSION}", width_mm=panel_width, height_mm=panel_height))
        plt.close(panel_fig)
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_296')}{VERSION}.json")
    if standalone_a_clearance is None:
        raise RuntimeError('Standalone Figure 7A clearance audit was not generated')
    clearance_path = write_json({'full': full_clearance, 'standalone': standalone_a_clearance}, OUTPUT_DIR / f"{get_value('figures.config.item_302')}{VERSION}.json")
    scientific_path = write_json(scientific_summary(bundle), OUTPUT_DIR / f"{get_value('figures.config.item_298')}{VERSION}.json")
    tracked = full_paths + panel_paths + [layout_path, panel_audit_path, clearance_path, scientific_path]
    manifest = {'figure': 'Figure 7', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked frozen CSV snapshots', 'panel_letters': 'assembly only', 'layout_summary': layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_audits.items()}, 'panel_a_text_graphic_clearance': {'required_mm': PANEL_A_TEXT_GRAPHIC_CLEARANCE_MM, 'full_minimum_mm': full_clearance['minimum_clearance_mm'], 'standalone_minimum_mm': standalone_a_clearance['minimum_clearance_mm'], 'pass': bool(full_clearance['pass'] and standalone_a_clearance['pass'])}, 'provenance_documents': [str(input_manifest_path), str(scientific_contract_path)], 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in tracked]}
    manifest_path = write_json(manifest, OUTPUT_DIR / f"{get_value('figures.config.item_299')}{VERSION}.json")
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'panel_a_text_graphic_clearance': str(clearance_path), 'input_manifest': str(input_manifest_path), 'scientific_contract': str(scientific_contract_path), 'scientific_guardrail_report': str(scientific_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_audits.items()}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
