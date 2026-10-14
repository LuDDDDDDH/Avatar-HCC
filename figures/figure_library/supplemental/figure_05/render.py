from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.patheffects as path_effects
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
VERSION = 'v002'
DATE_TAG = get_value('figures.config.item_350')
INPUT_DIR = PILOT_ROOT / 'inputs'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f'Supplementary_Figure_S5_full_a_Nature_{VERSION}'
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 150.0
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
STAR_FONT_PT = 6.0
STAR_CONTRAST_TARGET = 4.5
BUBBLE_SIZE_PT2_V001 = 30.0
BUBBLE_DIAMETER_SCALE = 1.12
BUBBLE_SIZE_PT2 = BUBBLE_SIZE_PT2_V001 * BUBBLE_DIAMETER_SCALE ** 2
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
INPUT_FILES = {'paired': INPUT_DIR / get_value('figures.config.item_164'), 'correlation': INPUT_DIR / get_value('figures.config.item_115'), 'correlation_family': INPUT_DIR / get_value('figures.config.item_114'), 'legend': INPUT_DIR / get_value('figures.config.item_152'), 'selected_ko': INPUT_DIR / get_value('figures.config.item_202'), 'selected_metabolite': INPUT_DIR / get_value('figures.config.item_204')}
for _slug in AXIS_SLUG.values():
    INPUT_FILES[f'{_slug}_nodes'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_155')}"
    INPUT_FILES[f'{_slug}_nr'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_103')}"
    INPUT_FILES[f'{_slug}_r'] = INPUT_DIR / f"{_slug}{get_value('figures.config.item_104')}"
SOURCE_ROOT = Path(__file__).resolve().parent / 'inputs' / 'source_inputs'
BIOMAP_ROOT = SOURCE_ROOT / get_value('figures.config.item_257')
RADIOMICS_V2_ROOT = SOURCE_ROOT / get_value('figures.config.item_258')
ORIGINAL_SOURCES = {'paired': BIOMAP_ROOT / get_value('figures.path.part_006') / get_value('figures.path.part_003') / get_value('figures.config.item_089'), 'correlation': RADIOMICS_V2_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_038'), 'correlation_family': RADIOMICS_V2_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_037'), 'legend': BIOMAP_ROOT / get_value('figures.config.item_259') / get_value('figures.path.part_003') / get_value('figures.config.item_041'), 'selected_ko': BIOMAP_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_203'), 'selected_metabolite': BIOMAP_ROOT / get_value('figures.path.part_003') / get_value('figures.config.item_035')}
_SANKY_SOURCE_STEM = {'amino': get_value('figures.path.part_010'), 'lipid': get_value('figures.path.part_011'), 'scfa': get_value('figures.path.part_012')}
for _slug, _stem in _SANKY_SOURCE_STEM.items():
    _sankey_root = BIOMAP_ROOT / get_value('figures.path.part_008') / get_value('figures.path.part_005')
    ORIGINAL_SOURCES[f'{_slug}_nodes'] = _sankey_root / f"{_stem}{get_value('figures.config.item_100')}"
    ORIGINAL_SOURCES[f'{_slug}_nr'] = _sankey_root / f"{_stem}{get_value('figures.config.item_098')}"
    ORIGINAL_SOURCES[f'{_slug}_r'] = _sankey_root / f"{_stem}{get_value('figures.config.item_099')}"
HTML_REFERENCE_LOCAL = PILOT_ROOT / get_value('figures.config.item_032')
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
    bundle: dict[str, Any] = {'paired': _read_csv('paired'), 'correlation': _read_csv('correlation'), 'correlation_family': _read_csv('correlation_family'), 'legend': _read_csv('legend'), 'selected_ko': _read_csv('selected_ko'), 'selected_metabolite': _read_csv('selected_metabolite'), 'nodes': {}, 'flows': {}}
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
    _require_columns(bundle['selected_ko'], ['axis', 'ko_id', 'gene_symbol', 'subpathway_en', 'main_rank', 'in_main'], 'selected KO master')
    _require_columns(bundle['selected_metabolite'], ['axis', 'metabolite', 'metabolic_module_plot', 'main_rank', 'in_main'], 'selected metabolite master')
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
    if significance_value < 0.001:
        return '***'
    if significance_value < 0.01:
        return '**'
    if significance_value < 0.05:
        return '*'
    return ''

def _relative_luminance(color: Any) -> float:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

def _contrast_ratio(foreground: Any, background: Any) -> float:
    lighter = max(_relative_luminance(foreground), _relative_luminance(background))
    darker = min(_relative_luminance(foreground), _relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)

def _bubble_star_colors(effect: float) -> tuple[str, str, str, float]:
    fill = mpl.colors.to_hex(EFFECT_CMAP(EFFECT_NORM(effect)), keep_alpha=False).upper()
    candidates = ('#000000', '#FFFFFF')
    text_color = max(candidates, key=lambda candidate: _contrast_ratio(candidate, fill))
    stroke_color = '#FFFFFF' if text_color == '#000000' else '#000000'
    return (fill, text_color, stroke_color, _contrast_ratio(text_color, fill))

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
    for xi, (cohort, effect_col, p_col) in enumerate((('Internal', 'internal_effect', 'internal_p'), ('External', 'external_effect', 'external_p'))):
        vals = pd.to_numeric(rows[effect_col], errors='raise').to_numpy(float)
        ax.scatter(np.full(n, xi), y, s=BUBBLE_SIZE_PT2, c=vals, cmap=EFFECT_CMAP, norm=EFFECT_NORM, edgecolors='white', linewidths=0.35, zorder=2)
        for yi, p_value, value, feature, axis_name in zip(y, rows[p_col], vals, rows['feature'], rows['axis']):
            stars = _stars(p_value)
            if stars:
                fill, text_color, stroke_color, contrast = _bubble_star_colors(float(value))
                artist = ax.text(xi, yi, stars, ha='center', va='center', fontsize=STAR_FONT_PT, fontweight='bold', color=text_color, alpha=1.0, clip_on=False, path_effects=[path_effects.Stroke(linewidth=0.35, foreground=stroke_color), path_effects.Normal()], zorder=3)
                artist._supps5_star_metadata = {'axis': str(axis_name), 'feature': str(feature), 'side': side, 'cohort': cohort, 'stars': stars, 'p_value': float(p_value), 'effect': float(value), 'fill': fill, 'text': text_color, 'stroke': stroke_color, 'contrast_ratio': float(contrast)}
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
    ax.pcolormesh(np.arange(matrix.shape[1] + 1) - 0.5, np.arange(matrix.shape[0] + 1) - 0.5, matrix, cmap=EFFECT_CMAP, norm=RHO_NORM, shading='flat', edgecolors='none', antialiased=False, rasterized=False)
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
    headers = ((0.13, 'KO response effect\nInternal | External'), (0.405, 'KO–metabolite association\nNR | R'), (0.728, 'Metabolite response effect\nInternal | External'), (0.905, 'Imaging phenotype\nLow-gray | Coarse | Fine'))
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
    gradient_values = np.linspace(-1, 1, 129)
    gradient = np.tile(gradient_values, (2, 1))
    effect_x, effect_y = np.meshgrid(gradient_values, np.array([0.0, 1.0]))
    effect_ax.pcolormesh(effect_x, effect_y, gradient, cmap=EFFECT_CMAP, norm=Normalize(vmin=-1.0, vmax=1.0), shading='gouraud', edgecolors='none', rasterized=False)
    effect_ax.set_xticks([-1, 0, 1], labels=['−1', '0', '1'])
    effect_ax.set_yticks([])
    effect_ax.set_title('Response effect z', fontsize=MIN_FONT_PT, pad=1.5)
    effect_ax.tick_params(axis='x', length=1.5, pad=1)
    corr_x, corr_y = np.meshgrid(np.linspace(-0.4, 0.4, 129), np.array([0.0, 1.0]))
    corrbar_ax.pcolormesh(corr_x, corr_y, gradient, cmap=EFFECT_CMAP, norm=Normalize(vmin=-1.0, vmax=1.0), shading='gouraud', edgecolors='none', rasterized=False)
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

def audit_bubble_star_accessibility(fig: mpl.figure.Figure, bundle: Mapping[str, Any]) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    records: list[dict[str, Any]] = []
    tolerance_px = 0.75
    for ax in fig.axes:
        axis_box = ax.get_window_extent(renderer)
        for artist in ax.texts:
            metadata = getattr(artist, '_supps5_star_metadata', None)
            if metadata is None:
                continue
            bbox = artist.get_window_extent(renderer)
            not_clipped = bool(bbox.x0 >= axis_box.x0 - tolerance_px and bbox.x1 <= axis_box.x1 + tolerance_px and (bbox.y0 >= axis_box.y0 - tolerance_px) and (bbox.y1 <= axis_box.y1 + tolerance_px))
            alpha = 1.0 if artist.get_alpha() is None else float(artist.get_alpha())
            record = {**metadata, 'font_size_pt': float(artist.get_fontsize()), 'font_weight': str(artist.get_fontweight()), 'font_weight_numeric': 700, 'alpha': alpha, 'zorder': float(artist.get_zorder()), 'bbox_within_axes': not_clipped, 'bbox_pixels': [round(float(value), 3) for value in bbox.bounds], 'axes_bbox_pixels': [round(float(value), 3) for value in axis_box.bounds]}
            record['pass'] = bool(float(record['contrast_ratio']) >= STAR_CONTRAST_TARGET and alpha == 1.0 and (float(record['font_size_pt']) >= 5.9) and (int(record['font_weight_numeric']) >= 700) and (float(record['zorder']) > 2.0) and not_clipped)
            records.append(record)
    if not records:
        raise RuntimeError('No circular response-effect stars were found for accessibility QA')
    expected: dict[tuple[str, str, str], str] = {}
    for row in bundle['paired'].itertuples(index=False):
        for cohort, p_column in (('Internal', 'internal_p'), ('External', 'external_p')):
            stars = _stars(getattr(row, p_column))
            if stars:
                expected[str(row.axis), str(row.feature), cohort] = stars
    actual = {(str(record['axis']), str(record['feature']), str(record['cohort'])): str(record['stars']) for record in records}
    exact_nominal_p_star_match = actual == expected
    for record in records:
        key = (str(record['axis']), str(record['feature']), str(record['cohort']))
        record['expected_stars_from_nominal_p'] = expected.get(key, '')
        record['exact_nominal_p_star_match'] = record['stars'] == record['expected_stars_from_nominal_p']
        record['pass'] = bool(record['pass'] and record['exact_nominal_p_star_match'])
    minimum_contrast = min((float(record['contrast_ratio']) for record in records))
    median_contrast = float(np.median([float(record['contrast_ratio']) for record in records]))
    report = {'target_contrast_ratio': STAR_CONTRAST_TARGET, 'candidate_text_colors': ['#000000', '#FFFFFF'], 'stroke_width_pt': 0.35, 'bubble_marker_area_pt2': BUBBLE_SIZE_PT2, get_value('figures.config.item_348'): (BUBBLE_DIAMETER_SCALE - 1.0) * 100.0, 'star_count': len(records), 'expected_star_count': len(expected), 'star_text_distribution': {symbol: sum((record['stars'] == symbol for record in records)) for symbol in ('*', '**', '***')}, 'cohort_star_counts': {cohort: sum((record['cohort'] == cohort for record in records)) for cohort in ('Internal', 'External')}, 'exact_nominal_p_star_match': exact_nominal_p_star_match, 'minimum_contrast_ratio': minimum_contrast, 'median_contrast_ratio': median_contrast, 'all_alpha_one': all((record['alpha'] == 1.0 for record in records)), 'all_font_weight_at_least_700': all((int(record['font_weight_numeric']) >= 700 for record in records)), 'all_font_size_at_least_5_9_pt': all((float(record['font_size_pt']) >= 5.9 for record in records)), 'all_zorder_above_bubbles': all((float(record['zorder']) > 2.0 for record in records)), 'none_clipped': all((record['bbox_within_axes'] for record in records)), 'pass': bool(len(records) == 91 and len(expected) == 91 and exact_nominal_p_star_match and (minimum_contrast >= STAR_CONTRAST_TARGET) and (median_contrast >= 7.0) and all((record['pass'] for record in records))), 'stars': records}
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
    ax.text(0.5, 0.98, f'ρ={trend_rho:.2f}, q{_format_q(trend_q)}\nI–III q{_format_q(pair_q)}', transform=ax.transAxes, ha='center', va='top', fontsize=MIN_FONT_PT, linespacing=0.95)
    ax.set_title(AXIS_LABEL[axis_name], fontsize=5.8, pad=1.5)
    ax.set_xticks(positions, labels=('I', 'II', 'III'))
    if show_ylabel:
        ax.set_ylabel('Response-oriented ecosystem score', fontsize=MIN_FONT_PT, labelpad=1.5)
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
FULL_PANELS_MM = {'a': (5.0, 4.0, 171.0, 142.0)}
PANEL_LAYOUTS = {'a': {'size': (180.0, 150.0), 'rect': (5.0, 4.0, 171.0, 142.0)}}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    draw_panel_a(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANELS_MM['a'], bundle, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'a', 1.5, 2.0, registry)
    return (fig, registry)

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    panel = panel.lower()
    if panel != 'a':
        raise ValueError(panel)
    configure_style()
    width_mm, height_mm = PANEL_LAYOUTS[panel]['size']
    rect = PANEL_LAYOUTS[panel]['rect']
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    draw_panel_a(fig, width_mm, height_mm, rect, bundle, registry)
    _panel_letter(fig, width_mm, height_mm, panel, 1.5, 2.0, registry)
    return (fig, registry)

def _save_exact(fig: mpl.figure.Figure, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S5a {VERSION} - {PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
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
        raise FileNotFoundError('Supplementary Figure S5 HTML reference image is missing')
    reference_hash = sha256_file(HTML_REFERENCE_LOCAL).upper()
    reference_source_hash = sha256_file(HTML_REFERENCE_SOURCE).upper()
    if reference_hash != reference_source_hash:
        raise ValueError('Local Supplementary Figure S5 HTML reference does not match the authoritative former Figure 7A source')
    return {'figure': 'Supplementary Figure S5', 'snapshot_date': '2026-09-02', 'snapshot_root': INPUT_DIR.as_posix(), 'copy_policy': 'Fifteen byte-for-byte copies of the minimum frozen tables required for former Figure 7A only; no source file or v001 artifact was edited.', 'source_version_policy': get_value('figures.provenance.source_policy'), 'authoritative_html_reference': {'snapshot': HTML_REFERENCE_LOCAL.name, 'source': HTML_REFERENCE_SOURCE.as_posix(), 'bytes': int(HTML_REFERENCE_LOCAL.stat().st_size), 'sha256': reference_hash, 'source_sha256': reference_source_hash, 'byte_for_byte_match': True}, 'files': records}

def scientific_contract() -> dict[str, Any]:
    return {'figure': 'Supplementary Figure S5', 'version': VERSION, 'render_mode': 'frozen-source render-only', 'content_origin': 'Author-approved former Figure 7A, moved intact to Supplementary Figure S5a in v002.', 'authoritative_content_and_order': HTML_REFERENCE_LOCAL.name, 'working_master_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'ecosystem_axis_order': list(AXIS_ORDER), 'panel_mapping': {'a': {'inputs': [get_value('figures.config.item_164'), get_value('figures.config.item_202'), get_value('figures.config.item_204'), get_value('figures.config.item_056'), get_value('figures.config.item_054'), get_value('figures.config.item_055'), get_value('figures.config.item_127'), get_value('figures.config.item_125'), get_value('figures.config.item_126'), get_value('figures.config.item_199'), get_value('figures.config.item_197'), get_value('figures.config.item_198'), get_value('figures.config.item_115'), get_value('figures.config.item_114'), get_value('figures.config.item_152')], 'meaning': 'Three ecosystem axes with frozen internal/external response effects, NR/R KO-to-metabolite links, and individual-metabolite correlations with three imaging phenotypes.', 'locked_counts': {'paired_effect_rows': 54, 'main_correlation_rows': 78, 'main_axis_row_counts': [33, 27, 18], 'full_fdr_family_rows': 138, 'main_axis_metabolite_entries': 26, 'metabolic_legend_items': 13}, 'correlation_guardrail': 'Plot rho from 11_MAIN directly. 11_MAIN must exactly equal in_main=TRUE from 10_ALL. q_BH_within_axis belongs to the full 138-test, per-axis BH families.', 'star_guardrail': 'Display star_nominal copied from the frozen table; these stars encode nominal p, not BH q.', 'bubble_star_guardrail': "For both Internal and External bubbles, derive display stars only from the already frozen internal_p/external_p values using the original notebook's NOMINAL mode; internal_q_display/external_q_display are provenance only.", 'anchor_guardrail': 'Ornithine must be the first Amino row with rho -0.281086741467582, 0.12528061644139, -0.174089922366148 in Low-gray, Coarse, Fine order (displayed as -0.28, 0.13, -0.17).', 'geometry_guardrail': 'Reconstruct ribbon geometry only from frozen node order and flow thickness; do not calculate correlations or select edges.'}}, 'star_accessibility_contract': {'scope': 'All 91 non-empty stars inside Internal/External response-effect circles only.', 'text_source': 'Zero-tolerance derivation from frozen internal_p/external_p nominal P values; star strings cannot change.', 'expected_counts': {'all': 91, 'Internal': 53, 'External': 38, '*': 19, '**': 39, '***': 33}, 'color_rule': 'For each final bubble fill, select whichever of #000000 and #FFFFFF has the larger WCAG sRGB contrast ratio; pure black is the deep blue-black accessibility fallback needed to guarantee >=4.5:1 on every frozen fill.', 'minimum_contrast_ratio': STAR_CONTRAST_TARGET, 'minimum_median_contrast_ratio': 7.0, 'font_size_pt': STAR_FONT_PT, 'font_weight': 700, 'alpha': 1.0, 'opposite_color_stroke_pt': 0.35, get_value('figures.config.item_348'): (BUBBLE_DIAMETER_SCALE - 1.0) * 100.0}, 'display_label_aliases': {key: {'display': value, 'reason': 'HTML-authoritative display alias; underlying frozen identifier and all values remain unchanged.'} for key, value in DISPLAY_LABEL_ALIASES.items()}, 'allowed_render_operations': ['One-to-one palette remapping while preserving all categories.', 'Sankey ribbon path geometry from frozen node order, sign, and thickness.', 'Numeric formatting; bubble significance symbols from frozen nominal p fields in both cohorts; correlation stars copied from star_nominal.', 'Per-bubble high-contrast star color selection, full opacity, bold weight, vector outline, and marker-size increase without changing any star string.', 'HTML-authoritative display aliases that do not alter source identifiers.'], 'forbidden_operations': ['Model fitting or refitting', 'SHAP calculation or recalculation', 'Correlation calculation or recalculation', 'FDR calculation or recalculation', 'Feature or edge selection', 'Changes to frozen effects, p values, q values, or sample sizes']}

def scientific_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    paired = bundle['paired']
    bubble_nominal_only = {cohort: int((pd.to_numeric(paired[f'{cohort}_p'], errors='raise').le(0.05) & pd.to_numeric(paired[f'{cohort}_q_display'], errors='raise').gt(0.05)).sum()) for cohort in ('internal', 'external')}
    return {'figure': 'Supplementary Figure S5', 'version': VERSION, 'render_only': True, 'content_origin': 'former Figure 7A', 'authoritative_reference': HTML_REFERENCE_LOCAL.name, 'palette': PALETTE_ID, 'panel_mapping': {'a': 'Three HTML-ordered ecosystem axes: paired internal/external response effects, paired NR/R KO-to-metabolite Sankey geometry, and frozen main individual-metabolite-to-imaging-phenotype correlations.'}, 'scientific_locks': {'radiomics_content': '78-row 11_MAIN frozen individual-metabolite x imaging-phenotype table; exactly the in_main subset of the 138-row within-axis BH family', 'individual_metabolite_radiomics_table_used': True, 'module_aggregate_14E_used': False, 'correlation_stars': 'star_nominal copied verbatim from the frozen table; q_BH_within_axis retained for provenance and not used to generate stars', 'bubble_stars': 'internal_p and external_p in NOMINAL mode for both cohorts; q_display retained for provenance only', 'bubble_nominal_p_le_0.05_but_q_gt_0.05_counts': bubble_nominal_only, 'paired_effect_rows': int(len(bundle['paired'])), 'bubble_star_count': int(pd.to_numeric(bundle['paired']['internal_p'], errors='raise').le(0.05).sum() + pd.to_numeric(bundle['paired']['external_p'], errors='raise').le(0.05).sum())}, 'display_only_operations': ['one-to-one palette remapping', 'Sankey ribbon geometry from frozen node and subgroup flow tables', 'numeric formatting; bubble significance symbols from frozen nominal p values in both cohorts; correlation stars copied from frozen nominal-p star_nominal', 'WCAG contrast-based black/white star color selection with full opacity, bold weight and vector halo', '12% response-bubble diameter increase for improved star legibility'], 'display_label_aliases': DISPLAY_LABEL_ALIASES, 'forbidden_operations_executed': []}

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
        raise RuntimeError(f"Supplementary Figure S5a full-figure text/graphic clearance failed: minimum={full_clearance['minimum_clearance_pt']} pt")
    full_star_qa = audit_bubble_star_accessibility(fig, bundle)
    if not full_star_qa['pass']:
        raise RuntimeError(f'Supplementary Figure S5a full bubble-star accessibility failed: {full_star_qa}')
    layout = audit_figure_layout(fig, f'Supplementary_Figure_S5_full_a_Nature_{VERSION}', registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = write_json(layout, OUTPUT_DIR / f'Supplementary_Figure_S5_layout_and_text_audit_{VERSION}.json')
    full_paths = _save_exact(fig, FIGURE_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    plt.close(fig)
    panel_fig, panel_registry = build_panel_figure('a', bundle)
    standalone_a_clearance = audit_panel_a_text_graphic_clearance(panel_fig)
    if not standalone_a_clearance['pass']:
        raise RuntimeError(f"Supplementary Figure S5a standalone text/graphic clearance failed: minimum={standalone_a_clearance['minimum_clearance_pt']} pt")
    standalone_star_qa = audit_bubble_star_accessibility(panel_fig, bundle)
    if not standalone_star_qa['pass']:
        raise RuntimeError(f'Supplementary Figure S5a standalone bubble-star accessibility failed: {standalone_star_qa}')
    panel_width, panel_height = PANEL_LAYOUTS['a']['size']
    panel_report = audit_figure_layout(panel_fig, f'Supplementary_Figure_S5_panel_a_Nature_{VERSION}', panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    panel_paths = _save_exact(panel_fig, PANEL_DIR / f'Supplementary_Figure_S5_panel_a_Nature_{VERSION}', width_mm=panel_width, height_mm=panel_height)
    plt.close(panel_fig)
    panel_audits = {'a': panel_report}
    panel_audit_path = write_json(panel_audits, OUTPUT_DIR / f'Supplementary_Figure_S5_panel_layout_audit_{VERSION}.json')
    clearance_path = write_json({'full': full_clearance, 'standalone': standalone_a_clearance}, OUTPUT_DIR / f'Supplementary_Figure_S5_panel_a_text_graphic_clearance_{VERSION}.json')
    star_qa_path = write_json({'full': full_star_qa, 'standalone': standalone_star_qa}, OUTPUT_DIR / f'Supplementary_Figure_S5_bubble_star_accessibility_QA_{VERSION}.json')
    scientific_path = write_json(scientific_summary(bundle), OUTPUT_DIR / f'Supplementary_Figure_S5_scientific_guardrail_report_{VERSION}.json')
    tracked = full_paths + panel_paths + [layout_path, panel_audit_path, clearance_path, star_qa_path, scientific_path]
    manifest = {'figure': 'Supplementary Figure S5', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked frozen CSV snapshots', 'formal_panels': ['a'], 'full_and_panel_a_same_content_geometry': True, 'layout_summary': layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_audits.items()}, 'panel_a_text_graphic_clearance': {'required_mm': PANEL_A_TEXT_GRAPHIC_CLEARANCE_MM, 'full_minimum_mm': full_clearance['minimum_clearance_mm'], 'standalone_minimum_mm': standalone_a_clearance['minimum_clearance_mm'], 'pass': bool(full_clearance['pass'] and standalone_a_clearance['pass'])}, 'bubble_star_accessibility': {'full_star_count': full_star_qa['star_count'], 'standalone_star_count': standalone_star_qa['star_count'], 'full_minimum_contrast_ratio': full_star_qa['minimum_contrast_ratio'], 'standalone_minimum_contrast_ratio': standalone_star_qa['minimum_contrast_ratio'], 'full_median_contrast_ratio': full_star_qa['median_contrast_ratio'], 'standalone_median_contrast_ratio': standalone_star_qa['median_contrast_ratio'], 'exact_nominal_p_star_match': bool(full_star_qa['exact_nominal_p_star_match'] and standalone_star_qa['exact_nominal_p_star_match']), 'pass': bool(full_star_qa['pass'] and standalone_star_qa['pass'])}, 'provenance_documents': [str(input_manifest_path), str(scientific_contract_path)], 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in tracked]}
    manifest_path = write_json(manifest, OUTPUT_DIR / f'Supplementary_Figure_S5_export_manifest_{VERSION}.json')
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'panel_a_text_graphic_clearance': str(clearance_path), 'bubble_star_accessibility_qa': str(star_qa_path), 'input_manifest': str(input_manifest_path), 'scientific_contract': str(scientific_contract_path), 'scientific_guardrail_report': str(scientific_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_audits.items()}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
