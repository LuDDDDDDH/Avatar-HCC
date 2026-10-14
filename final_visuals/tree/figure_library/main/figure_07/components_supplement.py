from __future__ import annotations
from avatar_paths import get_value, get_column
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, MBP_COLORS, NEUTRAL_COLORS, PALETTE_ID, apply_theme
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
INPUT_DIR = ROOT / 'inputs'
VERSION = 'v001'
OUTPUT_DIR = ROOT / 'outputs' / get_value('figures.config.item_355')
PANEL_DIR = OUTPUT_DIR / 'panels'
FULL_STEM = OUTPUT_DIR / f'Supplementary_Figure_S5_full_a_Nature_{VERSION}'
PANEL_STEM = PANEL_DIR / f'Supplementary_Figure_S5_panel_a_Nature_{VERSION}'
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 145.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
FONT = {'base': 6.0, 'panel': 8.5, 'facet': 5.6, 'axis': 6.0, 'tick': 5.4, 'label': 5.6, 'small': 5.4}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
REFERENCE = NEUTRAL_COLORS['reference']
LIGHT = NEUTRAL_COLORS['light_background']
NEGATIVE = '#2F6FB0'
POSITIVE = '#DE646C'
NEUTRAL = DIVERGING_COLORS[len(DIVERGING_COLORS) // 2]
MODULE_COLOR_MAP = {'#00A087': '#2A8F78', '#20854E': '#4F9D73', '#3C5488': '#3E5F91', '#4DBBD5': '#2F91C3', '#5B9BD5': '#6CA8D1', '#6A6599': '#7163A6', '#6F6FAE': '#7D70D1', '#7E6148': '#95735E', '#7F95B8': '#7892B6', '#8491B4': '#617FC4', '#8F7700': '#A68A38', '#91D1C2': '#78B8AB', '#DC0000': '#C44A58', '#E64B35': '#DE646C', '#F39B7F': '#E49592', '#FFDC91': '#D9B866'}
AXES = OrderedDict([('Amino', {'axis': 'Amino Acids & TME Regulation Axis', 'display': 'Amino acids & TME', 'ko_pdf': get_value('figures.config.item_001'), 'met_pdf': get_value('figures.config.item_002'), 'flow_i': get_value('figures.config.item_051'), 'flow_iii': get_value('figures.config.item_052'), 'nodes': get_value('figures.config.item_053')}), ('Lipid', {'axis': 'Lipid & 1-Carbon Metabolism Axis', 'display': 'Lipid & 1-carbon', 'ko_pdf': get_value('figures.config.item_024'), 'met_pdf': get_value('figures.config.item_025'), 'flow_i': get_value('figures.config.item_122'), 'flow_iii': get_value('figures.config.item_123'), 'nodes': get_value('figures.config.item_124')}), ('SCFA', {'axis': 'SCFA & BCAA Energy Metabolism Axis', 'display': 'SCFA & BCAA', 'ko_pdf': get_value('figures.config.item_030'), 'met_pdf': get_value('figures.config.item_031'), 'flow_i': get_value('figures.config.item_194'), 'flow_iii': get_value('figures.config.item_195'), 'nodes': get_value('figures.config.item_196')})])
MODULE_DISPLAY = {'Arginine / polyamines': 'Arginine →\npolyamines', 'Purine nucleosides': 'Purine nucleoside\nmetabolism', 'Tryptophan / indole': 'Tryptophan →\nindole', 'One-carbon / methylation': 'One-carbon\nmetabolism →\nmethylation', 'Phospholipid remodeling': 'Phospholipid\nremodeling', 'Acetate production': 'Acetate\nproduction', 'Butyrate production': 'Butyrate\nproduction', 'Propionate metabolism': 'Propionate\nmetabolism'}
MODULE_ORDER = ['Low-gray-weighted phenotype', 'Coarse structural organization', 'Fine-scale heterogeneity']
MODULE_TITLE = {'Low-gray-weighted phenotype': 'Low-gray-weighted\nphenotype', 'Coarse structural organization': 'Coarse structural\norganization', 'Fine-scale heterogeneity': 'Fine-scale\nheterogeneity'}
MBP_ORDER = ['MBP-I', 'MBP-II', 'MBP-III']

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT['base'], 'axes.labelsize': FONT['axis'], 'axes.titlesize': FONT['facet'], 'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'], 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False, 'figure.max_open_warning': 0})

def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing columns: {missing}')

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def load_inputs() -> dict[str, Any]:
    manifest = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8-sig'))
    visual = json.loads((INPUT_DIR / get_value('figures.config.item_105')).read_text(encoding='utf-8'))
    bundle: dict[str, Any] = {'manifest': manifest, 'visual': visual, 'ko_master': pd.read_csv(INPUT_DIR / get_value('figures.config.item_203')), 'met_master': pd.read_csv(INPUT_DIR / get_value('figures.config.item_205')), 'radiomics': pd.read_csv(INPUT_DIR / get_value('figures.config.item_183')), 'axes': {}}
    for slug, spec in AXES.items():
        bundle['axes'][slug] = {'flow_i': pd.read_csv(INPUT_DIR / spec['flow_i']), 'flow_iii': pd.read_csv(INPUT_DIR / spec['flow_iii']), 'nodes': pd.read_csv(INPUT_DIR / spec['nodes'])}
    validate_inputs(bundle)
    return bundle

def validate_inputs(bundle: Mapping[str, Any]) -> None:
    manifest = bundle['manifest']
    if manifest.get('figure') != 'Supplementary Figure S5' or len(manifest.get('files', [])) != 20:
        raise ValueError('S5 input manifest identity or minimal file count drifted')
    for entry in manifest['files']:
        local = INPUT_DIR / entry['snapshot']
        if not local.exists() or _sha(local) != str(entry['sha256']).upper():
            raise ValueError(f"Frozen input hash drift: {entry['snapshot']}")
        if entry.get('kind') != 'derived_display_encoding':
            source = Path(entry['source'])
            if not source.exists() or _sha(source) != str(entry['sha256']).upper():
                raise ValueError(f"Authoritative source hash drift: {entry['snapshot']}")
    visual = bundle['visual']
    if visual.get('schema') != get_value('figures.config.item_371') or not visual.get('render_only'):
        raise ValueError('Frozen bubble visual schema drifted')
    expected_rows = {('Amino', 'ko'): 9, ('Amino', 'metabolite'): 11, ('Lipid', 'ko'): 15, ('Lipid', 'metabolite'): 9, ('SCFA', 'ko'): 4, ('SCFA', 'metabolite'): 6}
    found = {}
    for panel in visual['panels']:
        key = (panel['axis_slug'], panel['layer'])
        found[key] = len(panel['rows'])
        source = INPUT_DIR / panel['source_pdf']
        if _sha(source) != panel['source_pdf_sha256']:
            raise ValueError(f'Bubble source hash drift: {source.name}')
        labels = [row['label'] for row in panel['rows']]
        if len(labels) != len(set(labels)) or any((row['order'] != i for i, row in enumerate(panel['rows'], 1))):
            raise ValueError(f'Bubble label/order drift in {key}')
        if any((row['significance'] not in {'', '*', '**', '***'} for row in panel['rows'])):
            raise ValueError(f'Unexpected bubble significance string in {key}')
    if found != expected_rows:
        raise ValueError(f'Bubble row membership drifted: {found}')
    flow_counts = {'Amino': 13, 'Lipid': 19, 'SCFA': 8}
    for slug, values in bundle['axes'].items():
        nodes = values['nodes']
        _require_columns(nodes, ['node_id', 'display_name', 'side', 'module', 'color', 'plot_order'], f'{slug} nodes')
        if nodes['node_id'].duplicated().any():
            raise ValueError(f'{slug} node dictionary is not unique')
        for group_key, group_label, n_pair in (('flow_i', 'MBP-I', 37), ('flow_iii', 'MBP-III', 36)):
            flow = values[group_key]
            _require_columns(flow, ['source_node', 'target_node', 'rho', 'thickness', 'direction', 'panel', 'n_pair', 'axis'], f'{slug} {group_label} flow')
            if len(flow) != flow_counts[slug] or set(flow['panel']) != {group_label}:
                raise ValueError(f'{slug} {group_label} flow row/group drift')
            if set(flow['n_pair'].astype(int)) != {n_pair}:
                raise ValueError(f'{slug} {group_label} pair count drift')
            if not np.allclose(flow['thickness'], flow['rho'].abs(), atol=1e-12):
                raise ValueError(f'{slug} {group_label} frozen thickness no longer equals abs(rho)')
            active = set(flow['source_node']) | set(flow['target_node'])
            if not active.issubset(set(nodes['node_id'])):
                raise ValueError(f'{slug} {group_label} contains nodes absent from dictionary')
    radiomics = bundle['radiomics']
    _require_columns(radiomics, ['sample', 'module', 'module_score', 'MBP'], 'radiomics scores')
    if len(radiomics) != 327 or radiomics['sample'].nunique() != 109:
        raise ValueError('Radiomics module score dimensions drifted')
    if set(radiomics['module']) != set(MODULE_ORDER) or set(radiomics['MBP']) != set(MBP_ORDER):
        raise ValueError('Radiomics module or MBP membership drifted')
    counts = radiomics.groupby(['module', 'MBP']).size().unstack()
    if not (counts.loc[:, MBP_ORDER].to_numpy() == np.array([37, 36, 36])).all():
        raise ValueError(f'Radiomics MBP counts drifted: {counts.to_dict()}')

def visual_panel(bundle: Mapping[str, Any], slug: str, layer: str) -> dict:
    return next((panel for panel in bundle['visual']['panels'] if panel['axis_slug'] == slug and panel['layer'] == layer))

def module_lookup(bundle: Mapping[str, Any], axis: str, layer: str, label: str) -> str:
    if layer == 'ko':
        frame = bundle['ko_master']
        match = frame.loc[(frame['axis'] == axis) & (frame['ko_alias_display'].astype(str) == label)]
        column = 'subpathway_en'
    else:
        frame = bundle['met_master']
        candidates = {label}
        if label == 'Coumaric acid (U)':
            candidates.add('Coumaric acid (isomer of 192, 194) (U)')
        if label == 'Allysine (U)':
            candidates.add('Allysine (not validated) (U)')
        match = frame.loc[(frame['axis'] == axis) & frame['metabolite'].astype(str).isin(candidates)]
        column = 'metabolic_module_plot'
    if len(match) != 1:
        raise ValueError(f'Expected one module match for {axis} / {layer} / {label}; found {len(match)}')
    return str(match.iloc[0][column])

def _interp(start: str, end: str, fraction: float) -> str:
    a = np.array(to_rgb(start), dtype=float)
    b = np.array(to_rgb(end), dtype=float)
    return to_hex(a + np.clip(fraction, 0.0, 1.0) * (b - a), keep_alpha=False).upper()

def aurora_module_colour(original: str) -> str:
    return MODULE_COLOR_MAP.get(str(original).upper(), str(original).upper())

def aurora_effect_colour(original: str) -> str:
    red, green, blue = (int(original[i:i + 2], 16) for i in (1, 3, 5))
    if red >= blue:
        strength = min(1.0, max(0.1, abs(red - blue) / 177.0))
        return _interp(NEUTRAL, POSITIVE, strength)
    strength = min(1.0, max(0.1, abs(red - blue) / 76.0))
    return _interp(NEUTRAL, NEGATIVE, strength)

def _contiguous_groups(rows: list[dict]) -> list[tuple[int, int, str, str]]:
    groups: list[tuple[int, int, str, str]] = []
    start = 0
    current = rows[0]['module']
    colour = rows[0]['module_color_original']
    for index, row in enumerate(rows[1:], start=1):
        if row['module'] != current:
            groups.append((start, index - 1, current, colour))
            start, current, colour = (index, row['module'], row['module_color_original'])
    groups.append((start, len(rows) - 1, current, colour))
    return groups

def draw_bubble_panel(ax: mpl.axes.Axes, bundle: Mapping[str, Any], slug: str, layer: str, registry: ArtistRegistry) -> None:
    panel = visual_panel(bundle, slug, layer)
    rows = [dict(row) for row in panel['rows']]
    for row in rows:
        row['module'] = module_lookup(bundle, panel['axis'], layer, row['label'])
    n = len(rows)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(n - 0.35, -1.05)
    ax.axis('off')
    if layer == 'ko':
        strip_x, dot_x, label_x, module_x = (0.57, 0.675, 0.755, 0.52)
        header = ax.text(0.015, -0.72, AXES[slug]['display'], ha='left', va='center', fontsize=FONT['facet'], fontweight='bold', color=TEXT, clip_on=False)
        registry.tag(header, name=f'{slug}-axis-name', role='direct_label', panel='a')
    else:
        strip_x, dot_x, label_x, module_x = (0.035, 0.135, 0.225, 0.0)
    for start, end, module, colour in _contiguous_groups(rows):
        ax.add_patch(Rectangle((strip_x - 0.022, start - 0.43), 0.044, end - start + 0.86, facecolor=aurora_module_colour(colour), edgecolor='white', linewidth=0.35, clip_on=True))
        if layer == 'ko':
            module_artist = ax.text(module_x, (start + end) / 2.0, MODULE_DISPLAY.get(module, module.replace(' / ', ' /\n')), ha='right', va='center', fontsize=FONT['small'], fontweight='semibold', linespacing=0.94, color=TEXT, clip_on=True)
            registry.tag(module_artist, name=f'{slug}-{layer}-module-{start}', role='direct_label', panel='a')
    for index, row in enumerate(rows):
        diameter = max(7.7, float(row['bubble_diameter_pt_original']) * 1.02)
        ax.scatter([dot_x], [index], s=diameter ** 2, marker='o', facecolor=aurora_effect_colour(row['bubble_color_original']), edgecolor='#6C7885', linewidth=0.32, zorder=3, clip_on=True)
        if row['significance']:
            ax.text(dot_x, index - 0.01, row['significance'], ha='center', va='center', fontsize=FONT['small'], fontweight='bold', color='#101820', zorder=4, clip_on=True)
        label_artist = ax.text(label_x, index, row['label'], ha='left', va='center', fontsize=FONT['label'], color=TEXT, clip_on=True)
        registry.tag(label_artist, name=f'{slug}-{layer}-feature-{index}', role='direct_label', panel='a')
    compare = ax.text(dot_x, -0.72, 'III vs I', ha='center', va='center', fontsize=FONT['small'], color=MUTED, clip_on=True)
    registry.tag(compare, name=f'{slug}-{layer}-comparison', role='small_value', panel='a')

def _paired_connectivity(flow_i: pd.DataFrame, flow_iii: pd.DataFrame) -> dict[str, float]:
    panel_weights = []
    for flow in (flow_i, flow_iii):
        weights: dict[str, float] = {}
        for row in flow.itertuples(index=False):
            weight = float(row.thickness)
            weights[str(row.source_node)] = weights.get(str(row.source_node), 0.0) + weight
            weights[str(row.target_node)] = weights.get(str(row.target_node), 0.0) + weight
        panel_weights.append(weights)
    all_nodes = set(panel_weights[0]) | set(panel_weights[1])
    return {node: max(panel_weights[0].get(node, 0.0), panel_weights[1].get(node, 0.0)) for node in all_nodes}

def _paired_y_map(nodes: pd.DataFrame, active: set[str], connectivity: Mapping[str, float], side: str) -> dict[str, float]:
    meta = nodes.loc[(nodes['side'] == side) & nodes['node_id'].isin(active)].copy()
    meta = meta.sort_values(['plot_order', 'node_id'], kind='stable')
    module_order = meta.groupby('module', sort=False)['plot_order'].min().sort_values(kind='stable').index.tolist()
    meta['module_rank'] = meta['module'].map({module: rank for rank, module in enumerate(module_order)})
    meta = meta.sort_values(['module_rank', 'plot_order', 'node_id'], kind='stable')
    available = 1.0 - 2 * 0.045 - 0.035 * max(0, len(module_order) - 1)
    masses = {str(node): max(0.1, float(connectivity.get(str(node), 0.0))) ** 0.45 for node in meta['node_id']}
    total_mass = sum(masses.values())
    cursor = 0.045
    y_map: dict[str, float] = {}
    for module_index, module in enumerate(module_order):
        subset = meta.loc[meta['module'] == module]
        module_mass = sum((masses[str(node)] for node in subset['node_id']))
        module_height = available * module_mass / total_mass
        local = cursor
        for node in subset['node_id']:
            node_height = module_height * masses[str(node)] / module_mass
            y_map[str(node)] = 1.0 - (local + 0.5 * node_height)
            local += node_height
        cursor += module_height
        if module_index < len(module_order) - 1:
            cursor += 0.035
    return y_map

def _paired_layout(flow_i: pd.DataFrame, flow_iii: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, dict[str, float]]:
    active = set(flow_i['source_node']) | set(flow_i['target_node'])
    active |= set(flow_iii['source_node']) | set(flow_iii['target_node'])
    connectivity = _paired_connectivity(flow_i, flow_iii)
    return {'source': _paired_y_map(nodes, active, connectivity, 'source'), 'target': _paired_y_map(nodes, active, connectivity, 'target')}

def _group_scale(flow: pd.DataFrame, y_maps: Mapping[str, Mapping[str, float]]) -> float:
    total = float(flow['thickness'].sum())
    scale = 0.78 / total
    for side, id_col in (('source', 'source_node'), ('target', 'target_node')):
        masses = flow.groupby(id_col)['thickness'].sum().to_dict()
        ordered = sorted(y_maps[side], key=lambda node: y_maps[side][node], reverse=True)
        for first, second in zip(ordered, ordered[1:]):
            gap = y_maps[side][first] - y_maps[side][second]
            denom = float(masses.get(first, 0.0) + masses.get(second, 0.0))
            if denom > 0:
                scale = min(scale, max(0.001, 2.0 * (gap - 0.004) / denom))
        for node, center in y_maps[side].items():
            mass = float(masses.get(node, 0.0))
            if mass > 0:
                scale = min(scale, 2.0 * min(center - 0.012, 0.988 - center) / mass)
    return max(scale, 0.001)

def _draw_flow_group(ax: mpl.axes.Axes, flow: pd.DataFrame, nodes: pd.DataFrame, x_source: float, x_target: float, group_label: str, y_maps: Mapping[str, Mapping[str, float]]) -> None:
    scale = _group_scale(flow, y_maps)
    order = nodes.set_index('node_id')['plot_order'].to_dict()
    work = flow.copy()
    work['source_order'] = work['source_node'].map(order)
    work['target_order'] = work['target_node'].map(order)
    work = work.sort_values(['source_order', 'target_order', 'source_node', 'target_node'], kind='stable')
    source_mass = flow.groupby('source_node')['thickness'].sum().to_dict()
    target_mass = flow.groupby('target_node')['thickness'].sum().to_dict()
    source_cursor = {node: center + float(source_mass.get(node, 0.0)) * scale / 2.0 for node, center in y_maps['source'].items()}
    target_cursor = {node: center + float(target_mass.get(node, 0.0)) * scale / 2.0 for node, center in y_maps['target'].items()}
    bar_width = 0.022
    curve = (x_target - x_source) * 0.43
    for _, row in work.iterrows():
        source, target = (str(row['source_node']), str(row['target_node']))
        height = float(row['thickness']) * scale
        sy1, sy0 = (source_cursor[source], source_cursor[source] - height)
        ty1, ty0 = (target_cursor[target], target_cursor[target] - height)
        source_cursor[source] = sy0
        target_cursor[target] = ty0
        vertices = [(x_source + bar_width, sy1), (x_source + bar_width + curve, sy1), (x_target - curve, ty1), (x_target, ty1), (x_target, ty0), (x_target - curve, ty0), (x_source + bar_width + curve, sy0), (x_source + bar_width, sy0), (x_source + bar_width, sy1)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        rho = float(row['rho'])
        colour = POSITIVE if rho >= 0 else NEGATIVE
        alpha = float(np.clip(0.22 + 1.8 * abs(rho), 0.22, 0.88))
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=colour, edgecolor='none', alpha=alpha, zorder=1, rasterized=False))
    node_meta = nodes.set_index('node_id')
    for side, x, masses in (('source', x_source, source_mass), ('target', x_target, target_mass)):
        for node, center in y_maps[side].items():
            height = max(0.012, float(masses.get(node, 0.0)) * scale)
            ax.add_patch(Rectangle((x, center - height / 2.0), bar_width, height, facecolor=aurora_module_colour(str(node_meta.loc[node, 'color'])), edgecolor='white', linewidth=0.35, zorder=3))
    colour = MBP_COLORS[group_label]
    ax.text((x_source + x_target + bar_width) / 2.0, 1.01, group_label, ha='center', va='bottom', fontsize=FONT['small'], fontweight='semibold', color=colour, clip_on=False)

def draw_sankey_pair(ax: mpl.axes.Axes, bundle: Mapping[str, Any], slug: str) -> None:
    values = bundle['axes'][slug]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.06)
    ax.axis('off')
    y_maps = _paired_layout(values['flow_i'], values['flow_iii'], values['nodes'])
    _draw_flow_group(ax, values['flow_i'], values['nodes'], 0.015, 0.455, 'MBP-I', y_maps)
    _draw_flow_group(ax, values['flow_iii'], values['nodes'], 0.535, 0.975, 'MBP-III', y_maps)
    ax.plot([0.505, 0.505], [0.03, 0.97], color=REFERENCE, linewidth=0.45, zorder=0)

def _stable_jitter(sample: str, module: str, group: str) -> float:
    token = f'{sample}|{module}|{group}'.encode('utf-8')
    value = int(hashlib.sha256(token).hexdigest()[:8], 16) / 4294967295
    return (value - 0.5) * 0.2

def draw_radiomics_facets(fig: mpl.figure.Figure, bundle: Mapping[str, Any]) -> list[mpl.axes.Axes]:
    frame = bundle['radiomics'].copy()
    frame['module_score_z'] = frame.groupby('module')['module_score'].transform(lambda series: (series - series.mean()) / series.std(ddof=1))
    axes: list[mpl.axes.Axes] = []
    left, bottom, total_width, height = (0.665, 0.305, 0.325, 0.405)
    gap = 0.014
    width = (total_width - 2 * gap) / 3.0
    for module_index, module in enumerate(MODULE_ORDER):
        ax = fig.add_axes([left + module_index * (width + gap), bottom, width, height])
        axes.append(ax)
        values = [frame.loc[(frame['module'] == module) & (frame['MBP'] == group), 'module_score_z'].to_numpy() for group in MBP_ORDER]
        box = ax.boxplot(values, positions=np.arange(3), widths=0.52, whis=1.5, showfliers=False, patch_artist=True, manage_ticks=False, medianprops={'linewidth': 0.9}, whiskerprops={'linewidth': 0.65}, capprops={'linewidth': 0.65}, boxprops={'linewidth': 0.7})
        for box_index, (patch, colour) in enumerate(zip(box['boxes'], [MBP_COLORS[g] for g in MBP_ORDER])):
            patch.set_facecolor(mpl.colors.to_rgba(colour, 0.18))
            patch.set_edgecolor(colour)
            box['medians'][box_index].set_color(colour)
            for item in box['whiskers'][2 * box_index:2 * box_index + 2]:
                item.set_color(colour)
            for item in box['caps'][2 * box_index:2 * box_index + 2]:
                item.set_color(colour)
        for group_index, group in enumerate(MBP_ORDER):
            subset = frame.loc[(frame['module'] == module) & (frame['MBP'] == group)]
            xs = [group_index + _stable_jitter(row.sample, module, group) for row in subset.itertuples()]
            ax.scatter(xs, subset['module_score_z'], s=7.2, marker='o', facecolor=mpl.colors.to_rgba(MBP_COLORS[group], 0.68), edgecolor='white', linewidth=0.18, zorder=3, rasterized=False)
        module_values = frame.loc[frame['module'] == module, 'module_score_z']
        span = float(module_values.max() - module_values.min())
        ax.set_ylim(float(module_values.min() - 0.1 * span), float(module_values.max() + 0.1 * span))
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(range(3), ['I', 'II', 'III'])
        ax.tick_params(axis='x', labelsize=FONT['tick'], length=2.2, width=0.55, pad=1.7)
        ax.tick_params(axis='y', labelsize=FONT['tick'], length=2.2, width=0.55, pad=1.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(MODULE_TITLE[module], fontsize=FONT['facet'], fontweight='semibold', color=TEXT, pad=3.2)
        if module_index == 0:
            ax.set_ylabel('Imaging phenotype score (z)', fontsize=FONT['axis'], color=TEXT, labelpad=2.2)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(TEXT)
            ax.spines[spine].set_linewidth(0.6)
        ax.grid(False)
    return axes

def build_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    panel_letter = fig.text(0.009, 0.982, 'a', ha='left', va='top', fontsize=FONT['panel'], fontweight='bold', color=TEXT)
    registry.tag(panel_letter, name='panel-letter-a', role='panel_label', panel='a')
    row_layout = {'Amino': (0.676, 0.275), 'Lipid': (0.315, 0.335), 'SCFA': (0.05, 0.235)}
    for slug, (bottom, height) in row_layout.items():
        ko_ax = fig.add_axes([0.026, bottom, 0.205, height])
        sankey_ax = fig.add_axes([0.236, bottom + 0.012, 0.205, height - 0.022])
        met_ax = fig.add_axes([0.445, bottom, 0.195, height])
        draw_bubble_panel(ko_ax, bundle, slug, 'ko', registry)
        draw_sankey_pair(sankey_ax, bundle, slug)
        draw_bubble_panel(met_ax, bundle, slug, 'metabolite', registry)
    draw_radiomics_facets(fig, bundle)
    mbp_axis_label = fig.text(0.827, 0.282, 'MBP group', ha='center', va='top', fontsize=FONT['axis'], color=TEXT)
    registry.tag(mbp_axis_label, name='radiomics-mbp-axis-label', role='axis_label', panel='a')
    assert_figure_size(fig, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    return (fig, registry)

def export_figure(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(suffix) for suffix in ('.pdf', '.svg', '.png')]
    fig.savefig(outputs[0], format='pdf', facecolor='white', transparent=False)
    fig.savefig(outputs[1], format='svg', facecolor='white', transparent=False)
    fig.savefig(outputs[2], format='png', dpi=EXPORT_DPI, facecolor='white', transparent=False)
    return outputs

def _geometry_report(bundle: Mapping[str, Any]) -> dict[str, Any]:
    radiomics = bundle['radiomics']
    z_summary = {}
    for module, group in radiomics.groupby('module', sort=False):
        values = (group['module_score'] - group['module_score'].mean()) / group['module_score'].std(ddof=1)
        z_summary[module] = {'n': int(len(group)), 'mean_z': round(float(values.mean()), 12), 'sd_z': round(float(values.std(ddof=1)), 12)}
    return {'figure': 'Supplementary Figure S5', 'version': VERSION, 'render_only': True, 'palette': PALETTE_ID, 'authoritative_composition': get_value('figures.config.item_020'), 'formal_panel_letters': ['a'], 'forbidden_operations_executed': [], 'forbidden_operations': ['correlation or FDR recomputation', 'feature selection', 'model fitting or SHAP calculation', 'MBP regrouping', 'patient selection'], 'render_translations': ['Bubble labels, significance stars, module strips, relative diameters and effect-hue sign/intensity are transcribed from six frozen vector PDFs.', get_value('figures.annotation.figure_07_2'), 'Sankey edges use frozen within-group rho and thickness=abs(rho) rows with no filtering or aggregation.', 'Radiomics module_score_z repeats the approved source-code display transform: module-wise mean centering divided by sample SD; all 109 patients are shown.', 'Boxplot summaries and deterministic display jitter are graphical primitives only.'], 'bubble_rows': {f"{panel['axis_slug']}_{panel['layer']}": len(panel['rows']) for panel in bundle['visual']['panels']}, 'sankey_rows_per_group': {slug: {'MBP-III': int(len(values['flow_iii'])), 'MBP-I': int(len(values['flow_i']))} for slug, values in bundle['axes'].items()}, 'radiomics_all_patient_counts': {f'{module} | {mbp}': int(value) for (module, mbp), value in radiomics.groupby(['module', 'MBP']).size().items()}, 'radiomics_display_z_audit': z_summary}

def run_render() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_inputs()
    fig, registry = build_figure(bundle)
    full_layout = audit_figure_layout(fig, 'Supplementary Figure S5 full a', registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.5, check_collisions=True)
    panel_layout = json.loads(json.dumps(full_layout))
    panel_layout['figure']['name'] = 'Supplementary Figure S5 panel a'
    if not full_layout['summary']['strict_pass']:
        raise RuntimeError(f"S5 strict layout QA failed before export: {full_layout['summary']}")
    full_outputs = export_figure(fig, FULL_STEM)
    panel_outputs = export_figure(fig, PANEL_STEM)
    plt.close(fig)
    layout_path = OUTPUT_DIR / f"{get_value('figures.config.item_303')}{VERSION}.json"
    panel_layout_path = OUTPUT_DIR / f"{get_value('figures.config.item_305')}{VERSION}.json"
    scientific_path = OUTPUT_DIR / f"{get_value('figures.config.item_306')}{VERSION}.json"
    write_json(full_layout, layout_path)
    write_json({'a': panel_layout}, panel_layout_path)
    write_json(_geometry_report(bundle), scientific_path)
    manifest_entries = []
    for path in full_outputs + panel_outputs + [layout_path, panel_layout_path, scientific_path]:
        manifest_entries.append({'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    export_manifest = {'figure': 'Supplementary Figure S5', 'version': VERSION, 'outputs': manifest_entries}
    export_manifest_path = OUTPUT_DIR / f"{get_value('figures.config.item_307')}{VERSION}.json"
    write_json(export_manifest, export_manifest_path)
    result = {'figure': 'Supplementary Figure S5', 'version': VERSION, 'full_stem': str(FULL_STEM), 'panel_stem': str(PANEL_STEM), 'layout_summary': full_layout['summary'], 'output_count': len(full_outputs) + len(panel_outputs), 'export_manifest': str(export_manifest_path)}
    return result
if __name__ == '__main__':
    run_render()
