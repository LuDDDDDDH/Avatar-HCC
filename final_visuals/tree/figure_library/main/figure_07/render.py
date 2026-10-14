from __future__ import annotations
from avatar_paths import get_value, get_column
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
HARMONIZATION_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HARMONIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(HARMONIZATION_ROOT))
import components_supplement as s5
import components_primary as old7
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, write_json
from figure_style import MBP_COLORS, NEUTRAL_COLORS, PALETTE_ID
VERSION = 'v003'
DATE_TAG = get_value('figures.config.item_350')
INPUT_ROOT = ROOT / 'inputs'
S5_INPUT_DIR = INPUT_ROOT / get_value('figures.path.part_014')
F7_INPUT_DIR = INPUT_ROOT / get_value('figures.path.part_015')
OUTPUT_DIR = ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
PANEL_DIR = OUTPUT_DIR / 'panels'
FULL_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_291')}{VERSION}"
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 200.0
EXPORT_DPI = 600
MIN_FONT_PT = 5.4
STAR_MIN_FONT_PT = 5.9
MM_PER_INCH = 25.4
TEXT = NEUTRAL_COLORS['text']
STAR_DARK = '#000000'
STAR_LIGHT = '#FFFFFF'
STAR_MIN_CONTRAST = 4.5
FULL_PANEL_RECTS = {'a': (0.0, 0.0, 180.0, 145.0), 'c_inset': (109.5, 108.0, 66.5, 37.0), 'b': (5.0, 150.0, 101.0, 46.0), 'b_legend': (112.0, 149.5, 64.0, 47.0)}
PANEL_LAYOUTS = {'a': {'size': (180.0, 145.0), 'rect': (0.0, 0.0, 180.0, 145.0)}, 'b': {'size': (125.0, 75.0), 'rect': (5.0, 5.0, 115.0, 65.0)}, 'c': {'size': (150.0, 102.0), 'rect': (8.0, 5.0, 135.0, 68.0), 'legend': (8.0, 77.0, 135.0, 20.0)}}
F7_TABLES = OrderedDict([('shap', get_value('figures.config.item_153')), ('legend', get_value('figures.config.item_152')), ('scores', get_value('figures.config.item_136')), ('stats', get_value('figures.config.item_137')), ('pairwise', get_value('figures.config.item_135'))])

def configure_style() -> None:
    s5.configure_style()
    mpl.rcParams.update({'font.size': 6.0, 'axes.labelsize': 6.0, 'axes.titlesize': 6.2, 'xtick.labelsize': MIN_FONT_PT, 'ytick.labelsize': MIN_FONT_PT, 'legend.fontsize': MIN_FONT_PT, 'savefig.dpi': EXPORT_DPI, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.max_open_warning': 0})

def _add_axes_local(fig: mpl.figure.Figure, canvas_width_mm: float, canvas_height_mm: float, panel_rect: Sequence[float], local_rect: Sequence[float], panel: str) -> mpl.axes.Axes:
    px, py_top, pw, ph = (float(value) for value in panel_rect)
    lx, ly, lw, lh = (float(value) for value in local_rect)
    left = (px + lx * pw) / canvas_width_mm
    bottom = (canvas_height_mm - py_top - ph + ly * ph) / canvas_height_mm
    width = lw * pw / canvas_width_mm
    height = lh * ph / canvas_height_mm
    ax = fig.add_axes([left, bottom, width, height])
    ax._qa_meta = {'panel': panel}
    return ax

def _figure_text_local(fig: mpl.figure.Figure, canvas_width_mm: float, canvas_height_mm: float, panel_rect: Sequence[float], x_local: float, y_local_from_top: float, text: str, **kwargs: Any) -> mpl.text.Text:
    px, py_top, pw, ph = (float(value) for value in panel_rect)
    x = (px + x_local * pw) / canvas_width_mm
    y = 1.0 - (py_top + y_local_from_top * ph) / canvas_height_mm
    return fig.text(x, y, text, **kwargs)

def _relative_luminance(colour: str) -> float:
    values = []
    for component in to_rgb(colour):
        values.append(component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4)
    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]

def _contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)

def _star_style(fill: str) -> tuple[str, str, float]:
    dark_ratio = _contrast_ratio(fill, STAR_DARK)
    light_ratio = _contrast_ratio(fill, STAR_LIGHT)
    if dark_ratio >= light_ratio:
        return (STAR_DARK, STAR_LIGHT, dark_ratio)
    return (STAR_LIGHT, STAR_DARK, light_ratio)

def _enhanced_bubble_diameter(row: Mapping[str, Any]) -> float:
    source = float(row['bubble_diameter_pt_original'])
    return max(7.7, source * 1.02) * 1.1

def draw_bubble_panel_enhanced(ax: mpl.axes.Axes, bundle: Mapping[str, Any], slug: str, layer: str, registry: ArtistRegistry) -> None:
    panel = s5.visual_panel(bundle, slug, layer)
    rows = [dict(row) for row in panel['rows']]
    for row in rows:
        row['module'] = s5.module_lookup(bundle, panel['axis'], layer, row['label'])
    n = len(rows)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(n - 0.35, -1.05)
    ax.axis('off')
    if layer == 'ko':
        strip_x, dot_x, label_x, module_x = (0.57, 0.675, 0.755, 0.52)
        header = ax.text(0.015, -0.72, s5.AXES[slug]['display'], ha='left', va='center', fontsize=s5.FONT['facet'], fontweight='bold', color=TEXT, clip_on=False)
        registry.tag(header, name=f'{slug}-axis-name', role='direct_label', panel='a')
    else:
        strip_x, dot_x, label_x, module_x = (0.035, 0.135, 0.225, 0.0)
    for start, end, module, colour in s5._contiguous_groups(rows):
        ax.add_patch(Rectangle((strip_x - 0.022, start - 0.43), 0.044, end - start + 0.86, facecolor=s5.aurora_module_colour(colour), edgecolor='white', linewidth=0.35, clip_on=True))
        if layer == 'ko':
            module_artist = ax.text(module_x, (start + end) / 2.0, s5.MODULE_DISPLAY.get(module, module.replace(' / ', ' /\n')), ha='right', va='center', fontsize=s5.FONT['small'], fontweight='semibold', linespacing=0.94, color=TEXT, clip_on=True)
            registry.tag(module_artist, name=f'{slug}-{layer}-module-{start}', role='direct_label', panel='a')
    for index, row in enumerate(rows):
        diameter = _enhanced_bubble_diameter(row)
        fill = to_hex(s5.aurora_effect_colour(row['bubble_color_original']), keep_alpha=False).upper()
        ax.scatter([dot_x], [index], s=diameter ** 2, marker='o', facecolor=fill, edgecolor='#52606D', linewidth=0.38, alpha=1.0, zorder=3, clip_on=True, rasterized=False)
        significance = str(row['significance'])
        if significance:
            star_colour, stroke_colour, contrast = _star_style(fill)
            star = ax.text(dot_x, index - 0.01, significance, ha='center', va='center', fontsize=STAR_MIN_FONT_PT, fontweight='bold', color=star_colour, alpha=1.0, zorder=4, clip_on=True, path_effects=[path_effects.withStroke(linewidth=0.3, foreground=stroke_colour)])
            star._star_qa = {'axis': slug, 'layer': layer, 'order': index + 1, 'label': row['label'], 'symbol': significance, 'fill': fill, 'text': star_colour, 'stroke': stroke_colour, 'contrast_ratio': float(contrast), 'font_pt': STAR_MIN_FONT_PT, 'diameter_pt': float(diameter), 'v001_diameter_pt': float(max(7.7, float(row['bubble_diameter_pt_original']) * 1.02)), 'alpha': 1.0}
            registry.tag(star, name=f'{slug}-{layer}-star-{index}', role='small_value', panel='a')
        label_artist = ax.text(label_x, index, row['label'], ha='left', va='center', fontsize=s5.FONT['label'], color=TEXT, clip_on=True)
        registry.tag(label_artist, name=f'{slug}-{layer}-feature-{index}', role='direct_label', panel='a')
    compare = ax.text(dot_x, -0.72, 'III vs I', ha='center', va='center', fontsize=s5.FONT['small'], color=s5.MUTED, clip_on=True)
    registry.tag(compare, name=f'{slug}-{layer}-comparison', role='small_value', panel='a')

def draw_radiomics_facets_local(fig: mpl.figure.Figure, canvas_width_mm: float, canvas_height_mm: float, panel_rect: Sequence[float], bundle: Mapping[str, Any]) -> list[mpl.axes.Axes]:
    frame = bundle['radiomics'].copy()
    frame['module_score_z'] = frame.groupby('module')['module_score'].transform(lambda series: (series - series.mean()) / series.std(ddof=1))
    axes: list[mpl.axes.Axes] = []
    left, bottom, total_width, height = (0.64, 0.305, 0.35, 0.405)
    gap = 0.014
    width = (total_width - 2 * gap) / 3.0
    for module_index, module in enumerate(s5.MODULE_ORDER):
        ax = _add_axes_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, (left + module_index * (width + gap), bottom, width, height), 'a')
        axes.append(ax)
        values = [frame.loc[(frame['module'] == module) & (frame['MBP'] == group), 'module_score_z'].to_numpy() for group in s5.MBP_ORDER]
        box = ax.boxplot(values, positions=np.arange(3), widths=0.52, whis=1.5, showfliers=False, patch_artist=True, manage_ticks=False, medianprops={'linewidth': 0.9}, whiskerprops={'linewidth': 0.65}, capprops={'linewidth': 0.65}, boxprops={'linewidth': 0.7})
        for box_index, (patch, colour) in enumerate(zip(box['boxes'], [MBP_COLORS[g] for g in s5.MBP_ORDER])):
            patch.set_facecolor(mpl.colors.to_rgba(colour, 0.18))
            patch.set_edgecolor(colour)
            box['medians'][box_index].set_color(colour)
            for item in box['whiskers'][2 * box_index:2 * box_index + 2]:
                item.set_color(colour)
            for item in box['caps'][2 * box_index:2 * box_index + 2]:
                item.set_color(colour)
        for group_index, group in enumerate(s5.MBP_ORDER):
            subset = frame.loc[(frame['module'] == module) & (frame['MBP'] == group)]
            xs = [group_index + s5._stable_jitter(row.sample, module, group) for row in subset.itertuples()]
            ax.scatter(xs, subset['module_score_z'], s=7.2, marker='o', facecolor=mpl.colors.to_rgba(MBP_COLORS[group], 0.68), edgecolor='white', linewidth=0.18, zorder=3, rasterized=False)
        module_values = frame.loc[frame['module'] == module, 'module_score_z']
        span = float(module_values.max() - module_values.min())
        ax.set_ylim(float(module_values.min() - 0.1 * span), float(module_values.max() + 0.1 * span))
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(range(3), ['I', 'II', 'III'])
        ax.tick_params(axis='x', labelsize=s5.FONT['tick'], length=2.2, width=0.55, pad=1.7)
        ax.tick_params(axis='y', labelsize=s5.FONT['tick'], length=2.2, width=0.55, pad=1.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(s5.MODULE_TITLE[module], fontsize=s5.FONT['facet'], fontweight='semibold', color=TEXT, pad=3.2)
        if module_index == 0:
            ax.set_ylabel('Imaging score (z)', fontsize=s5.FONT['axis'], color=TEXT, labelpad=2.2)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(TEXT)
            ax.spines[spine].set_linewidth(0.6)
        ax.grid(False)
    return axes

def draw_panel_a(fig: mpl.figure.Figure, canvas_width_mm: float, canvas_height_mm: float, panel_rect: Sequence[float], bundle: Mapping[str, Any], registry: ArtistRegistry) -> None:
    row_layout = {'Amino': (0.676, 0.275), 'Lipid': (0.315, 0.335), 'SCFA': (0.05, 0.235)}
    for slug, (bottom, height) in row_layout.items():
        ko_ax = _add_axes_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, (0.026, bottom, 0.205, height), 'a')
        sankey_ax = _add_axes_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, (0.236, bottom + 0.012, 0.205, height - 0.022), 'a')
        met_ax = _add_axes_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, (0.445, bottom, 0.195, height), 'a')
        draw_bubble_panel_enhanced(ko_ax, bundle, slug, 'ko', registry)
        s5.draw_sankey_pair(sankey_ax, bundle, slug)
        draw_bubble_panel_enhanced(met_ax, bundle, slug, 'metabolite', registry)
    draw_radiomics_facets_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, bundle)
    mbp_label = _figure_text_local(fig, canvas_width_mm, canvas_height_mm, panel_rect, 0.815, 0.718, 'MBP group', ha='center', va='top', fontsize=s5.FONT['axis'], color=TEXT)
    registry.tag(mbp_label, name='a-radiomics-mbp-label', role='axis_label', panel='a')

def draw_module_legend(fig: mpl.figure.Figure, canvas_width_mm: float, canvas_height_mm: float, rect_mm: Sequence[float], data: pd.DataFrame, registry: ArtistRegistry, *, ncols: int=4) -> None:
    ax = old7._axis(fig, canvas_width_mm, canvas_height_mm, rect_mm, 'b')
    ax.axis('off')
    work = data.copy()
    work['row_order'] = pd.to_numeric(work['row_order'], errors='raise')
    work = work.sort_values('row_order', kind='stable').drop_duplicates('display_label')
    work = work[['display_label', 'row_color']].reset_index(drop=True)
    nrows = int(math.ceil(len(work) / ncols))
    for idx, row in work.iterrows():
        col = idx // nrows
        row_idx = idx % nrows
        x = col / ncols + 0.006
        y = 1.0 - (row_idx + 0.5) / nrows
        ax.add_patch(Rectangle((x, y - 0.033), 0.013, 0.066, transform=ax.transAxes, facecolor=old7._hex_display(row['row_color']), edgecolor='none'))
        label = ax.text(x + 0.019, y, old7._short(row['display_label'], 31), transform=ax.transAxes, ha='left', va='center', fontsize=MIN_FONT_PT, color=TEXT)
        registry.tag(label, name=f'b-legend-{idx}', role='direct_label', panel='b')

def _panel_letter(fig: mpl.figure.Figure, width_mm: float, height_mm: float, panel: str, x_mm: float, y_top_mm: float, registry: ArtistRegistry) -> None:
    artist = fig.text(x_mm / width_mm, 1.0 - y_top_mm / height_mm, panel, ha='left', va='top', fontsize=8.5, fontweight='bold', color=TEXT)
    registry.tag(artist, name=f'panel-letter-{panel}', role='panel_label', panel=panel)

def load_inputs() -> dict[str, Any]:
    s5.INPUT_DIR = S5_INPUT_DIR
    s5_bundle = s5.load_inputs()
    f7_bundle = {key: pd.read_csv(F7_INPUT_DIR / filename) for key, filename in F7_TABLES.items()}
    old7._require_columns(f7_bundle['shap'], ['axis', 'display_label', 'row_color', 'row_order', 'outcome', 'plot_value', 'abs_value'], 'v002 SHAP')
    old7._require_columns(f7_bundle['scores'], ['sample', 'axis', 'MBP', 'integrated_score', 'score_direction'], 'v002 scores')
    old7._require_columns(f7_bundle['stats'], ['axis', 'n', 'trend_rho', 'trend_p', 'trend_q_BH', 'kw_q_BH'], 'v002 stats')
    old7._require_columns(f7_bundle['pairwise'], ['axis', 'group1', 'group2', 'q_value'], 'v002 pairwise')
    if len(f7_bundle['shap']) != 99:
        raise ValueError('Panel B frozen SHAP table must contain 99 rows')
    if len(f7_bundle['scores']) != 327 or f7_bundle['scores']['sample'].nunique() != 109:
        raise ValueError('Panel C frozen score table must contain 327 rows / 109 patients')
    return {'a': s5_bundle, 'bc': f7_bundle}

def build_full_figure(bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    draw_panel_a(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANEL_RECTS['a'], bundle['a'], registry)
    old7.draw_panel_c(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANEL_RECTS['c_inset'], bundle['bc'], registry)
    old7.draw_panel_b(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANEL_RECTS['b'], bundle['bc'], registry)
    draw_module_legend(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, FULL_PANEL_RECTS['b_legend'], bundle['bc']['shap'], registry, ncols=2)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'a', 1.5, 2.0, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'b', 104.0, 104.5, registry)
    _panel_letter(fig, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM, 'c', 1.5, 148.0, registry)
    assert_figure_size(fig, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    return (fig, registry)

def build_panel_figure(panel: str, bundle: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    panel = panel.lower()
    if panel not in PANEL_LAYOUTS:
        raise ValueError(panel)
    configure_style()
    width_mm, height_mm = PANEL_LAYOUTS[panel]['size']
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    rect = PANEL_LAYOUTS[panel]['rect']
    if panel == 'a':
        draw_panel_a(fig, width_mm, height_mm, rect, bundle['a'], registry)
    elif panel == 'b':
        old7.draw_panel_c(fig, width_mm, height_mm, rect, bundle['bc'], registry)
    else:
        old7.draw_panel_b(fig, width_mm, height_mm, rect, bundle['bc'], registry)
        draw_module_legend(fig, width_mm, height_mm, PANEL_LAYOUTS['c']['legend'], bundle['bc']['shap'], registry)
    _panel_letter(fig, width_mm, height_mm, panel, 1.5, 2.0, registry)
    assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    return (fig, registry)

def audit_star_visibility(fig: mpl.figure.Figure, figure_name: str) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows: list[dict[str, Any]] = []
    for artist in fig.findobj(match=mpl.text.Text):
        metadata = getattr(artist, '_star_qa', None)
        if not metadata:
            continue
        bbox = artist.get_window_extent(renderer)
        axes_bbox = artist.axes.bbox
        clipped = bool(bbox.x0 < axes_bbox.x0 - 0.5 or bbox.x1 > axes_bbox.x1 + 0.5 or bbox.y0 < axes_bbox.y0 - 0.5 or (bbox.y1 > axes_bbox.y1 + 0.5))
        row = dict(metadata)
        row.update({'clip_on': bool(artist.get_clip_on()), 'bbox_inside_axes': not clipped, 'fontweight': str(artist.get_fontweight()), 'alpha_actual': 1.0 if artist.get_alpha() is None else float(artist.get_alpha()), 'diameter_increase_fraction': float(metadata['diameter_pt'] / metadata['v001_diameter_pt'] - 1.0)})
        rows.append(row)
    if not rows:
        raise RuntimeError(f'No significance-star artists found in {figure_name}')
    expected_visual = json.loads((S5_INPUT_DIR / get_value('figures.config.item_105')).read_text(encoding='utf-8'))
    expected = [{'axis': panel['axis_slug'], 'layer': panel['layer'], 'order': int(row['order']), 'label': row['label'], 'symbol': row['significance']} for panel in expected_visual['panels'] for row in panel['rows'] if row['significance']]
    actual_identity = [{key: row[key] for key in ('axis', 'layer', 'order', 'label', 'symbol')} for row in rows]
    frozen_identity_match = actual_identity == expected
    report = {'figure': figure_name, 'version': VERSION, 'rule': 'Choose black or white by the larger WCAG sRGB contrast ratio against the final opaque bubble fill; no effect-size threshold.', 'required_minimum_contrast_ratio': STAR_MIN_CONTRAST, 'required_minimum_font_pt': STAR_MIN_FONT_PT, get_value('figures.config.item_367'): 0.08, 'summary': {'stars': len(rows), 'expected_frozen_stars': len(expected), 'frozen_label_order_star_match': frozen_identity_match, 'minimum_contrast_ratio': min((row['contrast_ratio'] for row in rows)), 'minimum_font_pt': min((row['font_pt'] for row in rows)), 'minimum_diameter_pt': min((row['diameter_pt'] for row in rows)), 'minimum_diameter_increase_fraction': min((row['diameter_increase_fraction'] for row in rows)), 'clipped_count': sum((not row['bbox_inside_axes'] for row in rows)), 'alpha_not_one_count': sum((abs(row['alpha_actual'] - 1.0) > 1e-09 for row in rows)), 'pass': len(rows) == 49 and len(expected) == 49 and frozen_identity_match and all((row['contrast_ratio'] >= STAR_MIN_CONTRAST and row['font_pt'] >= STAR_MIN_FONT_PT and (row['diameter_increase_fraction'] >= 0.08 - 1e-09) and row['bbox_inside_axes'] and (abs(row['alpha_actual'] - 1.0) <= 1e-09) and (row['fontweight'] in {'bold', '700'}) for row in rows))}, 'rows': rows}
    if not report['summary']['pass']:
        raise RuntimeError(f"Star visibility QA failed for {figure_name}: {report['summary']}")
    return report

def input_manifest() -> dict[str, Any]:
    s5_manifest = json.loads((S5_INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8-sig'))
    old7_manifest = json.loads((ROOT.parent / get_value('figures.path.part_013') / 'inputs' / get_value('figures.config.item_116')).read_text(encoding='utf-8-sig'))
    old7_by_name = {Path(row['snapshot']).name: row for row in old7_manifest['files']}
    records: list[dict[str, Any]] = []
    for row in s5_manifest['files']:
        snapshot = S5_INPUT_DIR / row['snapshot']
        digest = sha256_file(snapshot).upper()
        if digest != str(row['sha256']).upper():
            raise ValueError(f'Panel A frozen snapshot hash drift: {snapshot.name}')
        source = Path(row['source'])
        source_hash = sha256_file(source).upper()
        kind = row.get('kind', 'source_snapshot')
        record = {'panel': 'A', 'snapshot': snapshot.relative_to(INPUT_ROOT).as_posix(), 'source': row.get('source'), 'bytes': snapshot.stat().st_size, 'sha256': digest, 'source_sha256': source_hash, 'kind': kind, 'byte_for_byte_match': digest == source_hash if kind != 'derived_display_encoding' else None}
        if kind == 'derived_display_encoding':
            record['schema'] = row.get('schema')
            record['upstream_snapshots'] = row.get('upstream_snapshots', [])
        records.append(record)
    for filename in F7_TABLES.values():
        row = old7_by_name[filename]
        snapshot = F7_INPUT_DIR / filename
        digest = sha256_file(snapshot).upper()
        if digest != str(row['sha256']).upper():
            raise ValueError(f'Panel B/C frozen snapshot hash drift: {filename}')
        records.append({'panel': 'C' if filename.startswith('module_') else 'B', 'snapshot': snapshot.relative_to(INPUT_ROOT).as_posix(), 'source': row.get('source'), 'rows': row.get('rows'), 'columns': row.get('columns'), 'bytes': snapshot.stat().st_size, 'sha256': digest, 'source_sha256': row.get('source_sha256'), 'byte_for_byte_match': True})
    return {'figure': 'Figure 7', 'version': VERSION, 'snapshot_date': '2026-09-02', 'render_only': True, 'source_versions': {'panel_A': 'Supplementary Figure S5 v001 frozen package', 'panels_B_C': 'Figure 7 v001 frozen package, with source panels C/B renumbered as B/C'}, 'files': records}

def scientific_contract() -> dict[str, Any]:
    return {'figure': 'Figure 7', 'version': VERSION, 'render_mode': 'frozen-source render-only vector reassembly', 'working_master_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'panel_mapping': {'A': {'source_panel': 'Supplementary Figure S5A v001', 'meaning': 'MBP-III versus MBP-I KO and metabolite effects, paired MBP-I/MBP-III KO-to-metabolite networks, and three imaging phenotype distributions across MBP groups.', 'locked_content': 'Six frozen bubble visual encodings, six frozen Sankey flow tables plus three node dictionaries, and 327 radiomics rows from 109 patients.', 'star_guardrail': 'Significance strings are copied verbatim from the frozen bubble visual encoding. Only opacity, contrast-aware text colour, outline, font weight/size, and a uniform 10% diameter enlargement versus v001 are changed.', 'star_scientific_source': 'Frozen two-sided Wilcoxon nominal P symbols using p<0.05/*, p<0.01/**, p<0.001/***; no FDR. All 54 bubble rows and 49 non-empty symbols must match the locked JSON by axis, layer, order and label.', 'radiomics_guardrail': 'The approved display-only module-wise z transform uses sample SD (ddof=1); MBP assignments and all patients remain frozen.'}, 'B': {'source_panel': 'Figure 7C v001', 'meaning': 'Frozen response-oriented ecosystem scores across MBP-I, MBP-II, and MBP-III.', 'locked_counts': {'score_rows': 327, 'patients': 109, 'stats_rows': 3, 'pairwise_rows': 9}, 'display_role': "Placed in the upper composition's open lower-right region and renumbered as formal Panel B.", 'guardrail': 'Use frozen integrated_score, trend statistics and pairwise q values directly; no regrouping or statistical recalculation.'}, 'C': {'source_panel': 'Figure 7B v001', 'meaning': 'Frozen signed module-level SHAP contributions by ecosystem axis and endpoint.', 'locked_counts': {'rows': 99, 'non_null_plot_value': 99}, 'guardrail': 'Plot frozen plot_value directly; no model fitting or SHAP calculation.'}}, 'star_visibility': {'alpha': 1.0, 'fontweight': 'bold', 'minimum_font_pt': STAR_MIN_FONT_PT, 'candidate_text_colours': [STAR_DARK, STAR_LIGHT], 'selection': 'larger WCAG sRGB contrast ratio against final fill', 'minimum_contrast_ratio': STAR_MIN_CONTRAST, 'stroke_pt': 0.3, 'diameter_change': '+10% versus v001 rendered diameter', 'effect_threshold_used': False}, 'allowed_render_operations': ['Vector layout reassembly', get_value('figures.annotation.figure_07'), 'Deterministic geometry and point jitter from frozen rows', 'Approved display z transform for Panel A imaging phenotype plots', 'Typography, contrast and marker-size visibility changes'], 'forbidden_operations': ['Model fitting or refitting', 'SHAP calculation or recalculation', 'Correlation or FDR calculation', 'Feature or edge selection', 'MBP assignment or patient grouping', 'Changes to frozen effects, p values, q values or sample sizes', 'Raster image/PDF compositing']}

def scientific_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    visual_panels = bundle['a']['visual']['panels']
    star_counts = {f"{row['axis_slug']}_{row['layer']}": sum((bool(item['significance']) for item in row['rows'])) for row in visual_panels}
    radiomics = bundle['a']['radiomics']
    return {'figure': 'Figure 7', 'version': VERSION, 'render_only': True, 'palette': PALETTE_ID, 'panel_mapping': {'a': 'Former formal S5A content.', 'b': 'Former Figure 7C frozen 327-score/109-patient MBP panel, renumbered and positioned at upper right.', 'c': 'Former Figure 7B frozen 99-row signed module-level SHAP panel, renumbered with its complete key beside it.'}, 'scientific_locks': {'panel_a_bubble_row_counts': {f"{row['axis_slug']}_{row['layer']}": len(row['rows']) for row in visual_panels}, 'panel_a_star_counts': star_counts, 'panel_a_radiomics_rows': int(len(radiomics)), 'panel_a_unique_patients': int(radiomics['sample'].nunique()), 'panel_b_score_rows': int(len(bundle['bc']['scores'])), 'panel_b_unique_patients': int(bundle['bc']['scores']['sample'].nunique()), 'panel_c_shap_rows': int(len(bundle['bc']['shap']))}, get_value('figures.config.item_361'): ['reference-matched compact A/B/C vector layout', 'former Figure 7C renumbered as Panel B and positioned at upper right', 'former Figure 7B renumbered as Panel C with its complete module legend beside it in two columns', 'opaque contrast-aware bubble stars with 0.30-pt opposite-colour stroke', 'bubble diameters enlarged uniformly by 10% relative to S5 v001 rendered diameters'], 'forbidden_operations_executed': []}

def _save_exact(fig: mpl.figure.Figure, stem: Path, width_mm: float, height_mm: float) -> list[Path]:
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

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_input_path = write_json(input_manifest(), INPUT_ROOT / get_value('figures.config.item_116'))
    contract_path = write_json(scientific_contract(), INPUT_ROOT / get_value('figures.config.item_201'))
    full_fig, full_registry = build_full_figure(bundle)
    full_star = audit_star_visibility(full_fig, f"{get_value('figures.config.item_291')}{VERSION}")
    full_layout = audit_figure_layout(full_fig, f"{get_value('figures.config.item_291')}{VERSION}", full_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    full_paths = _save_exact(full_fig, FULL_STEM, FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM)
    plt.close(full_fig)
    panel_paths: list[Path] = []
    panel_layouts: dict[str, Any] = {}
    star_reports: dict[str, Any] = {'full': full_star}
    for panel in 'abc':
        panel_fig, panel_registry = build_panel_figure(panel, bundle)
        width_mm, height_mm = PANEL_LAYOUTS[panel]['size']
        if panel == 'a':
            star_reports['panel_a'] = audit_star_visibility(panel_fig, f"{get_value('figures.config.item_292')}{VERSION}")
        panel_layouts[panel] = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_294')}{panel}_Nature_{VERSION}", panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        panel_paths.extend(_save_exact(panel_fig, PANEL_DIR / f"{get_value('figures.config.item_294')}{panel}_Nature_{VERSION}", width_mm, height_mm))
        plt.close(panel_fig)
    layout_path = write_json(full_layout, OUTPUT_DIR / f"{get_value('figures.config.item_295')}{VERSION}.json")
    panel_layout_path = write_json(panel_layouts, OUTPUT_DIR / f"{get_value('figures.config.item_296')}{VERSION}.json")
    star_path = write_json(star_reports, OUTPUT_DIR / f"{get_value('figures.config.item_297')}{VERSION}.json")
    scientific_path = write_json(scientific_summary(bundle), OUTPUT_DIR / f"{get_value('figures.config.item_298')}{VERSION}.json")
    for label, report in [('full', full_layout), *panel_layouts.items()]:
        if report['summary']['critical'] or report['summary']['warning']:
            raise RuntimeError(f"Strict layout QA failed for {label}: {report['summary']}")
    if not all((report['summary']['pass'] for report in star_reports.values())):
        raise RuntimeError('Star visibility QA failed')
    tracked = full_paths + panel_paths + [layout_path, panel_layout_path, star_path, scientific_path, manifest_input_path, contract_path]
    export_manifest = {'figure': 'Figure 7', 'version': VERSION, 'palette': PALETTE_ID, 'working_master_size_mm': [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked frozen inputs; vector artist reassembly', 'panel_mapping': {'a': 'former S5a', 'b': 'former Figure7C, renumbered and positioned at upper right', 'c': 'former Figure7B, renumbered with complete key beside it'}, 'layout_summary': full_layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_layouts.items()}, 'star_visibility_summary': {key: value['summary'] for key, value in star_reports.items()}, 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in tracked]}
    export_manifest_path = write_json(export_manifest, OUTPUT_DIR / f"{get_value('figures.config.item_299')}{VERSION}.json")
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_layout_audit': str(panel_layout_path), 'star_visibility_qa': str(star_path), 'scientific_guardrail_report': str(scientific_path), 'input_manifest': str(manifest_input_path), 'scientific_contract': str(contract_path), 'export_manifest': str(export_manifest_path), 'layout_summary': full_layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_layouts.items()}, 'star_visibility_summary': {key: value['summary'] for key, value in star_reports.items()}}
configure_style()
if __name__ == '__main__':
    export_all(load_inputs())
