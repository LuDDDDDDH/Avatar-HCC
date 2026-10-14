from __future__ import annotations
from avatar_paths import get_value, get_column
import sysconfig
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_qa import ArtistRegistry, assert_figure_size, audit_figure_layout, sha256_file, verify_export_bundle, write_json
from figure_style import MASTER_SPECTRUM, NEUTRAL_COLORS, PALETTE_ID, apply_theme
VERSION = 'v001'
DATE_TAG = get_value('figures.config.item_350')
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 79.0
EXPORT_DPI = 600
MM_PER_INCH = 25.4
INPUT_DIR = ROOT / 'inputs'
CONTENT_LOCK = INPUT_DIR / get_value('figures.config.item_071')
AUTHOR_SOURCE = INPUT_DIR / get_value('figures.module_path.author_source')
OUTPUT_DIR = ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
OUTPUT_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_260')}{VERSION}"
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
CONNECTOR = '#8796A5'
WHITE = NEUTRAL_COLORS['background']
LIGHT = NEUTRAL_COLORS['light_background']
CORAL = '#DE646C'
CENTER_COLORS = {'SCH': '#2F6FB0', 'QLH': '#2A8F78', 'PUMCH': '#8F59C0'}
FONT = {'panel': 8.5, 'top': 5.6, 'top_count': 5.9, 'header': 5.7, 'body': 5.4, 'emphasis': 5.6}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _blend_with_white(colour: str, strength: float) -> str:
    rgb = to_rgb(colour)
    mixed = tuple((1.0 - float(strength) * (1.0 - channel) for channel in rgb))
    return to_hex(mixed, keep_alpha=False).upper()

def configure_style() -> None:
    apply_theme()
    mpl.rcParams.update({'font.size': FONT['body'], 'axes.linewidth': 0.6, 'patch.linewidth': 0.65, 'savefig.dpi': EXPORT_DPI, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.max_open_warning': 0})

def enable_local_pdf_qa() -> None:
    if importlib.util.find_spec('pypdf') or importlib.util.find_spec('PyPDF2'):
        return
    bundled = Path(get_value('figures.runtime.site_packages', sysconfig.get_paths()['purelib']))
    if bundled.is_dir():
        sys.path.insert(0, str(bundled))

def load_content() -> dict[str, Any]:
    content = json.loads(CONTENT_LOCK.read_text(encoding='utf-8'))
    if content.get('version') != VERSION:
        raise RuntimeError('Unexpected Figure 1A content-lock version')
    if content.get('palette') != PALETTE_ID:
        raise RuntimeError('Unexpected Figure 1A palette')
    if content.get('render_only') is not True:
        raise RuntimeError('Figure 1A must remain render-only')
    centers = content.get('centers', [])
    if [row.get('id') for row in centers] != ['SCH', 'QLH', 'PUMCH']:
        raise RuntimeError('Unexpected Figure 1A center order')
    source_sum = sum((int(row['source_n']) for row in centers))
    if source_sum != int(content['top_cohort']['n']):
        raise RuntimeError('Source-population counts do not match the locked total')
    return content

def _box(ax: mpl.axes.Axes, *, x: float, y: float, width: float, height: float, face: str, edge: str, linewidth: float=0.65, radius: float=1.15, zorder: float=2.0) -> FancyBboxPatch:
    patch = FancyBboxPatch((x, y), width, height, boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=face, edgecolor=edge, linewidth=linewidth, joinstyle='round', zorder=zorder)
    ax.add_patch(patch)
    return patch

def _line(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], *, linewidth: float=0.55, linestyle: str='-', zorder: float=0.8) -> None:
    ax.plot([start[0], end[0]], [start[1], end[1]], color=CONNECTOR, linewidth=linewidth, linestyle=linestyle, solid_capstyle='round', zorder=zorder)

def _arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], *, linewidth: float=0.55, mutation_scale: float=5.4, zorder: float=1.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=mutation_scale, linewidth=linewidth, color=CONNECTOR, shrinkA=0, shrinkB=0, capstyle='round', joinstyle='round', zorder=zorder))

def _text(ax: mpl.axes.Axes, registry: ArtistRegistry, x: float, y: float, label: str, *, name: str, fontsize: float=FONT['body'], fontweight: str='normal', color: str=TEXT, ha: str='center', va: str='center', linespacing: float=1.05, zorder: float=3.0) -> mpl.text.Text:
    artist = ax.text(x, y, label, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight, color=color, linespacing=linespacing, clip_on=False, zorder=zorder)
    return registry.tag(artist, name=name, role='direct_label', panel='a')

def _stacked_text(ax: mpl.axes.Axes, registry: ArtistRegistry, *, x: float, center_y: float, lines: Sequence[str], name: str, styles: Sequence[Mapping[str, Any]] | None=None, gap_mm: float=2.2) -> None:
    count = len(lines)
    styles = styles or [{} for _ in lines]
    if len(styles) != count:
        raise ValueError('Each stacked line must have one style mapping')
    first_y = center_y - gap_mm * (count - 1) / 2.0
    for index, (label, style) in enumerate(zip(lines, styles)):
        _text(ax, registry, x, first_y + index * gap_mm, label, name=f'{name}-line-{index + 1}', **dict(style))

def build_figure(content: Mapping[str, Any]) -> tuple[mpl.figure.Figure, ArtistRegistry]:
    configure_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH), facecolor=WHITE)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, FIGURE_WIDTH_MM)
    ax.set_ylim(FIGURE_HEIGHT_MM, 0.0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax._qa_meta = {'panel': 'a'}
    registry = ArtistRegistry()
    panel = _text(ax, registry, 1.6, 1.5, 'a', name='panel-label', fontsize=FONT['panel'], fontweight='bold', ha='left', va='top')
    panel.set_gid('qa:a:panel-label')
    top_x, top_y, top_w, top_h = (50.0, 1.5, 80.0, 8.0)
    _box(ax, x=top_x, y=top_y, width=top_w, height=top_h, face=_blend_with_white(MASTER_SPECTRUM[3], 0.11), edge=MASTER_SPECTRUM[3], linewidth=0.72, radius=1.2)
    top = content['top_cohort']
    _stacked_text(ax, registry, x=90.0, center_y=top_y + top_h / 2.0, lines=[str(top['line_1']), str(top['line_2'])], name='top-cohort', styles=[{'fontsize': FONT['top'], 'color': TEXT}, {'fontsize': FONT['top_count'], 'fontweight': 'bold', 'color': TEXT}], gap_mm=2.45)
    centers: Sequence[Mapping[str, Any]] = content['centers']
    columns = [(4.5, 50.0), (65.0, 50.0), (125.5, 50.0)]
    center_xs = [x + width / 2.0 for x, width in columns]
    header_y, header_h = (12.5, 5.0)
    source_y, source_h = (20.0, 8.5)
    criteria_y, criteria_h = (32.3, 11.4)
    data_y, data_h = (49.1, 9.4)
    final_y, final_h = (64.4, 10.8)
    split_y = 10.9
    _line(ax, (90.0, top_y + top_h), (90.0, split_y))
    _line(ax, (center_xs[0], split_y), (center_xs[-1], split_y))
    for center_x in center_xs:
        _arrow(ax, (center_x, split_y), (center_x, header_y - 0.25))
    for index, (row, (column_x, column_w)) in enumerate(zip(centers, columns)):
        center_id = str(row['id'])
        colour = CENTER_COLORS[center_id]
        center_x = column_x + column_w / 2.0
        _box(ax, x=column_x, y=header_y, width=column_w, height=header_h, face=colour, edge=colour, linewidth=0.65, radius=1.0)
        _text(ax, registry, center_x, header_y + header_h / 2.0, str(row['header']), name=f'{center_id}-header', fontsize=FONT['header'], fontweight='bold', color=WHITE)
        _arrow(ax, (center_x, header_y + header_h + 0.2), (center_x, source_y - 0.25))
        _box(ax, x=column_x, y=source_y, width=column_w, height=source_h, face=_blend_with_white(colour, 0.1), edge=colour, linewidth=0.62, radius=1.05)
        _stacked_text(ax, registry, x=center_x, center_y=source_y + source_h / 2.0, lines=[str(row['source_label']), f"n = {int(row['source_n']):,}"], name=f'{center_id}-source', styles=[{'fontsize': FONT['body'], 'color': MUTED}, {'fontsize': FONT['emphasis'], 'fontweight': 'bold', 'color': TEXT}], gap_mm=2.25)
    source_bus_y = 30.15
    _line(ax, (center_xs[0], source_bus_y), (center_xs[-1], source_bus_y))
    for center_x in center_xs:
        _line(ax, (center_x, source_y + source_h), (center_x, source_bus_y))
    _arrow(ax, (90.0, source_bus_y), (90.0, criteria_y - 0.25))
    inclusion_x, inclusion_w = (7.5, 101.0)
    exclusion_x, exclusion_w = (113.0, 61.5)
    _box(ax, x=inclusion_x, y=criteria_y, width=inclusion_w, height=criteria_h, face=_blend_with_white(MASTER_SPECTRUM[2], 0.085), edge=MASTER_SPECTRUM[2], linewidth=0.65, radius=1.1)
    inclusion = content['criteria']['inclusion']
    _stacked_text(ax, registry, x=inclusion_x + inclusion_w / 2.0, center_y=criteria_y + criteria_h / 2.0, lines=[str(inclusion['title']), *[str(item) for item in inclusion['lines']]], name='inclusion', styles=[{'fontsize': FONT['emphasis'], 'fontweight': 'bold'}, {'fontsize': FONT['body']}, {'fontsize': FONT['body']}], gap_mm=2.25)
    _box(ax, x=exclusion_x, y=criteria_y, width=exclusion_w, height=criteria_h, face=_blend_with_white(CORAL, 0.085), edge=CORAL, linewidth=0.65, radius=1.1)
    exclusion = content['criteria']['exclusion']
    _stacked_text(ax, registry, x=exclusion_x + exclusion_w / 2.0, center_y=criteria_y + criteria_h / 2.0, lines=[str(exclusion['title']), *[str(item) for item in exclusion['lines']]], name='exclusion', styles=[{'fontsize': FONT['emphasis'], 'fontweight': 'bold'}, {'fontsize': FONT['body']}, {'fontsize': FONT['body']}], gap_mm=2.25)
    _line(ax, (inclusion_x + inclusion_w + 0.8, criteria_y + criteria_h / 2.0), (exclusion_x - 0.8, criteria_y + criteria_h / 2.0), linewidth=0.55, linestyle=(0, (2.0, 1.5)))
    data_bus_y = 46.4
    _line(ax, (90.0, criteria_y + criteria_h), (90.0, data_bus_y))
    _line(ax, (center_xs[0], data_bus_y), (center_xs[-1], data_bus_y))
    for center_x in center_xs:
        _arrow(ax, (center_x, data_bus_y), (center_x, data_y - 0.25))
    for row, (column_x, column_w) in zip(centers, columns):
        center_id = str(row['id'])
        colour = CENTER_COLORS[center_id]
        center_x = column_x + column_w / 2.0
        _box(ax, x=column_x, y=data_y, width=column_w, height=data_h, face=_blend_with_white(colour, 0.065), edge=_blend_with_white(colour, 0.72), linewidth=0.6, radius=1.05)
        _stacked_text(ax, registry, x=center_x, center_y=data_y + data_h / 2.0, lines=[str(item) for item in row['data_lines']], name=f'{center_id}-data', styles=[{'fontsize': FONT['body'], 'fontweight': 'semibold'}, {'fontsize': FONT['body'], 'color': MUTED}], gap_mm=2.3)
        _arrow(ax, (center_x, data_y + data_h + 0.2), (center_x, final_y - 0.25))
        _box(ax, x=column_x, y=final_y, width=column_w, height=final_h, face=_blend_with_white(colour, 0.13), edge=colour, linewidth=0.72, radius=1.15)
        _stacked_text(ax, registry, x=center_x, center_y=final_y + final_h / 2.0, lines=[str(item) for item in row['final_lines']], name=f'{center_id}-final', styles=[{'fontsize': FONT['body'], 'fontweight': 'semibold'}, {'fontsize': FONT['body']}, {'fontsize': FONT['emphasis'], 'fontweight': 'bold'}], gap_mm=2.25)
    return (fig, registry)

def export_figure(fig: mpl.figure.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {'bbox_inches': None, 'pad_inches': 0, 'facecolor': WHITE}
    fig.savefig(OUTPUT_STEM.with_suffix('.pdf'), format='pdf', **common)
    fig.savefig(OUTPUT_STEM.with_suffix('.svg'), format='svg', **common)
    fig.savefig(OUTPUT_STEM.with_suffix('.png'), format='png', dpi=EXPORT_DPI, **common)

def build_reports(fig: mpl.figure.Figure, registry: ArtistRegistry, content: Mapping[str, Any]) -> dict[str, Any]:
    enable_local_pdf_qa()
    actual_size = assert_figure_size(fig, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM)
    layout = audit_figure_layout(fig, get_value('figures.config.item_271'), registry, minimum_font_pt=5.4, safe_pad_pt=1.5, check_collisions=True)
    write_json(layout, OUTPUT_DIR / f"{get_value('figures.config.item_282')}{VERSION}.json")
    exports = verify_export_bundle(OUTPUT_STEM, width_mm=FIGURE_WIDTH_MM, height_mm=FIGURE_HEIGHT_MM, dpi=EXPORT_DPI, minimum_svg_text_nodes=20)
    write_json(exports, OUTPUT_DIR / f"{get_value('figures.config.item_293')}{VERSION}.json")
    input_manifest = {'figure': 'Figure 1A', 'version': VERSION, 'palette': PALETTE_ID, 'render_only': True, 'files': [{'name': CONTENT_LOCK.name, 'sha256': _sha256(CONTENT_LOCK), 'role': 'frozen displayed content'}, {'name': AUTHOR_SOURCE.name, 'sha256': _sha256(AUTHOR_SOURCE), 'role': 'unaltered author source snapshot'}], 'numeric_checks': {'source_sum': sum((int(row['source_n']) for row in content['centers'])), 'locked_total': int(content['top_cohort']['n']), 'pass': True}}
    write_json(input_manifest, OUTPUT_DIR / f"{get_value('figures.config.item_304')}{VERSION}.json")
    summary = {'figure': 'Figure 1A', 'version': VERSION, 'palette': PALETTE_ID, 'render_only': True, 'size_mm': [round(actual_size[0], 3), round(actual_size[1], 3)], 'dpi': EXPORT_DPI, 'layout_summary': layout['summary'], 'output_stem': str(OUTPUT_STEM), 'files': {suffix: {'path': str(OUTPUT_STEM.with_suffix(f'.{suffix}')), 'sha256': sha256_file(OUTPUT_STEM.with_suffix(f'.{suffix}'))} for suffix in ('pdf', 'svg', 'png')}}
    write_json(summary, OUTPUT_DIR / f"{get_value('figures.config.item_315')}{VERSION}.json")
    return summary

def main() -> dict[str, Any]:
    content = load_content()
    fig, registry = build_figure(content)
    export_figure(fig)
    summary = build_reports(fig, registry, content)
    plt.close(fig)
    return summary
if __name__ == '__main__':
    main()
