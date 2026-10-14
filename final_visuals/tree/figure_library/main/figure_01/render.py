from __future__ import annotations
from avatar_paths import get_value, get_column
import sysconfig
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, MASTER_SPECTRUM, MBP_COLORS, MODALITY_COLORS, NEUTRAL_COLORS, PALETTE_ID
VERSION = 'v002'
PANEL_A_VERSION = 'v001'
DATE_TAG = get_value('figures.config.item_350')
EXPORT_DPI = 600
MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0
B_WIDTH_MM = 180.0
B_HEIGHT_MM = 150.0
A_HEIGHT_MM = 79.0
PANEL_GAP_MM = 2.0
FULL_WIDTH_MM = B_WIDTH_MM
FULL_HEIGHT_MM = A_HEIGHT_MM + PANEL_GAP_MM + B_HEIGHT_MM
CONTRACT_PATH = ROOT / 'inputs' / get_value('figures.config.item_200')
ASSET_PROVENANCE_PATH = ROOT / 'inputs' / get_value('figures.config.item_086')
HEALTH_ROOT = STYLE_ROOT / 'assets' / 'external' / 'healthicons'
HEALTH_SVG_ROOT = HEALTH_ROOT / 'public' / 'icons' / 'svg' / 'outline'
OUTPUT_DIR = ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
PANEL_DIR = OUTPUT_DIR / 'panels'
B_STEM = PANEL_DIR / f"{get_value('figures.config.item_326')}{VERSION}"
FULL_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_337')}{VERSION}"
A_OUTPUT_DIR = ROOT.parent / 'figure1a_flowchart' / 'outputs' / f'{DATE_TAG}_{PANEL_A_VERSION}'
A_STEM = A_OUTPUT_DIR / f"{get_value('figures.config.item_260')}{PANEL_A_VERSION}"
SVG_NS = get_value('figures.config.item_379')
XLINK_NS = get_value('figures.config.item_382')
ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', XLINK_NS)

def qname(tag: str) -> str:
    return f'{{{SVG_NS}}}{tag}'
WHITE = NEUTRAL_COLORS['background']
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
LIGHT = NEUTRAL_COLORS['light_background']
EDGE = '#D8E1E8'
RULE = '#C8D2DC'
ARROW = '#81909E'
CLINICAL = '#607384'
STAGE_BLUE = ENDPOINT_COLORS['PFS']
META = MODALITY_COLORS['Metagenomics']
RAD = MODALITY_COLORS['Radiomics']
BLOOD = MODALITY_COLORS['Blood']
URINE = MODALITY_COLORS['Urine']
RESPONSE = ENDPOINT_COLORS['Response']
PFS = ENDPOINT_COLORS['PFS']
OS = ENDPOINT_COLORS['OS']
DATA_BLOCK_COLORS = {'Clinical': CLINICAL, 'CT': RAD, 'Species': '#3F9E80', 'KO': META, 'Blood': BLOOD, 'Urine': URINE}
FONT_PT = {'panel': 8.8, 'stage': 7.1, 'module': 5.9, 'small': 5.3, 'micro': 5.0}
ICON_PATHS = {'body': HEALTH_SVG_ROOT / 'body' / get_value('figures.config.item_011'), 'liver': HEALTH_SVG_ROOT / 'body' / get_value('figures.config.item_026'), 'intestine': HEALTH_SVG_ROOT / 'body' / get_value('figures.config.item_021'), 'bacteria': HEALTH_SVG_ROOT / 'body' / get_value('figures.config.item_010'), 'medical-records': HEALTH_SVG_ROOT / 'specialties' / get_value('figures.config.item_027'), 'radiology': HEALTH_SVG_ROOT / 'specialties' / get_value('figures.config.item_029'), 'test-tubes': HEALTH_SVG_ROOT / 'devices' / get_value('figures.config.item_033'), 'urine-sample': HEALTH_SVG_ROOT / 'devices' / get_value('figures.config.item_034')}

@dataclass(frozen=True)
class Region:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def inset(self, dx: float=0.8, dy: float | None=None) -> 'Region':
        dy = dx if dy is None else dy
        return Region(self.x + dx, self.y + dy, self.w - 2.0 * dx, self.h - 2.0 * dy)

    def split_x(self, fractions: Sequence[float], gap: float=1.8) -> list['Region']:
        total = float(sum(fractions))
        if total <= 0 or self.w <= gap * (len(fractions) - 1):
            raise ValueError('Invalid horizontal split')
        available = self.w - gap * (len(fractions) - 1)
        regions: list[Region] = []
        cursor = self.x
        for fraction in fractions:
            width = available * float(fraction) / total
            regions.append(Region(cursor, self.y, width, self.h))
            cursor += width + gap
        return regions

@dataclass(frozen=True)
class StageSpec:
    number: int
    title: str
    region: Region
    fractions: tuple[float, ...]
STAGES = {1: StageSpec(1, 'Pretreatment multimodal profiling and patient alignment', Region(2.5, 1.5, 175.0, 22.0), (0.59, 0.27, 0.14)), 2: StageSpec(2, 'Multimodal geometry, clinical associations and complementarity', Region(2.5, 25.0, 175.0, 26.0), (0.16, 0.24, 0.28, 0.32)), 3: StageSpec(3, 'Feature prioritisation, multimodal screening and MRU selection', Region(2.5, 52.5, 175.0, 30.0), (0.23, 0.31, 0.14, 0.19, 0.13)), 4: StageSpec(4, 'MBP phenotyping and clinical relevance', Region(2.5, 84.0, 175.0, 25.5), (0.26, 0.27, 0.23, 0.24)), 5: StageSpec(5, 'Determinants, ecosystem axes and patient-level representation', Region(2.5, 111.0, 175.0, 37.5), (0.22, 0.46, 0.32))}

def fmt(value: float) -> str:
    return f'{float(value):.3f}'.rstrip('0').rstrip('.')

def pt_to_unit(value: float) -> float:
    return float(value) * MM_PER_INCH / POINTS_PER_INCH

def hex_rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip('#')
    return tuple((int(value[index:index + 2], 16) for index in (0, 2, 4)))

def mix(colour: str, target: str=WHITE, fraction: float=0.85) -> str:
    c0 = hex_rgb(colour)
    c1 = hex_rgb(target)
    rgb = tuple((round(a * (1.0 - fraction) + b * fraction) for a, b in zip(c0, c1)))
    return '#' + ''.join((f'{channel:02X}' for channel in rgb))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def enable_bundled_pdf_runtime() -> None:
    if importlib.util.find_spec('pypdf') or importlib.util.find_spec('PyPDF2'):
        return
    bundled = Path(get_value('figures.runtime.site_packages', sysconfig.get_paths()['purelib']))
    if bundled.is_dir() and str(bundled) not in sys.path:
        sys.path.insert(0, str(bundled))

class SvgCanvas:

    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)
        self.root = ET.Element(qname('svg'), {'width': f'{fmt(width)}mm', 'height': f'{fmt(height)}mm', 'viewBox': f'0 0 {fmt(width)} {fmt(height)}', 'version': '1.1'})
        self.defs = ET.SubElement(self.root, qname('defs'))
        self._add_marker('arrow-grey', ARROW)
        self._add_marker('arrow-blue', STAGE_BLUE)
        self.rect(0, 0, width, height, fill=WHITE, stroke='none')

    def _add_marker(self, marker_id: str, colour: str) -> None:
        marker = ET.SubElement(self.defs, qname('marker'), {'id': marker_id, 'viewBox': '0 0 6 6', 'refX': '5.2', 'refY': '3', 'markerWidth': '4.6', 'markerHeight': '4.6', 'orient': 'auto', 'markerUnits': 'strokeWidth'})
        ET.SubElement(marker, qname('path'), {'d': 'M0 0 L6 3 L0 6 Z', 'fill': colour})

    def group(self, *, parent: ET.Element | None=None, **attrs: str) -> ET.Element:
        return ET.SubElement(parent or self.root, qname('g'), {key.replace('_', '-'): str(value) for key, value in attrs.items()})

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str='none', stroke: str='none', stroke_width: float=0.4, rx: float=0.0, opacity: float=1.0, parent: ET.Element | None=None, **extra: str) -> ET.Element:
        attrs = {'x': fmt(x), 'y': fmt(y), 'width': fmt(w), 'height': fmt(h), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity)}
        if rx:
            attrs['rx'] = fmt(rx)
        attrs.update({key.replace('_', '-'): str(value) for key, value in extra.items()})
        return ET.SubElement(parent or self.root, qname('rect'), attrs)

    def circle(self, cx: float, cy: float, r: float, *, fill: str='none', stroke: str='none', stroke_width: float=0.4, opacity: float=1.0, parent: ET.Element | None=None, **extra: str) -> ET.Element:
        attrs = {'cx': fmt(cx), 'cy': fmt(cy), 'r': fmt(r), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity)}
        attrs.update({key.replace('_', '-'): str(value) for key, value in extra.items()})
        return ET.SubElement(parent or self.root, qname('circle'), attrs)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, *, fill: str='none', stroke: str='none', stroke_width: float=0.4, opacity: float=1.0, parent: ET.Element | None=None) -> ET.Element:
        return ET.SubElement(parent or self.root, qname('ellipse'), {'cx': fmt(cx), 'cy': fmt(cy), 'rx': fmt(rx), 'ry': fmt(ry), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity)})

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str=TEXT, stroke_width: float=0.45, dash: str | None=None, opacity: float=1.0, marker_end: str | None=None, parent: ET.Element | None=None) -> ET.Element:
        attrs = {'x1': fmt(x1), 'y1': fmt(y1), 'x2': fmt(x2), 'y2': fmt(y2), 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'stroke-linecap': 'round', 'opacity': fmt(opacity)}
        if dash:
            attrs['stroke-dasharray'] = dash
        if marker_end:
            attrs['marker-end'] = f'url(#{marker_end})'
        return ET.SubElement(parent or self.root, qname('line'), attrs)

    def path(self, d: str, *, fill: str='none', stroke: str='none', stroke_width: float=0.45, opacity: float=1.0, marker_end: str | None=None, parent: ET.Element | None=None, **extra: str) -> ET.Element:
        attrs = {'d': d, 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'opacity': fmt(opacity)}
        if marker_end:
            attrs['marker-end'] = f'url(#{marker_end})'
        attrs.update({key.replace('_', '-'): str(value) for key, value in extra.items()})
        return ET.SubElement(parent or self.root, qname('path'), attrs)

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str='none', stroke: str='none', stroke_width: float=0.4, opacity: float=1.0, parent: ET.Element | None=None) -> ET.Element:
        return ET.SubElement(parent or self.root, qname('polygon'), {'points': ' '.join((f'{fmt(x)},{fmt(y)}' for x, y in points)), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity), 'stroke-linejoin': 'round'})

    def text(self, x: float, y: float, label: str, *, size_pt: float=FONT_PT['small'], fill: str=TEXT, anchor: str='middle', weight: str='normal', italic: bool=False, line_height: float=1.15, parent: ET.Element | None=None, qa_name: str | None=None, opacity: float=1.0) -> ET.Element:
        attrs = {'x': fmt(x), 'y': fmt(y), 'font-family': 'Arial, Helvetica, sans-serif', 'font-size': fmt(pt_to_unit(size_pt)), 'fill': fill, 'text-anchor': anchor, 'font-weight': weight, 'font-style': 'italic' if italic else 'normal', 'dominant-baseline': 'middle', 'opacity': fmt(opacity)}
        if qa_name:
            attrs['data-qa-name'] = qa_name
        node = ET.SubElement(parent or self.root, qname('text'), attrs)
        lines = str(label).split('\n')
        if len(lines) == 1:
            node.text = lines[0]
        else:
            step = pt_to_unit(size_pt) * line_height
            start_y = y - step * (len(lines) - 1) / 2.0
            for index, item in enumerate(lines):
                tspan = ET.SubElement(node, qname('tspan'), {'x': fmt(x), 'y': fmt(start_y + index * step)})
                tspan.text = item
        return node

    def curve(self, start: tuple[float, float], end: tuple[float, float], *, stroke: str=ARROW, stroke_width: float=0.45, bend: float=0.0, opacity: float=1.0, marker_end: str | None=None, parent: ET.Element | None=None) -> ET.Element:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        d = f'M {fmt(x0)} {fmt(y0)} C {fmt(x0 + dx * 0.35)} {fmt(y0 + bend)}, {fmt(x0 + dx * 0.68)} {fmt(y1 - bend)}, {fmt(x1)} {fmt(y1)}'
        return self.path(d, stroke=stroke, stroke_width=stroke_width, opacity=opacity, marker_end=marker_end, parent=parent)

    def icon(self, name: str, region: Region, *, colour: str, opacity: float=1.0, parent: ET.Element | None=None) -> ET.Element:
        if name not in ICON_PATHS:
            raise KeyError(f'Unknown Health Icon: {name}')
        source = load_icon_tree(ICON_PATHS[name])
        view_box = [float(item) for item in source.attrib.get('viewBox', '0 0 48 48').split()]
        vx, vy, vw, vh = view_box
        scale = min(region.w / vw, region.h / vh)
        tx = region.x + (region.w - vw * scale) / 2.0 - vx * scale
        ty = region.y + (region.h - vh * scale) / 2.0 - vy * scale
        group = self.group(parent=parent, transform=f'translate({fmt(tx)} {fmt(ty)}) scale({fmt(scale)})', opacity=fmt(opacity), data_source=f'Health Icons:{name}', data_license='CC0-1.0')
        for child in source:
            clone = deepcopy(child)
            recolour_svg_tree(clone, colour)
            group.append(clone)
        return group

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.root, space='  ')
        ET.ElementTree(self.root).write(path, encoding='utf-8', xml_declaration=True)

@lru_cache(maxsize=None)
def load_icon_tree(path: Path) -> ET.Element:
    if not path.exists():
        raise FileNotFoundError(f'Required CC0 Health Icon missing: {path}')
    return ET.parse(path).getroot()

def recolour_svg_tree(node: ET.Element, colour: str) -> None:
    for item in node.iter():
        for key, value in list(item.attrib.items()):
            if 'currentColor' in value:
                item.set(key, value.replace('currentColor', colour))
        item.attrib.pop('class', None)

def stage_shell(canvas: SvgCanvas, spec: StageSpec) -> Region:
    r = spec.region
    canvas.rect(r.x, r.y, r.w, r.h, fill='#FBFCFD', stroke=EDGE, stroke_width=0.42, rx=0.8)
    number_x = r.x + 3.2
    title_y = r.y + 3.05
    canvas.circle(number_x, title_y, 2.05, fill=STAGE_BLUE, stroke=WHITE, stroke_width=0.5)
    canvas.text(number_x, title_y, str(spec.number), size_pt=6.2, fill=WHITE, weight='bold', qa_name=f'stage-{spec.number}-number')
    canvas.text(r.x + 6.2, title_y, spec.title, size_pt=FONT_PT['stage'], anchor='start', weight='bold', qa_name=f'stage-{spec.number}-title')
    canvas.line(r.x + 1.0, r.y + 5.35, r.right - 1.0, r.y + 5.35, stroke=RULE, stroke_width=0.35)
    return Region(r.x + 2.0, r.y + 6.0, r.w - 4.0, r.h - 7.1)

def module_guides(canvas: SvgCanvas, regions: Sequence[Region], labels: Sequence[str]) -> None:
    if len(regions) != len(labels):
        raise ValueError('Module label count does not match layout')
    for index, (region, label) in enumerate(zip(regions, labels)):
        canvas.text(region.cx, region.y + 1.25, label, size_pt=FONT_PT['module'], weight='bold', qa_name=f"module-{label.lower().replace(' ', '-')}")
        if index:
            x = (regions[index - 1].right + region.x) / 2.0
            canvas.line(x, region.y + 0.3, x, region.bottom - 0.1, stroke=EDGE, stroke_width=0.35, dash='1.1 1.1')

def process_arrow(canvas: SvgCanvas, left: Region, right: Region, y: float | None=None) -> None:
    yy = y if y is not None else (left.cy + right.cy) / 2.0
    canvas.line(left.right + 0.25, yy, right.x - 0.25, yy, stroke=ARROW, stroke_width=0.52, marker_end='arrow-grey')

def draw_tumour_liver_ct(canvas: SvgCanvas, region: Region, colour: str=RAD) -> None:
    canvas.rect(region.x, region.y, region.w, region.h, fill=mix(colour, fraction=0.94), stroke=colour, stroke_width=0.42, rx=0.7)
    canvas.ellipse(region.cx, region.cy, region.w * 0.36, region.h * 0.34, fill='#EEF1F4', stroke='#82909D', stroke_width=0.32)
    d = f'M {fmt(region.cx - region.w * 0.22)} {fmt(region.cy - region.h * 0.04)} C {fmt(region.cx - region.w * 0.09)} {fmt(region.cy - region.h * 0.28)}, {fmt(region.cx + region.w * 0.22)} {fmt(region.cy - region.h * 0.18)}, {fmt(region.cx + region.w * 0.22)} {fmt(region.cy + region.h * 0.1)} C {fmt(region.cx + region.w * 0.1)} {fmt(region.cy + region.h * 0.26)}, {fmt(region.cx - region.w * 0.19)} {fmt(region.cy + region.h * 0.22)}, {fmt(region.cx - region.w * 0.22)} {fmt(region.cy - region.h * 0.04)} Z'
    canvas.path(d, fill=mix(colour, fraction=0.7), stroke=colour, stroke_width=0.34)
    canvas.circle(region.cx + region.w * 0.08, region.cy - region.h * 0.03, min(region.w, region.h) * 0.075, fill=mix(RESPONSE, fraction=0.4), stroke=RESPONSE, stroke_width=0.32)

def draw_metabolite_cluster(canvas: SvgCanvas, cx: float, cy: float, colour: str, scale: float=1.0) -> None:
    points = [(0, 0), (-2.0, -1.1), (2.0, -1.2), (-1.7, 1.5), (2.0, 1.4)]
    for dx, dy in points[1:]:
        canvas.line(cx, cy, cx + dx * scale, cy + dy * scale, stroke=colour, stroke_width=0.35)
    for index, (dx, dy) in enumerate(points):
        canvas.circle(cx + dx * scale, cy + dy * scale, (0.58 if index == 0 else 0.45) * scale, fill=mix(colour, fraction=0.65), stroke=colour, stroke_width=0.32)

def draw_patient_alignment(canvas: SvgCanvas, region: Region) -> None:
    colours = [CLINICAL, RAD, META, BLOOD, URINE]
    labels = ['C', 'CT', 'M', 'B', 'U']
    grid = Region(region.x + 4.8, region.y + 3.7, region.w - 6.0, region.h - 6.0)
    cell_gap = 0.55
    cell_w = (grid.w - 4 * cell_gap) / 5.0
    cell_h = 1.42
    row_gap = 0.62
    for col, (label, colour) in enumerate(zip(labels, colours)):
        x = grid.x + col * (cell_w + cell_gap) + cell_w / 2.0
        canvas.text(x, grid.y - 1.15, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
    for row in range(4):
        cy = grid.y + row * (cell_h + row_gap) + cell_h / 2.0
        canvas.circle(region.x + 2.0, cy, 0.52, fill='#AEB9C3')
        canvas.line(region.x + 2.8, cy, grid.x - 0.6, cy, stroke='#AEB9C3', stroke_width=0.32)
        for col, colour in enumerate(colours):
            x = grid.x + col * (cell_w + cell_gap)
            canvas.rect(x, cy - cell_h / 2.0, cell_w, cell_h, fill=mix(colour, fraction=0.82), stroke=colour, stroke_width=0.28, rx=0.28)
    canvas.text(region.cx, region.bottom - 0.95, 'Patient-aligned internal cohort', size_pt=FONT_PT['small'], fill=MUTED)

def draw_observed_outcomes(canvas: SvgCanvas, region: Region) -> None:
    line_x = region.x + 3.0
    y0 = region.y + 3.9
    y1 = region.bottom - 1.4
    canvas.line(line_x, y0, line_x, y1, stroke='#9AA8B5', stroke_width=0.45)
    outcomes = [(y0 + 0.8, RESPONSE, 'mRECIST response', 'circle'), ((y0 + y1) / 2.0, PFS, 'PFS', 'square'), (y1 - 0.8, OS, 'OS', 'triangle')]
    for yy, colour, label, shape in outcomes:
        if shape == 'circle':
            canvas.circle(line_x, yy, 0.75, fill=colour, stroke=WHITE, stroke_width=0.28)
        elif shape == 'square':
            canvas.rect(line_x - 0.7, yy - 0.7, 1.4, 1.4, fill=colour, stroke=WHITE, stroke_width=0.28, rx=0.15)
        else:
            canvas.polygon([(line_x, yy - 0.85), (line_x - 0.82, yy + 0.62), (line_x + 0.82, yy + 0.62)], fill=colour, stroke=WHITE, stroke_width=0.28)
        canvas.text(line_x + 2.0, yy, label, size_pt=FONT_PT['small'], fill=colour, anchor='start', weight='bold')

def draw_stage_1(canvas: SvgCanvas, contract_stage: Mapping[str, Any]) -> None:
    content = stage_shell(canvas, STAGES[1])
    modules = content.split_x(STAGES[1].fractions, gap=2.2)
    module_guides(canvas, modules, contract_stage['modules'])
    inputs, aligned, outcomes = modules
    art_top = inputs.y + 3.2
    art_h = inputs.h - 4.2
    patient_box = Region(inputs.x + 0.4, art_top + 0.1, 10.4, art_h - 0.2)
    canvas.icon('body', patient_box, colour=CLINICAL)
    canvas.icon('liver', Region(patient_box.x + 3.8, patient_box.y + 4.0, 4.1, 3.6), colour=RESPONSE)
    canvas.text(patient_box.cx, inputs.bottom - 0.65, 'HCC patient', size_pt=FONT_PT['micro'], fill=CLINICAL, weight='bold')
    icon_specs = [('medical-records', 'Clinical', CLINICAL), ('radiology', 'CT\nradiomics', RAD), ('intestine', 'Stool', META), ('test-tubes', 'Blood\nmetabolomics', BLOOD), ('urine-sample', 'Urine\nmetabolomics', URINE)]
    usable_x0 = patient_box.right + 3.0
    icon_step = (inputs.right - usable_x0 - 1.2) / len(icon_specs)
    hub_x = inputs.right - 0.5
    hub_y = inputs.bottom - 3.0
    icon_centres = [usable_x0 + icon_step * (index + 0.48) for index in range(len(icon_specs))]
    canvas.line(patient_box.right - 0.5, hub_y, hub_x, hub_y, stroke='#B4BEC7', stroke_width=0.34)
    for cx, (_, _, colour) in zip(icon_centres, icon_specs):
        canvas.line(cx, art_top + 7.7, cx, hub_y, stroke=colour, stroke_width=0.32, opacity=0.68)
    for index, (icon_name, label, colour) in enumerate(icon_specs):
        cx = icon_centres[index]
        box = Region(cx - 3.6, art_top + 0.3, 7.2, 7.2)
        if icon_name == 'radiology':
            draw_tumour_liver_ct(canvas, box, colour=colour)
        else:
            canvas.icon(icon_name, box, colour=colour)
        if label == 'Stool':
            canvas.icon('bacteria', Region(cx + 1.8, art_top + 3.8, 3.2, 3.2), colour=META)
            canvas.text(cx, inputs.bottom - 1.02, 'Metagenomics\nSpecies · KO', size_pt=FONT_PT['micro'], fill=colour, weight='bold')
        else:
            canvas.text(cx, inputs.bottom - 0.72, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
    canvas.circle(hub_x, hub_y, 0.72, fill=STAGE_BLUE, stroke=WHITE, stroke_width=0.25)
    draw_patient_alignment(canvas, aligned)
    draw_observed_outcomes(canvas, outcomes)
    process_arrow(canvas, inputs, aligned, y=inputs.cy + 0.6)
    process_arrow(canvas, aligned, outcomes, y=aligned.cy + 0.6)

def draw_qc(canvas: SvgCanvas, region: Region) -> None:
    y_base = region.bottom - 2.5
    left_x0 = region.x + 1.1
    right_x0 = region.cx + 1.0
    colours = [META, RAD, BLOOD, URINE]
    for index, colour in enumerate(colours):
        offset = index * 0.22
        d1 = f'M {fmt(left_x0)} {fmt(y_base - offset)} C {fmt(left_x0 + 2.0)} {fmt(y_base - 4.2 + index * 0.32)}, {fmt(left_x0 + 4.0)} {fmt(y_base - 5.0 + index * 0.25)}, {fmt(left_x0 + 6.4)} {fmt(y_base - offset)}'
        d2 = f'M {fmt(right_x0)} {fmt(y_base - offset)} C {fmt(right_x0 + 1.9)} {fmt(y_base - 4.6)}, {fmt(right_x0 + 4.1)} {fmt(y_base - 4.6)}, {fmt(right_x0 + 6.2)} {fmt(y_base - offset)}'
        canvas.path(d1, stroke=colour, stroke_width=0.42, opacity=0.72)
        canvas.path(d2, stroke=colour, stroke_width=0.48, opacity=0.88)
    canvas.line(left_x0 - 0.2, y_base, left_x0 + 6.7, y_base, stroke='#9AA8B5', stroke_width=0.3)
    canvas.line(right_x0 - 0.2, y_base, right_x0 + 6.5, y_base, stroke='#9AA8B5', stroke_width=0.3)
    canvas.line(left_x0 + 7.3, y_base - 2.5, right_x0 - 0.8, y_base - 2.5, stroke=ARROW, stroke_width=0.45, marker_end='arrow-grey')
    canvas.text(left_x0 + 3.2, region.y + 3.7, 'raw', size_pt=FONT_PT['micro'], fill=MUTED)
    canvas.text(right_x0 + 3.1, region.y + 3.7, 'harmonised', size_pt=FONT_PT['micro'], fill=MUTED)
    canvas.circle(region.right - 2.0, region.y + 4.0, 1.15, fill=mix(META, fraction=0.85), stroke=META, stroke_width=0.35)
    canvas.path(f'M {fmt(region.right - 2.55)} {fmt(region.y + 4.0)} L {fmt(region.right - 2.12)} {fmt(region.y + 4.45)} L {fmt(region.right - 1.35)} {fmt(region.y + 3.45)}', stroke=META, stroke_width=0.48)

def draw_patient_geometry(canvas: SvgCanvas, region: Region) -> None:
    plot = Region(region.x + 1.0, region.y + 3.5, region.w - 2.0, region.h - 4.5)
    canvas.line(plot.x + 1.0, plot.bottom - 0.8, plot.right - 0.6, plot.bottom - 0.8, stroke='#9CAAB7', stroke_width=0.32)
    canvas.line(plot.x + 1.0, plot.bottom - 0.8, plot.x + 1.0, plot.y + 0.8, stroke='#9CAAB7', stroke_width=0.32)
    clusters = [(0.25, 0.58, RESPONSE), (0.55, 0.34, PFS), (0.77, 0.62, OS)]
    offsets = [(-0.08, 0.02), (-0.04, -0.1), (0.0, 0.06), (0.05, -0.03), (0.09, 0.09), (0.02, 0.14)]
    for cx_f, cy_f, colour in clusters:
        for index, (dx, dy) in enumerate(offsets):
            cx = plot.x + 1.0 + (cx_f + dx) * (plot.w - 2.0)
            cy = plot.y + 0.8 + (cy_f + dy) * (plot.h - 2.0)
            if index % 3 == 0:
                canvas.circle(cx, cy, 0.52, fill=colour, stroke=WHITE, stroke_width=0.2, opacity=0.86)
            elif index % 3 == 1:
                canvas.rect(cx - 0.46, cy - 0.46, 0.92, 0.92, fill=colour, stroke=WHITE, stroke_width=0.18, rx=0.12, opacity=0.78)
            else:
                canvas.polygon([(cx, cy - 0.55), (cx - 0.52, cy + 0.42), (cx + 0.52, cy + 0.42)], fill=colour, stroke=WHITE, stroke_width=0.18, opacity=0.78)
    modality_positions = [(plot.x + 2.0, plot.y + 1.2, META, 'M'), (plot.right - 2.0, plot.y + 1.2, RAD, 'R'), (plot.x + 2.0, plot.bottom - 2.0, BLOOD, 'B'), (plot.right - 2.0, plot.bottom - 2.0, URINE, 'U')]
    for x, y, colour, label in modality_positions:
        canvas.circle(x, y, 0.9, fill=mix(colour, fraction=0.82), stroke=colour, stroke_width=0.35)
        canvas.text(x, y, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
        canvas.curve((x, y), (plot.cx, plot.cy), stroke=colour, stroke_width=0.28, opacity=0.28, bend=(plot.cy - y) * 0.18)

def draw_domain_heatmap(canvas: SvgCanvas, region: Region) -> None:
    labels = ['Tumour', 'Liver', 'Inflamm.', 'Host']
    columns = ['KO', 'Sp.', 'CT', 'B', 'U', 'Resp.', 'PFS', 'OS']
    column_colours = [META, DATA_BLOCK_COLORS['Species'], RAD, BLOOD, URINE, RESPONSE, PFS, OS]
    grid = Region(region.x + 8.4, region.y + 3.7, region.w - 9.2, region.h - 5.0)
    gap = 0.36
    cell_w = (grid.w - gap * (len(columns) - 1)) / len(columns)
    cell_h = (grid.h - gap * (len(labels) - 1)) / len(labels)
    values = [[0, 1, 2, 4, 5, 6, 5, 4], [1, 2, 3, 5, 4, 2, 1, 0], [5, 4, 3, 2, 1, 0, 2, 4], [6, 5, 4, 3, 2, 1, 3, 5]]
    for col, (label, colour) in enumerate(zip(columns, column_colours)):
        x = grid.x + col * (cell_w + gap) + cell_w / 2.0
        canvas.rect(x - cell_w * 0.35, grid.y - 1.55, cell_w * 0.7, 0.48, fill=colour, stroke='none', rx=0.18)
        canvas.text(x, grid.y - 0.55, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
    for row, label in enumerate(labels):
        cy = grid.y + row * (cell_h + gap) + cell_h / 2.0
        canvas.text(grid.x - 0.8, cy, label, size_pt=FONT_PT['micro'], fill=MUTED, anchor='end')
        for col, value in enumerate(values[row]):
            x = grid.x + col * (cell_w + gap)
            y = grid.y + row * (cell_h + gap)
            canvas.rect(x, y, cell_w, cell_h, fill=DIVERGING_COLORS[value], stroke=WHITE, stroke_width=0.18, rx=0.12)

def draw_mini_increment_network(canvas: SvgCanvas, cx: float, cy: float, radius: float, endpoint: str, colour: str) -> None:
    nodes = [('KO', META), ('Sp', DATA_BLOCK_COLORS['Species']), ('R', RAD), ('B', BLOOD), ('U', URINE)]
    positions = [(cx + radius * math.cos(-math.pi / 2.0 + index * 2.0 * math.pi / len(nodes)), cy + radius * math.sin(-math.pi / 2.0 + index * 2.0 * math.pi / len(nodes))) for index in range(len(nodes))]
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for source, target in links:
        canvas.line(positions[source][0], positions[source][1], positions[target][0], positions[target][1], stroke='#AEBAC5', stroke_width=0.3, opacity=0.46)
    for (x, y), (label, node_colour) in zip(positions, nodes):
        canvas.circle(x, y, 0.72, fill=mix(node_colour, fraction=0.72), stroke=node_colour, stroke_width=0.28)
        canvas.text(x, y, label, size_pt=FONT_PT['micro'], fill=node_colour, weight='bold')
    canvas.circle(cx, cy, 1.32, fill=WHITE, stroke=colour, stroke_width=0.48)
    canvas.text(cx, cy, endpoint, size_pt=FONT_PT['micro'], fill=colour, weight='bold')

def draw_complementarity(canvas: SvgCanvas, region: Region) -> None:
    centres = [region.x + region.w * fraction for fraction in (0.18, 0.5, 0.82)]
    labels = [('Resp.', RESPONSE), ('PFS', PFS), ('OS', OS)]
    cy = region.y + 8.4
    radius = min(region.w * 0.105, region.h * 0.27)
    for cx, (label, colour) in zip(centres, labels):
        draw_mini_increment_network(canvas, cx, cy, radius, label, colour)
    canvas.text(region.cx, region.bottom - 0.8, 'complementarity · leave-one-block-out', size_pt=FONT_PT['micro'], fill=MUTED)
    base_y = region.bottom - 2.5
    for index, colour in enumerate((META, RAD, BLOOD, URINE)):
        x = region.x + 2.5 + index * (region.w - 5.0) / 4.0
        canvas.rect(x, base_y - 0.35, (region.w - 7.5) / 5.2, 0.7, fill=mix(colour, fraction=0.55), stroke='none', rx=0.25)

def draw_stage_2(canvas: SvgCanvas, contract_stage: Mapping[str, Any]) -> None:
    content = stage_shell(canvas, STAGES[2])
    modules = content.split_x(STAGES[2].fractions, gap=2.0)
    module_guides(canvas, modules, contract_stage['modules'])
    draw_qc(canvas, modules[0])
    draw_patient_geometry(canvas, modules[1])
    draw_domain_heatmap(canvas, modules[2])
    draw_complementarity(canvas, modules[3])
    for left, right in zip(modules[:-1], modules[1:]):
        process_arrow(canvas, left, right, y=content.cy + 0.7)

def draw_cv_shap(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.8, region.y + 3.1, region.w - 1.6, region.h - 4.0)
    patient_x = art.x + 1.4
    patient_ys = [art.y + 1.2 + index * (art.h - 2.4) / 6.0 for index in range(7)]
    for index, yy in enumerate(patient_ys):
        canvas.circle(patient_x, yy, 0.47, fill=STAGE_BLUE if index < 5 else mix(STAGE_BLUE, fraction=0.72), stroke=WHITE, stroke_width=0.16)
        canvas.line(patient_x + 0.75, yy, patient_x + 2.1, yy, stroke='#A9B5C0', stroke_width=0.28)
    fold_x = art.x + 4.4
    fold_w = max(7.5, art.w * 0.29)
    fold_h = (art.h - 1.0) / 5.0
    for resample in range(5):
        y = art.y + 0.5 + resample * fold_h
        colour = MASTER_SPECTRUM[2 + resample % 3]
        canvas.rect(fold_x, y, fold_w, fold_h - 0.55, fill=mix(colour, fraction=0.88), stroke=colour, stroke_width=0.25, rx=0.28)
        for dot in range(5):
            dot_x = fold_x + 0.9 + dot * (fold_w - 1.8) / 4.0
            retained = (dot + resample) % 4 != 0
            canvas.circle(dot_x, y + (fold_h - 0.55) / 2.0, 0.24, fill=colour if retained else WHITE, stroke=colour, stroke_width=0.18)
    canvas.line(patient_x + 2.45, art.cy, fold_x - 0.5, art.cy, stroke=ARROW, stroke_width=0.44, marker_end='arrow-grey')
    feature_x = fold_x + fold_w + 3.0
    feature_w = art.right - feature_x - 0.5
    endpoint_marks = [RESPONSE, PFS, OS]
    for index, colour in enumerate(endpoint_marks):
        x = feature_x + 1.0 + index * 2.8
        if index == 0:
            canvas.circle(x, art.y + 1.2, 0.56, fill=colour)
        elif index == 1:
            canvas.rect(x - 0.5, art.y + 0.7, 1.0, 1.0, fill=colour, rx=0.12)
        else:
            canvas.polygon([(x, art.y + 0.65), (x - 0.55, art.y + 1.65), (x + 0.55, art.y + 1.65)], fill=colour)
    canvas.text(feature_x + feature_w / 2.0, art.y + 2.55, '3 endpoints', size_pt=FONT_PT['micro'], fill=MUTED)
    bar_colours = [META, RAD, URINE, BLOOD, META, RAD]
    for index, colour in enumerate(bar_colours):
        y = art.y + 4.0 + index * (art.h - 5.0) / 5.6
        width = feature_w * (0.38 + index % 4 * 0.13)
        canvas.rect(feature_x, y, width, 0.72, fill=mix(colour, fraction=0.38), stroke='none', rx=0.26)
        canvas.circle(feature_x - 0.65, y + 0.36, 0.24, fill=colour)
    canvas.text(art.x + 6.4, art.bottom - 0.25, 'bootstrap resamples', size_pt=FONT_PT['micro'], fill=MUTED)
    canvas.text(feature_x + feature_w / 2.0, art.bottom - 0.25, 'feature retention', size_pt=FONT_PT['micro'], fill=MUTED)

def draw_combination_grid(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.7, region.y + 3.2, region.w - 1.4, region.h - 4.0)
    modality_labels = ['M', 'R', 'B', 'U']
    modality_colours = [META, RAD, BLOOD, URINE]
    all_combinations = [combo for size in range(1, 5) for combo in combinations(modality_labels, size)]
    columns = 5
    rows = 3
    gap_x = 0.65
    gap_y = 0.75
    cell_w = (art.w - gap_x * (columns - 1)) / columns
    cell_h = (art.h - gap_y * (rows - 1) - 1.2) / rows
    for index, combo in enumerate(all_combinations):
        row = index // columns
        col = index % columns
        x = art.x + col * (cell_w + gap_x)
        y = art.y + row * (cell_h + gap_y)
        selected = set(combo) == {'M', 'R', 'U'}
        canvas.rect(x, y, cell_w, cell_h, fill=mix(STAGE_BLUE, fraction=0.94) if selected else '#F8FAFC', stroke=STAGE_BLUE if selected else 'none', stroke_width=0.62 if selected else 0.0, rx=0.45)
        dot_y = y + 1.35
        for item_index, (label, colour) in enumerate(zip(modality_labels, modality_colours)):
            cx = x + 1.0 + item_index * (cell_w - 2.0) / 3.0
            included = label in combo
            canvas.circle(cx, dot_y, 0.42, fill=colour if included else WHITE, stroke=colour if included else '#C9D2DA', stroke_width=0.25)
        endpoint_y = y + cell_h - 1.05
        for endpoint_index, colour in enumerate((RESPONSE, PFS, OS)):
            bar_x = x + 0.85 + endpoint_index * (cell_w - 1.7) / 3.0
            canvas.rect(bar_x, endpoint_y, max(0.7, (cell_w - 2.5) / 4.1), 0.46, fill=mix(colour, fraction=0.3 if selected else 0.66), stroke='none', rx=0.18)
        if selected:
            canvas.text(x + cell_w - 0.7, y + 0.72, 'MRU', size_pt=FONT_PT['micro'], fill=STAGE_BLUE, anchor='end', weight='bold')
    canvas.text(art.cx, art.bottom - 0.05, '15 subsets · repeated 5 × 5 cross-validation', size_pt=FONT_PT['micro'], fill=MUTED)

def draw_mru_basis(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.5, region.y + 3.0, region.w - 1.0, region.h - 3.9)
    cx, cy = (art.cx, art.cy - 0.1)
    radius = min(art.w, art.h) * 0.27
    satellites = [(-math.pi / 2.0, 'M', META, 'bacteria'), (math.pi * 0.16, 'R', RAD, 'radiology'), (math.pi * 0.84, 'U', URINE, 'urine-sample')]
    for angle, label, colour, icon_name in satellites:
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        canvas.line(cx, cy, x, y, stroke=colour, stroke_width=0.48, opacity=0.75)
        canvas.circle(x, y, 2.25, fill=mix(colour, fraction=0.88), stroke=colour, stroke_width=0.45)
        canvas.icon(icon_name, Region(x - 1.45, y - 1.45, 2.9, 2.9), colour=colour)
        canvas.circle(x + 1.55, y - 1.55, 0.72, fill=colour, stroke=WHITE, stroke_width=0.2)
        canvas.text(x + 1.55, y - 1.55, label, size_pt=FONT_PT['micro'], fill=WHITE, weight='bold')
    canvas.circle(cx, cy, 2.55, fill=WHITE, stroke=STAGE_BLUE, stroke_width=0.58)
    canvas.icon('body', Region(cx - 1.3, cy - 1.55, 2.6, 3.1), colour=STAGE_BLUE)
    canvas.text(cx, art.bottom - 0.55, 'M + R + U', size_pt=FONT_PT['small'], fill=STAGE_BLUE, weight='bold')

def draw_step_curve(canvas: SvgCanvas, region: Region, colour: str, level_shift: float=0.0) -> None:
    x0 = region.x + 1.0
    y0 = region.bottom - 0.7
    x1 = region.right - 0.4
    y1 = region.y + 0.6
    canvas.line(x0, y1, x0, y0, stroke='#9DAAB6', stroke_width=0.28)
    canvas.line(x0, y0, x1, y0, stroke='#9DAAB6', stroke_width=0.28)
    points = [(x0, y1 + 0.55 + level_shift), (x0 + (x1 - x0) * 0.24, y1 + 0.55 + level_shift), (x0 + (x1 - x0) * 0.24, y1 + (y0 - y1) * 0.34 + level_shift), (x0 + (x1 - x0) * 0.55, y1 + (y0 - y1) * 0.34 + level_shift), (x0 + (x1 - x0) * 0.55, y1 + (y0 - y1) * 0.58 + level_shift), (x1, y1 + (y0 - y1) * 0.58 + level_shift)]
    d = 'M ' + ' L '.join((f'{fmt(x)} {fmt(y)}' for x, y in points))
    canvas.path(d, stroke=colour, stroke_width=0.55)

def draw_endpoint_projection(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.7, region.y + 3.0, region.w - 1.4, region.h - 3.7)
    lane_h = art.h / 3.0
    items = [('Response', RESPONSE), ('PFS', PFS), ('OS', OS)]
    for index, (label, colour) in enumerate(items):
        lane = Region(art.x, art.y + index * lane_h, art.w, lane_h)
        canvas.text(lane.x + 0.2, lane.cy, label, size_pt=FONT_PT['small'], fill=colour, anchor='start', weight='bold')
        plot = Region(lane.x + art.w * 0.42, lane.y + 0.35, art.w * 0.55, lane.h - 0.7)
        if label == 'Response':
            canvas.icon('liver', Region(plot.x, plot.y + 0.1, 4.0, plot.h - 0.2), colour=RAD)
            canvas.circle(plot.x + 2.55, plot.cy - 0.15, 0.75, fill=mix(RESPONSE, fraction=0.48), stroke=RESPONSE, stroke_width=0.3)
            canvas.line(plot.x + 4.6, plot.cy, plot.x + 7.0, plot.cy, stroke=ARROW, stroke_width=0.36, marker_end='arrow-grey')
            canvas.circle(plot.x + 8.0, plot.cy, 0.48, fill=mix(RESPONSE, fraction=0.55), stroke=RESPONSE, stroke_width=0.3)
        else:
            draw_step_curve(canvas, plot, colour)
        if index < 2:
            canvas.line(lane.x, lane.bottom, lane.right, lane.bottom, stroke=EDGE, stroke_width=0.25)

def draw_external_transfer(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.7, region.y + 3.1, region.w - 1.4, region.h - 4.0)
    labels = [('M', META), ('R', RAD), ('B', BLOOD), ('U', URINE)]
    step = art.h / 4.0
    for index, (label, colour) in enumerate(labels):
        cy = art.y + (index + 0.5) * step
        canvas.circle(art.x + 1.6, cy, 0.82, fill=mix(colour, fraction=0.72), stroke=colour, stroke_width=0.34)
        canvas.text(art.x + 1.6, cy, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
        canvas.line(art.x + 2.7, cy, art.right - 5.0, cy, stroke=colour, stroke_width=0.38, dash='1.0 0.8', marker_end='arrow-grey', opacity=0.72)
        for person in range(3):
            px = art.right - 3.8 + person * 1.25
            canvas.circle(px, cy - 0.35, 0.3, fill='#9EABB7')
            canvas.path(f'M {fmt(px - 0.52)} {fmt(cy + 0.48)} Q {fmt(px)} {fmt(cy - 0.05)} {fmt(px + 0.52)} {fmt(cy + 0.48)}', stroke='#9EABB7', stroke_width=0.32)
    canvas.text(art.cx, art.bottom + 0.05, 'separate cohorts', size_pt=FONT_PT['micro'], fill=MUTED)

def draw_stage_3(canvas: SvgCanvas, contract_stage: Mapping[str, Any]) -> None:
    content = stage_shell(canvas, STAGES[3])
    modules = content.split_x(STAGES[3].fractions, gap=1.85)
    labels = ['Endpoint-specific SHAP\nprioritisation', contract_stage['modules'][1], contract_stage['modules'][2], contract_stage['modules'][3], 'Modality-specific\ntransferability']
    module_guides(canvas, modules, labels)
    draw_cv_shap(canvas, modules[0])
    draw_combination_grid(canvas, modules[1])
    draw_mru_basis(canvas, modules[2])
    draw_endpoint_projection(canvas, modules[3])
    draw_external_transfer(canvas, modules[4])
    for left, right in zip(modules[:3], modules[1:4]):
        process_arrow(canvas, left, right, y=content.cy + 1.0)
    branch_y = content.bottom + 0.05
    canvas.line(modules[1].right - 1.0, branch_y, modules[4].x + 1.0, branch_y, stroke=ARROW, stroke_width=0.38, dash='1.1 0.9', opacity=0.72, marker_end='arrow-grey')

def draw_mbp_score(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.8, region.y + 3.1, region.w - 1.6, region.h - 4.0)
    centre_x = art.x + 8.2
    endpoint_y = [art.y + 1.8, art.y + 5.0, art.y + 8.2]
    endpoints = [('Response', RESPONSE), ('PFS', PFS), ('OS', OS)]
    for yy, (label, colour) in zip(endpoint_y, endpoints):
        canvas.circle(art.x + 1.5, yy, 0.72, fill=colour)
        canvas.text(art.x + 2.7, yy, label, size_pt=FONT_PT['micro'], fill=colour, anchor='start', weight='bold')
        canvas.curve((art.x + 7.0, yy), (centre_x, art.y + 5.0), stroke=colour, stroke_width=0.48, bend=(art.y + 5.0 - yy) * 0.18, marker_end='arrow-grey', opacity=0.74)
    canvas.circle(centre_x, art.y + 5.0, 1.2, fill=WHITE, stroke=STAGE_BLUE, stroke_width=0.46)
    canvas.text(centre_x, art.y + 5.0, '=', size_pt=FONT_PT['small'], fill=STAGE_BLUE, weight='bold')
    strip = Region(art.x + 11.1, art.y + 2.2, art.w - 12.0, 5.7)
    segment_w = strip.w / 3.0
    for index, (label, colour) in enumerate(MBP_COLORS.items()):
        x = strip.x + index * segment_w
        canvas.rect(x, strip.y, segment_w, strip.h, fill=mix(colour, fraction=0.82), stroke=WHITE, stroke_width=0.28)
        for patient in range(5):
            px = x + (patient + 0.5) * segment_w / 5.0
            canvas.circle(px, strip.y + 2.25 + patient % 2 * 0.55, 0.46, fill=colour, stroke=WHITE, stroke_width=0.16)
        canvas.text(x + segment_w / 2.0, strip.bottom + 1.0, label, size_pt=FONT_PT['micro'], fill=colour, weight='bold')
    canvas.text(strip.cx, strip.bottom + 2.25, 'high benefit  ←                       →  low benefit', size_pt=FONT_PT['micro'], fill=MUTED)
    canvas.text(strip.cx, art.bottom - 0.2, 'equal-weight patient-level score', size_pt=FONT_PT['micro'], fill=MUTED)

def draw_km_family(canvas: SvgCanvas, region: Region, endpoint: str) -> None:
    canvas.text(region.cx, region.y + 0.65, endpoint, size_pt=FONT_PT['micro'], fill=PFS if endpoint == 'PFS' else OS, weight='bold')
    plot = Region(region.x + 0.4, region.y + 1.5, region.w - 0.8, region.h - 1.9)
    x0, y0 = (plot.x + 0.9, plot.bottom - 0.5)
    x1, y1 = (plot.right - 0.2, plot.y + 0.4)
    canvas.line(x0, y1, x0, y0, stroke='#9CAAB6', stroke_width=0.28)
    canvas.line(x0, y0, x1, y0, stroke='#9CAAB6', stroke_width=0.28)
    for index, colour in enumerate(MBP_COLORS.values()):
        top = y1 + 0.4 + index * 0.55
        mid = y1 + (y0 - y1) * (0.3 + index * 0.12)
        low = y1 + (y0 - y1) * (0.48 + index * 0.16)
        points = [(x0, top), (x0 + (x1 - x0) * 0.28, top), (x0 + (x1 - x0) * 0.28, mid), (x0 + (x1 - x0) * 0.65, mid), (x0 + (x1 - x0) * 0.65, low), (x1, low)]
        canvas.path('M ' + ' L '.join((f'{fmt(x)} {fmt(y)}' for x, y in points)), stroke=colour, stroke_width=0.46)

def draw_response_survival(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.6, region.y + 3.1, region.w - 1.2, region.h - 4.0)
    matrix = Region(art.x, art.y + 1.3, art.w * 0.27, art.h - 2.1)
    canvas.text(matrix.cx, art.y + 0.5, 'mRECIST', size_pt=FONT_PT['micro'], fill=RESPONSE, weight='bold')
    cell_gap = 0.42
    cell_w = (matrix.w - 2 * cell_gap) / 3.0
    cell_h = (matrix.h - cell_gap) / 2.0
    for row in range(2):
        for col, colour in enumerate(MBP_COLORS.values()):
            shade = 0.5 + 0.16 * ((row + col) % 3)
            canvas.rect(matrix.x + col * (cell_w + cell_gap), matrix.y + row * (cell_h + cell_gap), cell_w, cell_h, fill=mix(colour, fraction=shade), stroke=WHITE, stroke_width=0.22, rx=0.22)
    plot_gap = 0.9
    remaining_x = matrix.right + 1.4
    remaining_w = art.right - remaining_x
    pfs_region = Region(remaining_x, art.y, (remaining_w - plot_gap) / 2.0, art.h)
    os_region = Region(pfs_region.right + plot_gap, art.y, pfs_region.w, art.h)
    draw_km_family(canvas, pfs_region, 'PFS')
    draw_km_family(canvas, os_region, 'OS')

def draw_adjusted_associations(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.8, region.y + 3.2, region.w - 1.6, region.h - 4.1)
    labels = [('mRECIST', RESPONSE), ('PFS', PFS), ('OS', OS)]
    plot_x0 = art.x + 8.0
    plot_x1 = art.right - 0.7
    positions = [(0.52, 0.24, 0.8), (0.52, 0.3, 0.74), (0.52, 0.2, 0.84)]
    for index, ((label, colour), (point, low, high)) in enumerate(zip(labels, positions)):
        cy = art.y + 2.0 + index * (art.h - 4.0) / 2.0
        canvas.text(art.x + 0.2, cy, label, size_pt=FONT_PT['small'], fill=colour, anchor='start', weight='bold')
        x_low = plot_x0 + low * (plot_x1 - plot_x0)
        x_high = plot_x0 + high * (plot_x1 - plot_x0)
        x_point = plot_x0 + point * (plot_x1 - plot_x0)
        canvas.line(x_low, cy, x_high, cy, stroke=colour, stroke_width=0.62)
        canvas.line(x_low, cy - 0.55, x_low, cy + 0.55, stroke=colour, stroke_width=0.4)
        canvas.line(x_high, cy - 0.55, x_high, cy + 0.55, stroke=colour, stroke_width=0.4)
        canvas.circle(x_point, cy, 0.66, fill=colour, stroke=WHITE, stroke_width=0.22)

def draw_incremental_context(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.7, region.y + 3.2, region.w - 1.4, region.h - 4.0)
    bar_area = Region(art.x, art.y + 0.7, art.w * 0.42, art.h * 0.55)
    baseline = bar_area.bottom
    endpoints = [(PFS, 0.68), (OS, 0.78)]
    for index, (colour, fraction) in enumerate(endpoints):
        x = bar_area.x + 1.6 + index * (bar_area.w - 4.2) / 2.0
        canvas.rect(x, baseline - bar_area.h * (fraction - 0.16), 1.05, bar_area.h * (fraction - 0.16), fill=mix(CLINICAL, fraction=0.55), stroke='none', rx=0.18)
        canvas.rect(x + 1.35, baseline - bar_area.h * fraction, 1.05, bar_area.h * fraction, fill=colour, stroke='none', rx=0.18)
    canvas.text(bar_area.cx, bar_area.bottom + 0.85, 'clinical  |  + MBP', size_pt=FONT_PT['micro'], fill=MUTED)
    curve = Region(art.x + art.w * 0.5, art.y + 0.7, art.w * 0.48, art.h * 0.55)
    trajectory = [(curve.x + 0.7, curve.bottom - 0.9), (curve.x + curve.w * 0.23, curve.bottom - 1.5), (curve.x + curve.w * 0.42, curve.cy + 0.3), (curve.x + curve.w * 0.61, curve.y + 2.0), (curve.x + curve.w * 0.79, curve.y + 1.1), (curve.right - 0.5, curve.y + 0.7)]
    canvas.path('M ' + ' L '.join((f'{fmt(x)} {fmt(y)}' for x, y in trajectory)), stroke='#A8B4BF', stroke_width=0.42)
    trajectory_colours = [MBP_COLORS['MBP-I'], MBP_COLORS['MBP-I'], MBP_COLORS['MBP-II'], MBP_COLORS['MBP-II'], MBP_COLORS['MBP-III'], MBP_COLORS['MBP-III']]
    for (x, y), colour in zip(trajectory, trajectory_colours):
        canvas.circle(x, y, 0.56, fill=colour, stroke=WHITE, stroke_width=0.18)
    canvas.text(curve.cx, curve.bottom + 0.85, 'continuous score', size_pt=FONT_PT['micro'], fill=MUTED)
    chip_y = art.bottom - 1.4
    chip_specs = [('Tumour', RESPONSE), ('Liver', RAD), ('Inflamm.', META)]
    chip_w = (art.w - 1.2) / 3.0
    for index, (label, colour) in enumerate(chip_specs):
        x = art.x + index * (chip_w + 0.6)
        canvas.rect(x, chip_y - 0.75, chip_w, 1.5, fill=mix(colour, fraction=0.9), stroke=colour, stroke_width=0.26, rx=0.6)
        canvas.circle(x + 0.8, chip_y, 0.28, fill=colour)
        canvas.text(x + 1.35, chip_y, label, size_pt=FONT_PT['micro'], fill=colour, anchor='start')

def draw_stage_4(canvas: SvgCanvas, contract_stage: Mapping[str, Any]) -> None:
    content = stage_shell(canvas, STAGES[4])
    modules = content.split_x(STAGES[4].fractions, gap=1.95)
    labels = ['MRU benefit score\nto MBP', 'mRECIST, PFS and OS', contract_stage['modules'][2], 'Incremental information\nand clinical context']
    module_guides(canvas, modules, labels)
    draw_mbp_score(canvas, modules[0])
    draw_response_survival(canvas, modules[1])
    draw_adjusted_associations(canvas, modules[2])
    draw_incremental_context(canvas, modules[3])
    for left, right in zip(modules[:-1], modules[1:]):
        process_arrow(canvas, left, right, y=content.cy + 0.9)

def draw_determinants(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.8, region.y + 4.0, region.w - 1.6, region.h - 4.9)
    top = Region(art.x, art.y, art.w, art.h * 0.4)
    colours = [META, RAD, URINE]
    endpoint_info = [('Resp.', RESPONSE), ('PFS', PFS), ('OS', OS)]
    endpoint_bin_w = top.w / 3.0
    for index, (endpoint, endpoint_colour) in enumerate(endpoint_info):
        x = top.x + index * endpoint_bin_w + (endpoint_bin_w - 2.4) / 2.0
        canvas.text(x + 1.2, top.y + 0.8, endpoint, size_pt=FONT_PT['micro'], fill=endpoint_colour, weight='bold')
        base_y = top.bottom - 0.4
        segment_heights = [2.0 + 0.35 * ((index + j) % 3) for j in range(3)]
        cursor = base_y
        for colour, height in zip(colours, segment_heights):
            cursor -= height
            canvas.rect(x, cursor, 2.4, height, fill=colour, stroke=WHITE, stroke_width=0.18)
    bottom = Region(art.x, top.bottom + 0.6, art.w, art.bottom - top.bottom - 0.6)
    swarm = Region(bottom.x, bottom.y, bottom.w * 0.58, bottom.h)
    canvas.text(swarm.x + 0.2, swarm.y + 0.65, 'Patient-level SHAP', size_pt=FONT_PT['micro'], fill=MUTED, anchor='start', weight='bold')
    centre_x = swarm.x + swarm.w * 0.55
    canvas.line(centre_x, swarm.y + 1.7, centre_x, swarm.bottom - 0.5, stroke='#A9B5C0', stroke_width=0.3, dash='0.7 0.7')
    swarm_colours = [META, RAD, URINE, META, RAD]
    for row, colour in enumerate(swarm_colours):
        cy = swarm.y + 2.2 + row * (swarm.h - 3.0) / 4.0
        canvas.line(swarm.x + 0.3, cy, swarm.right - 0.3, cy, stroke='#EEF1F4', stroke_width=0.26)
        offsets = [-3.1, -1.7, -0.7, 0.5, 1.4, 2.8]
        for index, offset in enumerate(offsets):
            canvas.circle(centre_x + offset, cy + (index % 2 - 0.5) * 0.45, 0.4, fill=mix(colour, fraction=0.15 + index * 0.08), stroke=WHITE, stroke_width=0.14)
    simplex = Region(swarm.right + 0.8, bottom.y, bottom.right - swarm.right - 0.8, bottom.h)
    canvas.text(simplex.cx, simplex.y + 0.65, 'Shared / preferred', size_pt=FONT_PT['micro'], fill=MUTED, weight='bold')
    vertices = [(simplex.cx, simplex.y + 2.2), (simplex.x + 1.0, simplex.bottom - 0.8), (simplex.right - 1.0, simplex.bottom - 0.8)]
    canvas.polygon(vertices, fill=mix(STAGE_BLUE, fraction=0.96), stroke='#8D9BA8', stroke_width=0.34)
    for (x, y), label, colour in zip(vertices, ('M', 'R', 'U'), (META, RAD, URINE)):
        canvas.circle(x, y, 0.65, fill=colour, stroke=WHITE, stroke_width=0.18)
        canvas.text(x, y, label, size_pt=FONT_PT['micro'], fill=WHITE, weight='bold')
    points = [(0.48, 0.35), (0.36, 0.55), (0.6, 0.58), (0.49, 0.72), (0.67, 0.44)]
    for index, (px, py) in enumerate(points):
        x = simplex.x + 1.0 + px * (simplex.w - 2.0)
        y = simplex.y + 2.2 + py * (simplex.h - 3.0)
        canvas.circle(x, y, 0.36, fill=(RESPONSE, PFS, OS)[index % 3], stroke=WHITE, stroke_width=0.12)

def draw_ecosystem_axes(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.5, region.y + 3.0, region.w - 1.0, region.h - 3.8)
    label_w = 16.0
    x_positions = [art.x + label_w + 4.0, art.x + label_w + 21.0, art.x + label_w + 39.0, art.right - 5.0]
    headings = ['KO / functions', 'Metabolites', 'Radiomic phenotypes', 'MBP / outcomes']
    for x, heading in zip(x_positions, headings):
        canvas.text(x, art.y + 0.75, heading, size_pt=FONT_PT['micro'], fill=MUTED, weight='bold')
    lane_top = art.y + 3.6
    lane_gap = 0.65
    lane_h = (art.h - 4.1 - lane_gap * 2.0) / 3.0
    lane_specs = [('Amino acids\n& TME', '#2A8F78', RESPONSE), ('Lipid &\n1-carbon', '#617FC4', PFS), ('SCFA &\nBCAA', '#B45A9A', OS)]
    for lane_index, (axis_label, axis_colour, endpoint_colour) in enumerate(lane_specs):
        lane = Region(art.x, lane_top + lane_index * (lane_h + lane_gap), art.w, lane_h)
        canvas.rect(lane.x, lane.y, lane.w, lane.h, fill=mix(axis_colour, fraction=0.95), stroke='none', stroke_width=0.0, rx=0.55)
        canvas.text(lane.x + label_w / 2.0, lane.cy, axis_label, size_pt=FONT_PT['small'], fill=axis_colour, weight='bold')
        canvas.line(lane.x + label_w, lane.y + 0.5, lane.x + label_w, lane.bottom - 0.5, stroke=mix(axis_colour, fraction=0.68), stroke_width=0.3)
        ko_nodes = [(x_positions[0] - 1.6, lane.cy - 1.2), (x_positions[0] + 1.2, lane.cy + 1.1)]
        metabolite_nodes = [(x_positions[1] - 1.7, lane.cy - 1.25), (x_positions[1] + 1.2, lane.cy), (x_positions[1] - 0.3, lane.cy + 1.45)]
        imaging_nodes = [(x_positions[2] - 1.5, lane.cy - 0.8), (x_positions[2] + 1.4, lane.cy + 0.9)]
        outcome_nodes = [(x_positions[3], lane.cy - 1.0), (x_positions[3], lane.cy + 1.1)]
        for index, (x, y) in enumerate(ko_nodes):
            canvas.polygon([(x, y - 0.65), (x - 0.58, y - 0.3), (x - 0.58, y + 0.3), (x, y + 0.65), (x + 0.58, y + 0.3), (x + 0.58, y - 0.3)], fill=mix(META, fraction=0.55 + index * 0.12), stroke=META, stroke_width=0.24)
        metabolite_colours = [axis_colour, URINE, BLOOD]
        for index, ((x, y), colour) in enumerate(zip(metabolite_nodes, metabolite_colours)):
            canvas.circle(x, y, 0.62, fill=mix(colour, fraction=0.62), stroke=colour, stroke_width=0.28)
        for index, (x, y) in enumerate(imaging_nodes):
            canvas.rect(x - 0.8, y - 0.7, 1.6, 1.4, fill=mix(RAD, fraction=0.72 + index * 0.1), stroke=RAD, stroke_width=0.28, rx=0.28)
            canvas.circle(x + 0.25, y - 0.1, 0.23, fill=RESPONSE, stroke='none')
        for index, (x, y) in enumerate(outcome_nodes):
            colour = list(MBP_COLORS.values())[(lane_index + index) % 3]
            canvas.circle(x, y, 0.72, fill=mix(colour, fraction=0.55), stroke=colour, stroke_width=0.3)
            if index == 1:
                canvas.circle(x + 1.45, y, 0.42, fill=endpoint_colour, stroke=WHITE, stroke_width=0.12)
        links = [(ko_nodes[0], metabolite_nodes[0], axis_colour, 0.72), (ko_nodes[0], metabolite_nodes[1], META, 0.46), (ko_nodes[1], metabolite_nodes[1], axis_colour, 0.62), (ko_nodes[1], metabolite_nodes[2], URINE, 0.48), (metabolite_nodes[0], imaging_nodes[0], axis_colour, 0.72), (metabolite_nodes[1], imaging_nodes[1], BLOOD, 0.52), (metabolite_nodes[2], imaging_nodes[0], URINE, 0.46), (imaging_nodes[0], outcome_nodes[0], RAD, 0.62), (imaging_nodes[1], outcome_nodes[1], endpoint_colour, 0.58)]
        for index, (start, end, colour, width) in enumerate(links):
            canvas.curve(start, end, stroke=colour, stroke_width=width, opacity=0.42, bend=(-1) ** index * 0.45)

def draw_benefit_avatar(canvas: SvgCanvas, region: Region) -> None:
    art = Region(region.x + 0.7, region.y + 3.0, region.w - 1.4, region.h - 3.8)
    canvas.text(art.cx, art.y + 0.8, 'Patient-level Benefit Avatar', size_pt=FONT_PT['small'], fill=STAGE_BLUE, weight='bold')
    patient_cx = art.x + art.w * 0.34
    patient_cy = art.y + art.h * 0.52
    aura_r = min(art.w * 0.25, art.h * 0.36)
    canvas.circle(patient_cx, patient_cy, aura_r, fill=mix(STAGE_BLUE, fraction=0.97), stroke='#CAD7E4', stroke_width=0.38)

    def arc_path(radius: float, start_deg: float, end_deg: float, colour: str) -> None:
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        x0 = patient_cx + radius * math.cos(start)
        y0 = patient_cy + radius * math.sin(start)
        x1 = patient_cx + radius * math.cos(end)
        y1 = patient_cy + radius * math.sin(end)
        large = 1 if abs(end_deg - start_deg) > 180 else 0
        canvas.path(f'M {fmt(x0)} {fmt(y0)} A {fmt(radius)} {fmt(radius)} 0 {large} 1 {fmt(x1)} {fmt(y1)}', stroke=colour, stroke_width=0.95)
    arc_path(aura_r + 1.1, -88, 22, MBP_COLORS['MBP-I'])
    arc_path(aura_r + 1.1, 30, 140, MBP_COLORS['MBP-II'])
    arc_path(aura_r + 1.1, 148, 258, MBP_COLORS['MBP-III'])
    body_box = Region(patient_cx - 4.2, patient_cy - 7.0, 8.4, 14.0)
    canvas.icon('body', body_box, colour=CLINICAL)
    canvas.icon('liver', Region(patient_cx - 1.9, patient_cy - 1.7, 5.0, 4.0), colour=RESPONSE)
    canvas.icon('intestine', Region(patient_cx - 1.4, patient_cy + 1.7, 4.0, 4.0), colour=META)
    right_x = art.x + art.w * 0.63
    canvas.text(right_x, art.y + 3.2, 'MBP', size_pt=FONT_PT['micro'], fill=MUTED, anchor='start', weight='bold')
    for index, (label, colour) in enumerate(MBP_COLORS.items()):
        cx = right_x + 7.3 + index * 4.1
        canvas.circle(cx, art.y + 3.2, 0.85, fill=mix(colour, fraction=0.68), stroke=colour, stroke_width=0.3)
        canvas.text(cx, art.y + 3.2, label.replace('MBP-', ''), size_pt=FONT_PT['micro'], fill=colour, weight='bold')
    endpoint_y0 = art.y + 7.1
    for index, (label, colour, shape) in enumerate((('mRECIST resp.', RESPONSE, 'circle'), ('PFS', PFS, 'square'), ('OS', OS, 'triangle'))):
        cy = endpoint_y0 + index * 3.0
        if shape == 'circle':
            canvas.circle(right_x + 0.9, cy, 0.66, fill=colour)
        elif shape == 'square':
            canvas.rect(right_x + 0.25, cy - 0.65, 1.3, 1.3, fill=colour, rx=0.16)
        else:
            canvas.polygon([(right_x + 0.9, cy - 0.76), (right_x + 0.18, cy + 0.58), (right_x + 1.62, cy + 0.58)], fill=colour)
        canvas.text(right_x + 2.3, cy, label, size_pt=FONT_PT['small'], fill=colour, anchor='start', weight='bold')
        canvas.curve((patient_cx + aura_r * 0.75, patient_cy), (right_x - 0.2, cy), stroke=colour, stroke_width=0.32, opacity=0.34, bend=(cy - patient_cy) * 0.22)
    determinant_y = art.bottom - 6.8
    canvas.text(right_x, determinant_y, 'Determinants', size_pt=FONT_PT['micro'], fill=MUTED, anchor='start', weight='bold')
    determinant_nodes = [(right_x + 1.2, determinant_y + 2.0, META), (right_x + 4.0, determinant_y + 2.7, RAD), (right_x + 6.8, determinant_y + 1.9, URINE)]
    for index, (x, y, colour) in enumerate(determinant_nodes):
        if index:
            canvas.line(determinant_nodes[index - 1][0], determinant_nodes[index - 1][1], x, y, stroke='#A8B4BF', stroke_width=0.32)
        canvas.circle(x, y, 0.62, fill=mix(colour, fraction=0.48), stroke=colour, stroke_width=0.25)
    axis_y = art.bottom - 2.65
    axis_specs = [META, '#617FC4', '#B45A9A']
    canvas.text(right_x, axis_y - 1.0, 'Ecosystem context', size_pt=FONT_PT['micro'], fill=MUTED, anchor='start', weight='bold')
    for index, colour in enumerate(axis_specs):
        x = right_x + 2.8 + index * 5.8
        canvas.path(f'M {fmt(x - 2.3)} {fmt(axis_y)} C {fmt(x - 1.2)} {fmt(axis_y - 0.65)}, {fmt(x + 0.9)} {fmt(axis_y + 0.65)}, {fmt(x + 2.2)} {fmt(axis_y)}', stroke=colour, stroke_width=0.48)

def draw_stage_5(canvas: SvgCanvas, contract_stage: Mapping[str, Any]) -> None:
    content = stage_shell(canvas, STAGES[5])
    modules = content.split_x(STAGES[5].fractions, gap=2.1)
    labels = ['Cross-endpoint\ndeterminants', contract_stage['modules'][1], contract_stage['modules'][2]]
    module_guides(canvas, modules, labels)
    draw_determinants(canvas, modules[0])
    draw_ecosystem_axes(canvas, modules[1])
    draw_benefit_avatar(canvas, modules[2])
    process_arrow(canvas, modules[0], modules[1], y=content.cy + 1.0)
    canvas.line(modules[1].right + 0.25, content.cy + 1.0, modules[2].x - 0.25, content.cy + 1.0, stroke=ARROW, stroke_width=0.48, dash='1.0 0.8', marker_end='arrow-grey')

def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if contract.get('version') != VERSION:
        raise RuntimeError(f"Expected contract {VERSION}, got {contract.get('version')!r}")
    if contract.get('palette') != PALETTE_ID:
        raise RuntimeError('Figure 1B palette contract changed unexpectedly')
    if contract.get('render_only') is not True or contract.get('all_miniatures_are_schematic') is not True:
        raise RuntimeError('Figure 1B must remain an explicitly schematic render')
    stages = contract.get('stages', [])
    if [int(item['number']) for item in stages] != [1, 2, 3, 4, 5]:
        raise RuntimeError('Figure 1B must contain exactly five ordered stages')
    if [str(item['title']) for item in stages] != [STAGES[index].title for index in range(1, 6)]:
        raise RuntimeError('Stage titles do not match the approved v002 layout')
    expected_counts = [3, 4, 5, 4, 3]
    actual_counts = [len(item.get('modules', [])) for item in stages]
    if actual_counts != expected_counts:
        raise RuntimeError(f'Unexpected module counts: {actual_counts}')
    display_strings = [str(item['title']) for item in stages]
    display_strings.extend((str(module) for item in stages for module in item['modules']))
    display_blob = '\n'.join(display_strings).lower()
    for forbidden in contract.get('forbidden_display_terms', []):
        if str(forbidden).lower() in display_blob:
            raise RuntimeError(f'Forbidden display term found in contract content: {forbidden}')
    return contract

def validate_assets() -> dict[str, Any]:
    provenance = json.loads(ASSET_PROVENANCE_PATH.read_text(encoding='utf-8'))
    if provenance.get('license') != 'CC0 1.0 Universal':
        raise RuntimeError('Unexpected external asset license')
    if not (HEALTH_ROOT / 'LICENSE').exists():
        raise FileNotFoundError('Health Icons license file is missing')
    assets: dict[str, Any] = {}
    for name, path in ICON_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing required Health Icon {name}: {path}')
        assets[name] = {'path': str(path), 'sha256': sha256_file(path), 'license': 'CC0-1.0'}
    return {'source': provenance['source_collection'], 'official_repository': provenance['official_repository'], 'license': provenance['license'], 'license_sha256': sha256_file(HEALTH_ROOT / 'LICENSE'), 'assets': assets}

def build_panel_b(contract: Mapping[str, Any]) -> SvgCanvas:
    canvas = SvgCanvas(B_WIDTH_MM, B_HEIGHT_MM)
    canvas.text(0.8, 1.2, 'b', size_pt=FONT_PT['panel'], fill=TEXT, anchor='start', weight='bold', qa_name='panel-b')
    stages = {int(item['number']): item for item in contract['stages']}
    draw_stage_1(canvas, stages[1])
    draw_stage_2(canvas, stages[2])
    draw_stage_3(canvas, stages[3])
    draw_stage_4(canvas, stages[4])
    draw_stage_5(canvas, stages[5])
    return canvas

def expected_pixels(width_mm: float, height_mm: float, dpi: int=EXPORT_DPI) -> tuple[int, int]:
    return (round(width_mm / MM_PER_INCH * dpi), round(height_mm / MM_PER_INCH * dpi))

def convert_svg_bundle(stem: Path) -> None:
    import cairosvg
    from PIL import Image
    svg_path = stem.with_suffix('.svg')
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(stem.with_suffix('.pdf')), dpi=EXPORT_DPI)
    width_px, height_px = expected_pixels(B_WIDTH_MM, B_HEIGHT_MM)
    png_path = stem.with_suffix('.png')
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width_px, output_height=height_px)
    with Image.open(png_path) as rendered:
        rendered.save(png_path, dpi=(EXPORT_DPI, EXPORT_DPI))
URL_ID_PATTERN = re.compile('url\\(#([^)]+)\\)')

def prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    mapping: dict[str, str] = {}
    for node in root.iter():
        old_id = node.attrib.get('id')
        if old_id:
            mapping[old_id] = f'{prefix}-{old_id}'
    for node in root.iter():
        old_id = node.attrib.get('id')
        if old_id:
            node.set('id', mapping[old_id])
        for key, value in list(node.attrib.items()):
            updated = URL_ID_PATTERN.sub(lambda match: f'url(#{mapping.get(match.group(1), match.group(1))})', value)
            if updated.startswith('#') and updated[1:] in mapping:
                updated = f'#{mapping[updated[1:]]}'
            node.set(key, updated)
        if node.text and 'url(#' in node.text:
            node.text = URL_ID_PATTERN.sub(lambda match: f'url(#{mapping.get(match.group(1), match.group(1))})', node.text)

def require_panel_a() -> None:
    missing = [str(A_STEM.with_suffix(f'.{suffix}')) for suffix in ('svg', 'pdf', 'png') if not A_STEM.with_suffix(f'.{suffix}').exists()]
    if missing:
        raise RuntimeError(f'Approved Figure 1A outputs are missing: {missing}')

def compose_svg() -> None:
    outer = ET.Element(qname('svg'), {'width': f'{fmt(FULL_WIDTH_MM)}mm', 'height': f'{fmt(FULL_HEIGHT_MM)}mm', 'viewBox': f'0 0 {fmt(FULL_WIDTH_MM)} {fmt(FULL_HEIGHT_MM)}', 'version': '1.1'})
    ET.SubElement(outer, qname('rect'), {'x': '0', 'y': '0', 'width': fmt(FULL_WIDTH_MM), 'height': fmt(FULL_HEIGHT_MM), 'fill': WHITE})
    specifications = [(A_STEM.with_suffix('.svg'), 'panel-a', 0.0, A_HEIGHT_MM), (B_STEM.with_suffix('.svg'), 'panel-b', A_HEIGHT_MM + PANEL_GAP_MM, B_HEIGHT_MM)]
    for path, prefix, y_position, target_height in specifications:
        source_root = ET.parse(path).getroot()
        prefix_svg_ids(source_root, prefix)
        nested = ET.SubElement(outer, qname('svg'), {'x': '0', 'y': fmt(y_position), 'width': fmt(FULL_WIDTH_MM), 'height': fmt(target_height), 'viewBox': source_root.attrib['viewBox'], 'preserveAspectRatio': 'xMinYMin meet'})
        for child in source_root:
            nested.append(deepcopy(child))
    ET.indent(outer, space='  ')
    ET.ElementTree(outer).write(FULL_STEM.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)

def compose_pdf() -> None:
    enable_bundled_pdf_runtime()
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        from PyPDF2 import PdfReader, PdfWriter
    reader_a = PdfReader(str(A_STEM.with_suffix('.pdf')))
    reader_b = PdfReader(str(B_STEM.with_suffix('.pdf')))
    if len(reader_a.pages) != 1 or len(reader_b.pages) != 1:
        raise RuntimeError('Figure 1 panels must each contain exactly one PDF page')
    width_pt = FULL_WIDTH_MM / MM_PER_INCH * POINTS_PER_INCH
    height_pt = FULL_HEIGHT_MM / MM_PER_INCH * POINTS_PER_INCH
    a_height_pt = A_HEIGHT_MM / MM_PER_INCH * POINTS_PER_INCH
    writer = PdfWriter()
    page = writer.add_blank_page(width=width_pt, height=height_pt)
    page.merge_translated_page(reader_a.pages[0], 0.0, height_pt - a_height_pt, expand=False)
    page.merge_translated_page(reader_b.pages[0], 0.0, 0.0, expand=False)
    writer.add_metadata({'/Title': 'Avatar-HCC Figure 1', '/Subject': 'Multicentre cohorts and the Avatar-HCC multimodal framework'})
    with FULL_STEM.with_suffix('.pdf').open('wb') as stream:
        writer.write(stream)

def compose_png() -> None:
    from PIL import Image
    full_width_px, full_height_px = expected_pixels(FULL_WIDTH_MM, FULL_HEIGHT_MM)
    _, a_height_px = expected_pixels(FULL_WIDTH_MM, A_HEIGHT_MM)
    _, b_height_px = expected_pixels(FULL_WIDTH_MM, B_HEIGHT_MM)
    canvas = Image.new('RGB', (full_width_px, full_height_px), 'white')
    with Image.open(A_STEM.with_suffix('.png')) as source:
        panel_a = source.convert('RGB')
        if panel_a.size != (full_width_px, a_height_px):
            panel_a = panel_a.resize((full_width_px, a_height_px), Image.Resampling.LANCZOS)
    with Image.open(B_STEM.with_suffix('.png')) as source:
        panel_b = source.convert('RGB')
        if panel_b.size != (full_width_px, b_height_px):
            panel_b = panel_b.resize((full_width_px, b_height_px), Image.Resampling.LANCZOS)
    canvas.paste(panel_a, (0, 0))
    canvas.paste(panel_b, (0, full_height_px - b_height_px))
    canvas.save(FULL_STEM.with_suffix('.png'), dpi=(EXPORT_DPI, EXPORT_DPI))

def compose_full_figure() -> None:
    require_panel_a()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compose_svg()
    compose_pdf()
    compose_png()

def svg_report(path: Path, *, expected_width: float, expected_height: float, required_terms: Iterable[str], forbidden_terms: Iterable[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    text_nodes = list(root.iter(qname('text')))
    visible_text = '\n'.join((''.join(node.itertext()) for node in text_nodes))
    lower_text = visible_text.lower()
    required = {term: term.lower() in lower_text for term in required_terms}
    forbidden = {term: term.lower() in lower_text for term in forbidden_terms}
    image_nodes = list(root.iter(qname('image')))
    external_asset_groups = [node for node in root.iter(qname('g')) if node.attrib.get('data-source', '').startswith('Health Icons:')]
    font_sizes_pt: list[float] = []
    non_arial: list[str] = []
    for node in text_nodes:
        if 'font-size' in node.attrib:
            font_sizes_pt.append(float(node.attrib['font-size']) / MM_PER_INCH * POINTS_PER_INCH)
        family = node.attrib.get('font-family', '')
        if family and 'Arial' not in family:
            non_arial.append(family)
    width_match = re.fullmatch('([0-9.]+)mm', root.attrib.get('width', ''))
    height_match = re.fullmatch('([0-9.]+)mm', root.attrib.get('height', ''))
    actual_width = float(width_match.group(1)) if width_match else None
    actual_height = float(height_match.group(1)) if height_match else None
    metric_patterns = {'p_or_q_value': bool(re.search('\\b[pq]\\s*[<=>]', visible_text, re.IGNORECASE)), 'effect_estimate': bool(re.search('\\b(?:HR|OR)\\b', visible_text)), 'performance_metric': bool(re.search('\\b(?:AUC|AUROC|C-index)\\b', visible_text, re.IGNORECASE)), 'percentage': '%' in visible_text}
    passed = actual_width == expected_width and actual_height == expected_height and all(required.values()) and (not any(forbidden.values())) and (not any(metric_patterns.values())) and (not image_nodes) and (not non_arial) and bool(font_sizes_pt) and (min(font_sizes_pt) >= 4.35)
    return {'path': str(path), 'sha256': sha256_file(path), 'declared_size_mm': [actual_width, actual_height], 'expected_size_mm': [expected_width, expected_height], 'text_node_count': len(text_nodes), 'minimum_font_pt': min(font_sizes_pt) if font_sizes_pt else None, 'non_arial_font_families': sorted(set(non_arial)), 'embedded_image_count': len(image_nodes), 'external_cc0_asset_instances': len(external_asset_groups), 'required_terms': required, 'forbidden_terms_present': forbidden, 'metric_like_text_present': metric_patterns, 'pass': passed}

def pdf_report(path: Path, *, expected_width: float, expected_height: float) -> dict[str, Any]:
    enable_bundled_pdf_runtime()
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(str(path))
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / POINTS_PER_INCH * MM_PER_INCH
    height_mm = float(page.mediabox.height) / POINTS_PER_INCH * MM_PER_INCH
    return {'path': str(path), 'sha256': sha256_file(path), 'pages': len(reader.pages), 'size_mm': [width_mm, height_mm], 'pass': len(reader.pages) == 1 and abs(width_mm - expected_width) < 0.15 and (abs(height_mm - expected_height) < 0.15)}

def png_report(path: Path, *, expected_width: float, expected_height: float) -> dict[str, Any]:
    from PIL import Image
    expected = expected_pixels(expected_width, expected_height)
    with Image.open(path) as image:
        actual = image.size
        dpi = image.info.get('dpi')
    dpi_pass = dpi is not None and all((abs(float(value) - EXPORT_DPI) < 1.0 for value in dpi[:2]))
    return {'path': str(path), 'sha256': sha256_file(path), 'pixels': list(actual), 'expected_pixels': list(expected), 'dpi_metadata': list(dpi) if dpi else None, 'dpi_pass': dpi_pass, 'pass': actual == expected and dpi_pass}

def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def build_reports(contract: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, Any]:
    required_terms = ['Pretreatment', 'Metagenomics', 'Radiomics', 'Blood', 'Urine', 'MRU', 'Response', 'PFS', 'OS', 'MBP-I', 'MBP-II', 'MBP-III', 'Amino', 'Lipid', 'SCFA']
    forbidden_terms = list(contract.get('forbidden_display_terms', []))
    panel_svg = svg_report(B_STEM.with_suffix('.svg'), expected_width=B_WIDTH_MM, expected_height=B_HEIGHT_MM, required_terms=required_terms, forbidden_terms=forbidden_terms)
    full_svg = svg_report(FULL_STEM.with_suffix('.svg'), expected_width=FULL_WIDTH_MM, expected_height=FULL_HEIGHT_MM, required_terms=required_terms, forbidden_terms=forbidden_terms)
    panel_report = {'svg': panel_svg, 'pdf': pdf_report(B_STEM.with_suffix('.pdf'), expected_width=B_WIDTH_MM, expected_height=B_HEIGHT_MM), 'png': png_report(B_STEM.with_suffix('.png'), expected_width=B_WIDTH_MM, expected_height=B_HEIGHT_MM)}
    full_report = {'svg': full_svg, 'pdf': pdf_report(FULL_STEM.with_suffix('.pdf'), expected_width=FULL_WIDTH_MM, expected_height=FULL_HEIGHT_MM), 'png': png_report(FULL_STEM.with_suffix('.png'), expected_width=FULL_WIDTH_MM, expected_height=FULL_HEIGHT_MM)}
    write_json(panel_report, OUTPUT_DIR / f"{get_value('figures.config.item_340')}{VERSION}.json")
    write_json(full_report, OUTPUT_DIR / f"{get_value('figures.config.item_261')}{VERSION}.json")
    scientific_report = {'figure': 'Figure 1B', 'version': VERSION, 'palette': PALETTE_ID, 'render_only': True, 'schematic_only': True, 'displayed_numeric_results': 0, 'stage_count': len(contract['stages']), 'module_counts': [len(item['modules']) for item in contract['stages']], 'external_evidence_constraint': contract['external_evidence_constraint'], 'avatar_constraint': contract['avatar_constraint'], 'forbidden_terms_pass': not any(panel_svg['forbidden_terms_present'].values()), 'metric_text_pass': not any(panel_svg['metric_like_text_present'].values()), 'no_raster_embedding_pass': panel_svg['embedded_image_count'] == 0, 'contract_sha256': sha256_file(CONTRACT_PATH), 'asset_provenance_sha256': sha256_file(ASSET_PROVENANCE_PATH), 'pass': panel_svg['pass']}
    write_json(scientific_report, OUTPUT_DIR / f"{get_value('figures.config.item_262')}{VERSION}.json")
    write_json(assets, OUTPUT_DIR / f"{get_value('figures.config.item_263')}{VERSION}.json")
    all_pass = all((item['pass'] for item in panel_report.values())) and all((item['pass'] for item in full_report.values())) and scientific_report['pass']
    summary = {'figure': 'Figure 1', 'version': VERSION, 'panel_a_source_version': PANEL_A_VERSION, 'palette': PALETTE_ID, 'render_only': True, 'panel_b_size_mm': [B_WIDTH_MM, B_HEIGHT_MM], 'full_size_mm': [FULL_WIDTH_MM, FULL_HEIGHT_MM], 'panel_b_stem': str(B_STEM), 'full_stem': str(FULL_STEM), 'cc0_asset_source': assets['source'], 'qa_pass': all_pass}
    write_json(summary, OUTPUT_DIR / f"{get_value('figures.config.item_264')}{VERSION}.json")
    if not all_pass:
        raise RuntimeError('Figure 1 v002 QA failed; inspect the generated JSON reports')
    return summary

def main() -> dict[str, Any]:
    contract = load_contract()
    assets = validate_assets()
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    canvas = build_panel_b(contract)
    canvas.write(B_STEM.with_suffix('.svg'))
    convert_svg_bundle(B_STEM)
    compose_full_figure()
    summary = build_reports(contract, assets)
    return summary
if __name__ == '__main__':
    main()
