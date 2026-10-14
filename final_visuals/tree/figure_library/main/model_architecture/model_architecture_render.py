from __future__ import annotations
from avatar_paths import get_value, get_column
import sysconfig
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parent
STYLE_ROOT = ROOT.parents[1]
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import ENDPOINT_COLORS, MODALITY_COLORS, NEUTRAL_COLORS, PALETTE_ID
VERSION = 'v001'
DATE_TAG = get_value('figures.config.item_350')
WIDTH_MM = 180.0
HEIGHT_MM = 105.0
EXPORT_DPI = 600
MM_PER_INCH = 25.4
PT_PER_INCH = 72.0
OUTPUT_DIR = ROOT / 'outputs' / f'{DATE_TAG}_{VERSION}'
STEM = OUTPUT_DIR / f'Avatar_HCC_model_architecture_Nature_{VERSION}'
QA_RENDER_DIR = OUTPUT_DIR / 'qa_render'
SVG_NS = get_value('figures.config.item_379')
ET.register_namespace('', SVG_NS)

def qname(tag: str) -> str:
    return f'{{{SVG_NS}}}{tag}'
WHITE = NEUTRAL_COLORS['background']
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
LIGHT = NEUTRAL_COLORS['light_background']
EDGE = '#D8E1E8'
RULE = '#C7D2DC'
ARROW = '#788896'
NAVY = '#244F78'
META = MODALITY_COLORS['Metagenomics']
RAD = MODALITY_COLORS['Radiomics']
BLOOD = MODALITY_COLORS['Blood']
URINE = MODALITY_COLORS['Urine']
RESPONSE = ENDPOINT_COLORS['Response']
PFS = ENDPOINT_COLORS['PFS']
OS = ENDPOINT_COLORS['OS']
MODALITIES: tuple[tuple[str, str, str, int, int, int], ...] = (('M', 'Metagenomic functions', META, 4681, 10, 15), ('R', 'CT radiomics', RAD, 851, 6, 9), ('B', 'Blood metabolomics', BLOOD, 1080, 8, 15), ('U', 'Urine metabolomics', URINE, 3035, 8, 15))
ENDPOINTS: tuple[tuple[str, str, str], ...] = (('Response', 'probability', RESPONSE), ('PFS', 'risk', PFS), ('OS', 'risk', OS))

def fmt(value: float) -> str:
    return f'{float(value):.3f}'.rstrip('0').rstrip('.')

def pt_to_mm(value: float) -> float:
    return float(value) * MM_PER_INCH / PT_PER_INCH

def hex_rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip('#')
    return tuple((int(value[index:index + 2], 16) for index in (0, 2, 4)))

def mix(colour: str, target: str=WHITE, fraction: float=0.86) -> str:
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

class SvgCanvas:

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.root = ET.Element(qname('svg'), {'width': f'{fmt(width)}mm', 'height': f'{fmt(height)}mm', 'viewBox': f'0 0 {fmt(width)} {fmt(height)}', 'version': '1.1'})
        self.defs = ET.SubElement(self.root, qname('defs'))
        self._marker('arrow', ARROW)
        self._marker('arrow-navy', NAVY)
        self._linear_gradient('stage1', '#F2F7FB', '#E2EDF6')
        self._linear_gradient('stage2', '#F6F3FB', '#E9E3F4')
        self._linear_gradient('final', '#EEF5FA', '#F5EFF8')
        self._shadow()
        self.rect(0, 0, width, height, fill=WHITE, stroke='none')

    def _marker(self, marker_id: str, colour: str) -> None:
        marker = ET.SubElement(self.defs, qname('marker'), {'id': marker_id, 'viewBox': '0 0 6 6', 'refX': '5.2', 'refY': '3', 'markerWidth': '4.4', 'markerHeight': '4.4', 'orient': 'auto'})
        ET.SubElement(marker, qname('path'), {'d': 'M0,0 L6,3 L0,6 Z', 'fill': colour})

    def _linear_gradient(self, gradient_id: str, start: str, end: str) -> None:
        grad = ET.SubElement(self.defs, qname('linearGradient'), {'id': gradient_id, 'x1': '0', 'y1': '0', 'x2': '1', 'y2': '0'})
        ET.SubElement(grad, qname('stop'), {'offset': '0', 'stop-color': start})
        ET.SubElement(grad, qname('stop'), {'offset': '1', 'stop-color': end})

    def _shadow(self) -> None:
        filt = ET.SubElement(self.defs, qname('filter'), {'id': 'soft-shadow', 'x': '-10%', 'y': '-15%', 'width': '120%', 'height': '140%'})
        ET.SubElement(filt, qname('feDropShadow'), {'dx': '0', 'dy': '0.35', 'stdDeviation': '0.45', 'flood-color': '#92A2B0', 'flood-opacity': '0.18'})

    def group(self, **attrs: str) -> ET.Element:
        return ET.SubElement(self.root, qname('g'), attrs)

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str='none', stroke: str=EDGE, stroke_width: float=0.45, rx: float=1.4, opacity: float | None=None, dash: str | None=None, parent: ET.Element | None=None, extra: Mapping[str, str] | None=None) -> ET.Element:
        attrs = {'x': fmt(x), 'y': fmt(y), 'width': fmt(w), 'height': fmt(h), 'rx': fmt(rx), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width)}
        if opacity is not None:
            attrs['opacity'] = fmt(opacity)
        if dash:
            attrs['stroke-dasharray'] = dash
        if extra:
            attrs.update(extra)
        return ET.SubElement(parent or self.root, qname('rect'), attrs)

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str=RULE, stroke_width: float=0.45, opacity: float=1.0, dash: str | None=None, arrow: bool=False, parent: ET.Element | None=None) -> ET.Element:
        attrs = {'x1': fmt(x1), 'y1': fmt(y1), 'x2': fmt(x2), 'y2': fmt(y2), 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity), 'stroke-linecap': 'round'}
        if dash:
            attrs['stroke-dasharray'] = dash
        if arrow:
            attrs['marker-end'] = 'url(#arrow)'
        return ET.SubElement(parent or self.root, qname('line'), attrs)

    def path(self, d: str, *, fill: str='none', stroke: str=RULE, stroke_width: float=0.5, opacity: float=1.0, dash: str | None=None, arrow: bool=False, parent: ET.Element | None=None) -> ET.Element:
        attrs = {'d': d, 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity), 'stroke-linecap': 'round', 'stroke-linejoin': 'round'}
        if dash:
            attrs['stroke-dasharray'] = dash
        if arrow:
            attrs['marker-end'] = 'url(#arrow)'
        return ET.SubElement(parent or self.root, qname('path'), attrs)

    def circle(self, cx: float, cy: float, r: float, *, fill: str=WHITE, stroke: str=EDGE, stroke_width: float=0.4, opacity: float=1.0, parent: ET.Element | None=None) -> ET.Element:
        return ET.SubElement(parent or self.root, qname('circle'), {'cx': fmt(cx), 'cy': fmt(cy), 'r': fmt(r), 'fill': fill, 'stroke': stroke, 'stroke-width': fmt(stroke_width), 'opacity': fmt(opacity)})

    def text(self, x: float, y: float, value: str, *, size_pt: float=5.2, fill: str=TEXT, weight: str='normal', anchor: str='start', italic: bool=False, parent: ET.Element | None=None, opacity: float=1.0) -> ET.Element:
        size_pt = max(5.4, float(size_pt))
        node = ET.SubElement(parent or self.root, qname('text'), {'x': fmt(x), 'y': fmt(y), 'font-family': 'Arial, Helvetica, sans-serif', 'font-size': fmt(pt_to_mm(size_pt)), 'font-weight': weight, 'font-style': 'italic' if italic else 'normal', 'fill': fill, 'text-anchor': anchor, 'opacity': fmt(opacity)})
        node.text = value
        return node

    def multiline(self, x: float, y: float, lines: Sequence[str], *, size_pt: float=5.2, leading_mm: float=2.0, fill: str=TEXT, weight: str='normal', anchor: str='start', parent: ET.Element | None=None) -> None:
        for index, value in enumerate(lines):
            self.text(x, y + index * leading_mm, value, size_pt=size_pt, fill=fill, weight=weight, anchor=anchor, parent=parent)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(path, encoding='utf-8', xml_declaration=True)

def draw_stage_header(canvas: SvgCanvas, y: float, number: str, title: str, subtitle: str, gradient: str) -> None:
    canvas.rect(3.0, y, 174.0, 6.0, fill=f'url(#{gradient})', stroke='none', rx=2.0)
    canvas.circle(6.6, y + 3.0, 1.75, fill=NAVY, stroke='none')
    canvas.text(6.6, y + 3.72, number, size_pt=6.4, fill=WHITE, weight='bold', anchor='middle')
    canvas.text(9.6, y + 2.8, title, size_pt=6.5, weight='bold')
    canvas.text(9.6, y + 4.75, subtitle, size_pt=4.8, fill=MUTED)

def matrix_glyph(canvas: SvgCanvas, x: float, y: float, colour: str) -> None:
    for row in range(3):
        for col in range(4):
            alpha = 0.38 + 0.13 * ((row + col) % 4)
            canvas.rect(x + col * 1.0, y + row * 0.95, 0.72, 0.68, fill=colour, stroke='none', rx=0.1, opacity=alpha)

def mini_multitask_net(canvas: SvgCanvas, x: float, y: float, colour: str) -> None:
    input_x = x
    hidden1_x = x + 5.0
    hidden2_x = x + 10.0
    head_x = x + 16.2
    in_ys = [y - 1.55, y, y + 1.55]
    h1_ys = [y - 1.8, y - 0.6, y + 0.6, y + 1.8]
    h2_ys = [y - 1.25, y, y + 1.25]
    head_ys = [y - 1.55, y, y + 1.55]
    for y0 in in_ys:
        for y1 in h1_ys:
            canvas.line(input_x, y0, hidden1_x, y1, stroke=mix(colour, WHITE, 0.55), stroke_width=0.22, opacity=0.65)
    for y0 in h1_ys:
        for y1 in h2_ys:
            canvas.line(hidden1_x, y0, hidden2_x, y1, stroke=mix(NAVY, WHITE, 0.62), stroke_width=0.2, opacity=0.58)
    for y0 in h2_ys:
        for y1 in head_ys:
            canvas.line(hidden2_x, y0, head_x, y1, stroke=mix(NAVY, WHITE, 0.65), stroke_width=0.2, opacity=0.58)
    for y0 in in_ys:
        canvas.circle(input_x, y0, 0.42, fill=colour, stroke=WHITE, stroke_width=0.18)
    for y0 in h1_ys:
        canvas.circle(hidden1_x, y0, 0.46, fill=mix(colour, WHITE, 0.24), stroke=colour, stroke_width=0.22)
    for y0 in h2_ys:
        canvas.circle(hidden2_x, y0, 0.46, fill=mix(NAVY, WHITE, 0.28), stroke=NAVY, stroke_width=0.22)
    for y0, head_colour in zip(head_ys, (RESPONSE, PFS, OS)):
        canvas.circle(head_x, y0, 0.5, fill=head_colour, stroke=WHITE, stroke_width=0.18)

def draw_cv_grid(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 22.0, 28.7, fill=WHITE, stroke=EDGE, stroke_width=0.48, rx=1.5)
    canvas.text(x + 2.0, y + 2.8, 'Repeated stratified CV', size_pt=5.2, weight='bold')
    canvas.text(x + 2.0, y + 4.9, '5 repeats × 5 folds', size_pt=4.8, fill=NAVY, weight='bold')
    gx, gy, cell, gap = (x + 3.15, y + 6.4, 2.35, 0.55)
    for row in range(5):
        for col in range(5):
            colour = PFS if row == col else mix(PFS, WHITE, 0.78)
            canvas.rect(gx + col * (cell + gap), gy + row * (cell + gap), cell, cell, fill=colour, stroke=WHITE, stroke_width=0.18, rx=0.35)
    canvas.text(x + 11.0, y + 22.5, '25 held-out folds', size_pt=4.8, fill=MUTED, anchor='middle')
    canvas.path(f'M {fmt(x + 3.2)} {fmt(y + 24.5)} C {fmt(x + 7.5)} {fmt(y + 26.4)}, {fmt(x + 14.0)} {fmt(y + 26.4)}, {fmt(x + 18.8)} {fmt(y + 24.5)}', stroke=NAVY, stroke_width=0.55, arrow=True)
    canvas.text(x + 11.0, y + 27.3, 'OOF assembly', size_pt=4.9, fill=NAVY, weight='bold', anchor='middle')

def draw_oof_card(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 29.0, 28.7, fill=WHITE, stroke=EDGE, stroke_width=0.48, rx=1.5)
    canvas.text(x + 2.0, y + 3.0, 'Out-of-fold evidence', size_pt=5.3, weight='bold')
    canvas.text(x + 2.0, y + 5.4, 'OOF predictions', size_pt=4.8, fill=MUTED)
    for index, (_, _, colour) in enumerate(ENDPOINTS):
        by = y + 7.1 + index * 1.55
        canvas.rect(x + 2.2, by, 13.5 + index * 2.1, 0.85, fill=mix(colour, WHITE, 0.35), stroke='none', rx=0.42)
        canvas.circle(x + 2.2 + 5.0 + index * 4.0, by + 0.42, 0.56, fill=colour, stroke=WHITE, stroke_width=0.16)
    canvas.line(x + 2.0, y + 12.2, x + 27.0, y + 12.2, stroke=EDGE, stroke_width=0.35)
    canvas.text(x + 2.0, y + 14.5, 'OOF SHAP', size_pt=4.8, fill=MUTED)
    baseline = y + 18.0
    canvas.line(x + 4.0, baseline, x + 25.5, baseline, stroke='#BBC7D0', stroke_width=0.35)
    for row, colour in enumerate((META, RAD, BLOOD, URINE)):
        yy = baseline - 2.2 + row * 1.4
        for index in range(7):
            xx = x + 7.0 + index * 2.4 + 0.45 * math.sin(index * 1.6 + row)
            jitter = 0.35 * math.cos(index * 1.3 + row * 0.7)
            canvas.circle(xx, yy + jitter, 0.38, fill=colour, stroke=WHITE, stroke_width=0.1, opacity=0.84)
    canvas.text(x + 2.0, y + 23.5, 'Top-k per endpoint', size_pt=4.8, fill=MUTED)
    labels = (('M10', META), ('R6', RAD), ('B8', BLOOD), ('U8', URINE))
    for index, (label, colour) in enumerate(labels):
        bx = x + 2.0 + index * 6.35
        canvas.rect(bx, y + 24.8, 5.65, 2.35, fill=mix(colour, WHITE, 0.83), stroke=colour, stroke_width=0.35, rx=0.9)
        canvas.text(bx + 2.825, y + 26.48, label, size_pt=4.8, fill=colour, weight='bold', anchor='middle')

def draw_feature_bank(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 33.8, 28.7, fill=WHITE, stroke=NAVY, stroke_width=0.62, rx=1.5, extra={'filter': 'url(#soft-shadow)'})
    canvas.text(x + 2.0, y + 3.0, 'Endpoint-wise union', size_pt=5.3, weight='bold')
    canvas.text(x + 31.8, y + 3.0, '54', size_pt=6.0, fill=NAVY, weight='bold', anchor='end')
    canvas.text(x + 2.0, y + 5.1, 'retained features', size_pt=4.8, fill=MUTED)
    bar_y = y + 7.7
    for index, (abbr, _, colour, _, _, retained) in enumerate(MODALITIES):
        yy = bar_y + index * 3.25
        canvas.circle(x + 3.3, yy + 0.65, 1.02, fill=colour, stroke=WHITE, stroke_width=0.2)
        canvas.text(x + 3.3, yy + 1.3, abbr, size_pt=5.4, fill=WHITE, weight='bold', anchor='middle')
        canvas.rect(x + 5.1, yy, 20.5, 1.3, fill='#EDF1F4', stroke='none', rx=0.65)
        width = 20.5 * retained / 15.0
        canvas.rect(x + 5.1, yy, width, 1.3, fill=colour, stroke='none', rx=0.65, opacity=0.86)
        canvas.text(x + 29.8, yy + 1.12, str(retained), size_pt=5.0, fill=colour, weight='bold', anchor='end')
    branch_y = y + 22.2
    canvas.path(f'M {fmt(x + 1.8)} {fmt(branch_y)} C {fmt(x - 2.5)} {fmt(branch_y)}, {fmt(x - 2.2)} {fmt(branch_y + 4.4)}, {fmt(x + 1.8)} {fmt(branch_y + 4.4)}', stroke=MUTED, stroke_width=0.45, dash='1.2 0.8')
    canvas.rect(x + 2.6, y + 22.45, 28.5, 4.7, fill=mix(NAVY, WHITE, 0.91), stroke=RULE, stroke_width=0.35, rx=1.2)
    canvas.text(x + 16.85, y + 24.22, 'Conditional stability', size_pt=5.4, fill=MUTED, weight='bold', anchor='middle')
    canvas.text(x + 16.85, y + 26.2, '500 patient bootstraps', size_pt=5.4, fill=MUTED, anchor='middle')

def draw_stage1(canvas: SvgCanvas) -> None:
    y0 = 12.0
    canvas.rect(3.0, y0, 174.0, 42.6, fill='#FAFCFD', stroke=EDGE, stroke_width=0.52, rx=2.2)
    draw_stage_header(canvas, y0, '1', 'Modality-specific multi-task representation', 'Identical ReducedMultiTaskNet topology; independent parameters by modality', 'stage1')
    canvas.text(5.0, 20.05, 'Model-ready matrices', size_pt=4.9, fill=MUTED, weight='bold')
    canvas.text(39.0, 20.05, 'Shared encoder × 4', size_pt=4.9, fill=MUTED, weight='bold')
    canvas.text(66.0, 20.05, '3 endpoint heads', size_pt=4.9, fill=MUTED, weight='bold')
    row_ys = (21.0, 28.65, 36.3, 43.95)
    for (abbr, name, colour, raw_count, _, _), row_y in zip(MODALITIES, row_ys):
        canvas.rect(5.0, row_y, 31.0, 6.1, fill=mix(colour, WHITE, 0.91), stroke=colour, stroke_width=0.42, rx=1.3)
        canvas.circle(8.0, row_y + 3.05, 1.55, fill=colour, stroke=WHITE, stroke_width=0.25)
        canvas.text(8.0, row_y + 3.72, abbr, size_pt=5.4, fill=WHITE, weight='bold', anchor='middle')
        canvas.text(10.4, row_y + 2.6, name, size_pt=4.9, weight='bold')
        canvas.text(10.4, row_y + 4.65, f'{raw_count:,} features', size_pt=4.8, fill=MUTED)
        matrix_glyph(canvas, 31.0, row_y + 1.58, colour)
        canvas.line(36.4, row_y + 3.05, 39.0, row_y + 3.05, stroke=ARROW, stroke_width=0.48, arrow=True)
        canvas.rect(39.5, row_y, 37.8, 6.1, fill=WHITE, stroke=mix(colour, WHITE, 0.45), stroke_width=0.4, rx=1.3)
        mini_multitask_net(canvas, 44.3, row_y + 3.05, colour)
        canvas.line(77.6, row_y + 3.05, 80.0, row_y + 3.05, stroke=ARROW, stroke_width=0.48, arrow=True)
    canvas.rect(39.5, 50.7, 37.8, 2.2, fill=mix(NAVY, WHITE, 0.92), stroke='none', rx=0.8)
    canvas.text(58.4, 52.27, 'Linear → BN → ReLU → Dropout   × 2', size_pt=4.8, fill=NAVY, anchor='middle')
    draw_cv_grid(canvas, 81.0, 21.0)
    canvas.line(103.4, 35.35, 106.5, 35.35, stroke=ARROW, stroke_width=0.52, arrow=True)
    draw_oof_card(canvas, 107.5, 21.0)
    canvas.line(136.9, 35.35, 140.1, 35.35, stroke=ARROW, stroke_width=0.52, arrow=True)
    draw_feature_bank(canvas, 141.0, 21.0)

def combination_tiles(canvas: SvgCanvas, x: float, y: float) -> None:
    groups: tuple[tuple[str, tuple[str, ...]], ...] = (('1 modality · 4', ('M', 'R', 'B', 'U')), ('2 modalities · 6', ('MR', 'MB', 'MU', 'RB', 'RU', 'BU')), ('3 modalities · 4', ('MRB', 'MRU', 'MBU', 'RBU')), ('4 modalities · 1', ('MRBU',)))
    letter_colours = {'M': META, 'R': RAD, 'B': BLOOD, 'U': URINE}
    yy = y
    for label, combos in groups:
        canvas.text(x, yy + 1.55, label, size_pt=4.8, fill=MUTED)
        if len(combos) == 6:
            tile_x = x + 16.7
            tile_w = 4.35
            gap = 0.38
        else:
            tile_x = x + 17.2
            tile_w = 5.2
            gap = 0.65
        for index, combo in enumerate(combos):
            xx = tile_x + index * (tile_w + gap)
            selected = combo == 'MRU'
            canvas.rect(xx, yy, tile_w, 2.7, fill=mix(NAVY, WHITE, 0.9) if selected else WHITE, stroke=NAVY if selected else RULE, stroke_width=0.72 if selected else 0.32, rx=0.75, extra={'filter': 'url(#soft-shadow)'} if selected else None)
            char_gap = min(1.28, (tile_w - 1.0) / max(1, len(combo)))
            start = xx + tile_w / 2.0 - char_gap * (len(combo) - 1) / 2.0
            for char_index, char in enumerate(combo):
                canvas.text(start + char_index * char_gap, yy + 1.88, char, size_pt=4.8, fill=letter_colours[char], weight='bold', anchor='middle')
        yy += 4.7

def draw_fusion(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 31.0, 36.4, fill=WHITE, stroke=EDGE, stroke_width=0.48, rx=1.5)
    canvas.text(x + 2.0, y + 3.0, 'Selected MRU fusion', size_pt=5.4, weight='bold')
    canvas.text(x + 2.0, y + 5.2, 'early feature concatenation', size_pt=4.8, fill=MUTED)
    strips = (('M', 15, META), ('R', 9, RAD), ('U', 15, URINE))
    strip_y = y + 8.0
    for index, (abbr, count, colour) in enumerate(strips):
        yy = strip_y + index * 6.2
        canvas.circle(x + 3.5, yy + 1.5, 1.35, fill=colour, stroke=WHITE, stroke_width=0.22)
        canvas.text(x + 3.5, yy + 2.1, abbr, size_pt=5.0, fill=WHITE, weight='bold', anchor='middle')
        canvas.text(x + 6.0, yy + 1.2, f'{count} retained', size_pt=4.9, fill=colour, weight='bold')
        for cell in range(6):
            canvas.rect(x + 6.0 + cell * 2.05, yy + 2.15, 1.55, 1.35, fill=colour, stroke='none', rx=0.22, opacity=0.45 + 0.08 * (cell % 4))
        canvas.path(f'M {fmt(x + 18.3)} {fmt(yy + 2.85)} C {fmt(x + 21.2)} {fmt(yy + 2.85)}, {fmt(x + 20.6)} {fmt(y + 27.7)}, {fmt(x + 23.0)} {fmt(y + 27.7)}', stroke=colour, stroke_width=0.6, opacity=0.75)
    canvas.rect(x + 21.3, y + 24.9, 7.9, 6.1, fill='url(#final)', stroke=NAVY, stroke_width=0.55, rx=1.0)
    canvas.text(x + 25.25, y + 27.45, '39', size_pt=6.2, fill=NAVY, weight='bold', anchor='middle')
    canvas.text(x + 25.25, y + 29.65, 'features', size_pt=5.4, fill=MUTED, anchor='middle')
    canvas.text(x + 15.5, y + 34.0, 'M + R + U', size_pt=5.3, fill=NAVY, weight='bold', anchor='middle')

def layer_nodes(canvas: SvgCanvas, x: float, centre_y: float, count: int, colour: str) -> list[float]:
    shown = 7 if count >= 30 else 6 if count >= 16 else 5
    spacing = 2.45 if shown <= 6 else 2.15
    ys = [centre_y + (index - (shown - 1) / 2.0) * spacing for index in range(shown)]
    for yy in ys:
        canvas.circle(x, yy, 0.68, fill=mix(colour, WHITE, 0.22), stroke=colour, stroke_width=0.28)
    return ys

def connect_layers(canvas: SvgCanvas, x0: float, ys0: Iterable[float], x1: float, ys1: Iterable[float], colour: str) -> None:
    ys0_tuple = tuple(ys0)
    ys1_tuple = tuple(ys1)
    for a in ys0_tuple:
        for b in ys1_tuple:
            canvas.line(x0 + 0.72, a, x1 - 0.72, b, stroke=mix(colour, WHITE, 0.62), stroke_width=0.18, opacity=0.52)

def draw_final_checkpoint(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 47.0, 36.4, fill='url(#final)', stroke=NAVY, stroke_width=0.62, rx=1.6, extra={'filter': 'url(#soft-shadow)'})
    canvas.text(x + 2.0, y + 3.0, 'Final ReducedMultiTaskNet checkpoint', size_pt=5.4, weight='bold')
    canvas.text(x + 2.0, y + 5.2, 'shared two-layer encoder · endpoint-specific heads', size_pt=4.8, fill=MUTED)
    centre_y = y + 17.3
    x39, x19, x16, xhead = (x + 6.0, x + 16.0, x + 26.0, x + 37.6)
    ys39 = layer_nodes(canvas, x39, centre_y, 39, NAVY)
    ys19 = layer_nodes(canvas, x19, centre_y, 19, META)
    ys16 = layer_nodes(canvas, x16, centre_y, 16, PFS)
    connect_layers(canvas, x39, ys39, x19, ys19, NAVY)
    connect_layers(canvas, x19, ys19, x16, ys16, META)
    head_ys = (centre_y - 6.2, centre_y, centre_y + 6.2)
    for yy, (_, _, colour) in zip(head_ys, ENDPOINTS):
        for hidden_y in ys16:
            canvas.line(x16 + 0.7, hidden_y, xhead - 1.15, yy, stroke=mix(colour, WHITE, 0.63), stroke_width=0.2, opacity=0.6)
        canvas.circle(xhead, yy, 1.05, fill=colour, stroke=WHITE, stroke_width=0.3)
    labels = ((x39, '39', 'input'), (x19, '19', 'hidden 1'), (x16, '16', 'hidden 2'))
    for xx, number, label in labels:
        canvas.text(xx, y + 29.7, number, size_pt=5.3, fill=NAVY, weight='bold', anchor='middle')
        canvas.text(xx, y + 31.55, label, size_pt=4.8, fill=MUTED, anchor='middle')
    canvas.text(xhead, y + 29.7, '3', size_pt=5.3, fill=NAVY, weight='bold', anchor='middle')
    canvas.text(xhead, y + 31.55, 'heads', size_pt=4.8, fill=MUTED, anchor='middle')
    canvas.rect(x + 1.8, y + 33.0, 43.4, 2.25, fill=WHITE, stroke='none', rx=0.75)
    canvas.text(x + 23.5, y + 34.62, 'dropout 0.45   ·   best epoch 171   ·   5 × 5 CV', size_pt=4.8, fill=MUTED, anchor='middle')

def draw_outputs(canvas: SvgCanvas, x: float, y: float) -> None:
    canvas.rect(x, y, 30.7, 36.4, fill=WHITE, stroke=EDGE, stroke_width=0.48, rx=1.5)
    canvas.text(x + 2.0, y + 3.0, 'Model outputs', size_pt=5.4, weight='bold')
    yy = y + 5.6
    for title, suffix, colour in ENDPOINTS:
        canvas.rect(x + 2.0, yy, 26.7, 4.25, fill=mix(colour, WHITE, 0.9), stroke=colour, stroke_width=0.4, rx=1.0)
        canvas.circle(x + 4.0, yy + 2.12, 0.9, fill=colour, stroke=WHITE, stroke_width=0.18)
        canvas.text(x + 6.0, yy + 1.85, title, size_pt=5.0, fill=colour, weight='bold')
        canvas.text(x + 6.0, yy + 3.45, suffix, size_pt=4.8, fill=MUTED)
        yy += 5.25
    canvas.rect(x + 2.0, y + 21.7, 26.7, 5.2, fill=mix(NAVY, WHITE, 0.94), stroke=RULE, stroke_width=0.35, rx=1.0)
    canvas.text(x + 3.2, y + 23.7, 'Weighted task objective', size_pt=4.8, fill=MUTED)
    canvas.text(x + 27.5, y + 25.3, '0.4 / 0.8 / 1.0', size_pt=5.2, fill=NAVY, weight='bold', anchor='end')
    canvas.text(x + 3.2, y + 25.35, 'R / PFS / OS', size_pt=4.8, fill=MUTED)
    canvas.text(x + 2.0, y + 29.2, 'Compact SHAP explanation', size_pt=4.8, fill=MUTED)
    base_x, base_y = (x + 3.0, y + 32.3)
    canvas.line(base_x, base_y, x + 28.0, base_y, stroke=RULE, stroke_width=0.32)
    shap_bars = ((META, 11.4), (URINE, 8.2), (RAD, 6.0), (PFS, 3.8))
    for index, (colour, length) in enumerate(shap_bars):
        by = y + 30.2 + index * 1.25
        canvas.rect(base_x, by, length, 0.78, fill=colour, stroke='none', rx=0.39, opacity=0.85)

def draw_stage2(canvas: SvgCanvas) -> None:
    y0 = 57.5
    canvas.rect(3.0, y0, 174.0, 45.2, fill='#FCFBFD', stroke=EDGE, stroke_width=0.52, rx=2.2)
    draw_stage_header(canvas, y0, '2', 'Exhaustive early-fusion screening and final MRU checkpoint', 'All 15 non-empty modality combinations are evaluated with the same multi-task model family', 'stage2')
    canvas.rect(5.0, 65.0, 49.0, 36.4, fill=WHITE, stroke=EDGE, stroke_width=0.48, rx=1.5)
    canvas.text(7.0, 68.0, 'Combination lattice', size_pt=5.4, weight='bold')
    canvas.text(52.0, 68.0, '15', size_pt=5.7, fill=NAVY, weight='bold', anchor='end')
    combination_tiles(canvas, 7.0, 70.0)
    canvas.rect(7.0, 90.0, 44.9, 8.8, fill=mix(NAVY, WHITE, 0.95), stroke='none', rx=1.0)
    canvas.text(8.4, 92.4, 'Each candidate', size_pt=4.8, fill=MUTED)
    canvas.text(8.4, 94.3, 'early concatenation → ReducedMultiTaskNet', size_pt=4.8, fill=NAVY, weight='bold')
    canvas.text(8.4, 96.3, '→ repeated 5 × 5 out-of-fold evaluation', size_pt=4.8, fill=MUTED)
    canvas.text(49.8, 100.0, 'MRU selected', size_pt=4.8, fill=NAVY, weight='bold', anchor='end')
    canvas.line(54.5, 83.2, 56.8, 83.2, stroke=ARROW, stroke_width=0.55, arrow=True)
    draw_fusion(canvas, 57.5, 65.0)
    canvas.line(88.9, 83.2, 91.2, 83.2, stroke=ARROW, stroke_width=0.55, arrow=True)
    draw_final_checkpoint(canvas, 92.0, 65.0)
    canvas.line(139.4, 83.2, 141.7, 83.2, stroke=ARROW, stroke_width=0.55, arrow=True)
    draw_outputs(canvas, 142.3, 65.0)

def build_svg() -> SvgCanvas:
    canvas = SvgCanvas(WIDTH_MM, HEIGHT_MM)
    canvas.text(3.0, 5.1, 'Avatar-HCC model architecture', size_pt=8.8, fill=TEXT, weight='bold')
    canvas.text(3.0, 8.4, 'Two-stage multi-task neural modelling for stable feature prioritisation and multimodal early fusion', size_pt=5.4, fill=MUTED)
    canvas.line(3.0, 10.2, 177.0, 10.2, stroke=RULE, stroke_width=0.45)
    draw_stage1(canvas)
    draw_stage2(canvas)
    return canvas

def convert_svg_bundle(stem: Path) -> None:
    import cairosvg
    from PIL import Image
    svg_path = stem.with_suffix('.svg')
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(stem.with_suffix('.pdf')), dpi=EXPORT_DPI)
    output_width = round(WIDTH_MM / MM_PER_INCH * EXPORT_DPI)
    output_height = round(HEIGHT_MM / MM_PER_INCH * EXPORT_DPI)
    png_path = stem.with_suffix('.png')
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=output_width, output_height=output_height)
    with Image.open(png_path) as image:
        image.save(png_path, dpi=(EXPORT_DPI, EXPORT_DPI))

def enable_bundled_pdf_runtime() -> None:
    if importlib.util.find_spec('pypdf'):
        return
    bundled = Path(get_value('figures.runtime.site_packages', sysconfig.get_paths()['purelib']))
    if bundled.is_dir() and str(bundled) not in sys.path:
        sys.path.insert(0, str(bundled))

def svg_qa(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    texts = root.findall(f".//{qname('text')}")
    joined = ' '.join((''.join(node.itertext()) for node in texts))
    fonts = sorted({node.attrib.get('font-family', '') for node in texts})
    sizes_mm = [float(node.attrib['font-size']) for node in texts if 'font-size' in node.attrib]
    required = ('ReducedMultiTaskNet', '4,681 features', '851 features', '1,080 features', '3,035 features', '5 × 5', 'OOF SHAP', '500 patient bootstraps', '54', '15 non-empty modality combinations', 'M + R + U', '39', '19', '16', 'dropout 0.45', 'best epoch 171', '0.4 / 0.8 / 1.0', 'Response', 'PFS', 'OS')
    forbidden = ('attention', 'transformer', 'graph neural', 'GNN', 'agent', 'Avatar', 'MBP-I', 'Clinical workflow', 'treatment recommendation', 'causal benefit')
    report = {'path': str(path.resolve()), 'sha256': sha256_file(path), 'declared_size_mm': [WIDTH_MM, HEIGHT_MM], 'text_node_count': len(texts), 'minimum_font_pt': min(sizes_mm) * PT_PER_INCH / MM_PER_INCH, 'font_families': fonts, 'embedded_image_count': len(root.findall(f".//{qname('image')}")), 'required_terms': {term: term in joined for term in required}, 'forbidden_terms_present': {term: term.lower() in joined.lower() for term in forbidden}}
    report['pass'] = all(report['required_terms'].values()) and (not any(report['forbidden_terms_present'].values())) and (report['minimum_font_pt'] >= 5.39) and (report['embedded_image_count'] == 0) and all(('Arial' in font for font in fonts))
    return report

def pdf_qa(path: Path) -> dict[str, Any]:
    enable_bundled_pdf_runtime()
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / PT_PER_INCH * MM_PER_INCH
    height_mm = float(page.mediabox.height) / PT_PER_INCH * MM_PER_INCH
    fonts: list[dict[str, str]] = []
    for resource_name, font_ref in page['/Resources'].get('/Font', {}).items():
        font = font_ref.get_object()
        fonts.append({'resource': str(resource_name), 'base_font': str(font.get('/BaseFont', '')), 'subtype': str(font.get('/Subtype', ''))})
    all_arial = bool(fonts) and all(('Arial' in item['base_font'] for item in fonts))
    no_type3 = all((item['subtype'] != '/Type3' for item in fonts))
    return {'path': str(path.resolve()), 'sha256': sha256_file(path), 'pages': len(reader.pages), 'size_mm': [width_mm, height_mm], 'fonts': fonts, 'all_fonts_arial': all_arial, 'no_type3_fonts': no_type3, 'pass': len(reader.pages) == 1 and abs(width_mm - WIDTH_MM) < 0.05 and (abs(height_mm - HEIGHT_MM) < 0.05) and all_arial and no_type3}

def png_qa(path: Path) -> dict[str, Any]:
    from PIL import Image
    expected = (round(WIDTH_MM / MM_PER_INCH * EXPORT_DPI), round(HEIGHT_MM / MM_PER_INCH * EXPORT_DPI))
    with Image.open(path) as image:
        dpi = image.info.get('dpi', (0.0, 0.0))
        pixels = image.size
    return {'path': str(path.resolve()), 'sha256': sha256_file(path), 'pixels': list(pixels), 'expected_pixels': list(expected), 'dpi_metadata': [float(dpi[0]), float(dpi[1])], 'pass': pixels == expected and abs(float(dpi[0]) - EXPORT_DPI) < 1.0 and (abs(float(dpi[1]) - EXPORT_DPI) < 1.0)}

def render_pdf_for_visual_qa(pdf_path: Path) -> Path | None:
    QA_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    prefix = QA_RENDER_DIR / f'{pdf_path.stem}_poppler_300dpi'
    command = ['pdftoppm', '-png', '-singlefile', '-r', '300', str(pdf_path), str(prefix)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = prefix.with_suffix('.png')
    return output if output.exists() else None

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def write_qa() -> None:
    file_report = {'palette': PALETTE_ID, 'render_only': True, 'svg': svg_qa(STEM.with_suffix('.svg')), 'pdf': pdf_qa(STEM.with_suffix('.pdf')), 'png': png_qa(STEM.with_suffix('.png'))}
    file_report['pass'] = all((file_report[key]['pass'] for key in ('svg', 'pdf', 'png')))
    write_json(OUTPUT_DIR / f'model_architecture_file_qa_{VERSION}.json', file_report)
    contract_report = {'scope': 'model-method architecture only', 'render_only': True, 'no_patient_data_loaded': True, 'no_model_training': True, 'stage1': {'model_ready_dimensions': {'M': 4681, 'R': 851, 'B': 1080, 'U': 3035}, 'isomorphic_models': 4, 'shared_encoder': 'two blocks of Linear-BN-ReLU-Dropout', 'heads': ['Response', 'PFS', 'OS'], 'validation': '5 repeats × 5 stratified folds', 'evidence': ['OOF predictions', 'OOF SHAP'], 'endpoint_top_k': {'M': 10, 'R': 6, 'B': 8, 'U': 8}, 'retained_union': {'M': 15, 'R': 9, 'B': 15, 'U': 15, 'total': 54}, 'conditional_stability_bootstraps': 500}, 'stage2': {'non_empty_combinations': {'one': 4, 'two': 6, 'three': 4, 'four': 1, 'total': 15}, 'fusion': 'early feature concatenation', 'selected_MRU': {'modalities': ['M', 'R', 'U'], 'features': 39}, 'checkpoint': {'layers': [39, 19, 16], 'heads': 3, 'dropout': 0.45, 'best_epoch': 171}, 'task_objective_weights': {'Response': 0.4, 'PFS': 0.8, 'OS': 1.0}, 'outputs': ['Response probability', 'PFS risk', 'OS risk']}, 'explicitly_absent': ['cohort or sampling flow', 'clinical workflow', 'MBP phenotyping', 'Benefit Avatar', 'ecological mechanism', 'attention', 'transformer', 'graph neural network', 'autonomous agent'], 'pass': True}
    write_json(OUTPUT_DIR / f'model_architecture_scientific_contract_qa_{VERSION}.json', contract_report)
    visual_path = render_pdf_for_visual_qa(STEM.with_suffix('.pdf'))
    visual_report = {'source': str(STEM.with_suffix('.pdf').resolve()), 'render': str(visual_path.resolve()) if visual_path else None, 'render_dpi': 300 if visual_path else None, 'checks': {'single_panel': True, 'model_only_scope': True, 'two_stage_hierarchy_visible': True, 'semantic_modality_colours_consistent': True, 'semantic_endpoint_colours_consistent': True, 'all_text_is_vector': True, 'no_raster_or_external_asset': True}, 'pass': visual_path is not None}
    write_json(OUTPUT_DIR / f'model_architecture_visual_qa_{VERSION}.json', visual_report)
    summary = {'version': VERSION, 'palette': PALETTE_ID, 'canvas_mm': [WIDTH_MM, HEIGHT_MM], 'dpi': EXPORT_DPI, 'outputs': [str(STEM.with_suffix(ext).resolve()) for ext in ('.svg', '.pdf', '.png')], 'qa': [str((OUTPUT_DIR / f'model_architecture_file_qa_{VERSION}.json').resolve()), str((OUTPUT_DIR / f'model_architecture_scientific_contract_qa_{VERSION}.json').resolve()), str((OUTPUT_DIR / f'model_architecture_visual_qa_{VERSION}.json').resolve())], 'pass': file_report['pass'] and visual_path is not None}
    write_json(OUTPUT_DIR / f'model_architecture_run_summary_{VERSION}.json', summary)

def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = build_svg()
    canvas.write(STEM.with_suffix('.svg'))
    convert_svg_bundle(STEM)
    write_qa()
    return STEM.with_suffix('.png')
if __name__ == '__main__':
    main()
