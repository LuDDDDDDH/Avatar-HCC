from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
try:
    import matplotlib as mpl
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    _MATPLOTLIB_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    mpl = None
    Legend = None
    Text = None
    Bbox = None
    _MATPLOTLIB_IMPORT_ERROR = exc
MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0

@dataclass
class ArtistRegistry:
    artists: list[Any] = field(default_factory=list)

    def tag(self, artist: Any, *, name: str, role: str, panel: str) -> Any:
        gid = f'qa:{panel}:{name}'
        if hasattr(artist, 'set_gid'):
            artist.set_gid(gid)
        artist._qa_meta = {'name': str(name), 'role': str(role), 'panel': str(panel).lower()}
        self.artists.append(artist)
        return artist

def _require_matplotlib() -> None:
    if mpl is None:
        raise RuntimeError('Matplotlib is required for layout QA') from _MATPLOTLIB_IMPORT_ERROR

def _bbox_metrics(a: Bbox, b: Bbox) -> dict[str, float]:
    ix0, iy0 = (max(a.x0, b.x0), max(a.y0, b.y0))
    ix1, iy1 = (min(a.x1, b.x1), min(a.y1, b.y1))
    width = max(0.0, ix1 - ix0)
    height = max(0.0, iy1 - iy0)
    intersection = width * height
    area_a = max(a.width * a.height, 1e-09)
    area_b = max(b.width * b.height, 1e-09)
    union = area_a + area_b - intersection
    return {'intersection': intersection, 'overlap_min': intersection / min(area_a, area_b), 'iou': intersection / union if union else 0.0}

def _collision_threshold(role_a: str, role_b: str, *, cross_axes: bool) -> tuple[float, float]:
    dense_roles = {'tick', 'value', 'small_value', 'direct_label'}
    dense_count = sum((role in dense_roles for role in (role_a, role_b)))
    if dense_count == 2:
        warning, critical = (0.35, 0.75)
    elif dense_count == 1:
        warning, critical = (0.18, 0.5)
    else:
        warning, critical = (0.06, 0.28)
    if cross_axes and dense_count < 2:
        warning = min(warning, 0.03)
    return (warning, critical)

def audit_figure_layout(fig: mpl.figure.Figure, figure_name: str, registry: ArtistRegistry | Sequence[Any]=(), *, minimum_font_pt: float=5.4, safe_pad_pt: float=1.5, check_collisions: bool=True) -> dict[str, Any]:
    _require_matplotlib()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    one_pt_px = fig.dpi / POINTS_PER_INCH
    safe_pad_px = float(safe_pad_pt) * one_pt_px
    explicit = registry.artists if isinstance(registry, ArtistRegistry) else list(registry)
    items: list[dict[str, Any]] = []
    seen: set[int] = set()

    def append_item(artist: Any, role: str, panel: str, name: str) -> None:
        if id(artist) in seen or not artist.get_visible():
            return
        if isinstance(artist, Text) and (not artist.get_text().strip()):
            return
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            return
        if not all((math.isfinite(float(v)) for v in (bbox.x0, bbox.y0, bbox.x1, bbox.y1))):
            return
        if bbox.width <= 0 or bbox.height <= 0:
            return
        seen.add(id(artist))
        axes = getattr(artist, 'axes', None)
        items.append({'artist': artist, 'name': str(name), 'role': str(role), 'panel': str(panel), 'axes_id': id(axes) if axes is not None else None, 'bbox': bbox})
    for artist in explicit:
        if getattr(artist, 'get_figure', lambda: None)() is not fig:
            continue
        meta = getattr(artist, '_qa_meta', {})
        if isinstance(artist, (Text, Legend)):
            append_item(artist, meta.get('role', 'tagged'), meta.get('panel', 'unknown'), meta.get('name', f'artist-{id(artist)}'))
    for axis_index, ax in enumerate(fig.axes):
        if not ax.get_visible() or not ax.axison:
            continue
        panel = getattr(ax, '_qa_meta', {}).get('panel', 'unknown')
        append_item(ax.title, 'panel_title', panel, f'axis-{axis_index}-title')
        labels = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for label_index, artist in enumerate(labels):
            append_item(artist, 'tick', panel, f'axis-{axis_index}-tick-{label_index}')
        for label_name, artist in (('x', ax.xaxis.label), ('y', ax.yaxis.label), ('x-offset', ax.xaxis.get_offset_text()), ('y-offset', ax.yaxis.get_offset_text())):
            append_item(artist, 'axis_label', panel, f'axis-{axis_index}-{label_name}')
        legend = ax.get_legend()
        if legend is not None:
            append_item(legend, 'legend', panel, f'axis-{axis_index}-legend')
    for legend_index, legend in enumerate(fig.legends):
        append_item(legend, 'legend', 'figure', f'figure-legend-{legend_index}')
    for text_index, artist in enumerate(fig.texts):
        meta = getattr(artist, '_qa_meta', {})
        append_item(artist, meta.get('role', 'figure_text'), meta.get('panel', 'figure'), meta.get('name', f'figure-text-{text_index}'))
    issues: list[dict[str, Any]] = []
    safe_bbox = Bbox.from_extents(figure_bbox.x0 + safe_pad_px, figure_bbox.y0 + safe_pad_px, figure_bbox.x1 - safe_pad_px, figure_bbox.y1 - safe_pad_px)
    for item in items:
        bbox = item['bbox']
        outside = bbox.x0 < figure_bbox.x0 or bbox.y0 < figure_bbox.y0 or bbox.x1 > figure_bbox.x1 or (bbox.y1 > figure_bbox.y1)
        low_clearance = bbox.x0 < safe_bbox.x0 or bbox.y0 < safe_bbox.y0 or bbox.x1 > safe_bbox.x1 or (bbox.y1 > safe_bbox.y1)
        if outside:
            issues.append({'type': 'figure_outside', 'severity': 'critical', 'item': item['name'], 'panel': item['panel']})
        elif low_clearance:
            issues.append({'type': 'low_figure_clearance', 'severity': 'low_clearance', 'item': item['name'], 'panel': item['panel']})
    if check_collisions:
        for index, item_a in enumerate(items):
            for item_b in items[index + 1:]:
                metrics = _bbox_metrics(item_a['bbox'], item_b['bbox'])
                if metrics['intersection'] <= 0:
                    continue
                cross_axes = item_a['axes_id'] != item_b['axes_id']
                warning_threshold, critical_threshold = _collision_threshold(item_a['role'], item_b['role'], cross_axes=cross_axes)
                overlap = metrics['overlap_min']
                if overlap >= critical_threshold:
                    severity = 'critical'
                elif overlap >= warning_threshold:
                    severity = 'warning'
                else:
                    continue
                issues.append({'type': 'text_or_legend_overlap', 'severity': severity, 'item_a': item_a['name'], 'item_b': item_b['name'], 'panel_a': item_a['panel'], 'panel_b': item_b['panel'], 'overlap_min': round(float(overlap), 4), 'cross_axes': bool(cross_axes)})
    font_records: list[dict[str, Any]] = []
    for text_index, artist in enumerate(fig.findobj(match=Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        size = float(artist.get_fontsize())
        meta = getattr(artist, '_qa_meta', {})
        record = {'name': meta.get('name', f'text-{text_index}'), 'panel': meta.get('panel', 'unknown'), 'text': artist.get_text()[:120], 'font_pt': round(size, 3)}
        font_records.append(record)
        if size + 1e-06 < float(minimum_font_pt):
            issues.append({'type': 'font_below_minimum', 'severity': 'critical', **record, 'minimum_pt': float(minimum_font_pt)})
    gids = [artist.get_gid() for artist in fig.findobj() if hasattr(artist, 'get_gid')]
    gids = [gid for gid in gids if isinstance(gid, str) and gid]
    duplicate_gids = sorted({gid for gid in gids if gids.count(gid) > 1})
    if duplicate_gids:
        issues.append({'type': 'duplicate_gid', 'severity': 'critical', 'gids': duplicate_gids})
    counts = {severity: sum((issue['severity'] == severity for issue in issues)) for severity in ('critical', 'warning', 'low_clearance')}
    serialized_items = []
    for item in items:
        bbox = item['bbox']
        serialized_items.append({'name': item['name'], 'role': item['role'], 'panel': item['panel'], 'bbox_pt': [round(float(value) / one_pt_px, 3) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)]})
    minimum_seen = min((row['font_pt'] for row in font_records), default=None)
    return {'figure': {'name': str(figure_name), 'width_mm': round(float(fig.get_figwidth() * MM_PER_INCH), 3), 'height_mm': round(float(fig.get_figheight() * MM_PER_INCH), 3)}, 'summary': {**counts, 'minimum_font_pt': minimum_seen, 'required_minimum_font_pt': float(minimum_font_pt), 'pass': counts['critical'] == 0 and counts['warning'] == 0, 'strict_pass': all((counts[key] == 0 for key in counts))}, 'items': serialized_items, 'fonts': font_records, 'issues': issues}

def write_json(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(public_path_display(dict(report)), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output

def assert_figure_size(fig: mpl.figure.Figure, *, width_mm: float, height_mm: float, tolerance_mm: float=0.05) -> tuple[float, float]:
    _require_matplotlib()
    actual_width = float(fig.get_figwidth()) * MM_PER_INCH
    actual_height = float(fig.get_figheight()) * MM_PER_INCH
    if abs(actual_width - float(width_mm)) > float(tolerance_mm) or abs(actual_height - float(height_mm)) > float(tolerance_mm):
        raise RuntimeError(f'Unexpected figure size: {actual_width:.3f} x {actual_height:.3f} mm; expected {float(width_mm):.3f} x {float(height_mm):.3f} mm')
    return (actual_width, actual_height)

def expected_raster_pixels(width_mm: float, height_mm: float, dpi: int) -> tuple[int, int]:
    return (round(float(width_mm) / MM_PER_INCH * int(dpi)), round(float(height_mm) / MM_PER_INCH * int(dpi)))

def require_file(path: str | Path, minimum_bytes: int=1000) -> Path:
    value = Path(path)
    if not value.exists() or value.stat().st_size < int(minimum_bytes):
        raise RuntimeError(f'Missing or unexpectedly small output: {value}')
    return value

def verify_png_dimensions(path: str | Path, *, width_mm: float, height_mm: float, dpi: int, tolerance_px: int=1) -> dict[str, Any]:
    from PIL import Image
    value = require_file(path)
    with Image.open(value) as image:
        actual = tuple((int(item) for item in image.size))
    expected = expected_raster_pixels(width_mm, height_mm, dpi)
    if any((abs(a - e) > int(tolerance_px) for a, e in zip(actual, expected))):
        raise RuntimeError(f'Unexpected PNG dimensions for {value.name}: {actual}; expected {expected}')
    return {'path': str(value), 'actual_px': actual, 'expected_px': expected}

def verify_svg_editable_text(path: str | Path, *, minimum_text_nodes: int=5) -> dict[str, Any]:
    value = require_file(path)
    source = value.read_text(encoding='utf-8')
    nodes = len(re.findall('<text\\b', source))
    if nodes < int(minimum_text_nodes):
        raise RuntimeError(f'SVG appears non-editable: {nodes} text nodes in {value.name}')
    return {'path': str(value), 'text_nodes': nodes}

def verify_pdf_page(path: str | Path, *, width_mm: float, height_mm: float, tolerance_mm: float=0.08, require_embedded_fonts: bool=True) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    value = require_file(path)
    reader = PdfReader(str(value))
    if len(reader.pages) != 1:
        raise RuntimeError(f'Expected one PDF page, found {len(reader.pages)}')
    page = reader.pages[0]
    actual_width = float(page.mediabox.width) * MM_PER_INCH / POINTS_PER_INCH
    actual_height = float(page.mediabox.height) * MM_PER_INCH / POINTS_PER_INCH
    if abs(actual_width - float(width_mm)) > float(tolerance_mm) or abs(actual_height - float(height_mm)) > float(tolerance_mm):
        raise RuntimeError(f'Unexpected PDF page size: {actual_width:.3f} x {actual_height:.3f} mm')

    def font_descriptor(font: Any) -> Any:
        descriptor = font.get('/FontDescriptor')
        if descriptor is not None:
            return descriptor.get_object()
        descendants = font.get('/DescendantFonts')
        if descendants:
            descriptor = descendants[0].get_object().get('/FontDescriptor')
            if descriptor is not None:
                return descriptor.get_object()
        return None
    fonts: dict[str, dict[str, Any]] = {}
    resources = page.get('/Resources', {}).get_object()
    font_resources = resources.get('/Font', {})
    if hasattr(font_resources, 'get_object'):
        font_resources = font_resources.get_object()
    for name, reference in font_resources.items():
        font = reference.get_object()
        descriptor = font_descriptor(font)
        embedded = bool(descriptor and any((key in descriptor for key in ('/FontFile', '/FontFile2', '/FontFile3'))))
        fonts[str(name)] = {'subtype': str(font.get('/Subtype')), 'basefont': str(font.get('/BaseFont')), 'embedded': embedded}
    if not fonts:
        raise RuntimeError('No PDF font resources found')
    if any((record['subtype'] == '/Type3' for record in fonts.values())):
        raise RuntimeError(f'Type 3 font found: {fonts}')
    if require_embedded_fonts and (not all((record['embedded'] for record in fonts.values()))):
        raise RuntimeError(f'Unembedded PDF font resource found: {fonts}')
    return {'path': str(value), 'pages': len(reader.pages), 'width_mm': round(actual_width, 3), 'height_mm': round(actual_height, 3), 'fonts': fonts}

def sha256_file(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def verify_export_bundle(output_stem: str | Path, *, width_mm: float, height_mm: float, dpi: int=600, minimum_svg_text_nodes: int=5) -> dict[str, Any]:
    stem = Path(output_stem)
    pdf = require_file(stem.with_suffix('.pdf'))
    svg = require_file(stem.with_suffix('.svg'))
    png = require_file(stem.with_suffix('.png'))
    return {'png': verify_png_dimensions(png, width_mm=width_mm, height_mm=height_mm, dpi=dpi), 'svg': verify_svg_editable_text(svg, minimum_text_nodes=minimum_svg_text_nodes), 'pdf': verify_pdf_page(pdf, width_mm=width_mm, height_mm=height_mm), 'sha256': {'pdf': sha256_file(pdf), 'svg': sha256_file(svg), 'png': sha256_file(png)}}
__all__ = ['ArtistRegistry', 'assert_figure_size', 'audit_figure_layout', 'expected_raster_pixels', 'require_file', 'sha256_file', 'verify_export_bundle', 'verify_pdf_page', 'verify_png_dimensions', 'verify_svg_editable_text', 'write_json']
