from __future__ import annotations
from avatar_paths import get_value, get_column
import json
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import figureS4_render as legacy
ROOT = Path(__file__).resolve().parent
VERSION = 'v002'
OUTPUT_DIR = ROOT / 'outputs' / get_value('figures.config.item_372')
PANEL_DIR = OUTPUT_DIR / 'panels'
FULL_STEM = OUTPUT_DIR / f'Supplementary_Figure_S5_A_to_F_Nature_{VERSION}'
PANELS = 'abcdef'

def load_privacy_safe_bundle() -> tuple[dict[str, Any], list[str]]:
    bundle = legacy.load_inputs()
    original_ids = bundle['e_select']['sample'].astype(str).tolist()
    if len(original_ids) != 3 or len(set(original_ids)) != 3:
        raise ValueError('Expected exactly three distinct frozen representative patients')
    bundle['e_select'] = bundle['e_select'].copy()
    bundle['e_select'].loc[:, 'sample'] = 'representative patient'
    return (bundle, original_ids)

def _save(fig, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    legacy.assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S5 {VERSION} - {legacy.PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
    paths: list[Path] = []
    for suffix in ('pdf', 'svg', 'png'):
        path = stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, Any] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0.0}
        if suffix == 'png':
            kwargs['dpi'] = legacy.EXPORT_DPI
        elif suffix == 'pdf':
            kwargs['metadata'] = metadata
        else:
            kwargs['metadata'] = {'Title': metadata['Title'], 'Creator': metadata['Creator']}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def _verify_bundle(stem: Path, *, width_mm: float, height_mm: float, minimum_svg_text_nodes: int) -> dict[str, Any]:
    pdf = stem.with_suffix('.pdf')
    svg = stem.with_suffix('.svg')
    png = stem.with_suffix('.png')
    for path in (pdf, svg, png):
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty export: {path}')
    expected_px = (round(width_mm / legacy.MM_PER_INCH * legacy.EXPORT_DPI), round(height_mm / legacy.MM_PER_INCH * legacy.EXPORT_DPI))
    with Image.open(png) as image:
        observed_px = image.size
        mode = image.mode
    if any((abs(value - expected) > 1 for value, expected in zip(observed_px, expected_px))):
        raise ValueError(f'PNG dimensions {observed_px} != {expected_px}: {png}')
    svg_source = svg.read_text(encoding='utf-8')
    text_nodes = svg_source.count('<text')
    if text_nodes < minimum_svg_text_nodes:
        raise ValueError(f'SVG editable text count {text_nodes} < {minimum_svg_text_nodes}: {svg}')
    return {'pdf': {'path': str(pdf), 'bytes': pdf.stat().st_size}, 'svg': {'path': str(svg), 'bytes': svg.stat().st_size, 'editable_text_nodes': text_nodes}, 'png': {'path': str(png), 'bytes': png.stat().st_size, 'pixels': list(observed_px), 'mode': mode}, 'expected_size_mm': [width_mm, height_mm], 'pass': True}

def export_all(bundle: Mapping[str, Any], original_ids: list[str]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, registry = legacy.build_full_figure(bundle)
    layout = legacy.audit_figure_layout(fig, f'Supplementary_Figure_S5_A_to_F_Nature_{VERSION}', registry, minimum_font_pt=legacy.MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
    layout_path = legacy.write_json(layout, OUTPUT_DIR / f'Supplementary_Figure_S5_layout_and_text_audit_{VERSION}.json')
    full_paths = _save(fig, FULL_STEM, width_mm=legacy.FIGURE_WIDTH_MM, height_mm=legacy.FIGURE_HEIGHT_MM)
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in PANELS:
        panel_fig, panel_registry = legacy.build_panel_figure(panel, bundle)
        width_mm, height_mm = legacy.PANEL_LAYOUTS[panel]
        panel_audits[panel] = legacy.audit_figure_layout(panel_fig, f'Supplementary_Figure_S5_panel_{panel}_Nature_{VERSION}', panel_registry, minimum_font_pt=legacy.MIN_FONT_PT, safe_pad_pt=1.2, check_collisions=True)
        panel_paths.extend(_save(panel_fig, PANEL_DIR / f'Supplementary_Figure_S5_panel_{panel}_Nature_{VERSION}', width_mm=width_mm, height_mm=height_mm))
        plt.close(panel_fig)
    panel_audit_path = legacy.write_json(panel_audits, OUTPUT_DIR / f'Supplementary_Figure_S5_panel_layout_audit_{VERSION}.json')
    scientific = legacy.scientific_summary(bundle)
    scientific.update({'figure': 'Supplementary Figure S5', 'version': VERSION, 'privacy_display_only_change': True, 'representative_patient_selection_unchanged': True, 'representative_patient_identifier_count': len(original_ids), 'visible_label': '<MBP group> · representative patient'})
    scientific_path = legacy.write_json(scientific, OUTPUT_DIR / f'Supplementary_Figure_S5_scientific_guardrail_report_{VERSION}.json')
    full = _verify_bundle(FULL_STEM, width_mm=legacy.FIGURE_WIDTH_MM, height_mm=legacy.FIGURE_HEIGHT_MM, minimum_svg_text_nodes=100)
    panel_bundles = {panel: _verify_bundle(PANEL_DIR / f'Supplementary_Figure_S5_panel_{panel}_Nature_{VERSION}', width_mm=legacy.PANEL_LAYOUTS[panel][0], height_mm=legacy.PANEL_LAYOUTS[panel][1], minimum_svg_text_nodes=5) for panel in PANELS}
    svg_paths = [FULL_STEM.with_suffix('.svg'), PANEL_DIR / f'Supplementary_Figure_S5_panel_e_Nature_{VERSION}.svg']
    visible_svg_text: list[str] = []
    for path in svg_paths:
        root = ET.parse(path).getroot()
        visible_svg_text.extend((''.join(node.itertext()) for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'text'))
    svg_text = '\n'.join(visible_svg_text)
    leaked = [value for value in original_ids if value in svg_text]
    if leaked:
        raise ValueError('Representative patient identifier remained in an SVG export')
    expected_labels = [f'{group} · representative patient' for group in legacy.MBP_ORDER]
    missing_labels = [value for value in expected_labels if value not in svg_text]
    if missing_labels:
        raise ValueError(f'Privacy-safe labels missing from SVG: {missing_labels}')
    privacy_qa = {'figure': 'Supplementary Figure S5', 'version': VERSION, 'pass': True, 'display_only_change': True, 'patient_selection_unchanged': True, 'source_rows_and_values_unchanged': True, 'original_identifier_count': len(original_ids), 'original_identifiers_absent_from_full_and_panel_e_svg': True, 'generic_labels_present': expected_labels, 'full_export_bundle': full, 'panel_export_bundles': panel_bundles, 'layout_full': layout['summary'], 'layout_panels': {key: value['summary'] for key, value in panel_audits.items()}}
    privacy_qa_path = legacy.write_json(privacy_qa, OUTPUT_DIR / f'Supplementary_Figure_S5_privacy_and_file_QA_{VERSION}.json')
    outputs = full_paths + panel_paths + [layout_path, panel_audit_path, scientific_path, privacy_qa_path]
    for extra_qa in (OUTPUT_DIR / f'Supplementary_Figure_S5_visual_qa_{VERSION}.json', OUTPUT_DIR / f'Supplementary_Figure_S5_pdf_qa_{VERSION}.json'):
        if extra_qa.exists():
            outputs.append(extra_qa)
    source_paths = [Path(__file__).resolve(), ROOT / get_value('figures.config.item_182'), Path(legacy.__file__).resolve(), ROOT / 'inputs' / get_value('figures.config.item_116'), ROOT / 'inputs' / get_value('figures.config.item_201')]
    manifest = {'figure': 'Supplementary Figure S5', 'legacy_package': get_value('figures.path.part_016'), 'version': VERSION, 'palette': legacy.PALETTE_ID, 'working_master_size_mm': [legacy.FIGURE_WIDTH_MM, legacy.FIGURE_HEIGHT_MM], 'source_mode': 'render-only from SHA-256-locked frozen tables and matrices', 'layout_summary': layout['summary'], 'panel_layout_summaries': {key: value['summary'] for key, value in panel_audits.items()}, 'outputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': legacy.sha256_file(path)} for path in outputs], 'sources': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': legacy.sha256_file(path)} for path in source_paths]}
    manifest_path = legacy.write_json(manifest, OUTPUT_DIR / f'Supplementary_Figure_S5_export_manifest_{VERSION}.json')
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'scientific_guardrail_report': str(scientific_path), 'privacy_and_file_qa': str(privacy_qa_path), 'manifest': str(manifest_path)}

def main() -> None:
    bundle, original_ids = load_privacy_safe_bundle()
    export_all(bundle, original_ids)
if __name__ == '__main__':
    main()
