from __future__ import annotations
from avatar_paths import get_value, get_column
import json
from pathlib import Path
from typing import Any, Mapping
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import figureS3_render as legacy
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / 'inputs'
VERSION = 'v002'
OUTPUT_DIR = ROOT / 'outputs' / get_value('figures.config.item_356')
PANEL_DIR = OUTPUT_DIR / 'panels'
FULL_STEM = OUTPUT_DIR / f'Supplementary_Figure_S4_A_to_E_Nature_{VERSION}'
WIDTH_MM = 180.0
HEIGHT_MM = 240.0
EXPORT_DPI = 600
MIN_FONT_PT = legacy.MIN_FONT_PT
MM_PER_INCH = legacy.MM_PER_INCH
PANELS = 'abcde'
PANEL_SIZES_MM = {panel: legacy.PANEL_SIZES_MM[panel] for panel in PANELS}
FULL_LAYOUT: Mapping[str, tuple[float, float, float, float]] = {'a': (7.0, 6.0, 166.0, 32.0), 'b': (7.0, 41.0, 166.0, 33.0), 'c': (7.0, 77.0, 166.0, 54.0), 'd': (7.0, 134.0, 166.0, 43.0), 'e': (7.0, 180.0, 166.0, 54.0)}
INPUT_KEYS = ('landscape', 'boundaries', 'mrecist_cells', 'mrecist_assoc', 'within_mrecist', 'phenotype', 'phenotype_tests', 'multivariable', 'outer_summary', 'outer_folds', 'repeat_folds', 'repeat_summary', 'cutoff_values', 'cutoff_summary', 'excluded_effects', 'excluded_agreement', 'excluded_counts')
INPUT_FILES = {key: legacy.INPUT_FILES[key] for key in INPUT_KEYS}

def _require_rows(frame: pd.DataFrame, count: int, label: str) -> None:
    if len(frame) != int(count):
        raise ValueError(f'{label}: expected {count} rows, found {len(frame)}')

def validate_inputs(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle['manifest']
    if manifest['palette_id'] != legacy.PALETTE_ID:
        raise ValueError('Palette identifier differs from the shared style')
    records = {entry['snapshot']: entry for entry in manifest['files']}
    file_checks: list[dict[str, Any]] = []
    for filename in INPUT_FILES.values():
        path = INPUT_DIR / filename
        if filename not in records:
            raise ValueError(f'Manifest does not register {filename}')
        record = records[filename]
        actual_hash = legacy._sha256(path)
        frame = pd.read_csv(path)
        if actual_hash != record['sha256']:
            raise ValueError(f'SHA-256 mismatch for {filename}')
        if frame.shape != (int(record['rows']), int(record['columns'])):
            raise ValueError(f'Shape mismatch for {filename}')
        file_checks.append({'file': filename, 'shape': [int(frame.shape[0]), int(frame.shape[1])], 'sha256': actual_hash, 'pass': True})
    landscape = bundle['landscape']
    _require_rows(landscape, 109, 'patient landscape')
    legacy._require_columns(landscape, ['sample', 'MBP_raw', 'patient_order', 'Response_benefit', 'PFS_benefit', 'OS_benefit', 'Response_observed', 'AFP_favorability', 'TBS_favorability', 'PVTT_favorability', 'ALBI_favorability'], 'patient landscape')
    if sorted(landscape['patient_order'].astype(int).tolist()) != list(range(1, 110)):
        raise ValueError('Patient order is not the locked 1..109 sequence')
    group_n = landscape['MBP_raw'].value_counts().reindex(legacy.MBP_ORDER).astype(int)
    if group_n.tolist() != [37, 36, 36]:
        raise ValueError('MBP group sizes differ from 37/36/36')
    boundaries = bundle['boundaries'].set_index('group').reindex(legacy.MBP_ORDER)
    if boundaries['n'].astype(int).tolist() != group_n.tolist():
        raise ValueError('Landscape and boundary group counts disagree')
    _require_rows(bundle['mrecist_assoc'], 1, 'mRECIST association')
    _require_rows(bundle['mrecist_cells'], 6, 'mRECIST cells')
    if int(bundle['mrecist_cells']['Observed_n'].sum()) != 109:
        raise ValueError('mRECIST cell total differs from 109')
    if int(bundle['mrecist_assoc'].loc[0, 'n']) != 109:
        raise ValueError('mRECIST association total differs from 109')
    within = bundle['within_mrecist']
    _require_rows(within, 4, 'within-mRECIST summary')
    expected_ne = {('PFS', 'Responder'): (53, 44), ('PFS', 'Nonresponder'): (56, 54), ('OS', 'Responder'): (53, 22), ('OS', 'Nonresponder'): (56, 45)}
    for row in within.itertuples(index=False):
        if (int(row.n), int(row.events)) != expected_ne[row.Endpoint, row.mRECIST]:
            raise ValueError('Within-mRECIST n/event contract changed')
    phenotype = bundle['phenotype']
    phenotype_tests = bundle['phenotype_tests']
    _require_rows(phenotype, 69, 'phenotype effect matrix')
    _require_rows(phenotype_tests, 23, 'phenotype global tests')
    if phenotype['Variable'].nunique() != 23:
        raise ValueError('Phenotype matrix does not contain 23 variables')
    if set(phenotype['Variable']) != set(phenotype_tests['Variable']):
        raise ValueError('Phenotype rows and global tests differ')
    _require_rows(bundle['multivariable'], 12, 'full multivariable effects')
    expected_rows = {'outer_summary': 4, 'outer_folds': 200, 'repeat_folds': 100, 'repeat_summary': 4, 'cutoff_values': 200, 'cutoff_summary': 4, 'excluded_effects': 6, 'excluded_agreement': 3, 'excluded_counts': 27}
    for key, count in expected_rows.items():
        _require_rows(bundle[key], count, key)
    if not (bundle['outer_summary']['n_valid_outer_folds'].astype(int) == 50).all():
        raise ValueError('Outer-CV fold count differs from 50')
    if not (bundle['repeat_summary']['n_folds'].astype(int) == 25).all():
        raise ValueError('Repeated-CV fold count differs from 25')
    if not (bundle['cutoff_summary']['n_outer_training_folds'].astype(int) == 50).all():
        raise ValueError('Cutoff summary fold count differs from 50')
    if not (bundle['excluded_counts'].groupby('Endpoint_excluded')['n'].sum().astype(int) == 109).all():
        raise ValueError('Endpoint-excluded confusion tables do not total 109')
    if not (bundle['excluded_agreement']['n'].astype(int) == 109).all():
        raise ValueError('Endpoint-excluded agreement n differs from 109')
    if not (bundle['excluded_effects']['n'].astype(int) == 109).all():
        raise ValueError('Endpoint-excluded effects n differs from 109')
    return {'figure': 'Supplementary Figure S4', 'version': VERSION, 'panels': list(PANELS), 'source_mode': 'frozen snapshot validation only; no statistical estimation', 'files': file_checks, 'excluded_panel_f_inputs': [legacy.INPUT_FILES[key] for key in ('weight_agreement', 'weight_effects', 'cutoff_effects_extreme', 'cutoff_effects_category')], 'locked': {'cohort_n': 109, 'MBP_group_n': group_n.to_dict(), 'within_mRECIST_rows': 4, 'phenotype_variables': 23, 'outerCV_fold_rows': 200, 'repeatedCV_fold_rows': 100, 'cutoff_rows': 200, 'endpoint_excluded_confusion_total_per_endpoint': 109}, 'pass': True}

def load_inputs() -> dict[str, Any]:
    bundle: dict[str, Any] = {key: pd.read_csv(INPUT_DIR / filename) for key, filename in INPUT_FILES.items()}
    bundle['manifest'] = json.loads((INPUT_DIR / get_value('figures.config.item_116')).read_text(encoding='utf-8'))
    contract = json.loads((INPUT_DIR / get_value('figures.config.item_201')).read_text(encoding='utf-8'))
    contract['figure'] = 'Supplementary Figure S4'
    contract['panel_mapping'] = {key: value for key, value in contract['panel_mapping'].items() if key in PANELS}
    bundle['contract'] = contract
    bundle['numeric_qa'] = validate_inputs(bundle)
    return bundle

def build_full_figure(bundle: Mapping[str, Any] | None=None):
    if bundle is None:
        bundle = load_inputs()
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = legacy.ArtistRegistry()
    axes = []
    for panel in PANELS:
        axes.extend(legacy._draw_panel(panel, fig, (WIDTH_MM, HEIGHT_MM), FULL_LAYOUT[panel], bundle, registry))
        legacy._add_panel_letter(fig, panel, FULL_LAYOUT[panel], registry)
    return (fig, registry, axes)

def build_panel_figure(panel: str, bundle: Mapping[str, Any] | None=None):
    panel = panel.lower().strip()
    if panel not in PANEL_SIZES_MM:
        raise ValueError(f'Unknown panel: {panel}')
    if bundle is None:
        bundle = load_inputs()
    width, height = PANEL_SIZES_MM[panel]
    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), facecolor='white')
    registry = legacy.ArtistRegistry()
    axes = legacy._draw_panel(panel, fig, (width, height), (0, 0, width, height), bundle, registry)
    return (fig, registry, axes)

def _save(fig, stem: Path, *, width_mm: float, height_mm: float) -> list[Path]:
    legacy.assert_figure_size(fig, width_mm=width_mm, height_mm=height_mm)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S4 {VERSION} - {legacy.PALETTE_ID}', 'Creator': 'Avatar-HCC Figure Harmonization'}
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
        paths.append(path)
    return paths

def _file_qa(paths: list[Path], panel_paths: list[Path]) -> dict[str, Any]:
    expected_full_px = tuple((round(value / MM_PER_INCH * EXPORT_DPI) for value in (WIDTH_MM, HEIGHT_MM)))
    full_png = FULL_STEM.with_suffix('.png')
    with Image.open(full_png) as image:
        full_px = image.size
        full_mode = image.mode
    if any((abs(observed - expected) > 1 for observed, expected in zip(full_px, expected_full_px))):
        raise ValueError(f'Full PNG dimensions {full_px} != {expected_full_px}')
    panels: dict[str, Any] = {}
    for panel, (width, height) in PANEL_SIZES_MM.items():
        png = PANEL_DIR / f'Supplementary_Figure_S4_panel_{panel}_Nature_{VERSION}.png'
        expected_px = tuple((round(value / MM_PER_INCH * EXPORT_DPI) for value in (width, height)))
        with Image.open(png) as image:
            observed = image.size
            mode = image.mode
        if any((abs(value - expected) > 1 for value, expected in zip(observed, expected_px))):
            raise ValueError(f'Panel {panel} PNG dimensions {observed} != {expected_px}')
        panels[panel] = {'png': str(png), 'pixels': list(observed), 'mode': mode, 'pass': True}
    all_artifacts = paths + panel_paths
    missing = [str(path) for path in all_artifacts if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f'Missing or empty artifacts: {missing}')
    return {'figure': 'Supplementary Figure S4', 'version': VERSION, 'panels': list(PANELS), 'full_png': {'path': str(full_png), 'pixels': list(full_px), 'mode': full_mode}, 'standalone_panels': panels, 'artifacts_checked': len(all_artifacts), 'pass': True}

def export_all(bundle: Mapping[str, Any] | None=None) -> dict[str, Any]:
    if bundle is None:
        bundle = load_inputs()
    legacy.configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    numeric_path = legacy.write_json(bundle['numeric_qa'], OUTPUT_DIR / f'Supplementary_Figure_S4_numeric_qa_{VERSION}.json')
    fig, registry, _ = build_full_figure(bundle)
    layout = legacy.audit_figure_layout(fig, f'Supplementary_Figure_S4_A_to_E_Nature_{VERSION}', registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.0, check_collisions=True)
    layout_path = legacy.write_json(layout, OUTPUT_DIR / f'Supplementary_Figure_S4_layout_and_text_audit_{VERSION}.json')
    full_paths = _save(fig, FULL_STEM, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    plt.close(fig)
    panel_paths: list[Path] = []
    panel_audits: dict[str, Any] = {}
    for panel in PANELS:
        panel_fig, panel_registry, _ = build_panel_figure(panel, bundle)
        width, height = PANEL_SIZES_MM[panel]
        panel_audits[panel] = legacy.audit_figure_layout(panel_fig, f'Supplementary_Figure_S4_panel_{panel}_Nature_{VERSION}', panel_registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.0, check_collisions=True)
        stem = PANEL_DIR / f'Supplementary_Figure_S4_panel_{panel}_Nature_{VERSION}'
        panel_paths.extend(_save(panel_fig, stem, width_mm=width, height_mm=height))
        plt.close(panel_fig)
    panel_audit_path = legacy.write_json(panel_audits, OUTPUT_DIR / f'Supplementary_Figure_S4_panel_layout_audit_{VERSION}.json')
    file_qa = _file_qa(full_paths, panel_paths)
    file_qa_path = legacy.write_json(file_qa, OUTPUT_DIR / f'Supplementary_Figure_S4_file_qa_{VERSION}.json')
    outputs = full_paths + panel_paths + [numeric_path, layout_path, panel_audit_path, file_qa_path]
    visual_qa_path = OUTPUT_DIR / f'Supplementary_Figure_S4_visual_qa_{VERSION}.json'
    if visual_qa_path.exists():
        outputs.append(visual_qa_path)
    pdf_qa_path = OUTPUT_DIR / f'Supplementary_Figure_S4_pdf_qa_{VERSION}.json'
    if pdf_qa_path.exists():
        outputs.append(pdf_qa_path)
    source_paths = [Path(__file__).resolve(), ROOT / get_value('figures.config.item_042')]
    manifest = {'figure': 'Supplementary Figure S4', 'version': VERSION, 'panels': list(PANELS), 'palette': legacy.PALETTE_ID, 'working_master_size_mm': [WIDTH_MM, HEIGHT_MM], 'source_mode': 'render-only frozen CSV snapshots; no statistical recomputation', 'panel_mapping': bundle['contract']['panel_mapping'], 'excluded_panel': 'f', 'excluded_panel_sources': bundle['numeric_qa']['excluded_panel_f_inputs'], 'layout_summary': layout['summary'], 'numeric_qa': {'status': 'passed', 'report': str(numeric_path)}, 'file_qa': {'status': 'passed', 'report': str(file_qa_path)}, 'visual_qa': {'status': 'passed' if visual_qa_path.exists() else 'pending', 'report': str(visual_qa_path)}, 'pdf_qa': {'status': 'passed' if pdf_qa_path.exists() else 'pending', 'report': str(pdf_qa_path)}, 'outputs': [{'path': str(path), 'sha256': legacy.sha256_file(path), 'bytes': path.stat().st_size} for path in outputs], 'sources': [{'path': str(path), 'sha256': legacy.sha256_file(path), 'bytes': path.stat().st_size} for path in source_paths if path.exists()]}
    manifest_path = legacy.write_json(manifest, OUTPUT_DIR / f'Supplementary_Figure_S4_export_manifest_{VERSION}.json')
    return {'full_paths': [str(path) for path in full_paths], 'panel_paths': [str(path) for path in panel_paths], 'numeric_qa': str(numeric_path), 'layout_audit': str(layout_path), 'panel_audit': str(panel_audit_path), 'file_qa': str(file_qa_path), 'manifest': str(manifest_path), 'layout_summary': layout['summary'], 'panel_layout_summaries': {p: report['summary'] for p, report in panel_audits.items()}}

def main() -> None:
    bundle = load_inputs()
    export_all(bundle)
if __name__ == '__main__':
    main()
