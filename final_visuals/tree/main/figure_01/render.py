from avatar_paths import get_value, get_column
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ByteStringObject, ContentStream
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NS = get_value('figures.config.item_379')
ET.register_namespace('', NS)
ET.register_namespace('xlink', get_value('figures.config.item_382'))
COHORT = OUT / get_value('figures.config.item_005')
ORIGINAL_COHORT = ROOT / get_value('figures.config.item_028')
CLINICAL = ROOT / get_value('figures.config.item_017')
SOURCES = [COHORT, CLINICAL, COHORT.with_suffix('.pdf'), CLINICAL.with_suffix('.pdf'), ORIGINAL_COHORT, ORIGINAL_COHORT.with_suffix('.pdf')]
style_audit = json.loads((OUT / get_value('figures.config.item_215')).read_text(encoding='utf-8'))
required_style_checks = ['cohort_text_preserved', 'cohort_geometry_preserved', 'cohort_text_positions_preserved', 'cohort_criteria_font_sizes_preserved', 'only_allowed_text_enlarged', 'pdf_other_operations_preserved', 'source_files_unchanged']
assert all((style_audit['checks'][k] is True for k in required_style_checks))
assert style_audit['new_analysis'] is False
before_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
mapping = dict(zip('abcdefgh', 'bcdefghi'))
cohort_original = ET.parse(COHORT).getroot()
clinical_original = ET.parse(CLINICAL).getroot()
cohort = copy.deepcopy(cohort_original)
clinical = copy.deepcopy(clinical_original)
changed_svg_labels = []
for g in clinical.iter(f'{{{NS}}}g'):
    old = g.get('id', '').removeprefix('qa:').removesuffix('-label')
    if g.get('id') == f'qa:{old}-label' and old in mapping:
        text_node = next(g.iter(f'{{{NS}}}text'))
        assert text_node.text == old
        text_node.text = mapping[old]
        changed_svg_labels.append((old, mapping[old]))
assert sorted(changed_svg_labels) == sorted(mapping.items())

def prefix_ids(svg, prefix):
    ids = {n.get('id'): prefix + n.get('id') for n in svg.iter() if n.get('id')}
    for n in svg.iter():
        for key, value in list(n.attrib.items()):
            if key == 'id':
                n.set(key, ids[value])
            else:
                value = re.sub('url\\(#([^\\)]+)\\)', lambda m: 'url(#' + ids.get(m.group(1), m.group(1)) + ')', value)
                if value.startswith('#') and value[1:] in ids:
                    value = '#' + ids[value[1:]]
                n.set(key, value)
    for n in list(svg):
        if n.tag in (f'{{{NS}}}metadata', f'{{{NS}}}title', f'{{{NS}}}desc'):
            svg.remove(n)
prefix_ids(cohort, 'cohort_')
prefix_ids(clinical, 'clinical_')
W, GAP_MM, TRIM_CLINICAL_PT = (180.0, 4.0, 18.0)
avx, avy, avw, avh = map(float, cohort_original.get('viewBox').split())
cvx, cvy, cvw, cvh = map(float, clinical_original.get('viewBox').split())
cohort_h = W * avh / avw
clinical_h = W * (cvh - TRIM_CLINICAL_PT) / cvw
H = cohort_h + GAP_MM + clinical_h
root = ET.Element(f'{{{NS}}}svg', width=f'{W}mm', height=f'{H:.9f}mm', viewBox=f'0 0 {W} {H:.9f}')
ET.SubElement(root, f'{{{NS}}}title').text = 'Avatar-HCC Figure 1: complete cohort flow and clinical outcomes, a-i'
ET.SubElement(root, f'{{{NS}}}rect', x='0', y='0', width=str(W), height=str(H), fill='white')
cohort.set('x', '0')
cohort.set('y', '0')
cohort.set('width', str(W))
cohort.set('height', str(cohort_h))
root.append(cohort)
clinical.set('viewBox', f'{cvx} {cvy + TRIM_CLINICAL_PT} {cvw} {cvh - TRIM_CLINICAL_PT}')
clinical.set('x', '0')
clinical.set('y', str(cohort_h + GAP_MM))
clinical.set('width', str(W))
clinical.set('height', str(clinical_h))
root.append(clinical)
geometry_keys = ['d', 'points', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height', 'transform', 'style']
geometry_tags = {f'{{{NS}}}{t}' for t in ['path', 'line', 'polyline', 'polygon', 'circle', 'ellipse', 'rect', 'text', 'image']}

def geometry(svg):
    return [(n.tag, tuple(((k, n.get(k)) for k in geometry_keys if n.get(k) is not None))) for n in svg.iter() if n.tag in geometry_tags]

def all_text(svg):
    return [''.join(n.itertext()) for n in svg.iter(f'{{{NS}}}text')]
expected_clinical_text = copy.deepcopy(clinical_original)
for g in expected_clinical_text.iter(f'{{{NS}}}g'):
    for old, new in mapping.items():
        if g.get('id') == f'qa:{old}-label':
            next(g.iter(f'{{{NS}}}text')).text = new
assert geometry(cohort_original) == geometry(cohort)
assert all_text(cohort_original) == all_text(cohort)
assert geometry(clinical_original) == geometry(clinical)
assert all_text(expected_clinical_text) == all_text(clinical)
stem = OUT / get_value('figures.config.item_346')
ET.ElementTree(root).write(stem.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)
previous_svg = ROOT / get_value('figures.config.item_003')
previous_root = ET.parse(previous_svg).getroot()
assert root.attrib == previous_root.attrib
assert ET.tostring(root[-1]) == ET.tostring(previous_root[-1])
cohort_pdf = PdfReader(COHORT.with_suffix('.pdf'))
clinical_pdf = PdfReader(CLINICAL.with_suffix('.pdf'))
assert len(cohort_pdf.pages) == len(clinical_pdf.pages) == 1
clinical_writer = PdfWriter(clone_from=clinical_pdf)
cp = clinical_writer.pages[0]
operations = ContentStream(cp.get_contents(), clinical_writer)
before_operations = copy.deepcopy(operations.operations)
font, font_size = (None, 0)
pdf_label_updates = []
for i, (operands, operator) in enumerate(operations.operations):
    if operator == b'Tf':
        font, font_size = (str(operands[0]), float(operands[1]))
    if operator == b'TJ' and font == '/F1' and (font_size == 8.5) and (len(operands[0]) == 1):
        old = str(operands[0][0])
        if old in mapping:
            raw = operands[0][0].original_bytes
            assert raw == old.encode('utf-16-be')
            operands[0][0] = ByteStringObject(mapping[old].encode('utf-16-be'))
            pdf_label_updates.append({'operation_index': i, 'old': old, 'new': mapping[old]})
assert sorted(((r['old'], r['new']) for r in pdf_label_updates)) == sorted(mapping.items())
indices = {r['operation_index'] for r in pdf_label_updates}
assert all((a == b for i, (a, b) in enumerate(zip(before_operations, operations.operations)) if i not in indices))
assert len(before_operations) == len(operations.operations)
cp.replace_contents(operations)
pt_per_mm = 72 / 25.4
width_pt = W * pt_per_mm
scale_a = width_pt / float(cohort_pdf.pages[0].mediabox.width)
scale_c = width_pt / float(cp.mediabox.width)
clinical_height_pt = (float(cp.mediabox.height) - TRIM_CLINICAL_PT) * scale_c
cohort_height_pt = float(cohort_pdf.pages[0].mediabox.height) * scale_a
height_pt = clinical_height_pt + GAP_MM * pt_per_mm + cohort_height_pt
assert abs(height_pt / pt_per_mm - H) < 1e-05
cp.cropbox.upper_right = (float(cp.mediabox.width), float(cp.mediabox.height) - TRIM_CLINICAL_PT)
writer = PdfWriter()
page = writer.add_blank_page(width=width_pt, height=height_pt)
page.merge_transformed_page(cp, Transformation().scale(scale_c), expand=False)
page.merge_transformed_page(cohort_pdf.pages[0], Transformation().scale(scale_a).translate(0, clinical_height_pt + GAP_MM * pt_per_mm), expand=False)
writer.add_metadata({'/Title': 'Avatar-HCC Figure 1 - full original cohort flow and complete clinical outcomes', '/Author': 'Avatar-HCC', '/Subject': get_value('figures.annotation.figure_01')})
with stem.with_suffix('.pdf').open('wb') as f:
    writer.write(f)

def scientific_lines(text):
    headings = '|'.join((re.escape(h) for h in ['Survival and follow-up', 'Event and censoring', 'Objective response', 'Patient-level outcomes', 'PFS by cohort', 'OS by cohort', 'PFS by response', 'OS by response']))
    lines = [re.sub('^[a-i] (?=(?:' + headings + ')(?:\\s|$))', '', line.strip()) for line in text.splitlines() if line.strip() and line.strip() not in 'abcdefghi']
    return Counter(lines)
output_reader = PdfReader(stem.with_suffix('.pdf'))
assert len(output_reader.pages) == 1
baseline_reader = PdfReader(CLINICAL.with_suffix('.pdf'))
expected_lines = scientific_lines(cohort_pdf.pages[0].extract_text()) + scientific_lines(baseline_reader.pages[0].extract_text())
assert expected_lines == scientific_lines(output_reader.pages[0].extract_text())
assert before_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
manifest = {'purpose': get_value('figures.annotation.figure_01_2'), 'study': 'Avatar-HCC', 'canvas_mm': [W, H], 'panel_mapping': {'a': 'Original Fig1A, full content retained; presentation-only style refinement', 'b': 'Fig2a survival and follow-up', 'c': 'Fig2b events and censoring', 'd': 'Fig2c patient-level outcomes', 'e': 'Fig2d objective response', 'f': 'Fig2e PFS by cohort', 'g': 'Fig2f OS by cohort', 'h': 'Fig2g PFS by response', 'i': 'Fig2h OS by response'}, 'checks': {'original_cohort_geometry_preserved': True, 'original_cohort_text_preserved': True, 'complete_Fig1A_retained': True, 'original_clinical_geometry_preserved': True, 'clinical_statistics_text_preserved': True, 'complete_Fig2_retained': True, 'pdf_nonlabel_content_operations_preserved': True, 'pdf_scientific_text_preserved': True, 'source_files_unchanged': True, 'new_analysis': False, 'cohort_style_only_change': True, get_value('figures.config.item_349'): True}, 'cohort_style_audit': get_value('figures.config.item_215'), 'cohort_style_changes': ['26 panel-a text labels enlarged by 15%', 'Six inclusion/exclusion text labels unchanged', 'Panel-a letter size retained to match b-i', 'Box fills, borders, text colors and arrow paths retained from v005', 'Text center anchors and baselines retained'], 'font_adjustment': {'scale': 1.15, 'changed_text_count': 26, 'criteria_text_count_unchanged': 6, 'panel_letter_unchanged': True}, 'layout': {'cohort_height_mm': cohort_h, 'gap_mm': GAP_MM, 'clinical_height_mm': clinical_h, 'clinical_empty_top_margin_trim_pt': TRIM_CLINICAL_PT, 'cohort_crop': None, 'clinical_panels_rearranged': False, 'pdf_and_svg_use_matching_uniform_scales': True}, 'pdf_panel_letter_updates': pdf_label_updates, 'source_files': [{'path': str(p), 'sha256': before_hashes[str(p)]} for p in SOURCES], 'output_files': [stem.with_suffix('.svg').name, stem.with_suffix('.pdf').name]}
(OUT / get_value('figures.config.item_059')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
