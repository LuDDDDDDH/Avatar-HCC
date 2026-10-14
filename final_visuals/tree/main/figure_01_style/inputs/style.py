from avatar_paths import get_value, get_column
from pathlib import Path
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject, DecodedStreamObject
BASE = Path(__file__).resolve().parents[3]
SOURCE = BASE / get_value('figures.module_path.configured')
OUT = Path(__file__).resolve().parent
STEM = get_value('figures.config.item_331')
PUBLIC = get_value('figures.config.item_347')
SVG_NS = get_value('figures.config.item_379')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rgb(hex_color):
    return tuple((int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)))

def tint(hex_color, strength):
    return '#' + ''.join((f'{round(255 * (1 - strength + c * strength)):02x}' for c in rgb(hex_color)))

def style_replace(style, key, value):
    pattern = f'(?<![\\w-]){re.escape(key)}:\\s*[^;]+'
    assert re.search(pattern, style), (key, style)
    return re.sub(pattern, f'{key}: {value}', style)
COLORS = {'SCH': '#2f6fb0', 'QLH': '#2a8f78', 'PUMCH': '#8f59c0'}
FILL_COLORS = {15: '#edf2f6', 22: '#f0f4f7', 23: '#fbefef'}
for center, header, source, data, final in [('SCH', 16, 17, 24, 25), ('QLH', 18, 19, 26, 27), ('PUMCH', 20, 21, 28, 29)]:
    for patch, strength in [(header, 0.12), (source, 0.08), (data, 0.08), (final, 0.15)]:
        FILL_COLORS[patch] = tint(COLORS[center], strength)
FINAL_TINTS = {patch: FILL_COLORS[patch] for patch in (25, 27, 29)}
TEXT_COLORS = {'SCH internal discovery cohort': COLORS['SCH'], 'QLH external validation cohort': COLORS['QLH'], 'PUMCH external validation cohort': COLORS['PUMCH'], 'n = 109': COLORS['SCH'], 'CT, n = 200 | Blood, n = 72': COLORS['QLH'], 'Urine, n = 33': COLORS['QLH'], 'n = 133': COLORS['PUMCH'], 'Key exclusion criteria': '#b25f66'}
OUT.mkdir(parents=True, exist_ok=True)
source_hashes = {suffix: digest(SOURCE / f'{STEM}.{suffix}') for suffix in ('svg', 'pdf')}
source_svg = SOURCE / f'{STEM}.svg'
old_svg = ET.parse(source_svg).getroot()
svg = copy.deepcopy(old_svg)
for group in svg.iter(f'{{{SVG_NS}}}g'):
    gid = group.get('id', '')
    if gid.startswith('patch_') and 15 <= int(gid.removeprefix('patch_')) <= 29:
        number = int(gid.removeprefix('patch_'))
        path = group.find(f'{{{SVG_NS}}}path')
        assert path is not None
        style = path.get('style')
        style = style_replace(style, 'fill', FILL_COLORS[number])
        style = style_replace(style, 'stroke', '#cbd3db')
        style = style_replace(style, 'stroke-width', '0.45')
        path.set('style', style)
    text = group.find(f'{{{SVG_NS}}}text')
    if text is not None:
        content = ''.join(text.itertext())
        if content in TEXT_COLORS:
            text.set('style', style_replace(text.get('style'), 'fill', TEXT_COLORS[content]))
ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', get_value('figures.config.item_382'))
out_svg = OUT / f'{PUBLIC}.svg'
ET.ElementTree(svg).write(out_svg, encoding='utf-8', xml_declaration=True)
reader = PdfReader(SOURCE / f'{STEM}.pdf')
page = reader.pages[0]
old_ops = copy.deepcopy(ContentStream(page.get_contents(), reader).operations)
ops = copy.deepcopy(old_ops)
shape_ranges = [(395 + 21 * i, 416 + 21 * i, 15 + i) for i in range(15)]
assert shape_ranges[-1] == (689, 710, 29)
shape_ranges[-1] = (689, 707, 29)
for start, end, patch in shape_ranges:
    for i in range(start, end):
        operands, operator = ops[i]
        if operator == b'rg':
            color = rgb(FILL_COLORS[patch])
            ops[i] = ([FloatObject(c) for c in color], operator)
        elif operator == b'RG':
            ops[i] = ([FloatObject(c) for c in rgb('#cbd3db')], operator)
        elif operator == b'w':
            ops[i] = ([FloatObject(0.45)], operator)
recolored_text = []
patched_ops = []
fill_state = (b'g', [FloatObject(1)])
i = 0
while i < len(ops):
    operands, operator = ops[i]
    if operator in (b'rg', b'g'):
        fill_state = (operator, copy.deepcopy(operands))
    if operator == b'BT':
        j = i + 1
        while ops[j][1] != b'ET':
            j += 1
        block = ops[i:j + 1]
        strings = []
        for args, op in block:
            if op == b'TJ':
                strings.extend((str(x) for x in args[0] if isinstance(x, str)))
            elif op == b'Tj':
                strings.append(str(args[0]))
        content = ''.join(strings)
        if content in TEXT_COLORS:
            patched_ops.append(block[0])
            patched_ops.append(([FloatObject(c) for c in rgb(TEXT_COLORS[content])], b'rg'))
            patched_ops.extend(block[1:])
            patched_ops.append((copy.deepcopy(fill_state[1]), fill_state[0]))
            recolored_text.append(content)
        else:
            patched_ops.extend(block)
        i = j + 1
        continue
    patched_ops.append((operands, operator))
    i += 1
assert set(recolored_text) == set(TEXT_COLORS)
raw = page.get_contents().get_data().decode('latin1')
style_pattern = re.compile('(?P<prefix>Q q 0 0 [\\d.]+ [\\d.]+ re W n )(?P<fill>[.\\d\\s]+) rg 1 j (?P<width>[.\\d]+) w (?P<stroke>[.\\d\\s]+) RG\\s*(?P<fill2>[.\\d\\s]+) rg')
matched_shapes = list(style_pattern.finditer(raw))
assert len(matched_shapes) == 15
shape_index = 0

def pdf_rgb(hex_color):
    return ' '.join((f'{c:.12f}'.rstrip('0').rstrip('.') for c in rgb(hex_color)))

def replace_shape(match):
    global shape_index
    patch = 15 + shape_index
    shape_index += 1
    fill = pdf_rgb(FILL_COLORS[patch])
    return f"{match.group('prefix')}{fill} rg 1 j 0.45 w {pdf_rgb('#cbd3db')} RG {fill} rg"
raw = style_pattern.sub(replace_shape, raw)
block_contents = []
for start, (_, op) in enumerate(old_ops):
    if op == b'BT':
        end = start + 1
        while old_ops[end][1] != b'ET':
            end += 1
        parts = []
        for args, command in old_ops[start:end + 1]:
            if command == b'TJ':
                parts.extend((str(x) for x in args[0] if isinstance(x, str)))
            elif command == b'Tj':
                parts.append(str(args[0]))
        block_contents.append(''.join(parts))
text_block_index = 0

def replace_text_block(match):
    global text_block_index
    content = block_contents[text_block_index]
    text_block_index += 1
    original = match.group(0)
    if content in TEXT_COLORS:
        return 'BT\n' + pdf_rgb(TEXT_COLORS[content]) + ' rg' + original[2:]
    return original
raw = re.sub('\\bBT\\b.*?\\bET\\b', replace_text_block, raw, flags=re.S)
assert text_block_index == len(block_contents)
stream = DecodedStreamObject()
stream.set_data(raw.encode('latin1'))
page[NameObject('/Contents')] = stream
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({'/Title': 'Avatar-HCC cohort flow: visual style revision v005'})
out_pdf = OUT / f'{PUBLIC}.pdf'
with out_pdf.open('wb') as f:
    writer.write(f)
new_pdf = PdfReader(out_pdf)
new_ops = ContentStream(new_pdf.pages[0].get_contents(), new_pdf).operations
STYLE_OPS = {b'rg', b'RG', b'g', b'G', b'w'}

def nonstyle_operations(operations):
    return [(operands, operator) for operands, operator in operations if operator not in STYLE_OPS]

def texts(root):
    return [''.join(t.itertext()) for t in root.iter(f'{{{SVG_NS}}}text')]

def all_nonstyle_xml(root):
    return [(e.tag, sorted(((k, v) for k, v in e.attrib.items() if k != 'style')), e.text, e.tail) for e in root.iter()]
checks = {'cohort_text_preserved': texts(old_svg) == texts(svg), 'cohort_geometry_preserved': all_nonstyle_xml(old_svg) == all_nonstyle_xml(svg), 'cohort_text_positions_preserved': all_nonstyle_xml(old_svg) == all_nonstyle_xml(svg), 'cohort_font_sizes_preserved': all((re.sub(get_value('figures.config.item_368'), '', a.get('style')) == re.sub(get_value('figures.config.item_368'), '', b.get('style')) for a, b in zip(old_svg.iter(f'{{{SVG_NS}}}text'), svg.iter(f'{{{SVG_NS}}}text')))), 'pdf_nonstyle_operations_preserved': nonstyle_operations(old_ops) == nonstyle_operations(new_ops), 'pdf_all_strings_preserved': PdfReader(SOURCE / f'{STEM}.pdf').pages[0].extract_text() == new_pdf.pages[0].extract_text(), 'source_files_unchanged': source_hashes == {suffix: digest(SOURCE / f'{STEM}.{suffix}') for suffix in ('svg', 'pdf')}, 'new_analysis': False}
previous_svg = ET.parse(BASE / get_value('figures.config.item_004')).getroot()

def without_fill(root):
    elements = []
    for node in root.iter():
        attrs = dict(node.attrib)
        if 'style' in attrs:
            attrs['style'] = re.sub(get_value('figures.config.item_373'), '', attrs['style'])
        elements.append((node.tag, sorted(attrs.items()), node.text, node.tail))
    return elements
checks[get_value('figures.config.item_366')] = without_fill(previous_svg) == without_fill(svg)
assert all((value for key, value in checks.items() if key != 'new_analysis')), checks
audit = {'revision': 'v005', 'scope': 'Restore gently colored panel-a box fills relative to v004; all text, outlines, arrows and layout retained.', 'new_analysis': False, 'checks': checks, 'source_sha256': source_hashes, 'text_nodes': len(texts(svg)), 'svg_shapes_restyled': 15, 'pdf_nonstyle_operator_count': len(nonstyle_operations(old_ops)), 'recolored_text': recolored_text, 'final_fill_colors': FINAL_TINTS, 'box_fill_colors': FILL_COLORS, 'tint_strengths': {'heading': 0.12, 'source': 0.08, 'sampling': 0.08, 'final': 0.15}, 'svg': str(out_svg), 'pdf': str(out_pdf)}
(OUT / get_value('figures.config.item_214')).write_text(json.dumps(audit, indent=2), encoding='utf-8')
