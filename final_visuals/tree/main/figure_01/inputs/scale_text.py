from avatar_paths import get_value, get_column
from pathlib import Path
from decimal import Decimal
from io import BytesIO
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ContentStream, DecodedStreamObject, NameObject
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / get_value('figures.module_path.inputs')
DST = Path(__file__).resolve().parent
FACTOR = Decimal('1.15')
SVG_SRC = SRC / get_value('figures.config.item_006')
PDF_SRC = SRC / get_value('figures.config.item_007')
SVG_OUT = DST / get_value('figures.config.item_008')
PDF_OUT = DST / get_value('figures.config.item_009')
SVG_NS = get_value('figures.config.item_380')
FONT_RX = re.compile(get_value('figures.config.item_360'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fmt(n):
    return format(Decimal(str(n)), '.12f').rstrip('0').rstrip('.') or '0'

def frozen(gid):
    return gid == 'qa:a:panel-label' or gid.startswith('qa:a:inclusion-line-') or gid.startswith('qa:a:exclusion-line-')

def svg_records(raw):
    tree = ET.fromstring(raw)
    rows = []
    for group in tree.iter(SVG_NS + 'g'):
        gid = group.get('id', '')
        if gid.startswith('qa:a:'):
            text = group.find(SVG_NS + 'text')
            assert text is not None
            rows.append({'id': gid, 'text': ''.join(text.itertext()), 'x': text.get('x'), 'y': text.get('y'), 'style': text.get('style'), 'font_size': float(FONT_RX.search(text.get('style')).group(2))})
    return (tree, rows)

def width_map(font):
    subfont = font['/DescendantFonts'][0].get_object()
    seq = subfont.get('/W', [])
    seq = seq.get_object() if hasattr(seq, 'get_object') else seq
    widths = {}
    i = 0
    while i < len(seq):
        start = int(seq[i])
        spec = seq[i + 1]
        if isinstance(spec, ArrayObject):
            for j, val in enumerate(spec):
                widths[start + j] = float(val)
            i += 2
        else:
            end = int(spec)
            val = float(seq[i + 2])
            for cid in range(start, end + 1):
                widths[cid] = val
            i += 3
    return (widths, float(subfont.get('/DW', 1000)))

def text_bytes(obj):
    return obj.original_bytes if hasattr(obj, 'original_bytes') else bytes(obj)

def advance(tj, font, size):
    widths, default = width_map(font)
    total = 0.0
    for obj in tj:
        if isinstance(obj, (int, float)):
            total -= float(obj)
        elif hasattr(obj, 'original_bytes') or isinstance(obj, bytes):
            raw = text_bytes(obj)
            assert len(raw) % 2 == 0
            total += sum((widths.get(int.from_bytes(raw[j:j + 2], 'big'), default) for j in range(0, len(raw), 2)))
        else:
            total -= float(obj)
    return total * size / 1000.0

def pdf_text_blocks(page):
    operations = ContentStream(page.get_contents(), page.pdf).operations
    blocks = []
    last_cm = None
    block = None
    for i, (args, op) in enumerate(operations):
        if op == b'cm':
            last_cm = (i, [float(v) for v in args])
        elif op == b'BT':
            block = {'cm_index': last_cm[0], 'cm': last_cm[1], 'tj': [], 'tf_index': None}
        elif block is not None and op == b'Tf':
            block.update(font=str(args[0]), size=float(args[1]), tf_index=i)
        elif block is not None and op == b'TJ':
            block['tj'].extend(args[0])
        elif block is not None and op == b'ET':
            raw = b''.join((text_bytes(o) for o in block['tj'] if hasattr(o, 'original_bytes') or isinstance(o, bytes)))
            block['text'] = raw.decode('utf-16-be')
            blocks.append(block)
            block = None
    return (operations, blocks)

def main():
    DST.mkdir(parents=True, exist_ok=True)
    before_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (SVG_SRC, PDF_SRC)}
    svg_raw = SVG_SRC.read_bytes()
    _, originals = svg_records(svg_raw)
    assert len(originals) == 33
    enlarged = [r for r in originals if not frozen(r['id'])]
    unchanged = [r for r in originals if frozen(r['id'])]
    assert len(enlarged) == 26 and len(unchanged) == 7

    def replace_svg_group(m):
        chunk = m.group(0)
        if frozen(m.group('gid')):
            return chunk
        changed, count = FONT_RX.subn(lambda f: f.group(1) + fmt(Decimal(f.group(2)) * FACTOR) + 'px', chunk)
        assert count == 1
        return changed
    svg_text = svg_raw.decode('utf-8')
    group_rx = re.compile('<g id="(?P<gid>qa:a:[^"]+)">\\s*<text\\b.*?</text>\\s*</g>', re.S)
    svg_new = group_rx.sub(replace_svg_group, svg_text)
    SVG_OUT.write_text(svg_new, encoding='utf-8', newline='')
    before_tree, _ = svg_records(svg_raw)
    after_tree, after_rows = svg_records(SVG_OUT.read_bytes())
    for group in after_tree.iter(SVG_NS + 'g'):
        gid = group.get('id', '')
        if gid.startswith('qa:a:') and (not frozen(gid)):
            text = group.find(SVG_NS + 'text')
            original = next((r for r in originals if r['id'] == gid))
            text.set('style', original['style'])
    svg_normalized_equal = ET.tostring(before_tree) == ET.tostring(after_tree)
    reader = PdfReader(PDF_SRC)
    page = reader.pages[0]
    operations, pdf_blocks = pdf_text_blocks(page)
    assert len(pdf_blocks) == len(originals)
    data = page.get_contents().get_data()
    source_stream_hash = hashlib.sha256(data).hexdigest()
    block_rx = re.compile(b'q\\s+(?P<a>[-+\\d.]+)\\s+(?P<b>[-+\\d.]+)\\s+(?P<c>[-+\\d.]+)\\s+(?P<d>[-+\\d.]+)\\s+(?P<tx>[-+\\d.]+)\\s+(?P<ty>[-+\\d.]+)\\s+cm\\s+BT(?P<body>.*?)ET\\s+Q', re.S)
    stream_matches = list(block_rx.finditer(data))
    assert len(stream_matches) == 33
    changes = []
    replacements = []
    allowed_operation_indices = set()
    center_differences = []
    for idx, (row, block, match) in enumerate(zip(originals, pdf_blocks, stream_matches)):
        assert block['text'] == row['text'], (idx, row, block['text'])
        assert abs(block['size'] - row['font_size']) < 1e-09
        if frozen(row['id']):
            continue
        font = page['/Resources']['/Font'][block['font']].get_object()
        old_size = block['size']
        new_size = float(Decimal(str(old_size)) * FACTOR)
        old_advance = advance(block['tj'], font, old_size)
        new_advance = advance(block['tj'], font, new_size)
        old_tx = float(match.group('tx'))
        new_tx = old_tx - (new_advance - old_advance) / 2.0
        tx_bytes = fmt(new_tx).encode('ascii')
        center_differences.append(abs(old_tx + old_advance / 2 - (float(tx_bytes) + new_advance / 2)))
        tf_rx = re.compile(b'(?P<font>/F\\d+\\s+)(?P<size>[-+\\d.]+)(?P<op>\\s+Tf)')
        tf = tf_rx.search(match.group('body'))
        assert tf is not None
        size_start = match.start('body') + tf.start('size')
        size_end = match.start('body') + tf.end('size')
        replacements.extend([(match.start('tx'), match.end('tx'), tx_bytes), (size_start, size_end, fmt(new_size).encode('ascii'))])
        allowed_operation_indices.update([block['cm_index'], block['tf_index']])
        changes.append({'id': row['id'], 'text': row['text'], 'old_font_size_pt': old_size, 'new_font_size_pt': new_size, 'scale': 1.15, 'center_x_pt': old_tx + old_advance / 2, 'baseline_y_pdf_pt': float(match.group('ty')), 'old_advance_pt': old_advance, 'new_advance_pt': new_advance, 'old_pdf_tx_pt': old_tx, 'new_pdf_tx_pt': float(tx_bytes)})
    result = data
    for start, end, replacement in sorted(replacements, reverse=True):
        result = result[:start] + replacement + result[end:]
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    stream = DecodedStreamObject()
    stream.set_data(result)
    writer.pages[0][NameObject('/Contents')] = writer._add_object(stream)
    with PDF_OUT.open('wb') as f:
        writer.write(f)
    out_reader = PdfReader(PDF_OUT)
    out_operations, out_blocks = pdf_text_blocks(out_reader.pages[0])
    assert len(out_operations) == len(operations)
    other_operations_equal = all((a == b for i, (a, b) in enumerate(zip(operations, out_operations)) if i not in allowed_operation_indices))
    operation_changes_exact = all((operations[i][1] == out_operations[i][1] for i in allowed_operation_indices))
    for i, (old_op, new_op) in enumerate(zip(operations, out_operations)):
        if i not in allowed_operation_indices:
            continue
        old_args, op = old_op
        new_args, _ = new_op
        if op == b'Tf':
            operation_changes_exact &= old_args[0] == new_args[0] and abs(float(new_args[1]) - float(old_args[1]) * 1.15) < 1e-10
        elif op == b'cm':
            operation_changes_exact &= all((old_args[j] == new_args[j] for j in (0, 1, 2, 3, 5)))
        else:
            operation_changes_exact = False
    text_preserved = [r['text'] for r in originals] == [r['text'] for r in after_rows] == [r['text'] for r in out_blocks]
    positions_preserved = all(((a['x'], a['y']) == (b['x'], b['y']) for a, b in zip(originals, after_rows))) and max(center_differences) < 1e-09
    criteria_preserved = all((a['style'] == b['style'] and old_pdf['size'] == new_pdf['size'] and (old_pdf['cm'] == new_pdf['cm']) for a, b, old_pdf, new_pdf in zip(originals, after_rows, pdf_blocks, out_blocks) if frozen(a['id'])))
    only_allowed = operation_changes_exact and all((abs(b['font_size'] - a['font_size'] * (1 if frozen(a['id']) else 1.15)) < 1e-09 for a, b in zip(originals, after_rows)))
    hashes_after = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (SVG_SRC, PDF_SRC)}
    checks = {'cohort_text_preserved': text_preserved, 'cohort_geometry_preserved': svg_normalized_equal and other_operations_equal, 'cohort_text_positions_preserved': positions_preserved, 'cohort_criteria_font_sizes_preserved': criteria_preserved, 'only_allowed_text_enlarged': only_allowed, 'pdf_other_operations_preserved': other_operations_equal, 'source_files_unchanged': before_hashes == hashes_after}
    assert all(checks.values()), checks
    audit = {'change': 'Uniform 15% enlargement of cohort text outside inclusion/exclusion criteria; panel label unchanged.', 'new_analysis': False, 'changes': changes, 'unchanged': unchanged, 'number_of_enlarged_texts': len(changes), 'number_of_unchanged_texts': len(unchanged), 'source_hashes_before': before_hashes, 'source_hashes_after': hashes_after, 'source_pdf_content_sha256': source_stream_hash, 'output_pdf_content_sha256': hashlib.sha256(out_reader.pages[0].get_contents().get_data()).hexdigest(), 'pdf_center_alignment_method': 'Original embedded Arial CID widths (/DescendantFonts /W) and TJ kerning; tx reduced by half the advance increase.', 'pdf_max_center_difference_pt': max(center_differences), 'changed_pdf_operation_count': len(allowed_operation_indices), 'checks': checks, 'outputs': [str(p.relative_to(ROOT)).replace('\\', '/') for p in (SVG_OUT, PDF_OUT)]}
    (DST / get_value('figures.config.item_214')).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
if __name__ == '__main__':
    main()
