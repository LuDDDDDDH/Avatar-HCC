import argparse
from copy import deepcopy
from pathlib import Path
import re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject, ArrayObject, FloatObject, ByteStringObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
from _common import prepare_outputs, check_inputs_unchanged, digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pdf', required=True, type=Path)
    parser.add_argument('--font', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--png-dpi', type=int, default=600)
    args = parser.parse_args()
    if not 72 <= args.png_dpi <= 1200:
        raise ValueError('PNG resolution must be between 72 and 1200 dpi')
    targets, hashes = prepare_outputs(args.outdir, ['Avatar-HCC_NM_Supplementary_Figure_S2.pdf', f'Avatar-HCC_NM_Supplementary_Figure_S2_{args.png_dpi}dpi.png'], [args.source_pdf, args.font])
    pdf_target, png_target = targets
    pdfmetrics.registerFont(TTFont('AvatarArial', str(args.font)))
    reader = PdfReader(args.source_pdf)
    if len(reader.pages) != 1:
        raise ValueError('The approved S2 source must have exactly one page')
    page = reader.pages[0]
    stream = ContentStream(page.get_contents(), reader)
    original = deepcopy(stream.operations)
    operations = deepcopy(original)
    wrapped = {'Pooled PCA fusion': ('Pooled PCA', 'fusion'), 'Block-PCA early fusion': ('Block-PCA', 'early fusion'), 'Equal-weight late fusion': ('Equal-weight', 'late fusion')}
    targets_to_edit = []
    for index, (values, operator) in enumerate(operations):
        if operator != b'TJ' or index < 4 or operations[index - 4][1] != b'cm':
            continue
        text = ''.join((str(v) for v in values[0] if isinstance(v, str)))
        transform = operations[index - 4][0]
        x, y = (float(transform[4]), float(transform[5]))
        if text in wrapped and 370 < y < 410:
            targets_to_edit.append((index, text, 'wrap'))
        elif text in ('-0.03', '-0.07') and 375 < y < 400 and (x < 80):
            targets_to_edit.append((index, text, 'move'))
    if len(targets_to_edit) != 6:
        raise ValueError(f'Expected six original label targets; found {len(targets_to_edit)}')
    changes = []
    for index, text, kind in reversed(targets_to_edit):
        transform = list(operations[index - 4][0])
        before = [float(v) for v in transform]
        if kind == 'move':
            transform[4] = FloatObject(74.0)
        else:
            first, second = wrapped[text]
            width1 = pdfmetrics.stringWidth(first, 'AvatarArial', 5.6)
            width2 = pdfmetrics.stringWidth(second, 'AvatarArial', 5.6)
            transform[4] = FloatObject(177.054867 - width1)
            transform[5] = FloatObject(float(transform[5]) + 2.9)
            operations[index:index + 1] = [([ArrayObject([ByteStringObject(first.encode('utf-16-be'))])], b'TJ'), ([FloatObject(width1 - width2), FloatObject(-5.8)], b'Td'), ([ArrayObject([ByteStringObject(second.encode('utf-16-be'))])], b'TJ')]
        operations[index - 4] = (transform, b'cm')
        changes.append({'text': text, 'edit': kind, 'before_transform': before, 'after_transform': [float(v) for v in transform]})
    plot_operators = {b'm', b'l', b'c', b're', b'S', b'f', b'f*', b'rg', b'RG', b'w', b'd', b'W', b'W*'}
    before_plot = [(v, o) for v, o in original if o in plot_operators]
    after_plot = [(v, o) for v, o in operations if o in plot_operators]
    if before_plot != after_plot:
        raise RuntimeError('Scientific chart geometry or colours changed')
    stream.operations = operations
    page[NameObject('/Contents')] = stream
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'Avatar-HCC NM Supplementary Figure S2', '/Creator': 'Avatar-HCC'})
    with pdf_target.open('xb') as output:
        writer.write(output)
    normalize = lambda text: re.sub('\\s+', '', text)
    before_text = normalize(PdfReader(args.source_pdf).pages[0].extract_text())
    after_text = normalize(PdfReader(pdf_target).pages[0].extract_text())
    if before_text != after_text:
        raise RuntimeError('Scientific text or numeric labels changed')
    with pdfium.PdfDocument(str(pdf_target)) as rendered:
        image = rendered[0].render(scale=args.png_dpi / 72).to_pil().convert('RGB')
        image.save(png_target, dpi=(args.png_dpi, args.png_dpi))
    check_inputs_unchanged(hashes)
if __name__ == '__main__':
    main()
