import argparse
from pathlib import Path
from lxml import etree
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject
from _common import prepare_outputs, check_inputs_unchanged, digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-pdf', required=True, type=Path)
    parser.add_argument('--source-svg', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    args = parser.parse_args()
    targets, hashes = prepare_outputs(args.outdir, ['Avatar-HCC_NM_Supplementary_Figure_S1.pdf', 'Avatar-HCC_NM_Supplementary_Figure_S1.svg'], [args.source_pdf, args.source_svg])
    pdf_target, svg_target = targets
    reader = PdfReader(args.source_pdf)
    if len(reader.pages) != 1:
        raise ValueError('The approved workflow source must have exactly one page')
    page = reader.pages[0]
    before_text = page.extract_text()
    if not before_text.startswith('b\n'):
        raise ValueError('The expected standalone panel-b source was not supplied')
    stream = ContentStream(page.get_contents(), reader)
    matches = [i for i, (values, operator) in enumerate(stream.operations) if operator == b'Tj' and values == ['b']]
    if len(matches) != 1:
        raise ValueError('Expected exactly one panel-b text drawing operation')
    before_operations = list(stream.operations)
    del stream.operations[matches[0]]
    expected = before_operations[:matches[0]] + before_operations[matches[0] + 1:]
    if stream.operations != expected:
        raise RuntimeError('An unexpected PDF operation changed')
    page[NameObject('/Contents')] = stream
    svg = etree.fromstring(args.source_svg.read_bytes())
    labels = svg.xpath('//*[@data-qa-name="panel-b"]')
    if len(labels) != 1:
        raise ValueError('Expected exactly one panel-b SVG label')
    labels[0].getparent().remove(labels[0])
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': 'Avatar-HCC NM Supplementary Figure S1 Study workflow', '/Creator': 'Avatar-HCC'})
    with pdf_target.open('xb') as output:
        writer.write(output)
    etree.ElementTree(svg).write(str(svg_target), encoding='utf-8', xml_declaration=True)
    after_text = PdfReader(pdf_target).pages[0].extract_text()
    if after_text.strip() != before_text.removeprefix('b\n').strip():
        raise RuntimeError('Scientific workflow text changed')
    check_inputs_unchanged(hashes)
if __name__ == '__main__':
    main()
