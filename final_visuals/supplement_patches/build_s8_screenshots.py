import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from _common import prepare_outputs, check_inputs_unchanged, digest

def main():
    parser = argparse.ArgumentParser()
    for name in ('case-i', 'case-ii', 'case-iii', 'font', 'bold-font', 'outdir'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.case_i, args.case_ii, args.case_iii, args.font, args.bold_font]
    targets, hashes = prepare_outputs(args.outdir, ['Avatar-HCC_NM_Supplementary_Figure_S8.png', 'Avatar-HCC_NM_Supplementary_Figure_S8.pdf'], inputs)
    png_target, pdf_target = targets
    cases = []
    for path in inputs[:3]:
        with Image.open(path) as source:
            if source.width < 2310 or source.height < 1328:
                raise ValueError('Source screenshot is too small for the approved crop geometry')
            cases.append(source.convert('RGB'))
    font = ImageFont.truetype(str(args.font), 25)
    bold = ImageFont.truetype(str(args.bold_font), 34)
    overview = cases[0].crop((336, 96, 2310, 1328))
    width, margin, gap = (1974, 24, 24)
    column_width = (width - 2 * margin - 2 * gap) // 3
    panel_height = 1330
    overview_height = overview.height + 70
    height = overview_height + panel_height + 60
    plate = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(plate)
    draw.text((margin, 12), 'a', font=bold, fill='#152636')
    draw.text((margin + 52, 19), 'Avatar-Agent interface overview', font=font, fill='#152636')
    plate.paste(overview, (0, 65))
    crops = [('patient_core', (1180, 650, 1465, 780)), ('benefit_projections', (1732, 282, 2310, 780)), ('axis_1', (374, 868, 999, 984)), ('axis_2', (1011, 868, 1635, 984)), ('axis_3', (1648, 868, 2272, 984))]
    crop_records = []
    for index, (image, letter, phenotype) in enumerate(zip(cases, 'bcd', ('MBP-I', 'MBP-II', 'MBP-III'))):
        x = margin + index * (column_width + gap)
        y = overview_height + 24
        draw.text((x, y), letter, font=bold, fill='#152636')
        draw.text((x + 45, y + 7), f'{phenotype} representative patient', font=font, fill='#152636')
        y += 60
        for name, box in crops:
            crop = image.crop(box)
            crop_height = round(crop.height * column_width / crop.width)
            plate.paste(crop.resize((column_width, crop_height), Image.Resampling.LANCZOS), (x, y))
            y += crop_height + 14
            crop_records.append({'panel': letter, 'region': name, 'source_pixel_box': list(box)})
        if y >= height - 12:
            raise RuntimeError('Approved screenshot composition exceeds the page')
    plate.save(png_target, dpi=(300, 300))
    pdf_width = 720
    pdf_height = pdf_width * height / width
    output = canvas.Canvas(str(pdf_target), pagesize=(pdf_width, pdf_height))
    output.setTitle('Avatar-HCC Supplementary Figure S8')
    output.setAuthor('Avatar-HCC')
    output.drawImage(ImageReader(plate), 0, 0, width=pdf_width, height=pdf_height)
    output.save()
    check_inputs_unchanged(hashes)
if __name__ == '__main__':
    main()
