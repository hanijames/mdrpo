#!/usr/bin/env python3
"""Generate a PDF with the final iteration from each instance, one per page."""

import re
from pathlib import Path
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# === CONFIGURATION ===
PLOTS_DIR = "R20_Alpha2/plots"
OUTPUT_PDF = "R20_Alpha2/summary.pdf"
FINAL_ITER = 25
# =====================

PAGE_WIDTH = 9 * inch
PAGE_HEIGHT = 9.5 * inch


def parse_folder_name(folder_name):
    match = re.match(r'T(\d+)_O(\d+)_seed(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def generate_pdf(plots_dir, output_path, final_iter):
    base_path = Path(plots_dir)
    if not base_path.exists():
        raise ValueError(f"Directory not found: {plots_dir}")

    folders = []
    for item in base_path.iterdir():
        if item.is_dir() and parse_folder_name(item.name):
            img_path = item / f"iter_{final_iter:02d}.png"
            if img_path.exists():
                folders.append((item.name, img_path))
            else:
                print(f"Warning: iter_{final_iter:02d}.png not found in {item.name}")

    if not folders:
        raise ValueError("No valid experiment folders found!")

    folders.sort(key=lambda x: parse_folder_name(x[0]))
    print(f"Found {len(folders)} instances with iter_{final_iter:02d}.png")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        topMargin=0.25 * inch,
        bottomMargin=0.25 * inch,
        leftMargin=0.25 * inch,
        rightMargin=0.25 * inch
    )

    title_style = ParagraphStyle(
        'FigureTitle',
        parent=getSampleStyleSheet()['Heading1'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    max_img_width = PAGE_WIDTH - 0.5 * inch
    max_img_height = PAGE_HEIGHT - 0.75 * inch

    story = []
    for i, (folder_name, img_path) in enumerate(folders):
        targets, obstacles, seed = parse_folder_name(folder_name)

        title_text = f"T{targets} O{obstacles} S{seed}"
        story.append(Paragraph(title_text, title_style))

        img = Image(str(img_path))
        scale = min(max_img_width / img.imageWidth, max_img_height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = 'CENTER'
        story.append(img)

        if i < len(folders) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    generate_pdf(PLOTS_DIR, OUTPUT_PDF, FINAL_ITER)
