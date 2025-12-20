#!/usr/bin/env python3
"""Generate a PDF with iter_25.png from each experiment folder, one figure per page."""

# === CONFIGURATION ===
BASE_DIR = "R20_Alpha2/plots"
OUTPUT_PDF = "figures_summary.pdf"
# =====================

import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def parse_folder_name(folder_name):
    match = re.match(r'T(\d+)_O(\d+)_seed(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def generate_pdf(base_dir, output_path):
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Directory not found: {base_dir}")

    folders = []
    for item in base_path.iterdir():
        if item.is_dir() and parse_folder_name(item.name):
            img_path = item / "iter_25.png"
            if img_path.exists():
                folders.append((item.name, img_path))
            else:
                print(f"Warning: iter_25.png not found in {item.name}")

    if not folders:
        raise ValueError("No valid experiment folders found!")

    folders.sort(key=lambda x: parse_folder_name(x[0]))
    print(f"Found {len(folders)} experiment folders with iter_25.png")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch
    )

    title_style = ParagraphStyle(
        'FigureTitle',
        parent=getSampleStyleSheet()['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    page_width, page_height = letter
    max_img_width = page_width - 1 * inch
    max_img_height = page_height - 1.5 * inch  # leave room for title

    story = []
    for i, (folder_name, img_path) in enumerate(folders):
        targets, obstacles, seed = parse_folder_name(folder_name)

        title_text = f"Targets: {targets}, Obstacles: {obstacles}, Seed: {seed}"
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
    generate_pdf(BASE_DIR, OUTPUT_PDF)