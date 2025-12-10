#!/usr/bin/env python3
"""Generate a PDF showing iteration progression for a single instance."""

import re
from pathlib import Path
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# === CONFIGURATION ===
PLOTS_DIR = "R20_Alpha2/plots"
TARGETS = 5
OBSTACLES = 15
SEED = 15
# =====================

PAGE_WIDTH = 9 * inch
PAGE_HEIGHT = 9.5 * inch


def generate_progression_pdf(plots_dir, targets, obstacles, seed):
    base_path = Path(plots_dir)
    folder_name = f"T{targets}_O{obstacles}_seed{seed}"
    folder_path = base_path / folder_name

    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    files = []

    gtsp_file = folder_path / "iter_00_gtsp.png"
    if gtsp_file.exists():
        files.append((gtsp_file, "Iteration 0 (GTSP)"))

    constraint_file = folder_path / "iter_00_update_constraint.png"
    if constraint_file.exists():
        files.append((constraint_file, "Iteration 0 (Update Constraint)"))

    iter_files = sorted(
        [f for f in folder_path.glob("iter_*.png") if re.match(r'iter_\d+\.png$', f.name)],
        key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1))
    )
    for f in iter_files:
        iter_num = int(re.search(r'iter_(\d+)', f.name).group(1))
        files.append((f, f"Iteration {iter_num}"))

    if not files:
        raise ValueError(f"No iteration files found in {folder_path}")

    print(f"Found {len(files)} iteration files in {folder_name}")

    output_path = folder_path / f"{folder_name}_progression.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
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
    for i, (img_path, iter_label) in enumerate(files):
        title_text = f"T{targets} O{obstacles} S{seed} — {iter_label}"
        story.append(Paragraph(title_text, title_style))

        img = Image(str(img_path))
        scale = min(max_img_width / img.imageWidth, max_img_height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = 'CENTER'
        story.append(img)

        if i < len(files) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    generate_progression_pdf(PLOTS_DIR, TARGETS, OBSTACLES, SEED)
