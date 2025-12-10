#!/usr/bin/env python3
"""Generate a video showing iteration progression for a single instance."""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# === CONFIGURATION ===
PLOTS_DIR = "../R20_Alpha2/plots"
SECONDS_PER_IMAGE = 1
FADE_DURATION = 0.5
FPS = 30


# =====================


def generate_iteration_video(plots_dir, targets, obstacles, seed, seconds_per_image=1.5, fade_duration=0.5, fps=30):
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

    print(f"Found {len(files)} iteration images")

    images = []
    for img_path, label in files:
        img = mpimg.imread(str(img_path))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        images.append((img, label))

    frames_per_image = int(seconds_per_image * fps)
    fade_frames = int(fade_duration * fps)
    total_frames = len(images) * frames_per_image + (len(images) - 1) * fade_frames

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    im = ax.imshow(images[0][0])
    ax.set_title(f"{targets} Targets, {obstacles} Obstacles (Seed {seed})", fontsize=14)

    def animate(frame):
        segment_length = frames_per_image + fade_frames
        img_idx = frame // segment_length
        frame_in_segment = frame % segment_length

        if img_idx >= len(images) - 1:
            img_idx = len(images) - 1
            im.set_array(images[img_idx][0])
        elif frame_in_segment < frames_per_image:
            im.set_array(images[img_idx][0])
        else:
            fade_progress = (frame_in_segment - frames_per_image) / fade_frames
            blended = (1 - fade_progress) * images[img_idx][0] + fade_progress * images[img_idx + 1][0]
            im.set_array(blended)

        return [im]

    output_gif = folder_path / f"{folder_name}_iterations.gif"
    output_mp4 = folder_path / f"{folder_name}_iterations.mp4"

    print(f"Creating animation with {total_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps, blit=True)

    gif_writer = PillowWriter(fps=fps)
    anim.save(str(output_gif), writer=gif_writer, dpi=100)
    print(f"GIF saved: {output_gif}")

    mp4_writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_mp4), writer=mp4_writer, dpi=100)
    print(f"MP4 saved: {output_mp4}")

    plt.close(fig)


if __name__ == "__main__":
    targets_list = [5]
    obstacles_list = [15]
    seeds = [15]

    for targets in targets_list:
        for obstacles in obstacles_list:
            for seed in seeds:
                generate_iteration_video(PLOTS_DIR, targets, obstacles, seed, SECONDS_PER_IMAGE, FADE_DURATION, FPS)