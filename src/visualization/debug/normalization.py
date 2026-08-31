"""
Shows the effect of normalization on
two videos with very different scale/position.

Left:   video A, ORIGINAL keypoints (pixel coordinates)
Center: video B, ORIGINAL keypoints (pixel coordinates)
Right:  both skeletons AFTER normalization, overlaid on the same axes --
        they should come out nearly coincident, demonstrating the effect
        of normalization
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import normalize_keypoints


VIDEO_A_PKL = "outputs/keypoints/gLO_yt_04.pkl"
VIDEO_B_PKL = "outputs/keypoints/gHO_yt_02.pkl"

OUT_PATH = "outputs/normalization/normalizzazione_confronto.mp4"
PLAYBACK_FPS = 30
N_FRAMES_TO_SHOW = 420  


_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
JOINTS = list(range(17))  


def load_keypoints(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["keypoints2d"][0]


def flip_y(keypoints, reference):
    """Converts the y coordinate from 'image convention' (0 at the top,
    increasing downward) to 'standard Cartesian convention' (0 at the
    bottom, increasing upward): new_y = reference - old_y. Needed to use
    normally-oriented axes while keeping the skeleton right-side up
    (not upside down)."""
    kp = keypoints.copy()
    kp[..., 1] = reference - kp[..., 1]
    return kp


def make_artists(ax, linestyle="-", alpha=1.0):
    points = [ax.scatter([], [], s=25, color=np.array(_COLORS[i]) / 255.0,
                          alpha=alpha, zorder=3) for i in JOINTS]
    lines = [ax.plot([], [], color="black", linewidth=1.5, linestyle=linestyle,
                      alpha=alpha, zorder=2)[0] for _ in SKELETON]
    return points, lines


def update_panel(frame, kp, points, lines):
    x, y = kp[frame, :, 0], kp[frame, :, 1]
    for i, p in zip(JOINTS, points):
        p.set_offsets([[x[i], y[i]]])
    for line, (a, b) in zip(lines, SKELETON):
        line.set_data([x[a], x[b]], [y[a], y[b]])


def main():
    kp_a = load_keypoints(VIDEO_A_PKL)[:N_FRAMES_TO_SHOW]
    kp_b = load_keypoints(VIDEO_B_PKL)[:N_FRAMES_TO_SHOW]
    n_frames = min(len(kp_a), len(kp_b))
    kp_a, kp_b = kp_a[:n_frames], kp_b[:n_frames]

    kp_a_norm = normalize_keypoints(kp_a)
    kp_b_norm = normalize_keypoints(kp_b)

    # Range SHARED between the two videos, based on the REAL keypoints
    # (not the whole video frame): reduces empty whitespace, while still
    # keeping the scale comparable between the two videos.
    all_x = np.concatenate([kp_a[:, JOINTS, 0].flatten(), kp_b[:, JOINTS, 0].flatten()])
    all_y = np.concatenate([kp_a[:, JOINTS, 1].flatten(), kp_b[:, JOINTS, 1].flatten()])
    raw_x_min, raw_x_max = np.nanmin(all_x), np.nanmax(all_x)
    raw_y_min, raw_y_max = np.nanmin(all_y), np.nanmax(all_y)
    margin = 0.15 * max(raw_x_max - raw_x_min, raw_y_max - raw_y_min)

    # Flips y (see flip_y): allows standard-oriented axes (0 at bottom
    # left) while keeping the skeleton right-side up.
    kp_a_plot = flip_y(kp_a, raw_y_max)
    kp_b_plot = flip_y(kp_b, raw_y_max)
    y_plot_max = raw_y_max - raw_y_min  # extent after the flip

    # Width of each column proportional to the REAL width/height ratio
    # of that panel's data -- so no panel needs to be shrunk (empty
    # margins) or cropped (lost data) to respect the correct proportions.
    aspect_raw = ((raw_x_max - raw_x_min) + 2 * margin) / (y_plot_max + 2 * margin)
    aspect_norm = 1.0  # symmetric -norm_range/+norm_range range on both axes

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[aspect_raw, aspect_raw, aspect_norm])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_norm = fig.add_subplot(gs[2])

    for ax, title in [(ax_a, "Video A (originale)"), (ax_b, "Video B (originale)")]:
        ax.set_xlim(raw_x_min - margin, raw_x_max + margin)
        ax.set_ylim(0, y_plot_max + margin)  # standard orientation: 0 at the bottom
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

  
    # This is a placeholder, update it with the value for the 
    # actually used video pair. If left too small for the chosen 
    # videos, the right-hand ("after normalization") panel 
    # can silently clip some joints out of view
    norm_range = 1.0

    # Same flip applied to the normalized keypoints (already centered on
    # 0, so here the flip is simply a sign change).
    kp_a_norm_plot = flip_y(kp_a_norm, 0)
    kp_b_norm_plot = flip_y(kp_b_norm, 0)

    ax_norm.set_xlim(-norm_range, norm_range)
    ax_norm.set_ylim(-norm_range, norm_range)  # standard orientation
    ax_norm.set_aspect("equal", adjustable="box")
    ax_norm.set_title("Dopo la normalizzazione")

    points_a, lines_a = make_artists(ax_a)
    points_b, lines_b = make_artists(ax_b)
    # Identical style for both: the whole point here is to show that,
    # after normalization, they become visually indistinguishable.
    points_norm_a, lines_norm_a = make_artists(ax_norm)
    points_norm_b, lines_norm_b = make_artists(ax_norm)

    def update(frame):
        update_panel(frame, kp_a_plot, points_a, lines_a)
        update_panel(frame, kp_b_plot, points_b, lines_b)
        update_panel(frame, kp_a_norm_plot, points_norm_a, lines_norm_a)
        update_panel(frame, kp_b_norm_plot, points_norm_b, lines_norm_b)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / PLAYBACK_FPS)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    anim.save(OUT_PATH, writer="ffmpeg", fps=PLAYBACK_FPS)
    plt.close(fig)
    print(f"Salvato -> {OUT_PATH}")


if __name__ == "__main__":
    main()