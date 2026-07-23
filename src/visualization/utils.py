import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import normalize_keypoints

_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

N_KEYPOINTS = 17


def load_keypoints(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    keypoints = data["keypoints2d"][0]  # (T, 17, 3)
    fps = data.get("fps", 60)
    return keypoints, fps


def plot_skeleton_with_timeseries(keypoints, fps, feature_values, title, ylabel,
                                   out_path, feature_label="feature",
                                   hlines=None):
    kp = normalize_keypoints(keypoints)
    x, y = kp[:, :, 0], kp[:, :, 1]

    n_frames = len(x)

    fig, (ax_skel, ax_feat) = plt.subplots(1, 2, figsize=(11, 5))

    margin = 0.2
    tick_step = 0.5

    raw_half_range = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) + margin
    half_range = np.ceil(raw_half_range / tick_step) * tick_step

    ax_skel.set_title("Scheletro 2D (normalizzato)")
    ax_skel.set_xlim(-half_range, half_range)
    ax_skel.set_ylim(half_range, -half_range)  
    ax_skel.set_aspect("equal", adjustable="box")

    ticks = np.arange(-half_range, half_range + tick_step / 2, tick_step)
    ax_skel.set_xticks(ticks)
    ax_skel.set_yticks(ticks)

    points = []
    for i in range(N_KEYPOINTS):
        point = ax_skel.scatter([], [], s=18, color=np.array(_COLORS[i]) / 255.0, zorder=3)
        points.append(point)

    lines = []
    for _ in SKELETON:
        line, = ax_skel.plot([], [], color="black", linewidth=1, zorder=2)
        lines.append(line)

    ax_feat.set_title(title)
    time_axis = np.arange(n_frames) / fps
    ax_feat.plot(time_axis, feature_values, color="darkorange", lw=1.5,
                 label=feature_label)

    if hlines:
        for hl in hlines:
            ax_feat.axhline(hl["value"], color=hl.get("color", "gray"),
                             linestyle="--", lw=1.2,
                             label=f"{hl['label']} = {hl['value']:.3f}")

    ax_feat.set_xlabel("Tempo (s)")
    ax_feat.set_ylabel(ylabel)
    ax_feat.legend(loc="upper right", fontsize=8)
    time_marker = ax_feat.axvline(0, color="crimson", lw=1.5)

    def init():
        for point in points:
            point.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return points + lines + [time_marker]

    def update(frame_idx):
        frame_x = x[frame_idx]
        frame_y = y[frame_idx]

        for i, point in enumerate(points):
            if np.isnan(frame_x[i]) or np.isnan(frame_y[i]):
                point.set_offsets(np.empty((0, 2)))
            else:
                point.set_offsets([[frame_x[i], frame_y[i]]])

        for line, (a, b) in zip(lines, SKELETON):
            if (np.isnan(frame_x[a]) or np.isnan(frame_y[a]) or
                    np.isnan(frame_x[b]) or np.isnan(frame_y[b])):
                line.set_data([], [])
            else:
                line.set_data([frame_x[a], frame_x[b]], [frame_y[a], frame_y[b]])

        time_marker.set_xdata([frame_idx / fps, frame_idx / fps])
        return points + lines + [time_marker]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                          interval=1000 / fps, blit=True)

    plt.tight_layout()
    anim.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"salvata {out_path}")