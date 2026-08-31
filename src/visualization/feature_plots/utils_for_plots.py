import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import normalize_keypoints, JOINTS, LEFT_HIP, RIGHT_HIP

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


def compute_angle_arc(vertex, reference, angle_value, side, radius=0.12):
    """(x, y) points of the arc that visually represents an angle."""
    if np.isnan(angle_value):
        return [], []

    ref_vec = reference - vertex
    if np.isnan(ref_vec[0]) or np.isnan(ref_vec[1]):
        return [], []

    ref_angle = np.arctan2(ref_vec[1], ref_vec[0])

    if side == "right":
        theta = ref_angle - np.linspace(0, np.deg2rad(angle_value), 50)
    elif side == "left":
        theta = ref_angle + np.linspace(0, np.deg2rad(angle_value), 50)
    else:
        raise ValueError("side must be 'left' or 'right'")

    arc_x = vertex[0] + radius * np.cos(theta)
    arc_y = vertex[1] + radius * np.sin(theta)
    return arc_x, arc_y


def plot_skeleton_with_timeseries(keypoints, fps, feature_values, title, ylabel, out_path,
                                   feature_label="feature", feature_color="darkorange",
                                   hlines=None,
                                   show_bounding_box=False,
                                   highlight_joints=None,
                                   angle_arcs=None,
                                   distance_lines=None,
                                   extra_curves=None):
    """
    Draws, side by side, the animated skeleton (left) and one or more
    features over time with a current-instant marker (right).

    Optional parameters:
    - feature_color: color of the main curve (default "darkorange").
    - hlines: list of dicts {"value", "label", "color"} for simple
      horizontal reference lines (e.g. a median).
    - show_bounding_box: draws the rectangle enclosing the 12 joints
      used for the Body/Shape features.
    - highlight_joints: list of joint indices (0-16) to draw larger and
      with a black border, to highlight them.
    - angle_arcs: list of dicts to draw an angle's arc on the skeleton.
      Each dict: "vertex", "reference", "side", "values",
      "color" (optional), "radius" (optional).
    - distance_lines: list of dicts to draw a dashed segment from a
      joint to the hip center, in the SAME frame. Each dict:
      "joint" (joint index), "color".
    - extra_curves: list of dicts to draw additional curves in the
      right-hand panel, besides `feature_values`. Each dict: "values"
      (array T,), "label", "color" (defaults to "teal" if absent).
    """
    kp = normalize_keypoints(keypoints)
    x = kp[:, :, 0]
    y = kp[:, :, 1]

    hip_center_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_center_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    n_frames = len(x)

    fig, (ax_skel, ax_feat) = plt.subplots(1, 2, figsize=(11, 5))

    margin = 0.2
    tick_step = 0.5

    raw_half_range = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) + margin
    half_range = np.ceil(raw_half_range / tick_step) * tick_step

    ax_skel.set_title("Keypoints 2D")
    ax_skel.set_xlim(-half_range, half_range)
    ax_skel.set_ylim(half_range, -half_range)
    ax_skel.set_aspect("equal", adjustable="box")

    ticks = np.arange(-half_range, half_range + tick_step / 2, tick_step)
    ax_skel.set_xticks(ticks)
    ax_skel.set_yticks(ticks)

    # --- Skeleton points, with optional highlighting ---
    highlight_joints = highlight_joints or []
    points = []
    for i in range(N_KEYPOINTS):
        is_hl = i in highlight_joints
        point = ax_skel.scatter(
            [], [],
            s=55 if is_hl else 18,
            color=np.array(_COLORS[i]) / 255.0,
            edgecolor="black" if is_hl else "none",
            linewidth=1.3 if is_hl else 0,
            zorder=4 if is_hl else 3,
        )
        points.append(point)

    lines = []
    for _ in SKELETON:
        line, = ax_skel.plot([], [], color="black", linewidth=1, zorder=2)
        lines.append(line)

    # --- Optional bounding box ---
    bbox_patch = None
    bbox_min_x = bbox_max_x = bbox_min_y = bbox_max_y = None
    if show_bounding_box:
        with np.errstate(invalid="ignore"):
            bbox_min_x = np.nanmin(x[:, JOINTS], axis=1)
            bbox_max_x = np.nanmax(x[:, JOINTS], axis=1)
            bbox_min_y = np.nanmin(y[:, JOINTS], axis=1)
            bbox_max_y = np.nanmax(y[:, JOINTS], axis=1)

        bbox_patch = Rectangle((0, 0), 0, 0, fill=False, edgecolor="darkorange",
                                linewidth=1.5, linestyle="--", zorder=4)
        ax_skel.add_patch(bbox_patch)

    # --- Optional angle arcs ---
    angle_arcs = angle_arcs or []
    arc_lines = []
    for spec in angle_arcs:
        line, = ax_skel.plot([], [], color=spec.get("color", "darkorange"), linewidth=2, zorder=5)
        arc_lines.append(line)

    # --- Optional distance-from-hip-center lines ---
    distance_lines = distance_lines or []
    dist_lines = []
    for spec in distance_lines:
        line, = ax_skel.plot([], [], color=spec.get("color", "darkorange"),
                              linewidth=1.8, linestyle="--", zorder=5)
        dist_lines.append(line)

    # --- Right panel: feature(s) over time (one main curve + optional extras) ---
    ax_feat.set_title(title)
    time_axis = np.arange(n_frames) / fps
    ax_feat.plot(time_axis, feature_values, color=feature_color, lw=1.5,
                 label=feature_label)

    extra_curves = extra_curves or []
    for curve in extra_curves:
        ax_feat.plot(time_axis, curve["values"], color=curve.get("color", "teal"),
                     lw=1.5, label=curve["label"])

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
        for line in arc_lines:
            line.set_data([], [])
        for line in dist_lines:
            line.set_data([], [])

        artists = points + lines + arc_lines + dist_lines + [time_marker]
        if bbox_patch is not None:
            bbox_patch.set_bounds(0, 0, 0, 0)
            artists.append(bbox_patch)
        return artists

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

        for line, spec in zip(arc_lines, angle_arcs):
            vertex_pt = np.array([frame_x[spec["vertex"]], frame_y[spec["vertex"]]])
            reference_pt = np.array([frame_x[spec["reference"]], frame_y[spec["reference"]]])
            arc_x, arc_y = compute_angle_arc(
                vertex_pt, reference_pt, spec["values"][frame_idx],
                spec["side"], radius=spec.get("radius", 0.12)
            )
            line.set_data(arc_x, arc_y)

        hcx, hcy = hip_center_x[frame_idx], hip_center_y[frame_idx]
        for line, spec in zip(dist_lines, distance_lines):
            j = spec["joint"]
            if np.isnan(frame_x[j]) or np.isnan(frame_y[j]) or np.isnan(hcx) or np.isnan(hcy):
                line.set_data([], [])
            else:
                line.set_data([frame_x[j], hcx], [frame_y[j], hcy])

        time_marker.set_xdata([frame_idx / fps, frame_idx / fps])

        artists = points + lines + arc_lines + dist_lines + [time_marker]
        if bbox_patch is not None:
            bx0, bx1 = bbox_min_x[frame_idx], bbox_max_x[frame_idx]
            by0, by1 = bbox_min_y[frame_idx], bbox_max_y[frame_idx]
            if np.isnan(bx0) or np.isnan(bx1) or np.isnan(by0) or np.isnan(by1):
                bbox_patch.set_bounds(0, 0, 0, 0)
            else:
                bbox_patch.set_bounds(bx0, by0, bx1 - bx0, by1 - by0)
            artists.append(bbox_patch)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,
        blit=True
    )

    plt.tight_layout()
    anim.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"salvata {out_path}")