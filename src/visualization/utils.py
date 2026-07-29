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


def plot_skeleton_with_timeseries(keypoints, fps, feature_values, title, ylabel,out_path, feature_label="feature",hlines=None):
    kp = normalize_keypoints(keypoints)
    x = kp[:, :, 0]
    y = kp[:, :, 1]
    
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




# def plot_skeleton_with_curves(keypoints, fps, curves, title, ylabel, out_path,
#                                extra_edges=None, extra_points=None, max_frames=None):
#     """
#     Variante con PIÙ curve contemporaneamente (con fascia media±std),
#     e con la possibilità di disegnare linee/punti extra sullo scheletro
#     oltre alle 12 ossa fisse — usata per feature che coinvolgono più
#     distanze insieme (es. w2w/a2a/cross, o distanza dal centro-bacino).
 
#     curves: lista di dict {"values": array (T,), "label": str, "color": str}.
#     Ogni curva viene disegnata con la linea reale, una linea tratteggiata
#     sulla media e una fascia ombreggiata ampia ±1 std, coerente con la
#     convenzione mean/std usata per le feature di Body/Shape.
 
#     extra_edges: lista di tuple (a, b, color, label) per segmenti extra
#     disegnati sullo scheletro. `a`/`b` possono essere un indice intero
#     (joint reale, 0-16) oppure una stringa che referenzia una chiave in
#     `extra_points` (per punti virtuali come il centro-bacino).
 
#     extra_points: dict {nome: array (T, 2)} di punti virtuali, disegnati
#     come pallini neri a stella, distinti dai 17 keypoint colorati.
 
#     max_frames: come nell'altra funzione, accorcia la sola visualizzazione
#     senza alterare il calcolo di curves/extra_points (va fatto dal
#     chiamante sui dati completi, prima di passare gli array qui).
#     """
#     kp = normalize_keypoints(keypoints)
#     x, y = kp[:, :, 0], kp[:, :, 1]
 
#     if max_frames is not None:
#         x, y = x[:max_frames], y[:max_frames]
#         curves = [{**c, "values": c["values"][:max_frames]} for c in curves]
#         if extra_points:
#             extra_points = {k: v[:max_frames] for k, v in extra_points.items()}
 
#     n_frames = len(x)
 
#     fig, (ax_skel, ax_feat) = plt.subplots(1, 2, figsize=(11, 5))
 
#     # --- Pannello scheletro (stesso schema simmetrico/quadrato) ---
#     margin = 0.2
#     tick_step = 0.5
#     raw_half_range = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) + margin
#     half_range = np.ceil(raw_half_range / tick_step) * tick_step
 
#     ax_skel.set_title("Keypoints 2D")
#     ax_skel.set_xlim(-half_range, half_range)
#     ax_skel.set_ylim(half_range, -half_range)
#     ax_skel.set_aspect("equal", adjustable="box")
#     ticks = np.arange(-half_range, half_range + tick_step / 2, tick_step)
#     ax_skel.set_xticks(ticks)
#     ax_skel.set_yticks(ticks)
 
#     points = []
#     for i in range(N_KEYPOINTS):
#         point = ax_skel.scatter([], [], s=18, color=np.array(_COLORS[i]) / 255.0, zorder=3)
#         points.append(point)
 
#     bone_lines = []
#     for _ in SKELETON:
#         line, = ax_skel.plot([], [], color="black", linewidth=1, zorder=2)
#         bone_lines.append(line)
 
#     extra_edges = extra_edges or []
#     extra_lines = []
#     for (a, b, color, label) in extra_edges:
#         line, = ax_skel.plot([], [], color=color, linewidth=2, zorder=4, label=label)
#         extra_lines.append(line)
 
#     extra_points = extra_points or {}
#     extra_point_artists = {}
#     for name in extra_points:
#         artist = ax_skel.scatter([], [], s=45, color="black", marker="*", zorder=5)
#         extra_point_artists[name] = artist
 
#     if extra_edges:
#         ax_skel.legend(loc="upper right", fontsize=7)
 
#     # --- Pannello feature (una o più curve) ---
#     ax_feat.set_title(title)
#     time_axis = np.arange(n_frames) / fps
 
#     for c in curves:
#         vals = c["values"]
#         color = c["color"]
#         label = c["label"]
#         mean_v = np.nanmean(vals)
#         std_v = np.nanstd(vals)
 
#         ax_feat.plot(time_axis, vals, color=color, lw=1.3, alpha=0.9, label=label)
#         ax_feat.axhline(mean_v, color=color, linestyle="--", lw=1.1,
#                          label=f"{label} media={mean_v:.3f}")
#         ax_feat.fill_between(time_axis, mean_v - std_v, mean_v + std_v,
#                               color=color, alpha=0.12)
 
#     ax_feat.set_xlabel("Tempo (s)")
#     ax_feat.set_ylabel(ylabel)
#     ax_feat.legend(loc="upper right", fontsize=7)
#     time_marker = ax_feat.axvline(0, color="crimson", lw=1.5)
 
#     def get_xy(idx_or_name, frame_idx):
#         if isinstance(idx_or_name, str):
#             return extra_points[idx_or_name][frame_idx]
#         return x[frame_idx, idx_or_name], y[frame_idx, idx_or_name]
 
#     def init():
#         for point in points:
#             point.set_offsets(np.empty((0, 2)))
#         for line in bone_lines + extra_lines:
#             line.set_data([], [])
#         for artist in extra_point_artists.values():
#             artist.set_offsets(np.empty((0, 2)))
#         return (points + bone_lines + extra_lines +
#                 list(extra_point_artists.values()) + [time_marker])
 
#     def update(frame_idx):
#         frame_x = x[frame_idx]
#         frame_y = y[frame_idx]
 
#         for i, point in enumerate(points):
#             if np.isnan(frame_x[i]) or np.isnan(frame_y[i]):
#                 point.set_offsets(np.empty((0, 2)))
#             else:
#                 point.set_offsets([[frame_x[i], frame_y[i]]])
 
#         for line, (a, b) in zip(bone_lines, SKELETON):
#             if (np.isnan(frame_x[a]) or np.isnan(frame_y[a]) or
#                     np.isnan(frame_x[b]) or np.isnan(frame_y[b])):
#                 line.set_data([], [])
#             else:
#                 line.set_data([frame_x[a], frame_x[b]], [frame_y[a], frame_y[b]])
 
#         for line, (a, b, color, label) in zip(extra_lines, extra_edges):
#             pa = get_xy(a, frame_idx)
#             pb = get_xy(b, frame_idx)
#             if np.isnan(pa[0]) or np.isnan(pa[1]) or np.isnan(pb[0]) or np.isnan(pb[1]):
#                 line.set_data([], [])
#             else:
#                 line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
 
#         for name, artist in extra_point_artists.items():
#             px, py = extra_points[name][frame_idx]
#             if np.isnan(px) or np.isnan(py):
#                 artist.set_offsets(np.empty((0, 2)))
#             else:
#                 artist.set_offsets([[px, py]])
 
#         time_marker.set_xdata([frame_idx / fps, frame_idx / fps])
#         return (points + bone_lines + extra_lines +
#                 list(extra_point_artists.values()) + [time_marker])
 
#     anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
#                           interval=1000 / fps, blit=True)
 
#     plt.tight_layout()
#     anim.save(out_path, writer="ffmpeg", fps=fps)
#     plt.close(fig)
#     print(f"Animazione salvata → {out_path}")