"""
Utility condivise dagli script di plotting delle feature (uno script per
ciascun grafico: plot1_effort_wrist_speed.py, plot2_shape_elbow_angle.py,
plot3_body_area.py — così si possono lanciare/modificare indipendentemente).

Riusa normalize_keypoints da lma_extractor.py (stessa pipeline usata per
le feature reali), e la stessa palette/schema di scheletro usati nello
script di debug del progetto (animazioni normalizzate).
"""

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

# Palette per i 17 keypoint COCO (uno per indice, stessa usata nello script
# di debug del progetto). I punti del volto (naso/occhi/orecchie, indici
# 0-4) vengono mostrati come pallini colorati ma senza linee di collegamento.
_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85]
]

# Connessioni "ossee" tra i 12 joint usati per le feature (spalle, gomiti,
# polsi, fianchi, ginocchia, caviglie) — stessa lista usata nello script di
# debug del progetto.
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
                                   hlines=None, max_frames=None):
    """
    Crea un'animazione a due pannelli:
    - sinistra: scheletro 2D animato (tutti i 17 keypoint COCO come pallini
      colorati singolarmente + le 12 linee "ossee" tra i joint usati per
      le feature), normalizzato con la stessa scala usata nell'estrazione
      feature
    - destra: la feature scelta nel tempo, con una linea verticale rossa
      che segue il frame corrente, sincronizzata con l'animazione

    hlines: lista opzionale di dict {"value", "label", "color"} per marcare
    valori aggregati (mediana, media, massimo...) usati come feature reale
    nel dataset. Vanno calcolati dal chiamante SULLA SEQUENZA COMPLETA,
    prima di eventuali trim fatti solo per la visualizzazione, così la
    linea riflette il valore vero usato dal modello.

    max_frames: se impostato, mostra solo i primi N frame (utile per
    accorciare l'animazione). Non influenza il calcolo di `hlines`, che va
    fatto dal chiamante sui dati completi.

    Il salvataggio usa ffmpeg (output .mp4): rispetta correttamente i tempi
    reali tra frame, quindi la velocità di riproduzione è sempre quella
    corretta a qualunque fps nativo, senza bisogno di sotto-campionare.
    """
    kp = normalize_keypoints(keypoints)
    x, y = kp[:, :, 0], kp[:, :, 1]

    if max_frames is not None:
        x, y = x[:max_frames], y[:max_frames]
        feature_values = feature_values[:max_frames]

    n_frames = len(x)

    fig, (ax_skel, ax_feat) = plt.subplots(1, 2, figsize=(11, 5))

    # --- Pannello scheletro ---
    ax_skel.set_title("Scheletro 2D (normalizzato)")
    ax_skel.set_xlim(np.nanmin(x) - 0.2, np.nanmax(x) + 0.2)
    ax_skel.set_ylim(np.nanmax(y) + 0.2, np.nanmin(y) - 0.2)  # y invertito
    ax_skel.set_aspect("equal")

    points = []
    for i in range(N_KEYPOINTS):
        point = ax_skel.scatter([], [], s=18, color=np.array(_COLORS[i]) / 255.0, zorder=3)
        points.append(point)

    lines = []
    for _ in SKELETON:
        line, = ax_skel.plot([], [], color="black", linewidth=1, zorder=2)
        lines.append(line)

    # --- Pannello feature ---
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
    print(f"Animazione salvata → {out_path}")