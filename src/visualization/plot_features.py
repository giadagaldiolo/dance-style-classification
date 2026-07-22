"""
Genera animazioni a due pannelli (scheletro animato + grafico sincronizzato
della feature nel tempo) a supporto della spiegazione o del debug delle
feature LMA.

Riusa le stesse funzioni di lma_extractor.py (normalize_keypoints,
get_speed_accel, joint_angle) usate nella pipeline di classificazione, così
i plot mostrano ESATTAMENTE gli stessi calcoli usati per le feature reali,
non una riproduzione approssimativa.

Posizionare questo file in src/visualization/ (sorella di src/classification/)
perché lo script fa affidamento sulla stessa struttura a cartelle usata dagli
altri script del progetto (vedi SRC_DIR sotto).
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_DIR = "outputs/feature_plots"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import (
    normalize_keypoints, get_speed_accel, joint_angle,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, JOINTS,
)

# Connessioni tra joint per disegnare lo "scheletro" (coppie di indici)
SKELETON_EDGES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


def load_keypoints(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    keypoints = data["keypoints2d"][0]  # (T, 17, 3)
    fps = data.get("fps", 60)
    return keypoints, fps


def plot_skeleton_with_timeseries(keypoints, fps, feature_values, title, ylabel,
                                   out_path, raw_values=None, raw_label="grezzo",
                                   feature_label="smussato", max_frames=None):
    """
    Crea un'animazione a due pannelli:
    - sinistra: scheletro 2D animato (normalizzato, stessa scala usata
      nell'estrazione feature)
    - destra: la feature scelta nel tempo, con una linea verticale rossa
      che segue il frame corrente, sincronizzata con l'animazione

    Se `raw_values` è fornito, lo sovrappone in grigio chiaro dietro alla
    curva principale (utile per il confronto grezzo/smussato).

    `max_frames`: se impostato, mostra solo i primi N frame (utile per
    accorciare l'animazione in fase di presentazione).
    """
    kp = normalize_keypoints(keypoints)
    x, y = kp[:, :, 0], kp[:, :, 1]

    if max_frames is not None:
        x, y = x[:max_frames], y[:max_frames]
        feature_values = feature_values[:max_frames]
        if raw_values is not None:
            raw_values = raw_values[:max_frames]

    n_frames = len(x)

    fig, (ax_skel, ax_feat) = plt.subplots(1, 2, figsize=(11, 5))

    # --- Pannello scheletro ---
    ax_skel.set_title("Scheletro 2D (normalizzato)")
    ax_skel.set_xlim(np.nanmin(x) - 0.1, np.nanmax(x) + 0.1)
    # y invertito: nelle coordinate immagine y cresce verso il basso
    ax_skel.set_ylim(np.nanmax(y) + 0.1, np.nanmin(y) - 0.1)
    ax_skel.set_aspect("equal")
    ax_skel.axis("off")

    joint_scatter = ax_skel.scatter([], [], c="crimson", s=25, zorder=3)
    edge_lines = [ax_skel.plot([], [], c="steelblue", lw=2, zorder=2)[0]
                  for _ in SKELETON_EDGES]

    # --- Pannello feature ---
    ax_feat.set_title(title)
    time_axis = np.arange(n_frames) / fps
    if raw_values is not None:
        ax_feat.plot(time_axis, raw_values, color="lightgray", lw=1, label=raw_label)
    ax_feat.plot(time_axis, feature_values, color="darkorange", lw=1.5, label=feature_label)
    ax_feat.set_xlabel("Tempo (s)")
    ax_feat.set_ylabel(ylabel)
    ax_feat.legend(loc="upper right", fontsize=8)
    time_marker = ax_feat.axvline(0, color="crimson", lw=1.5)

    def update(frame_idx):
        pts_x = x[frame_idx, JOINTS]
        pts_y = y[frame_idx, JOINTS]
        joint_scatter.set_offsets(np.column_stack([pts_x, pts_y]))

        for line, (a, b) in zip(edge_lines, SKELETON_EDGES):
            line.set_data([x[frame_idx, a], x[frame_idx, b]],
                          [y[frame_idx, a], y[frame_idx, b]])

        time_marker.set_xdata([frame_idx / fps, frame_idx / fps])
        return [joint_scatter, *edge_lines, time_marker]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000 / fps, blit=True)

    plt.tight_layout()
    # Salvataggio in GIF con Pillow: non richiede ffmpeg installato.
    # Se hai ffmpeg disponibile, puoi ottenere qualità migliore/file più
    # piccoli con: anim.save(out_path_mp4, writer="ffmpeg", fps=fps, dpi=120)
    anim.save(out_path, writer="pillow", fps=fps, dpi=110)
    plt.close(fig)
    print(f"Animazione salvata → {out_path}")


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    # ----------------------------------------------------------------
    # PLOT 1 — EFFORT: velocità del polso destro nel tempo
    # ----------------------------------------------------------------
    wrist_speed, _ = get_speed_accel(x[:, RIGHT_WRIST], y[:, RIGHT_WRIST], fps)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=wrist_speed,
        feature_label="velocità polso destro",
        title="Effort: velocità del polso destro nel tempo",
        ylabel="velocità (unità normalizzate / s)",
        out_path=os.path.join(OUT_DIR, "plot1_effort_wrist_speed.gif"),
        max_frames=MAX_FRAMES,
    )

    # ----------------------------------------------------------------
    # PLOT 2 — SHAPE: angolo del gomito sinistro nel tempo
    # (si collega direttamente agli istogrammi angolari left_forearm_hist_*)
    # ----------------------------------------------------------------
    left_elbow_angle = joint_angle(kp_norm, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=left_elbow_angle,
        feature_label="angolo gomito sinistro",
        title="Shape: angolo del gomito sinistro nel tempo",
        ylabel="angolo (gradi, 0-360)",
        out_path=os.path.join(OUT_DIR, "plot2_shape_elbow_angle.gif"),
        max_frames=MAX_FRAMES,
    )

    # ----------------------------------------------------------------
    # PLOT 3 — BODY/SHAPE: area del corpo (bounding box) nel tempo
    # (mostra visivamente cosa catturano "mean" e "max" di shape_body_area)
    # ----------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        min_x, max_x = np.nanmin(x[:, JOINTS], axis=1), np.nanmax(x[:, JOINTS], axis=1)
        min_y, max_y = np.nanmin(y[:, JOINTS], axis=1), np.nanmax(y[:, JOINTS], axis=1)
    body_area = (max_x - min_x) * (max_y - min_y)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=body_area,
        feature_label="area del corpo (bounding box)",
        title="Body/Shape: area del corpo nel tempo",
        ylabel="area (unità normalizzate²)",
        out_path=os.path.join(OUT_DIR, "plot3_body_area.gif"),
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()