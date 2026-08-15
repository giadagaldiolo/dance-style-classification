"""
Visualizzazione per la presentazione: mostra l'effetto della
normalizzazione su due video con scala/posizione molto diverse.

Sinistra: video A, keypoint ORIGINALI (coordinate pixel)
Centro:   video B, keypoint ORIGINALI -- stesso range di assi del
          pannello sinistro apposta, così la differenza di scala tra i
          due video è visivamente evidente (non nascosta da assi che si
          adattano automaticamente a ciascun video)
Destra:   entrambi gli scheletri DOPO la normalizzazione, sovrapposti
          sullo stesso asse -- dovrebbero risultare quasi coincidenti,
          a dimostrazione dell'effetto della normalizzazione

Usa find_scale_contrast.py per scegliere due file con una differenza di
scala vistosa prima di lanciare questo script.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import normalize_keypoints

# <-- Scegli due sequenze con scala/posizione VISIBILMENTE diverse
#     (usa find_scale_contrast.py per trovarle)
VIDEO_A_PKL = "outputs/keypoints/gLO_yt_04.pkl"
VIDEO_B_PKL = "outputs/keypoints/gHO_yt_02.pkl"

OUT_PATH = "outputs/visualization/normalizzazione_confronto.mp4"
PLAYBACK_FPS = 30
N_FRAMES_TO_SHOW = 420  # 20s a 30fps -- alza/abbassa in base a quanto vuoi che duri

# Stessa palette usata altrove nel progetto, ora su tutti e 17 i keypoint
_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
# Scheletro completo: viso (naso/occhi/orecchie) + braccia/busto/gambe
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # viso
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
JOINTS = list(range(17))  # tutti e 17, non solo il sottoinsieme usato per le feature


def load_keypoints(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["keypoints2d"][0]


def flip_y(keypoints, reference):
    """Converte la coordinata y da 'convenzione immagine' (0 in alto,
    cresce verso il basso) a 'convenzione cartesiana standard' (0 in
    basso, cresce verso l'alto): y_nuova = reference - y_originale.
    Necessario per usare assi con orientamento normale mantenendo lo
    scheletro dritto (non capovolto)."""
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

    # Range CONDIVISO tra i due video, basato sui keypoint VERI (non
    # sull'intero frame): riduce lo spazio bianco, mantenendo comunque
    # comparabile la scala tra i due video.
    all_x = np.concatenate([kp_a[:, JOINTS, 0].flatten(), kp_b[:, JOINTS, 0].flatten()])
    all_y = np.concatenate([kp_a[:, JOINTS, 1].flatten(), kp_b[:, JOINTS, 1].flatten()])
    raw_x_min, raw_x_max = np.nanmin(all_x), np.nanmax(all_x)
    raw_y_min, raw_y_max = np.nanmin(all_y), np.nanmax(all_y)
    margin = 0.15 * max(raw_x_max - raw_x_min, raw_y_max - raw_y_min)

    # Flip della y (vedi flip_y): permette di usare assi con orientamento
    # standard (0 in basso a sinistra) mantenendo lo scheletro dritto.
    kp_a_plot = flip_y(kp_a, raw_y_max)
    kp_b_plot = flip_y(kp_b, raw_y_max)
    y_plot_max = raw_y_max - raw_y_min  # estensione dopo il flip

    # Larghezza di ciascuna colonna proporzionale al VERO rapporto
    # larghezza/altezza dei dati di quel pannello -- cosi' nessun pannello
    # ha bisogno di essere rimpicciolito (margini vuoti) o ritagliato
    # (dati persi) per rispettare le proporzioni corrette.
    aspect_raw = ((raw_x_max - raw_x_min) + 2 * margin) / (y_plot_max + 2 * margin)
    aspect_norm = 1.0  # range simmetrico -norm_range/+norm_range su entrambi gli assi

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[aspect_raw, aspect_raw, aspect_norm])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_norm = fig.add_subplot(gs[2])

    for ax, title in [(ax_a, "Video A (originale)"), (ax_b, "Video B (originale)")]:
        ax.set_xlim(raw_x_min - margin, raw_x_max + margin)
        ax.set_ylim(0, y_plot_max + margin)  # orientamento standard: 0 in basso
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

    # Valore FISSO, uguale per qualunque video -- non calcolato dai due
    # video specifici, cosi' e' sempre confrontabile tra run diversi.
    # Aggiorna con il valore suggerito da find_fixed_axis_range.py.
    norm_range = 1.0

    # Stesso flip applicato ai keypoint normalizzati (centrati sullo 0,
    # quindi qui il flip e' semplicemente un cambio di segno).
    kp_a_norm_plot = flip_y(kp_a_norm, 0)
    kp_b_norm_plot = flip_y(kp_b_norm, 0)

    ax_norm.set_xlim(-norm_range, norm_range)
    ax_norm.set_ylim(-norm_range, norm_range)  # orientamento standard
    ax_norm.set_aspect("equal", adjustable="box")
    ax_norm.set_title("Dopo la normalizzazione")

    points_a, lines_a = make_artists(ax_a)
    points_b, lines_b = make_artists(ax_b)
    # Stesso stile identico per entrambi: l'obiettivo qui è mostrare che
    # dopo la normalizzazione diventano visivamente indistinguibili.
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