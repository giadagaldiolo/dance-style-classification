"""
Plot 1 — EFFORT: velocità del polso destro nel tempo.

Mostra una linea orizzontale in corrispondenza della MEDIANA: non l'intera
curva ma solo quel singolo valore diventa la feature reale usata dal
modello (effort_wrist_speed_median).
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import normalize_keypoints, get_speed_accel, RIGHT_WRIST


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"  # <-- cambia con un tuo file
OUT_PATH = "outputs/feature_plots/plot1_effort_wrist_speed.gif"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    wrist_speed, _ = get_speed_accel(x[:, RIGHT_WRIST], y[:, RIGHT_WRIST], fps)
    median_speed = np.nanmedian(wrist_speed)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=wrist_speed,
        feature_label="velocità polso destro",
        title="Effort: velocità del polso destro nel tempo",
        ylabel="velocità (unità normalizzate / s)",
        out_path=OUT_PATH,
        hlines=[{"value": median_speed, "label": "mediana (= feature reale)", "color": "green"}],
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()