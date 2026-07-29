"""
Plot 6 — BODY: distanza di alcuni joint dal centro-bacino nel tempo.

Per non affollare il grafico mostriamo 2 joint rappresentativi (polso e
caviglia destri): lo stesso identico calcolo si applica a tutti e 12 i
joint usati per le feature (body_dist_hip_mean_X / body_dist_hip_std_X).

Il centro-bacino non è un keypoint reale ma il punto medio tra fianco
sinistro e destro: viene disegnato come un pallino nero a stella,
distinto dai 17 keypoint colorati.

Usa fasce media±std, coerente con la convenzione mean/std scelta per
questa famiglia di feature.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_keypoints, plot_skeleton_with_curves

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import (
    normalize_keypoints, LEFT_HIP, RIGHT_HIP, RIGHT_WRIST, RIGHT_ANKLE,
)


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"  # <-- cambia con un tuo file
OUT_PATH = "outputs/feature_plots/plot6_body_dist_hip.mp4"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    hip_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2
    hip_center = np.stack([hip_x, hip_y], axis=1)  # (T, 2)

    dist_wrist = np.sqrt((x[:, RIGHT_WRIST] - hip_x)**2 + (y[:, RIGHT_WRIST] - hip_y)**2)
    dist_ankle = np.sqrt((x[:, RIGHT_ANKLE] - hip_x)**2 + (y[:, RIGHT_ANKLE] - hip_y)**2)

    curves = [
        {"values": dist_wrist, "label": "polso destro - centro bacino", "color": "tab:orange"},
        {"values": dist_ankle, "label": "caviglia destra - centro bacino", "color": "tab:purple"},
    ]

    extra_edges = [
        (RIGHT_WRIST, "hip_center", "tab:orange", "dist. polso"),
        (RIGHT_ANKLE, "hip_center", "tab:purple", "dist. caviglia"),
    ]

    plot_skeleton_with_curves(
        keypoints, fps,
        curves=curves,
        title="Body: distanza dal centro-bacino (polso e caviglia destri)",
        ylabel="distanza (unità normalizzate)",
        out_path=OUT_PATH,
        extra_edges=extra_edges,
        extra_points={"hip_center": hip_center},
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()