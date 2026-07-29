"""
Plot 4 — SHAPE: distanza polso-polso (w2w), caviglia-caviglia (a2a) e
"cross distance" nel tempo, disegnate direttamente come segmenti colorati
sullo scheletro (stesso colore della curva corrispondente a destra).

Usa fasce media±std, coerente con la convenzione mean/std scelta per
queste feature (a differenza di Effort/Space, dove si usa la mediana per
la maggiore robustezza agli outlier delle quantità derivate dal tempo).
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
    normalize_keypoints, LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE,
)


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"  # <-- cambia con un tuo file
OUT_PATH = "outputs/feature_plots/plot4_shape_distances.mp4"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    w2w = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_WRIST])**2 +
                  (y[:, LEFT_WRIST] - y[:, RIGHT_WRIST])**2)
    a2a = np.sqrt((x[:, LEFT_ANKLE] - x[:, RIGHT_ANKLE])**2 +
                  (y[:, LEFT_ANKLE] - y[:, RIGHT_ANKLE])**2)

    cross1 = np.sqrt((x[:, LEFT_WRIST] - x[:, RIGHT_ANKLE])**2 +
                      (y[:, LEFT_WRIST] - y[:, RIGHT_ANKLE])**2)
    cross2 = np.sqrt((x[:, RIGHT_WRIST] - x[:, LEFT_ANKLE])**2 +
                      (y[:, RIGHT_WRIST] - y[:, LEFT_ANKLE])**2)
    cross_mean = (cross1 + cross2) / 2  # è questa la curva che diventa la feature reale

    curves = [
        {"values": w2w, "label": "polso-polso (w2w)", "color": "tab:blue"},
        {"values": a2a, "label": "caviglia-caviglia (a2a)", "color": "tab:green"},
        {"values": cross_mean, "label": "cross distance", "color": "tab:red"},
    ]

    # Segmenti disegnati sullo scheletro, stesso colore della curva a destra.
    # Le due diagonali della "cross distance" condividono lo stesso colore;
    # la seconda non ha etichetta in legenda (label=None) per non duplicarla.
    extra_edges = [
        (LEFT_WRIST, RIGHT_WRIST, "tab:blue", "w2w"),
        (LEFT_ANKLE, RIGHT_ANKLE, "tab:green", "a2a"),
        (LEFT_WRIST, RIGHT_ANKLE, "tab:red", "cross"),
        (RIGHT_WRIST, LEFT_ANKLE, "tab:red", None),
    ]

    plot_skeleton_with_curves(
        keypoints, fps,
        curves=curves,
        title="Shape: distanze polso-polso / caviglia-caviglia / cross",
        ylabel="distanza (unità normalizzate)",
        out_path=OUT_PATH,
        extra_edges=extra_edges,
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()