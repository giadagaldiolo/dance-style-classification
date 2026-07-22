"""
Plot 3 — BODY/SHAPE: area del corpo (bounding box) nel tempo.

Mostra due linee orizzontali, MEDIA e MASSIMO: sono i due valori che
diventano le feature reali usate dal modello
(shape_body_area_mean, shape_body_area_max).
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import normalize_keypoints, JOINTS


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"  # <-- cambia con un tuo file
OUT_PATH = "outputs/feature_plots/plot3_body_area.gif"
MAX_FRAMES = None  # es. 150 per accorciare l'animazione, None = tutta la sequenza


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)
    x, y = kp_norm[:, :, 0], kp_norm[:, :, 1]

    with np.errstate(invalid="ignore"):
        min_x, max_x = np.nanmin(x[:, JOINTS], axis=1), np.nanmax(x[:, JOINTS], axis=1)
        min_y, max_y = np.nanmin(y[:, JOINTS], axis=1), np.nanmax(y[:, JOINTS], axis=1)
    body_area = (max_x - min_x) * (max_y - min_y)

    mean_area = np.nanmean(body_area)
    max_area = np.nanmax(body_area)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=body_area,
        feature_label="area del corpo (bounding box)",
        title="Body/Shape: area del corpo nel tempo",
        ylabel="area (unità normalizzate²)",
        out_path=OUT_PATH,
        hlines=[
            {"value": mean_area, "label": "media (= feature reale)", "color": "green"},
            {"value": max_area, "label": "massimo (= feature reale)", "color": "purple"},
        ],
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()