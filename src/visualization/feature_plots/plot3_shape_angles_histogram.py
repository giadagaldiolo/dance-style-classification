import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualization.feature_plots.utils_for_plots import load_keypoints

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import (
    normalize_keypoints, joint_angle,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
)

PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot3b_shape_angle_histogram.png"

# Stesso intervallo e numero di bin usati davvero in angle_histogram()
# dentro lma_extractor.py -- NON -180/180: joint_angle() applica un
# modulo 360, quindi i valori sono sempre in [0, 360).
N_BINS = 8
ANGLE_RANGE = (0, 360)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)

    left_angle = joint_angle(kp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")
    right_angle = joint_angle(kp, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "right")

    left_valid = left_angle[~np.isnan(left_angle)]
    right_valid = right_angle[~np.isnan(right_angle)]

    left_hist, edges = np.histogram(left_valid, bins=N_BINS, range=ANGLE_RANGE)
    right_hist, _ = np.histogram(right_valid, bins=N_BINS, range=ANGLE_RANGE)
    left_freq = left_hist / left_hist.sum()
    right_freq = right_hist / right_hist.sum()

    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_labels = [f"{int(edges[i])}-{int(edges[i+1])}°" for i in range(N_BINS)]
    bar_h = (edges[1] - edges[0]) * 0.4  # spessore di ciascuna barra, per affiancare le due serie

    fig, ax = plt.subplots(figsize=(6, 6.5))

    ax.barh(bin_centers - bar_h/2, left_freq, height=bar_h, color="teal", label="gomito sinistro")
    ax.barh(bin_centers + bar_h/2, right_freq, height=bar_h, color="darkorange", label="gomito destro")

    ax.set_yticks(bin_centers)
    ax.set_yticklabels(bin_labels, fontsize=11)
    ax.set_ylim(ANGLE_RANGE[0] - 10, ANGLE_RANGE[1] + 10)
    ax.set_xlabel("Frequenza", fontsize=13)
    ax.set_ylabel("Angolo del gomito", fontsize=13)
    ax.set_title("Istogramma degli angoli del gomito", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.legend(fontsize=10.5)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Grafico salvato -> {OUT_PATH}")


if __name__ == "__main__":
    main()