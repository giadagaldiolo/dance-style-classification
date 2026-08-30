import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_for_plots import load_keypoints

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import (
    normalize_keypoints, LEFT_HIP, RIGHT_HIP, RIGHT_WRIST, LEFT_ANKLE,
)

PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot1b_body_distance_boxplot.png"


def draw_mean_std_box(ax, x_pos, mean, std, color, label, box_width=0.5):
    """Disegna un rettangolo centrato sulla media, alto 2*std (media±std),
    con una linea orizzontale sulla media -- non un boxplot statistico
    standard (che userebbe i quartili), ma la rappresentazione diretta
    di media e deviazione standard, le stesse due statistiche usate per
    aggregare le feature di Body in extract_features."""
    ax.add_patch(plt.Rectangle((x_pos - box_width/2, mean - std), box_width, 2*std,
                                facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.5))
    ax.plot([x_pos - box_width/2, x_pos + box_width/2], [mean, mean],
            color=color, linewidth=2.5)
    ax.scatter([x_pos], [mean], color=color, s=40, zorder=3, label=label)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)
    x, y = kp[:, :, 0], kp[:, :, 1]

    hip_center_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_center_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    wrist_dist = np.sqrt((x[:, RIGHT_WRIST] - hip_center_x)**2 +
                          (y[:, RIGHT_WRIST] - hip_center_y)**2)
    ankle_dist = np.sqrt((x[:, LEFT_ANKLE] - hip_center_x)**2 +
                          (y[:, LEFT_ANKLE] - hip_center_y)**2)

    wrist_mean, wrist_std = np.nanmean(wrist_dist), np.nanstd(wrist_dist)
    ankle_mean, ankle_std = np.nanmean(ankle_dist), np.nanstd(ankle_dist)

    fig, ax = plt.subplots(figsize=(5, 5.5))

    draw_mean_std_box(ax, 0, wrist_mean, wrist_std, "darkorange", "polso destro")
    draw_mean_std_box(ax, 1, ankle_mean, ankle_std, "teal", "caviglia sinistra")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Polso dx", "Caviglia sx"], fontsize=12)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel("Distanza dal centro del bacino (unità corporee)", fontsize=12.5)
    ax.set_title("Media \u00b1 deviazione standard", fontsize=14)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Grafico salvato -> {OUT_PATH}")
    print(f"Polso dx: media={wrist_mean:.3f}, std={wrist_std:.3f}")
    print(f"Caviglia sx: media={ankle_mean:.3f}, std={ankle_std:.3f}")


if __name__ == "__main__":
    main()