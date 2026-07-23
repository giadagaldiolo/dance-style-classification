"""
Plot 2 — SHAPE: angolo del gomito sinistro nel tempo.

Si collega direttamente agli istogrammi angolari left_forearm_hist_0..7:
questa curva è ciò da cui quegli istogrammi vengono costruiti.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import (
    normalize_keypoints, joint_angle, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
)


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl" 
OUT_PATH = "outputs/feature_plots/plot2_shape_elbow_angle.gif"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keypoints, fps = load_keypoints(PKL_PATH)
    kp_norm = normalize_keypoints(keypoints)

    left_elbow_angle = joint_angle(kp_norm, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=left_elbow_angle,
        feature_label="angolo gomito sinistro",
        title="Shape: angolo del gomito sinistro nel tempo",
        ylabel="angolo (gradi, 0-360)",
        out_path=OUT_PATH,
    )


if __name__ == "__main__":
    main()