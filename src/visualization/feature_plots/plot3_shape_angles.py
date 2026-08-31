import os
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from visualization.feature_plots.utils_for_plots import load_keypoints, plot_skeleton_with_timeseries
from classification.lma_extractor import (
    normalize_keypoints, joint_angle,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
)


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot3_shape_angles.mp4"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)

    # joint_angle() is the same
    # function already used inside extract_features to compute the
    # angle histograms, reused identically here for the graph.
    left_angle = joint_angle(kp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, "left")
    right_angle = joint_angle(kp, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "right")

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=right_angle,
        feature_label="angolo gomito destro",
        title="Angoli dei gomiti nel tempo",
        ylabel="angolo (gradi)",
        out_path=OUT_PATH,
        extra_curves=[
            {"values": left_angle, "label": "angolo gomito sinistro", "color": "teal"},
        ],
        angle_arcs=[
            {"vertex": LEFT_ELBOW, "reference": LEFT_SHOULDER, "side": "left",
             "values": left_angle, "color": "teal"},
            {"vertex": RIGHT_ELBOW, "reference": RIGHT_SHOULDER, "side": "right",
             "values": right_angle, "color": "darkorange"},
        ],
    )


if __name__ == "__main__":
    main()