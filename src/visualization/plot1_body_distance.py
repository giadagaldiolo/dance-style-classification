import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_for_plots import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import (
    normalize_keypoints, LEFT_HIP, RIGHT_HIP, RIGHT_WRIST, LEFT_ANKLE,
)


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot1_body_distance.mp4"


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

    wrist_mean = np.nanmean(wrist_dist)
    ankle_mean = np.nanmean(ankle_dist)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=wrist_dist,
        feature_label="distanza polso dx",
        feature_color="darkorange",
        title="Distanza dal centro del bacino nel tempo",
        ylabel="distanza (unità corporee)",
        out_path=OUT_PATH,
        distance_lines=[
            {"joint": RIGHT_WRIST, "color": "darkorange"},
            {"joint": LEFT_ANKLE, "color": "teal"},
        ],
        extra_curves=[
            {"values": ankle_dist, "label": "distanza caviglia sx", "color": "teal"},
        ],
        hlines=[
            {"value": wrist_mean, "label": "media polso dx", "color": "darkorange"},
            {"value": ankle_mean, "label": "media caviglia sx", "color": "teal"},
        ]
    )


if __name__ == "__main__":
    main()