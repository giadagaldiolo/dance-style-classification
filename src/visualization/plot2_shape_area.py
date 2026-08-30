import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_for_plots import load_keypoints, plot_skeleton_with_timeseries

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from classification.lma_extractor import normalize_keypoints, JOINTS


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot2_shape_area.mp4"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)
    x = kp[:, :, 0]
    y = kp[:, :, 1]

    with np.errstate(invalid="ignore"):
        min_x, max_x = np.nanmin(x[:, JOINTS], axis=1), np.nanmax(x[:, JOINTS], axis=1)
        min_y, max_y = np.nanmin(y[:, JOINTS], axis=1), np.nanmax(y[:, JOINTS], axis=1)

    body_area = (max_x - min_x) * (max_y - min_y)

    mean_area = np.nanmean(body_area)
    max_area = np.nanmax(body_area)

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=body_area,
        feature_label="area del corpo",
        title="Area del corpo nel tempo",
        ylabel="area (unità corporee²)",
        out_path=OUT_PATH,
        show_bounding_box=True,
        hlines=[
            {"value": mean_area, "label": "mean", "color": "green"},
            {"value": max_area, "label": "max", "color": "purple"},
        ],
    )


if __name__ == "__main__":
    main()