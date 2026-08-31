import os
import sys
import warnings

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from visualization.feature_plots.utils_for_plots import load_keypoints, plot_skeleton_with_timeseries
from classification.lma_extractor import normalize_keypoints, get_speed_accel, RIGHT_WRIST, LEFT_WRIST


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot4_effort_wrists.mp4"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)
    x = kp[:, :, 0]
    y = kp[:, :, 1]

    right_wrist_speed, _ = get_speed_accel(x[:, RIGHT_WRIST], y[:, RIGHT_WRIST], fps)
    left_wrist_speed, _ = get_speed_accel(x[:, LEFT_WRIST], y[:, LEFT_WRIST], fps)

    # Curve shown in the graph: per-frame mean of the two wrists,
    # computed ONLY for visualization (a single, readable line). This is
    # NOT the operation used for the real feature.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed_curve = np.nanmean(np.stack([left_wrist_speed, right_wrist_speed]), axis=0)

    median_speed = np.nanmedian(np.concatenate([left_wrist_speed, right_wrist_speed]))

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=mean_speed_curve,
        feature_label="velocità media polsi",
        feature_color="darkorange",
        title="Velocità dei polsi nel tempo",
        ylabel="velocità (unità corporee/s)",
        out_path=OUT_PATH,
        highlight_joints=[LEFT_WRIST, RIGHT_WRIST],
        hlines=[{"value": median_speed, "label": "mediana", "color": "green"}],
    )


if __name__ == "__main__":
    main()