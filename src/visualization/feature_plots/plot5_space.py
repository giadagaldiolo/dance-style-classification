import os
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from visualization.feature_plots.utils_for_plots import load_keypoints, plot_skeleton_with_timeseries
from classification.lma_extractor import normalize_keypoints, LEFT_HIP, RIGHT_HIP


PKL_PATH = "annotations/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
OUT_PATH = "outputs/feature_plots/plot5_space.mp4"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    keypoints, fps = load_keypoints(PKL_PATH)
    kp = normalize_keypoints(keypoints)
    x, y = kp[:, :, 0], kp[:, :, 1]

    hip_center_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
    hip_center_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2

    k = int(fps)  # 1 second, same definition used in extract_features

    # Same calculation as extract_features: for each frame, distance
    # between the hip center's position now and its position one second
    # earlier. dist_1sec has (n_frames - k) values; realigned to the
    # full timeline by prepending NaNs for the first second, where the
    # feature isn't computable yet (needs one second of "history" first).
    dx = hip_center_x[k:] - hip_center_x[:-k]
    dy = hip_center_y[k:] - hip_center_y[:-k]
    dist_1sec = np.sqrt(dx**2 + dy**2)
    dist_1sec_full = np.concatenate([np.full(k, np.nan), dist_1sec])

    with np.errstate(invalid="ignore"):
        median_vel_1sec = np.nanmedian(dist_1sec)
    if np.isnan(median_vel_1sec):
        median_vel_1sec = 0.0

    plot_skeleton_with_timeseries(
        keypoints, fps,
        feature_values=dist_1sec_full,
        feature_label="velocità bacino (1s)",
        title="Velocità del centro del bacino nel tempo",
        ylabel="velocità (unità corporee/s)",
        out_path=OUT_PATH,
        highlight_joints=[LEFT_HIP, RIGHT_HIP],
        hlines=[{"value": median_vel_1sec, "label": "median", "color": "purple"}],
    )


if __name__ == "__main__":
    main()