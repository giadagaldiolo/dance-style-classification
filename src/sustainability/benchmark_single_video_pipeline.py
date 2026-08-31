"""
ISOLATED benchmark: measures feature extraction + classification on a
SINGLE video. used to build the
"pipeline for one live session" chart. Extraction and classification are
measured together in the same track() block because, taken individually,
both are too short for a reliable energy reading.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import joblib

from sustainability_tracker import track, log_metric

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classification.lma_extractor import extract_features

# Uses the SAME video (or one of the same ~10s duration) as
# benchmark_pose_estimation.py, so the two charts stay consistent with
# each other (same session, same duration, used across all three phases
# shown in the "pipeline for one 10-second video" chart).
VIDEO_KEYPOINTS_PKL = "outputs/keypoints_live/live_20260811_012422.pkl"
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]


def main():
    # Loading the pre-extracted keypoints and the pretrained model both
    # happen BEFORE the tracked block 
    with open(VIDEO_KEYPOINTS_PKL, "rb") as f:
        data = pickle.load(f)
    keypoints = data["keypoints2d"][0]
    fps = data.get("fps", 30)
    duration_sec = len(keypoints) / fps
    print(f"Video: {len(keypoints)} frame, {duration_sec:.1f}s")

    clf = joblib.load(MODEL_PATH)

    # Only feature extraction + classification are measured 
    with track("single_video_pipeline", metadata={
        "approach": "estrazione feature + classificazione, un video",
        "video_duration_sec": round(duration_sec, 1),
    }):
        features = extract_features(keypoints, fps)
        df = pd.DataFrame([features])
        # Aligns the feature columns to the exact order/names the model
        # was trained on, same pattern used throughout the project.
        try:
            expected_cols = clf.feature_names_in_
        except AttributeError:
            expected_cols = clf.named_steps["imputer"].feature_names_in_
        df = df.reindex(columns=expected_cols, fill_value=np.nan)
        pred_idx = clf.predict(df)[0]

    print(f"Stile predetto: {CLASSES[pred_idx]}")
    log_metric("single_video_pipeline", predicted_style=CLASSES[pred_idx])


if __name__ == "__main__":
    main()