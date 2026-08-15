"""
Benchmark ISOLATO: misura estrazione feature + classificazione su UN
SINGOLO video (non l'intero dataset di training) -- serve per costruire
il grafico "pipeline per una sessione live". Estrazione e classificazione
sono misurate insieme nello stesso blocco track() perché, prese
singolarmente, sono entrambe troppo brevi per una lettura energetica
affidabile.
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

# <-- Usa lo STESSO video (o uno della stessa durata, ~10s) usato in
#     benchmark_pose_estimation.py, cosi' i due grafici restano coerenti
VIDEO_KEYPOINTS_PKL = "outputs/keypoints_live/live_20260811_012422.pkl"
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]


def main():
    with open(VIDEO_KEYPOINTS_PKL, "rb") as f:
        data = pickle.load(f)
    keypoints = data["keypoints2d"][0]
    fps = data.get("fps", 30)
    duration_sec = len(keypoints) / fps
    print(f"Video: {len(keypoints)} frame, {duration_sec:.1f}s")

    clf = joblib.load(MODEL_PATH)

    with track("single_video_pipeline", metadata={
        "approach": "estrazione feature + classificazione, un video",
        "video_duration_sec": round(duration_sec, 1),
    }):
        features = extract_features(keypoints, fps)
        df = pd.DataFrame([features])
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