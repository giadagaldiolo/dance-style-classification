import os
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

from lma_extractor import extract_features


MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
PKL_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

NUM_SEGMENTS = 10 

def get_label(filename):
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def main():
    clf = joblib.load(MODEL_PATH)
    rows = []

    for f in os.listdir(PKL_DIR):
        if not f.endswith(".pkl") or "_sMM_" in f:
            continue

        label = get_label(f)
        if label is None:
            continue

        with open(os.path.join(PKL_DIR, f), "rb") as file:
            data = pickle.load(file)

        keypoints = data["keypoints2d"][0]
        fps = data.get("fps", 60)

        segments = np.array_split(keypoints, NUM_SEGMENTS, axis=0)

        for segment_kp in segments:
            if len(segment_kp) < 10:
                continue

            feat = extract_features(segment_kp, fps)

            if feat is None:
                continue

            feat["label"] = label
            feat["sequence"] = f.replace(".pkl", "")
            rows.append(feat)

    if len(rows) == 0:
        print("Nessun segmento valido estratto dai file OOD.")
        return

    df = pd.DataFrame(rows)
    
    X_test = df.drop(columns=["label", "sequence"])
    y_test = df["label"]

    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        try:
            expected_cols = clf.named_steps['imputer'].feature_names_in_
        except AttributeError:
            expected_cols = clf.named_steps['scaler'].feature_names_in_
        
    X_test = X_test.reindex(columns=expected_cols, fill_value=np.nan)

    print("\n--- YOUTUBE TEST RESULTS (MAJORITY VOTING PER VIDEO) ---")
    
    pred_segments = clf.predict(X_test)
    proba_segments = clf.predict_proba(X_test)

    test_results = df[["label", "sequence"]].copy()
    test_results["pred_segment"] = pred_segments
    
    proba_cols = [f"prob_{c}" for c in range(len(CLASSES))]
    test_results[proba_cols] = proba_segments

    video_predictions = test_results.groupby("sequence")["pred_segment"].apply(
        lambda x: x.mode().iloc[0]
    )
    video_labels = test_results.groupby("sequence")["label"].first()
    
    video_probas = test_results.groupby("sequence")[proba_cols].mean().values


    print(classification_report(video_labels, video_predictions, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(video_labels, video_predictions, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("LMA Segments + Majority Voting — YouTube Test (OOD)")
    plt.tight_layout()
    plt.show()

    top3 = top_k_accuracy_score(video_labels, video_probas, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(video_labels, video_predictions, average='macro'):.4f}")


if __name__ == "__main__":
    main()