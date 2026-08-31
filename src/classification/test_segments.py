import os
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score

from lma_extractor import extract_features

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Pretrained "segments" model, produced by train_segments.py.
MODEL_PATH = "outputs/classification/segments_classification.pkl"
# Folder with the keypoints extracted from the YouTube out-of-domain videos 
PKL_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]
FULL_NAMES = {
    "gBR": "Break", "gHO": "House", "gJB": "Ballet Jazz", "gJS": "Street Jazz",
    "gKR": "Krump", "gLH": "LA-style Hip-hop", "gLO": "Lock", "gMH": "Middle Hip-hop",
    "gPO": "Pop", "gWA": "Waack",
}

# Same number of segments used at training time in train_segments.py
NUM_SEGMENTS = 10


def get_label(filename):
    # Returns the integer class index for a given filename, based on
    # its genre prefix (e.g. "gBR_..." -> index of "gBR" in CLASSES).
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def plot_confusion_matrix_styled(cm, classes):
    full_labels = [FULL_NAMES[c] for c in classes]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=full_labels, yticklabels=full_labels,
        cbar_kws={"label": "Predictions (%)"},
        ax=ax, square=True, linewidths=0.5, linecolor="white",
        vmin=0, vmax=100,
    )
    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("True Genre")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()


def main():
    clf = joblib.load(MODEL_PATH)
    rows = []

    # Feature extraction
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
            # Shared across all segments of the same video, used below to
            # regroup segment-level predictions back into a single
            # video-level prediction (majority voting).
            feat["sequence"] = f.replace(".pkl", "")
            rows.append(feat)

    if len(rows) == 0:
        print("Nessun segmento valido estratto dai file OOD.")
        return

    df = pd.DataFrame(rows)

    X_test = df.drop(columns=["label", "sequence"])
    y_test = df["label"]

    # Recovers the exact feature column order/names the model was
    # trained on, so that reindex() below can align this new data to it
    # even if some columns are missing or in a different order
    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps['imputer'].feature_names_in_

    X_test = X_test.reindex(columns=expected_cols, fill_value=np.nan)

    print("\n--- YOUTUBE TEST RESULTS (MAJORITY VOTING PER VIDEO) ---")

    # --- Segment-level predictions ---
    pred_segments = clf.predict(X_test)
    proba_segments = clf.predict_proba(X_test)

    test_results = df[["label", "sequence"]].copy()
    test_results["pred_segment"] = pred_segments

    proba_cols = [f"prob_{c}" for c in range(len(CLASSES))]
    test_results[proba_cols] = proba_segments

    # --- Recombining segment-level results into video-level results ---
    # Majority vote: the video's final predicted class is whichever class
    # was predicted most often among its segments. (x.mode().iloc[0] picks
        # the most frequent value; iloc[0] breaks ties by taking the first one).
    video_predictions = test_results.groupby("sequence")["pred_segment"].apply(
        lambda x: x.mode().iloc[0]
    )
    video_labels = test_results.groupby("sequence")["label"].first()

    # Top-3 accuracy below uses the MEAN of segment probabilities per video
    video_probas = test_results.groupby("sequence")[proba_cols].mean().values

    print(classification_report(video_labels, video_predictions, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(video_labels, video_predictions, labels=list(range(len(CLASSES))))
    plot_confusion_matrix_styled(cm, CLASSES)

    top3 = top_k_accuracy_score(video_labels, video_probas, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(video_labels, video_predictions, average='macro'):.4f}")

    plt.show()


if __name__ == "__main__":
    main()