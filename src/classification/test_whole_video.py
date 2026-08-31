import os
import pickle
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score

from lma_extractor import extract_features

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Pretrained "whole video" model, produced by train_whole_video.py.
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
# Folder with the keypoints extracted from the YouTube out-of-domain videos
KEYPOINT_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]
FULL_NAMES = {
    "gBR": "Break", "gHO": "House", "gJB": "Ballet Jazz", "gJS": "Street Jazz",
    "gKR": "Krump", "gLH": "LA-style Hip-hop", "gLO": "Lock", "gMH": "Middle Hip-hop",
    "gPO": "Pop", "gWA": "Waack",
}


def get_label(filename):
    # Returns the integer class index for a given filename, based on
    # its genre prefix (e.g. "gBR_..." -> index of "gBR" in CLASSES)
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

    X = []
    y = []

    # --- Feature extraction ---
    for filename in os.listdir(KEYPOINT_DIR):
        if not filename.endswith(".pkl") or "_sMM_" in filename:
            continue

        label = get_label(filename)
        if label is None:
            continue

        path = os.path.join(KEYPOINT_DIR, filename)
        with open(path, "rb") as f:
            data = pickle.load(f)

        keypoints = data["keypoints2d"][0]
        fps = data.get("fps", 60)

        features = extract_features(keypoints, fps)

        if features is None:
            print("Skipped (troppi NaN):", filename)
            continue

        X.append(features)
        y.append(label)

    df = pd.DataFrame(X)
    y = np.array(y)

    # Recovers the exact feature column order/names the model was
    # trained on, so that reindex() below can align this new data to it
    # even if some columns are missing or in a different order.
    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps['imputer'].feature_names_in_

    df = df.reindex(columns=expected_cols, fill_value=np.nan)

    print("\n--- TEST RESULTS (OOD DATA) ---")
    pred = clf.predict(df)

    print(classification_report(y, pred, target_names=CLASSES, zero_division=0, digits=4))

    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    plot_confusion_matrix_styled(cm, CLASSES)

    proba = clf.predict_proba(df)
    top3 = top_k_accuracy_score(y, proba, k=3)

    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(y, pred, average='macro'):.4f}")

    plt.show()


if __name__ == "__main__":
    main()