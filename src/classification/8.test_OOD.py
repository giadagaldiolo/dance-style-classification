import os
import pickle
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


def get_label(filename):
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def main():
    clf = joblib.load(MODEL_PATH)

    X = []
    y = []
    names = []

    for f in os.listdir(PKL_DIR):
        if not f.endswith(".pkl"):
            continue
            
        if "_sMM_" in f:
            continue

        label = get_label(f)
        if label is None:
            continue

        with open(os.path.join(PKL_DIR, f), "rb") as file:
            data = pickle.load(file)

        kp = data["keypoints2d"][0]
        fps = data.get("fps", 60)

        feat = extract_features(kp, fps)

        if feat is None:
            print("Skipped (troppi NaN):", f)
            continue

        X.append(feat)
        y.append(label)
        names.append(f)

    df = pd.DataFrame(X)
    y = np.array(y)

    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        expected_cols = clf.named_steps['imputer'].feature_names_in_
        
    df = df.reindex(columns=expected_cols, fill_value=np.nan)

    print("\n--- TEST RESULTS (OOD DATA) ---")
    pred = clf.predict(df)

    print(classification_report(y, pred, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    proba = clf.predict_proba(df)
    top3 = top_k_accuracy_score(y, proba, k=3)

    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(y, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()