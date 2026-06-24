import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

MODEL_PATH = "outputs/classification/rf_model.pkl"
PKL_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]
JOINTS = [
    5, 6,   # shoulders
    7, 8,   # elbows
    9, 10,  # wrists
    11, 12  # hips
]


def load_model():
    return joblib.load(MODEL_PATH)


def extract_features(keypoints):
    selected = keypoints[:, JOINTS, :2]

    x = selected[:, :, 0]
    y = selected[:, :, 1]

    valid = ~(np.isnan(x) | np.isnan(y))

    if np.sum(valid) == 0:
        return None

    features = {}

    for j in range(len(JOINTS)):
        xj = x[:, j][valid[:, j]]
        yj = y[:, j][valid[:, j]]

        if len(xj) == 0:
            continue

        features[f"mean_x_{j}"] = np.mean(xj)
        features[f"mean_y_{j}"] = np.mean(yj)

        features[f"std_x_{j}"] = np.std(xj)
        features[f"std_y_{j}"] = np.std(yj)

        features[f"min_x_{j}"] = np.min(xj)
        features[f"max_x_{j}"] = np.max(xj)

        features[f"range_x_{j}"] = np.max(xj) - np.min(xj)
        features[f"range_y_{j}"] = np.max(yj) - np.min(yj)

    return features



def get_label(filename):
    if filename.startswith("gBR"):
        return 0
    if filename.startswith("gHO"):
        return 1
    if filename.startswith("gJB"):
        return 2
    if filename.startswith("gJS"):
        return 3
    if filename.startswith("gKR"):
        return 4
    if filename.startswith("gLH"):
        return 5
    if filename.startswith("gLO"):
        return 6
    if filename.startswith("gMH"):
        return 7
    if filename.startswith("gPO"):
        return 8
    if filename.startswith("gWA"):
        return 9
    return None


def main():
    clf = load_model()

    X = []
    y = []

    for f in os.listdir(PKL_DIR):
        if not f.endswith(".pkl"):
            continue

        with open(os.path.join(PKL_DIR, f), "rb") as file:
            data = pickle.load(file)

        kp = data["keypoints2d"][0]
        feat = extract_features(kp)

        if feat is None:
            continue

        label = get_label(f)
        if label is None:
            continue

        X.append(feat)
        y.append(label)

    df = pd.DataFrame(X)
    y = np.array(y)


    print("\nYOUTUBE TEST RESULTS")
    pred = clf.predict(df)

    print(classification_report(y, pred))

    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Baseline — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("Macro F1:", f1_score(y, pred, average="macro"))


if __name__ == "__main__":
    main()