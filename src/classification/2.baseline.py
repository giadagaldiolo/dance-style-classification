import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc,classification_report,confusion_matrix, f1_score)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/binary_classification.csv"
os.makedirs(os.path.dirname(DATASET), exist_ok=True)
CLASSES = [
    "gHO", 
    "gLH",
]
ID = 10  # right wrist

def extract_features(keypoints):
    hand = keypoints[:, ID, :2]

    x = hand[:, 0]
    y = hand[:, 1]

    valid = ~(np.isnan(x) | np.isnan(y))

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None

    features = {
        "mean_x": np.mean(x),
        "mean_y": np.mean(y),

        "std_x": np.std(x),
        "std_y": np.std(y),

        "min_x": np.min(x),
        "max_x": np.max(x),

        "min_y": np.min(y),
        "max_y": np.max(y),

        "range_x": np.max(x) - np.min(x),
        "range_y": np.max(y) - np.min(y),
    }

    return features


def get_label(filename):
    if filename.startswith("gMH"):
        return 0  # Middle Hip Hop
    if filename.startswith("gLH"):
        return 1  # LA Hip Hop
    return None


def main():
    rows = []

    for filename in os.listdir(KEYPOINT_DIR):

        if not filename.endswith(".pkl"):
            continue

        label = get_label(filename)

        if label is None:
            continue

        path = os.path.join(KEYPOINT_DIR, filename)

        with open(path, "rb") as f:
            data = pickle.load(f)

        keypoints = data["keypoints2d"][0]

        features = extract_features(keypoints)

        if features is None:
            continue

        features["label"] = label
        features["sequence"] = filename.replace(".pkl", "")

        rows.append(features)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(DATASET), exist_ok=True)
    df.to_csv(DATASET, index=False)


    train_model()

def load_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def train_model():
    df = pd.read_csv(DATASET)
    df["base_name"] = df["sequence"].str.split("_ch").str[0]

    coreografie_uniche = df["base_name"].unique()

    train_coreo, test_coreo = train_test_split(
        coreografie_uniche,
        test_size=0.2,
        random_state=42
    )

    train_df = df[df["base_name"].isin(train_coreo)]
    test_df = df[df["base_name"].isin(test_coreo)]

    print("train samples:", len(train_df))
    print("test samples:", len(test_df))

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("\nTEST RESULTS")
    test_pred = clf.predict(X_test)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, test_pred))

    cm = confusion_matrix(y_test, test_pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Baseline — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print("AUC:", auc)
    print("Macro F1:", f1_score(y_test, test_pred, average="macro"))


if __name__ == "__main__":
    main()
