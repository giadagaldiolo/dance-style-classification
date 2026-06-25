import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.metrics import f1_score
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/multiclass_classification.csv"
MODEL_PATH = "outputs/classification/rf_model.pkl"

os.makedirs(os.path.dirname(DATASET), exist_ok=True)
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
    df.to_csv(DATASET, index=False)
    train_model()


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

    joblib.dump(clf, MODEL_PATH)

    print("\nTEST RESULTS")
    pred = clf.predict(X_test)

    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Baseline — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    proba = clf.predict_proba(X_test)
    top3 = top_k_accuracy_score(
        y_test,
        proba,
        k=3
    )

    print("Top-3 Accuracy:", top3)
    print("Macro F1:", f1_score(y_test, pred, average="macro"))


if __name__ == "__main__":
    main()