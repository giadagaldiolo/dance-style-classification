import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc,classification_report,confusion_matrix)
from sklearn.metrics import roc_auc_score

KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/binary_classification.csv"
os.makedirs(os.path.dirname(DATASET), exist_ok=True)

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
    if filename.startswith("gBR"):
        return 0  # Break

    if filename.startswith("gJB"):
        return 1  # Ballet Jazz

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

    print(df.head())
    print()
    print("Samples:", len(df))
    print("Saved:", DATASET)

    train_model()

def load_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def train_model():
    df = pd.read_csv(DATASET)
    coreografie_uniche = df["sequence"].str.split("_ch").str[0].unique()

    train_coreo, test_val_coreo = train_test_split(
        coreografie_uniche, test_size=0.3, random_state=42
    )
    val_coreo, test_coreo = train_test_split(
        test_val_coreo, test_size=0.5, random_state=42
    )

    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    train_df = df[df["base_name"].isin(train_coreo)]
    val_df = df[df["base_name"].isin(val_coreo)]
    test_df = df[df["base_name"].isin(test_coreo)]

    print("train samples:", len(train_df))
    print("val samples:", len(val_df))
    print("test samples:", len(test_df))

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]

    X_val = val_df.drop(columns=["label", "sequence", "base_name"])
    y_val = val_df["label"]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("\nVALIDATION RESULTS")
    val_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_pred))
    print(confusion_matrix(y_val, val_pred))

    print("\nTEST RESULTS")
    test_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print("AUC:", auc)


if __name__ == "__main__":
    main()
