import os
import pickle
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, top_k_accuracy_score

from lma_extractor import extract_features


KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/multiclass_classification.csv"
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"

os.makedirs(os.path.dirname(DATASET), exist_ok=True)

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
    rows = []

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

        features["label"] = label
        features["sequence"] = filename.replace(".pkl", "")
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(DATASET, index=False)
    print(f"{len(df)} samples e {len(df.columns)-2} feature.")
    
    train_model()

def train_model():
    df = pd.read_csv(DATASET)
    
    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    coreografie_uniche = df["base_name"].unique()

    train_coreo, test_coreo = train_test_split(
        coreografie_uniche, test_size=0.2, random_state=42
    )

    train_df = df[df["base_name"].isin(train_coreo)]
    test_df = df[df["base_name"].isin(test_coreo)]

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    print("\n--- TEST RESULTS ---")
    pred = pipeline.predict(X_test)
    print(classification_report(y_test, pred, target_names=CLASSES))

    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    proba = pipeline.predict_proba(X_test)
    top3 = top_k_accuracy_score(y_test, proba, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(y_test, pred, average='macro'):.4f}")

if __name__ == "__main__":
    main()