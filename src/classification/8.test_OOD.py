import os
import pickle
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

from lma_extractor import extract_features

DATASET = "outputs/classification/multiclass_classification.csv" 
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
KEYPOINT_DIR = "outputs/keypoints"

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
        names.append(filename)

    df = pd.DataFrame(X)
    y = np.array(y)
    ood_df = df.copy()

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

    # --- CONFRONTO SISTEMATICO SU TUTTE LE FEATURE ---
    train_df = pd.read_csv(DATASET)
    feature_cols = [c for c in expected_cols if c in ood_df.columns]

    shift_report = []
    for col in feature_cols:
        train_vals = train_df[col].dropna()
        ood_vals = ood_df[col].dropna()

        if len(train_vals) < 2 or len(ood_vals) < 1:
            continue

        train_mean, train_std = train_vals.mean(), train_vals.std()
        ood_mean = ood_vals.mean()

        # quanti "std del training" dista in media l'OOD dal training
        z = (ood_mean - train_mean) / train_std if train_std > 0 else np.nan

        # quota di valori OOD completamente fuori dal range [min, max] del training
        out_of_range = ((ood_vals < train_vals.min()) | (ood_vals > train_vals.max())).mean()

        shift_report.append({
            'feature': col,
            'train_mean': train_mean,
            'ood_mean': ood_mean,
            'z_score': z,
            'pct_out_of_range': out_of_range
        })

    shift_df = pd.DataFrame(shift_report).sort_values('z_score', key=abs, ascending=False)
    pd.set_option('display.max_rows', 100)
    print(shift_df.to_string(index=False))

    top_shifted = shift_df.head(8)['feature'].tolist()

    for col in top_shifted:
        plt.figure(figsize=(7, 4))
        plt.hist(train_df[col].dropna(), bins=25, alpha=0.5,
                  label='train', density=True, color='C0')
        for val in ood_df[col].dropna():
            plt.axvline(val, color='C1', alpha=0.7, linewidth=1.5)
        plt.plot([], [], color='C1', label='OOD (singoli campioni)')
        plt.legend()
        plt.title(col)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()