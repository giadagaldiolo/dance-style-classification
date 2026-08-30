import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from lma_extractor import extract_features

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from sustainability.sustainability_tracker import track, log_metric, get_file_size_mb

warnings.filterwarnings('ignore', category=RuntimeWarning)


KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/multiclass_classification.csv"
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"

os.makedirs(os.path.dirname(DATASET), exist_ok=True)

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
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def main():
    with track("whole_video", metadata={
            "model": "RandomForest",
            "n_estimators": 300,
            "approach": "video intero",
            }):
        t_extraction_start = time.time()
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
        t_extraction_end = time.time()

        # SOLO fit + salvataggio qui dentro -- la valutazione avviene
        # fuori dal blocco misurato, di proposito (non la conteggiamo).
        pipeline, X_test, y_test = train_model()

        t_training_end = time.time()

    # Valutazione: FUORI dal "with", non conteggiata nel confronto.
    accuracy = evaluate_model(pipeline, X_test, y_test)

    extraction_time = t_extraction_end - t_extraction_start
    training_time = t_training_end - t_extraction_end
    total_time = t_training_end - t_extraction_start

    log_metric("whole_video",
               accuracy=accuracy,
               model_size_mb=get_file_size_mb(MODEL_PATH),
               extraction_pct=round(100 * extraction_time / total_time, 1),
               training_pct=round(100 * training_time / total_time, 1))

    plt.show()


def train_model():
    """SOLO fit + salvataggio del modello. Nessuna valutazione qui --
    resta la parte che decidiamo di misurare per il confronto energetico."""
    df = pd.read_csv(DATASET)

    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["base_name"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, X_test, y_test




def plot_confusion_matrix_styled(cm, classes):
    """Confusion matrix con percentuali per riga, nomi degli stili per
    intero, e colori caldi sulla diagonale -- stile simile a quello
    usato in Hamscher et al. [6]."""
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

def evaluate_model(pipeline, X_test, y_test):
    """Valutazione (predict, metriche, confusion matrix) -- volutamente
    FUORI dal blocco track(), non la conteggiamo nel costo misurato."""
    proba = pipeline.predict_proba(X_test)
    pred = pipeline.classes_[np.argmax(proba, axis=1)]

    print("\nTEST RESULTS")
    print(classification_report(y_test, pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))
    plot_confusion_matrix_styled(cm, CLASSES)

    top3 = top_k_accuracy_score(y_test, proba, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(y_test, pred, average='macro'):.4f}")

    return accuracy_score(y_test, pred)


if __name__ == "__main__":
    main()