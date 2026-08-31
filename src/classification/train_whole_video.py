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
    # Returns the integer class index for a given AIST++ filename, based
    # on its genre prefix (e.g. "gBR_..." -> index of "gBR" in CLASSES)
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def main():
    # Everything inside this "with" block is timed and its energy
    # consumption measured (via CodeCarbon, see sustainability_tracker.py)
    with track("whole_video", metadata={
            "model": "RandomForest",
            "n_estimators": 300,
            "approach": "video intero",
            }):
        t_extraction_start = time.time()
        rows = []

        # --- Feature extraction ---
        for filename in os.listdir(KEYPOINT_DIR):
            if not filename.endswith(".pkl") or "_sMM_" in filename:
                # Skip non-keypoint files and sequences filmed with a moving camera
                continue

            label = get_label(filename)
            if label is None:
                continue

            path = os.path.join(KEYPOINT_DIR, filename)
            with open(path, "rb") as f:
                data = pickle.load(f)

            keypoints = data["keypoints2d"][0]  # (T, 17, 3): x, y, confidence
            fps = data.get("fps", 60)

            features = extract_features(keypoints, fps)

            if features is None:
                # extract_features returns None if too many frames are
                # invalid (NaN) to compute reliable features.
                print("Skipped (too many NaN):", filename)
                continue

            features["label"] = label
            features["sequence"] = filename.replace(".pkl", "")
            rows.append(features)

        df = pd.DataFrame(rows)
        df.to_csv(DATASET, index=False)
        print(f"{len(df)} samples and {len(df.columns)-2} features.")
        t_extraction_end = time.time()

        pipeline, X_test, y_test = train_model()

        t_training_end = time.time()

    # Evaluation is not included in the tracked energy/time measurement 
    accuracy = evaluate_model(pipeline, X_test, y_test)

    extraction_time = t_extraction_end - t_extraction_start
    training_time = t_training_end - t_extraction_end
    total_time = t_training_end - t_extraction_start

    # Logs the results to the sustainability log file (used later by
    # generate_report.py / plot_two_pipelines.py to build the charts).
    log_metric("whole_video",
               accuracy=accuracy,
               model_size_mb=get_file_size_mb(MODEL_PATH),
               extraction_pct=round(100 * extraction_time / total_time, 1),
               training_pct=round(100 * training_time / total_time, 1))

    plt.show()


def train_model():
    df = pd.read_csv(DATASET)

    # Groups sequences by genre + choreography, so that a GroupShuffleSplit
    # never puts the same choreography (danced by a different performer,
    # on different music) in both train and test.
    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["base_name"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    # Pipeline: median imputation for missing (NaN) feature values, then
    # a Random Forest with 300 trees. 
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
    full_labels = [FULL_NAMES[c] for c in classes]
    # Normalizes each row (true class) to percentages, so each row sums to 100%.
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
    # Runs predictions on the test set and reports accuracy, per-class
    # precision/recall/F1, Top-3 accuracy, and macro F1 -- plus the styled
    # confusion matrix. Deliberately called OUTSIDE the track() block in
    # main(), so evaluation cost is not counted in the energy comparison.
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