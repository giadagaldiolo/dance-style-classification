import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.metrics import accuracy_score
from lma_extractor import extract_features


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from sustainability.sustainability_tracker import track, log_metric, get_file_size_mb


warnings.filterwarnings('ignore', category=RuntimeWarning)

KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/segments_classification.csv"
MODEL_PATH = "outputs/classification/segments_classification.pkl"

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

# Number of equal-length segments each video is split into, inspired by
# Hamscher et al. [2]: features are computed per segment, and predictions
# are recombined at the end with a majority vote (see evaluate_model()).
NUM_SEGMENTS = 10


def get_label(filename):
    # Returns the integer class index for a given AIST++ filename, based
    # on its genre prefix (e.g. "gBR_..." -> index of "gBR" in CLASSES)
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
    # Everything inside this "with" block is timed and its energy
    # consumption measured (via CodeCarbon, see sustainability_tracker.py).
    with track("segments", metadata={
            "model": "RandomForest",
            "n_estimators": 300,
            "approach": f"{NUM_SEGMENTS} segmenti + majority voting",
            }):
        rows = []

        # --- Feature extraction: splits each video into NUM_SEGMENTS
        # equal parts, and computes a separate row of 73 LMA features
        # for EACH segment (not one row per video, unlike whole_video). ---
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

            # Splits the frame sequence into NUM_SEGMENTS chunks of
            # (roughly) equal length -- np.array_split handles sequences
            # whose length isn't evenly divisible by NUM_SEGMENTS too.
            segments = np.array_split(keypoints, NUM_SEGMENTS, axis=0)

            for segment_kp in segments:
                if len(segment_kp) < 10:
                    # Skips segments too short to compute reliable features from.
                    continue

                features = extract_features(segment_kp, fps)

                if features is None:
                    continue

                features["label"] = label
                # "sequence" here is the ORIGINAL video's filename,
                # shared by all of its segments -- this is what later
                # allows grouping segment-level predictions back together per video 
                features["sequence"] = filename.replace(".pkl", "")
                rows.append(features)

        df = pd.DataFrame(rows)
        df.to_csv(DATASET, index=False)
        print(f"Dataset salvato con {len(df)} segmenti e {len(df.columns)-2} feature.")

        pipeline, X_test, y_test, test_df = train_model()

    accuracy = evaluate_model(pipeline, X_test, y_test, test_df)

    # Logs the results to the sustainability log file (used later by
    # generate_report.py / plot_two_pipelines.py to build the charts).
    log_metric("segments",
               accuracy=accuracy,
               model_size_mb=get_file_size_mb(MODEL_PATH))
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
        ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, X_test, y_test, test_df


def evaluate_model(pipeline, X_test, y_test, test_df):
    # --- Segment-level predictions ---
    proba_segments = pipeline.predict_proba(X_test)
    pred_segments = pipeline.classes_[np.argmax(proba_segments, axis=1)]

    print("\nTEST RESULTS")
    test_results = test_df.copy()
    test_results["pred_segment"] = pred_segments

    proba_cols = [f"prob_{c}" for c in range(len(CLASSES))]
    test_results[proba_cols] = proba_segments

    # --- Recombining segment-level results into video-level results ---
    # Majority vote: the video's final predicted class is whichever class
    # was predicted most often among its segments (x.mode().iloc[0] picks
    # the most frequent value; iloc[0] breaks ties by taking the first one).
    video_predictions = test_results.groupby("sequence")["pred_segment"].apply(
        lambda x: x.mode().iloc[0]
    )

    video_labels = test_results.groupby("sequence")["label"].first()
    # Top-3 accuracy below uses the MEAN of segment probabilities per video
    video_probas = test_results.groupby("sequence")[proba_cols].mean().values

    print(classification_report(video_labels, video_predictions, target_names=CLASSES, digits=4))

    cm = confusion_matrix(video_labels, video_predictions, labels=list(range(len(CLASSES))))
    plot_confusion_matrix_styled(cm, CLASSES)


    top3 = top_k_accuracy_score(video_labels, video_probas, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(video_labels, video_predictions, average='macro'):.4f}")

    return accuracy_score(video_labels, video_predictions)


if __name__ == "__main__":
    main()