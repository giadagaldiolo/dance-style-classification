import os
import pickle
import warnings
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
from sustainability_tracker import track, log_metric, get_file_size_mb
from sklearn.metrics import accuracy_score

from lma_extractor import extract_features

warnings.filterwarnings('ignore', category=RuntimeWarning)

KEYPOINT_DIR = "annotations/keypoints2d"
DATASET = "outputs/classification/segments_classification.csv"
MODEL_PATH = "outputs/classification/segments_classification.pkl"

os.makedirs(os.path.dirname(DATASET), exist_ok=True)

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

NUM_SEGMENTS = 10  

def get_label(filename):
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None

def main():
    with track("segments", metadata={
            "model": "RandomForest",
            "n_estimators": 300,
            "approach": f"{NUM_SEGMENTS} segmenti + majority voting",
            }):
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
            
            segments = np.array_split(keypoints, NUM_SEGMENTS, axis=0)

            for segment_kp in segments:
                if len(segment_kp) < 10:
                    continue

                features = extract_features(segment_kp, fps)

                if features is None:
                    continue

                features["label"] = label
                features["sequence"] = filename.replace(".pkl", "")
                rows.append(features)

        df = pd.DataFrame(rows)
        df.to_csv(DATASET, index=False)
        print(f"Dataset salvato con {len(df)} segmenti e {len(df.columns)-2} feature.")
        
        accuracy =train_model()

    log_metric("segments",
                   accuracy=accuracy,
                   model_size_mb=get_file_size_mb(MODEL_PATH))
    plt.show()

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
        ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    proba_segments = pipeline.predict_proba(X_test)
    pred_segments = pipeline.classes_[np.argmax(proba_segments, axis=1)]

    print("\nTEST RESULTS")
    test_results = test_df.copy()
    test_results["pred_segment"] = pred_segments

    proba_cols = [f"prob_{c}" for c in range(len(CLASSES))]
    test_results[proba_cols] = proba_segments

    video_predictions = test_results.groupby("sequence")["pred_segment"].apply(
        lambda x: x.mode().iloc[0]
    )

    video_labels = test_results.groupby("sequence")["label"].first()
    video_probas = test_results.groupby("sequence")[proba_cols].mean().values

    print(classification_report(video_labels, video_predictions, target_names=CLASSES))

    cm = confusion_matrix(video_labels, video_predictions, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("LMA Segments + Majority Voting - Confusion Matrix")
    plt.tight_layout()

    top3 = top_k_accuracy_score(video_labels, video_probas, k=3)
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Macro F1 Score: {f1_score(video_labels, video_predictions, average='macro'):.4f}")

    return accuracy_score(video_labels, video_predictions)
    

if __name__ == "__main__":
    main()