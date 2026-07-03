import os
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

# Silenziamo i warning di NumPy per i segmenti che potrebbero avere tutti NaN
warnings.filterwarnings('ignore', category=RuntimeWarning)

from lma_extractor import extract_features

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = "outputs/classification/multiclass_classification.pkl"
PKL_DIR = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

NUM_SEGMENTS = 10  # Lo stesso numero di segmenti usato nel training

def get_label(filename):
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None

# ==========================================
# MAIN PREDICTION & EVALUATION
# ==========================================
def main():
    print(f"Caricamento pipeline modello da: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    rows = []

    print(f"Estrazione feature OOD in {NUM_SEGMENTS} segmenti dai file in: {PKL_DIR}")
    for f in os.listdir(PKL_DIR):
        if not f.endswith(".pkl") or "_sMM_" in f:
            continue

        label = get_label(f)
        if label is None:
            continue

        with open(os.path.join(PKL_DIR, f), "rb") as file:
            data = pickle.load(file)

        keypoints = data["keypoints2d"][0]
        fps = data.get("fps", 60)

        # Dividiamo il video di YouTube in 10 segmenti temporali
        segments = np.array_split(keypoints, NUM_SEGMENTS, axis=0)

        for segment_kp in segments:
            if len(segment_kp) < 10:
                continue

            feat = extract_features(segment_kp, fps)

            if feat is None:
                continue

            feat["label"] = label
            feat["sequence"] = f.replace(".pkl", "")
            rows.append(feat)

    if len(rows) == 0:
        print("Nessun segmento valido estratto dai file OOD.")
        return

    df = pd.DataFrame(rows)
    
    # Separiamo le feature dai metadati prima dell'allineamento
    X_test = df.drop(columns=["label", "sequence"])
    y_test = df["label"]

    # Allineamento dinamico delle colonne attese dalla Pipeline caricata
    try:
        expected_cols = clf.feature_names_in_
    except AttributeError:
        try:
            expected_cols = clf.named_steps['imputer'].feature_names_in_
        except AttributeError:
            expected_cols = clf.named_steps['scaler'].feature_names_in_
        
    X_test = X_test.reindex(columns=expected_cols, fill_value=np.nan)

    print("\n--- YOUTUBE TEST RESULTS (MAJORITY VOTING PER VIDEO) ---")
    
    # 1. Predizione sui singoli segmenti OOD
    pred_segments = clf.predict(X_test)
    proba_segments = clf.predict_proba(X_test)

    # 2. Creazione DataFrame di supporto per il calcolo del Majority Voting
    test_results = df[["label", "sequence"]].copy()
    test_results["pred_segment"] = pred_segments
    
    # Colonne per salvare le probabilità necessarie alla Top-3
    proba_cols = [f"prob_{c}" for c in range(len(CLASSES))]
    test_results[proba_cols] = proba_segments

    # 3. Raggruppamento per video (sequence) e calcolo della Moda
    video_predictions = test_results.groupby("sequence")["pred_segment"].apply(
        lambda x: x.mode().iloc[0]
    )
    video_labels = test_results.groupby("sequence")["label"].first()
    
    # Media delle probabilità dei segmenti per stimare la confidenza complessiva del video
    video_probas = test_results.groupby("sequence")[proba_cols].mean().values

    # 4. Report metriche e Confusion Matrix
    print(classification_report(video_labels, video_predictions, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(video_labels, video_predictions, labels=list(range(len(CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("LMA Segments + Majority Voting — YouTube Test (OOD)")
    plt.tight_layout()
    plt.show()

    top3 = top_k_accuracy_score(video_labels, video_probas, k=3)
    print(f"Top-3 Accuracy (Media Segmenti): {top3:.4f}")
    print(f"Macro F1 Score (Majority Voting): {f1_score(video_labels, video_predictions, average='macro'):.4f}")


if __name__ == "__main__":
    main()