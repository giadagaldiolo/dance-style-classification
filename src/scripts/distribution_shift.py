
import os
import pickle

import numpy as np
import pandas as pd

#from lma_extractor import extract_features
from lma_extractor_pre_smoothing import extract_features


KEYPOINT_DIR_TRAIN = "annotations/keypoints2d"
KEYPOINT_DIR_OOD = "outputs/keypoints"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]


def get_label(filename):
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return None


def build_features_dataframe(keypoint_dir, exclude_smm=True):
    """Estrae le feature LMA per tutte le sequenze in una cartella di .pkl.
    Ritorna anche un dataframe con lo score medio di confidenza per
    sequenza (terzo canale dei keypoint), usato per l'analisi di
    correlazione confidenza/effort."""
    rows = []
    confidence_rows = []

    for filename in sorted(os.listdir(keypoint_dir)):
        if not filename.endswith(".pkl"):
            continue
        if exclude_smm and "_sMM_" in filename:
            continue

        label = get_label(filename)
        if label is None:
            continue

        path = os.path.join(keypoint_dir, filename)
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

        if keypoints.shape[-1] >= 3:
            avg_conf = np.nanmean(keypoints[:, :, 2])
        else:
            avg_conf = np.nan
        confidence_rows.append({
            "sequence": filename.replace(".pkl", ""),
            "avg_confidence": avg_conf,
        })

    df = pd.DataFrame(rows)
    conf_df = pd.DataFrame(confidence_rows)
    return df, conf_df


def compute_shift_table(train_df, ood_df, feature_cols):
    shift_report = []
    for col in feature_cols:
        train_vals = train_df[col].dropna()
        ood_vals = ood_df[col].dropna()

        if len(train_vals) < 2 or len(ood_vals) < 1:
            continue

        train_mean, train_std = train_vals.mean(), train_vals.std()
        ood_mean = ood_vals.mean()

        z = (ood_mean - train_mean) / train_std if train_std > 0 else np.nan
        out_of_range = ((ood_vals < train_vals.min()) | (ood_vals > train_vals.max())).mean()

        shift_report.append({
            "feature": col,
            "train_mean": train_mean,
            "ood_mean": ood_mean,
            "z_score": z,
            "pct_out_of_range": out_of_range,
        })

    return pd.DataFrame(shift_report).sort_values("z_score", key=abs, ascending=False)


def compute_confidence_correlation(train_df, conf_df, effort_cols):
    merged = train_df.merge(conf_df, on="sequence", how="inner")
    correlations = {}
    for col in effort_cols:
        if col in merged.columns:
            correlations[col] = merged["avg_confidence"].corr(merged[col])
    return correlations


def main():
    print("Estrazione feature dal training set (AIST++)...")
    train_df, train_conf_df = build_features_dataframe(KEYPOINT_DIR_TRAIN)
    print(f"{len(train_df)} sequenze di training estratte.\n")

    print("Estrazione feature dai dati OOD...")
    ood_df, _ = build_features_dataframe(KEYPOINT_DIR_OOD, exclude_smm=False)
    print(f"{len(ood_df)} sequenze OOD estratte.\n")

    feature_cols = [c for c in train_df.columns if c not in ("label", "sequence")]

    print("--- TABELLA DI SHIFT TRAIN vs OOD ---")
    shift_df = compute_shift_table(train_df, ood_df, feature_cols)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    print(shift_df.to_string(index=False))

    print("\n--- CORRELAZIONE CONFIDENZA vs FEATURE DI EFFORT (training) ---")
    effort_cols = [c for c in feature_cols if c.startswith("effort_") or "angular_speed" in c]
    correlations = compute_confidence_correlation(train_df, train_conf_df, effort_cols)
    for col, corr in correlations.items():
        print(f"{col}: correlazione con confidenza = {corr:.3f}")

    # Salva anche su file, comodo per allegare tabelle/screenshot alla tesi
    shift_df.to_csv("outputs/classification/shift_report.csv", index=False)
    print("\nTabella di shift salvata in outputs/classification/shift_report.csv")


if __name__ == "__main__":
    main()