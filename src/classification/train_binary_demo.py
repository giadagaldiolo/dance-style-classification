"""
Addestra un classificatore BINARIO (2 stili) dedicato alla demo dal vivo,
riusando lo stesso dataset di feature già estratto per il classificatore
completo a 10 classi (outputs/classification/multiclass_classification.csv)
— nessuna nuova estrazione di keypoint necessaria.

Perché un modello separato per la demo:
con 10 stili il problema di classificazione è intrinsecamente più difficile
(l'abbiamo visto empiricamente per tutta la tesi: confusione tra classi
simili, accuracy OOD limitata attorno al 40-45%). Restringendo il compito
a solo 2 stili scelti per la demo, il problema diventa molto più semplice
e il modello può essere molto più affidabile — importante per una demo
dal vivo dove serve un risultato riproducibile davanti al professore.

IMPORTANTE: le etichette NON vengono rimappate a 0/1 — restano gli indici
originali (0-9) della lista CLASSES completa. Questo evita un bug subdolo
nello script realtime: se il modello binario restituisse un indice come "9"
(gWA) e lo si cercasse in una lista CLASSES troncata a solo 2 elementi,
si otterrebbe un IndexError proprio nel momento in cui il modello riconosce
correttamente lo stile — il peggior momento possibile per un crash.
"""

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


DATASET = "outputs/classification/multiclass_classification.csv"
MODEL_PATH_DEMO = "outputs/classification/binary_demo_classification.pkl"

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

# <-- SCEGLI QUI i due stili per la demo (nomi esatti come in CLASSES sopra)
CLASSES_DEMO = ["gBR", "gWA"]


def main():
    df = pd.read_csv(DATASET)

    demo_label_indices = [CLASSES.index(c) for c in CLASSES_DEMO]
    df_demo = df[df["label"].isin(demo_label_indices)].copy()

    print(f"Sequenze totali nel dataset: {len(df)}")
    print(f"Sequenze per la demo ({' vs '.join(CLASSES_DEMO)}): {len(df_demo)}")

    # Split per coreografia, stessa logica usata per il modello a 10 classi
    df_demo["base_name"] = df_demo["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_demo, groups=df_demo["base_name"]))

    train_df = df_demo.iloc[train_idx]
    test_df = df_demo.iloc[test_idx]

    X_train = train_df.drop(columns=["label", "sequence", "base_name"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    y_test = test_df["label"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH_DEMO), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH_DEMO)

    print("\n--- RISULTATI (classificatore binario per la demo) ---")
    pred = pipeline.predict(X_test)
    print(classification_report(y_test, pred,
                                  labels=demo_label_indices,
                                  target_names=CLASSES_DEMO))

    cm = confusion_matrix(y_test, pred, labels=demo_label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES_DEMO)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {' vs '.join(CLASSES_DEMO)}")
    plt.tight_layout()
    cm_path = "outputs/classification/binary_demo_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Matrice di confusione salvata → {cm_path}")

    print(f"\nModello binario salvato → {MODEL_PATH_DEMO}")


if __name__ == "__main__":
    main()