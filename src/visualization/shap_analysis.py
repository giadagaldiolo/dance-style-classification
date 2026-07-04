import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# CONFIGURATION
# ==========================================
DATASET = "outputs/classification/multiclass_classification.csv" # Modifica con il tuo dataset corretto (es. con o senza LMA custom)
MODEL_PATH = "outputs/classification/multiclass_classification.pkl" # Modifica con il path del tuo modello

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

def main():
    print("Caricamento dataset e modello...")
    
    # 1. Carica Dataset (usiamo lo stesso random_state per ricreare lo stesso X_test)
    df = pd.read_csv(DATASET)
    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["base_name"]))
    test_df = df.iloc[test_idx]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    
    # 2. Carica Modello (Pipeline)
    pipeline = joblib.load(MODEL_PATH)
    
    # IMPORTANTE: SHAP per gli alberi funziona sul classificatore puro.
    # Dobbiamo prima trasformare X_test passando attraverso imputer e scaler della pipeline.
    print("Trasformazione dei dati (Imputer + Scaler)...")
    X_test_transformed = X_test.copy()
    
    # Applica tutte le trasformazioni della pipeline tranne l'ultimo step (il classificatore)
    for name, step in pipeline.steps[:-1]:
        X_test_transformed = step.transform(X_test_transformed)
        
    # Ritrasformiamo in DataFrame per mantenere i nomi delle feature nei grafici
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    
    # Estraiamo il classificatore puro (la RandomForest)
    rf_model = pipeline.named_steps['classifier']

    print("Calcolo dei valori SHAP (potrebbe richiedere qualche minuto)...")
    # Inizializza l'explainer specifico per gli alberi decisionali
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # --- FIX PER LE NUOVE VERSIONI DI SHAP ---
    # Le versioni recenti restituiscono un array 3D (samples, features, classes) invece di una lista
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_list = shap_values
    # -----------------------------------------

    # ==========================================
    # GRAFICO 1: SHAP Feature Importance (Bar Plot Multiclasse) - Simile a Fig 3
    # ==========================================
    print("Generazione Fig 3: Global Feature Importance (Stacked Bar)...")
    plt.figure(figsize=(10, 8))
    # shap_values per multiclasse è una lista di array. Il bar plot li impila.
    shap.summary_plot(
        shap_values_list, 
        X_test_transformed, 
        plot_type="bar", 
        class_names=CLASSES,
        max_display=10, # Mostra solo le top 10 feature come nel paper
        show=False # show=False permette di personalizzare il titolo prima di renderizzare
    )
    plt.title("Impact and contribution of the top 10 features to the model predictions")
    plt.tight_layout()
    plt.show()

    # ==========================================
    # GRAFICO 2: SHAP Beeswarm Plot per una singola classe - Simile a Fig 4
    # ==========================================
    # Scegliamo la classe da analizzare (es. gMH - Middle Hip Hop, che è all'indice 7)
    target_class_idx = CLASSES.index("gMH")
    target_class_name = CLASSES[target_class_idx]
    
    print(f"Generazione Fig 4: Beeswarm Plot per la classe '{target_class_name}'...")
    plt.figure(figsize=(10, 8))
    
    # Passiamo solo i valori SHAP corrispondenti alla classe scelta
    shap.summary_plot(
        shap_values_list[target_class_idx], 
        X_test_transformed, 
        max_display=10,
        show=False
    )
    plt.title(f"SHAP values for {target_class_name} dance style predictions")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()