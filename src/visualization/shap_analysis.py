import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit


DATASET = "outputs/classification/multiclass_classification.csv" 
MODEL_PATH = "outputs/classification/multiclass_classification.pkl" 
PLOT_DIR = "outputs/shap" 

CLASSES = [
    "gBR", "gHO", "gJB", "gJS", "gKR",
    "gLH", "gLO", "gMH", "gPO", "gWA"
]

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET)
    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["base_name"]))
    test_df = df.iloc[test_idx]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])
    
    pipeline = joblib.load(MODEL_PATH)
    
    # IMPORTANTE: SHAP per gli alberi funziona sul classificatore puro.
    # Dobbiamo prima trasformare X_test passando attraverso imputer e scaler della pipeline.
    X_test_transformed = X_test.copy()
    
    # Applica tutte le trasformazioni della pipeline tranne l'ultimo step (il classificatore)
    for name, step in pipeline.steps[:-1]:
        X_test_transformed = step.transform(X_test_transformed)
        
    # Ritrasformiamo in DataFrame per mantenere i nomi delle feature nei grafici
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    
    # Estraiamo il classificatore puro (la RandomForest)
    rf_model = pipeline.named_steps['classifier']

    print("Calcolo dei valori SHAP in corso...")
    
    # -----------------------------------------------------------------------------------
    # FIX: 
    # 1. Se il test set è troppo grande, SHAP si blocca. Ne prendiamo un campione casuale.
    #    (Per capire l'importanza generale delle feature, 500-1000 samples sono sufficienti)
    # 2. check_additivity=False disabilita un controllo interno di SHAP che spesso causa 
    #    blocchi infiniti con le nuove versioni di scikit-learn.
    # -----------------------------------------------------------------------------------
    
    SAMPLE_SIZE = 500
    if len(X_test_transformed) > SAMPLE_SIZE:
        print(f"Dataset troppo grande ({len(X_test_transformed)} samples). Campionamento a {SAMPLE_SIZE} per SHAP...")
        X_test_shap = shap.sample(X_test_transformed, SAMPLE_SIZE, random_state=42)
    else:
        X_test_shap = X_test_transformed

    # Inizializza l'explainer 
    explainer = shap.TreeExplainer(rf_model)
    
    # Calcola i valori SHAP solo sul campione ridotto
    shap_values = explainer.shap_values(X_test_shap, check_additivity=False)

    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_list = shap_values


    # GRAFICO 1: SHAP Feature Importance (Bar Plot Multiclasse) 
    plt.figure(figsize=(10, 8))
    # shap_values per multiclasse è una lista di array. Il bar plot li impila.
    shap.summary_plot(
        shap_values_list, 
        X_test_shap,
        plot_type="bar", 
        class_names=CLASSES,
        max_display=10, 
        show=False 
    )
    plt.title("Impact and contribution of the top 10 features to the model predictions")
    plt.tight_layout()
    
    
    plot1_path = os.path.join(PLOT_DIR, "shap_global_feature_importance.png")
    plt.savefig(plot1_path, bbox_inches='tight', facecolor='white')
    plt.close() 


    # GRAFICO 2: SHAP Beeswarm Plot per una singola classe
    target_class_idx = CLASSES.index("gMH")
    target_class_name = CLASSES[target_class_idx]
    
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values_list[target_class_idx], 
        X_test_shap, # Usa il DataFrame campionato!
        max_display=10,
        show=False
    )
    plt.title(f"SHAP values for {target_class_name} dance style predictions")
    plt.tight_layout()
    

    plot2_path = os.path.join(PLOT_DIR, f"shap_beeswarm_{target_class_name}.png")
    plt.savefig(plot2_path, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    main()