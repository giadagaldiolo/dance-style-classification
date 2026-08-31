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
FULL_NAMES = {
    "gBR": "Break", "gHO": "House", "gJB": "Ballet Jazz", "gJS": "Street Jazz",
    "gKR": "Krump", "gLH": "LA-style Hip-hop", "gLO": "Lock", "gMH": "Middle Hip-hop",
    "gPO": "Pop", "gWA": "Waack",
}

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET)
    df["base_name"] = df["sequence"].str.split("_ch").str[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["base_name"]))
    test_df = df.iloc[test_idx]

    X_test = test_df.drop(columns=["label", "sequence", "base_name"])

    pipeline = joblib.load(MODEL_PATH)

    X_test_transformed = X_test.copy()

    # Applies every pipeline step EXCEPT the last one (the classifier).
    for name, step in pipeline.steps[:-1]:
        X_test_transformed = step.transform(X_test_transformed)

    X_test_transformed = pd.DataFrame(X_test_transformed, columns=X_test.columns)

    # Extracts the plain classifier (the Random Forest) from the pipeline.
    rf_model = pipeline.named_steps['classifier']

    print("Calcolo dei valori SHAP in corso...")

 
    # If the test set is too large, SHAP hangs. We take a random sample instead.
    #(To get a general sense of feature importance, 500-1000 samples are enough)
    # check_additivity=False disables an internal SHAP check that often causes
    # it to hang indefinitely with newer scikit-learn versions

    SAMPLE_SIZE = 500
    if len(X_test_transformed) > SAMPLE_SIZE:
        print(f"Dataset troppo grande ({len(X_test_transformed)} samples). Campionamento a {SAMPLE_SIZE} per SHAP...")
        X_test_shap = shap.sample(X_test_transformed, SAMPLE_SIZE, random_state=42)
    else:
        X_test_shap = X_test_transformed

    # Initializes the explainer.
    explainer = shap.TreeExplainer(rf_model)

    # Computes SHAP values only on the reduced sample.
    shap_values = explainer.shap_values(X_test_shap, check_additivity=False)

    # Different SHAP/scikit-learn versions return multiclass SHAP values
    # either as a single 3D array, or as a list of one 2D array per
    # class -- this handles both, normalizing to a list either way.
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_list = shap_values


    # PLOT 1: SHAP Feature Importance (Multiclass Bar Plot)
    plt.figure(figsize=(10, 8))
    # For multiclass, shap_values is a list of arrays -- the bar plot stacks them.
    shap.summary_plot(
        shap_values_list,
        X_test_shap,
        plot_type="bar",
        class_names=[FULL_NAMES[c] for c in CLASSES],
        max_display=10,
        show=False
    )
    plt.title("Impact and contribution of the top 10 features to the model predictions")
    plt.tight_layout()


    plot1_path = os.path.join(PLOT_DIR, "shap_global_feature_importance.png")
    plt.savefig(plot1_path, bbox_inches='tight', facecolor='white')
    plt.close()


    # PLOT 2: SHAP Beeswarm Plot for a single class.
    target_class_idx = CLASSES.index("gJS")
    target_class_name = CLASSES[target_class_idx]

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values_list[target_class_idx],
        X_test_shap,  
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