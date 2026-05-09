# =========================================================
# LOCAL SHAP FOR CLASSIFICATION
# =========================================================

from pathlib import Path
import pandas as pd
import shap
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
BASE_OUTPUT_PATH = Path("reports/shap_local_classifier")


# =========================
# DISCRETIZATION
# =========================
def discretize(endpoint, value):

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        if value <= 10:
            return 0
        elif value <= 40:
            return 1
        else:
            return 2

    elif endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        if value <= 0.5:
            return 0
        elif value <= 5:
            return 1
        else:
            return 2

    elif endpoint == "ADME_MDR1_ER":
        if value <= 2:
            return 0
        elif value <= 5:
            return 1
        else:
            return 2

    elif endpoint == "ADME_Sol":
        if value < 10:
            return 0
        elif value < 100:
            return 1
        else:
            return 2

    return 0


# =========================
# MAIN FUNCTION
# =========================
def run_local_shap_classifier(csv_filename: str):

    endpoint = csv_filename.replace("_rdkit.csv", "")
    print(f"\nRunning LOCAL SHAP (CLASSIFIER) for {endpoint}")

    df = pd.read_csv(PROCESSED_PATH / csv_filename)

    # escala real
    df["activity"] = 10 ** df["activity"]
    df["class"] = df["activity"].apply(lambda x: discretize(endpoint, x))

    X = df.drop(columns=["activity", "class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=84, stratify=y
    )

    # cargar modelo
    model = joblib.load(MODELS_PATH / f"{endpoint}_classifier.pkl")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # =========================
    # LOOP MOLECULAS
    # =========================
    for i in range(len(X_test)):

        sample = X_test.iloc[i:i+1]

        # predicción
        pred_class = int(model.predict(sample)[0])

        # 👉 shap de la clase predicha
        shap_vals = shap_values[pred_class][i]

        # carpeta
        sample_path = BASE_OUTPUT_PATH / endpoint / f"mol_{i}"
        sample_path.mkdir(parents=True, exist_ok=True)

        # =========================
        # TOP FEATURES
        # =========================
        feature_imp = list(zip(X.columns, shap_vals))
        feature_imp = sorted(feature_imp, key=lambda x: abs(x[1]), reverse=True)

        top_features = [
            {"feature": f, "impact": float(v)}
            for f, v in feature_imp[:10]
        ]

        with open(sample_path / "top_features.json", "w") as f:
            json.dump(top_features, f, indent=2)

        # =========================
        # PREDICTION
        # =========================
        with open(sample_path / "prediction.json", "w") as f:
            json.dump({
                "prediction_class": pred_class
            }, f, indent=2)

        # =========================
        # WATERFALL
        # =========================
        plt.figure()

        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[pred_class],
            shap_vals,
            feature_names=X.columns,
            max_display=10,
            show=False
        )

        plt.tight_layout()
        plt.savefig(sample_path / "waterfall.png")
        plt.close()

    print("Done.")


# =========================
# RUN ALL
# =========================
def run_all_local_shap_classifier():

    endpoints = [
        "ADME_HLM_rdkit.csv",
        "ADME_hPPB_rdkit.csv",
        "ADME_MDR1_ER_rdkit.csv",
        "ADME_RLM_rdkit.csv",
        "ADME_rPPB_rdkit.csv",
        "ADME_Sol_rdkit.csv",
    ]

    for ep in endpoints:
        run_local_shap_classifier(ep)