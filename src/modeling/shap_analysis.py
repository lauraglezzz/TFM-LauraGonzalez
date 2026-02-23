# src/modeling/shap_analysis.py

from pathlib import Path
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")
BASE_SHAP_PATH = Path("reports/shap")


def create_endpoint_folders(endpoint_name: str):

    endpoint_path = BASE_SHAP_PATH / endpoint_name

    data_path = endpoint_path / "data"
    summary_path = endpoint_path / "summary_plots"
    dependence_path = endpoint_path / "dependence_plots"

    data_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)
    dependence_path.mkdir(parents=True, exist_ok=True)

    return data_path, summary_path, dependence_path


def run_shap_for_endpoint(csv_filename: str):

    endpoint_name = csv_filename.replace("_rdkit.csv", "")
    print(f"\nRunning SHAP for {endpoint_name}")

    # Create structured folders
    data_path, summary_path, dependence_path = create_endpoint_folders(endpoint_name)

    df = pd.read_csv(PROCESSED_PATH / csv_filename)

    X = df.drop(columns=["activity"])
    y = df["activity"]

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=84
    )

    model_name = endpoint_name + "_lightgbm.pkl"
    model = joblib.load(MODELS_PATH / model_name)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # =========================
    #  Save raw SHAP values
    # =========================
    np.save(data_path / "shap_values.npy", shap_values)

    # =========================
    #  Save importance CSV
    # =========================
    shap_importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False)

    shap_importance.to_csv(data_path / "shap_importance.csv", index=False)

    # =========================
    #  Summary plot
    # =========================
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(summary_path / "shap_summary.png")
    plt.close()

    # =========================
    #  Beeswarm (bar style)
    # =========================
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(summary_path / "beeswarm_bar.png")
    plt.close()

    # =========================
    #  Top 5 dependence plots
    # =========================
    top_features = shap_importance["feature"].head(5)

    for feature in top_features:
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_train,
            show=False
        )
        plt.tight_layout()
        plt.savefig(dependence_path / f"{feature}.png")
        plt.close()

    print("Done.")


def run_all_shap():

    endpoints = [
        "ADME_HLM_rdkit.csv",
        "ADME_hPPB_rdkit.csv",
        "ADME_MDR1_ER_rdkit.csv",
        "ADME_RLM_rdkit.csv",
        "ADME_rPPB_rdkit.csv",
        "ADME_Sol_rdkit.csv",
    ]

    for endpoint in endpoints:
        run_shap_for_endpoint(endpoint)
