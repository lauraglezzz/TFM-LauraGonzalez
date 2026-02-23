from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

BASE_SHAP_PATH = Path("reports/shap")
ANALYSIS_PATH = BASE_SHAP_PATH / "analysis"
ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)


ENDPOINTS = ["HLM", "RLM", "hPPB", "rPPB", "MDR1_ER", "Sol"]


# =========================================================
# Generate Table 3 style summary (top 5 features)
# =========================================================
def generate_table3():

    summary = {}

    for endpoint in ENDPOINTS:

        csv_path = BASE_SHAP_PATH / endpoint / "data" / "shap_importance.csv"

        df = pd.read_csv(csv_path)
        top_features = df["feature"].head(5).tolist()

        summary[endpoint] = top_features

    table3 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in summary.items()]))

    table3.to_csv(ANALYSIS_PATH / "table3_like.csv", index=False)
    print("Saved Table 3 like summary.")


# =========================================================
# Correlation heatmap (Fig.16 style)
# =========================================================
def generate_correlation_figure():

    correlations = {}

    for endpoint in ENDPOINTS:

        data_path = Path("data/processed") / f"ADME_{endpoint}_rdkit.csv"
        df = pd.read_csv(data_path)

        y = df["activity"]
        X = df.drop(columns=["activity"])

        corr_values = {}

        for col in X.columns:
            try:
                corr_values[col] = pearsonr(X[col], y)[0]
            except:
                corr_values[col] = 0

        correlations[endpoint] = corr_values

    corr_df = pd.DataFrame(correlations)

    # Take only top 15 descriptors (by average abs correlation)
    corr_df["mean_abs"] = corr_df.abs().mean(axis=1)
    corr_df = corr_df.sort_values("mean_abs", ascending=False).drop(columns="mean_abs")
    corr_df = corr_df.head(15)

    corr_df.to_csv(ANALYSIS_PATH / "correlation_values.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, cmap="coolwarm", center=0)
    plt.title("Pearson Correlation of Descriptors with ADME activity")
    plt.tight_layout()
    plt.savefig(ANALYSIS_PATH / "correlation_heatmap.png")
    plt.close()

    print("Saved Fig.16-like correlation heatmap.")


# =========================================================
# RUN ALL
# =========================================================
def run_full_analysis():
    generate_table3()
    generate_correlation_figure()
