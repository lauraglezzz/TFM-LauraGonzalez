from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

BASE_SHAP_PATH = Path("reports/shap")
ANALYSIS_PATH = BASE_SHAP_PATH / "analysis"
ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)


# =========================================================
# Automatically detect endpoints (ADME_* folders)
# =========================================================
def get_available_endpoints():
    return [
        p.name for p in BASE_SHAP_PATH.iterdir()
        if p.is_dir() and p.name.startswith("ADME_")
    ]


# =========================================================
# Generate Table 3 style summary (Top 5 descriptors per endpoint)
# =========================================================
def generate_table3():

    endpoints = get_available_endpoints()
    summary = {}

    for endpoint in endpoints:

        csv_path = BASE_SHAP_PATH / endpoint / "data" / "mean_abs_shap.csv"

        if not csv_path.exists():
            print(f"Skipping {endpoint} (no SHAP file found)")
            continue

        df = pd.read_csv(csv_path)

        # If descriptor_name exists use it, otherwise fallback to feature
        if "descriptor_name" in df.columns:
            feature_col = "descriptor_name"
        else:
            feature_col = "feature"

        top_features = df[feature_col].head(5).tolist()
        summary[endpoint.replace("ADME_", "")] = top_features

    if not summary:
        print("No SHAP data found.")
        return

    table3 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in summary.items()]))

    table3.to_csv(ANALYSIS_PATH / "table3_like.csv", index=False)

    print("Saved Table 3 like summary.")


# =========================================================
# Correlation heatmap (Fig.16 style)
# =========================================================
def generate_correlation_figure():

    endpoints = get_available_endpoints()
    correlations = {}

    for endpoint in endpoints:

        data_path = Path("data/processed") / f"{endpoint}_rdkit.csv"

        if not data_path.exists():
            print(f"Skipping {endpoint} (dataset not found)")
            continue

        df = pd.read_csv(data_path)

        y = df["activity"]
        X = df.drop(columns=["activity"])

        corr_values = {}

        for col in X.columns:
            try:
                corr_values[col] = pearsonr(X[col], y)[0]
            except:
                corr_values[col] = 0

        correlations[endpoint.replace("ADME_", "")] = corr_values

    if not correlations:
        print("No datasets found for correlation analysis.")
        return

    corr_df = pd.DataFrame(correlations)

    # Take only top 15 descriptors by average absolute correlation
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
    print("\nRunning SHAP post-analysis...")
    generate_table3()
    generate_correlation_figure()
    print("Post-analysis complete.\n")
