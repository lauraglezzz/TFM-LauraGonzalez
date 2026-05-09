# =========================================================
# FILE:
# src/analysis/compare_regression_vs_classification.py
# =========================================================

from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

# =====================================
# PATHS
# =====================================
PRED_PATH = Path("reports/predictions")

RESULTS_PATH = Path("reports/results")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# =====================================
# DISCRETIZATION
# =====================================
def discretize(endpoint, value):

    # HLM / RLM
    if endpoint in ["ADME_HLM", "ADME_RLM"]:

        if value <= 10:
            return 0  # Low

        elif value <= 40:
            return 1  # Moderate

        else:
            return 2  # High

    # PPB
    if endpoint in ["ADME_hPPB", "ADME_rPPB"]:

        if value <= 0.5:
            return 0  # High binding

        elif value <= 5:
            return 1  # Intermediate

        else:
            return 2  # Low binding

    # MDR1
    if endpoint == "ADME_MDR1_ER":

        if value <= 2:
            return 0

        elif value <= 5:
            return 1

        else:
            return 2

    # Solubility
    if endpoint == "ADME_Sol":

        if value < 10:
            return 0

        elif value < 100:
            return 1

        else:
            return 2

    return 0


# =====================================
# COMPARE ONE ENDPOINT
# =====================================
def compare_endpoint(endpoint):

    print(f"\nComparing {endpoint}")

    # =====================================
    # LOAD REGRESSION PREDICTIONS
    # =====================================
    reg = pd.read_csv(
        PRED_PATH / f"{endpoint}_regression_predictions.csv"
    )

    # =====================================
    # CONVERT BACK FROM LOG SCALE
    # =====================================
    reg["y_true_real"] = 10 ** reg["y_true"]

    reg["prediction_real"] = 10 ** reg["prediction"]

    # =====================================
    # DISCRETIZE
    # =====================================
    reg["true_class"] = reg["y_true_real"].apply(
        lambda x: discretize(endpoint, x)
    )

    reg["pred_class"] = reg["prediction_real"].apply(
        lambda x: discretize(endpoint, x)
    )

    # =====================================
    # REGRESSION METRICS
    # =====================================
    reg_acc = accuracy_score(
        reg["true_class"],
        reg["pred_class"]
    )

    reg_f1 = f1_score(
        reg["true_class"],
        reg["pred_class"],
        average="weighted"
    )

    # =====================================
    # LOAD CLASSIFIER PREDICTIONS
    # =====================================
    clf = pd.read_csv(
        PRED_PATH / f"{endpoint}_classification_predictions.csv"
    )

    # =====================================
    # CLASSIFIER METRICS
    # =====================================
    clf_acc = accuracy_score(
        clf["true_class"],
        clf["prediction"]
    )

    clf_f1 = f1_score(
        clf["true_class"],
        clf["prediction"],
        average="weighted"
    )

    # =====================================
    # RESULTS
    # =====================================
    return {

        "endpoint": endpoint,

        "Regression_as_Classification_Accuracy": round(reg_acc, 4),

        "Classifier_Accuracy": round(clf_acc, 4),

        "Regression_as_Classification_F1": round(reg_f1, 4),

        "Classifier_F1": round(clf_f1, 4),
    }


# =====================================
# RUN ALL
# =====================================
def run_comparison():

    endpoints = [

        "ADME_HLM",
        "ADME_hPPB",
        "ADME_MDR1_ER",
        "ADME_RLM",
        "ADME_rPPB",
        "ADME_Sol",
    ]

    results = []

    for ep in endpoints:

        res = compare_endpoint(ep)

        results.append(res)

    # =====================================
    # SAVE
    # =====================================
    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_PATH / "regression_vs_classification.csv",
        index=False
    )

    print("\nComparison saved.")

    print(df)


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":

    run_comparison()