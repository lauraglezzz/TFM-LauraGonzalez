# =========================================================
# FILE:
# src/modeling/train_lightgbm_classifier.py
# =========================================================

from pathlib import Path
import pandas as pd
import numpy as np

import lightgbm as lgb
import joblib

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# =====================================
# PATHS
# =====================================
PROCESSED_PATH = Path("data/processed")

RESULTS_PATH = Path("reports/results")
MODELS_PATH = Path("models")
PREDICTIONS_PATH = Path("reports/predictions")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)


# =====================================
# DISCRETIZATION
# =====================================
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


# =====================================
# TRAIN CLASSIFIER
# =====================================
def train_classifier(csv_filename: str):

    endpoint = csv_filename.replace("_rdkit.csv", "")

    print(f"\nTraining CLASSIFIER for {endpoint}")

    df = pd.read_csv(PROCESSED_PATH / csv_filename)

    # =====================================
    # REAL SCALE
    # =====================================
    df["activity"] = 10 ** df["activity"]

    # =====================================
    # CREATE CLASSES
    # =====================================
    df["class"] = df["activity"].apply(
        lambda x: discretize(endpoint, x)
    )

    X = df.drop(columns=["activity", "class"])
    y = df["class"]

    # =====================================
    # SPLIT
    # =====================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=84,
        stratify=y
    )

    # =====================================
    # MODEL
    # =====================================
    model = lgb.LGBMClassifier(
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        subsample_freq=1,
        n_jobs=-1,
        random_state=42,
    )

    # =====================================
    # CROSS VALIDATION
    # =====================================
    rkf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=128
    )

    acc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="accuracy",
        cv=rkf
    )

    acc_cv = np.mean(acc_scores)

    # =====================================
    # TRAIN
    # =====================================
    model.fit(X_train, y_train)

    # =====================================
    # SAVE MODEL
    # =====================================
    model_name = csv_filename.replace(
        "_rdkit.csv",
        "_classifier.pkl"
    )

    joblib.dump(model, MODELS_PATH / model_name)

    # =====================================
    # PREDICT
    # =====================================
    y_pred = model.predict(X_test)

    # =====================================
    # SAVE PREDICTIONS
    # =====================================
    pred_df = pd.DataFrame({
        "true_class": y_test.values,
        "prediction": y_pred
    })

    pred_df.to_csv(
        PREDICTIONS_PATH / f"{endpoint}_classification_predictions.csv",
        index=False
    )

    # =====================================
    # METRICS
    # =====================================
    acc_test = accuracy_score(y_test, y_pred)

    f1_test = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # =====================================
    # CONFUSION MATRIX
    # =====================================
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm)

    cm_df.to_csv(
        RESULTS_PATH / f"{endpoint}_confusion_matrix.csv",
        index=False
    )

    # =====================================
    # RESULTS
    # =====================================
    results = {
        "endpoint": endpoint,
        "Accuracy_CV": acc_cv,
        "Accuracy_test": acc_test,
        "F1_test": f1_test,
    }

    return results


# =====================================
# RUN ALL
# =====================================
def run_all_classifiers():

    endpoints = [
        "ADME_HLM_rdkit.csv",
        "ADME_hPPB_rdkit.csv",
        "ADME_MDR1_ER_rdkit.csv",
        "ADME_RLM_rdkit.csv",
        "ADME_rPPB_rdkit.csv",
        "ADME_Sol_rdkit.csv",
    ]

    results = []

    for ep in endpoints:
        results.append(train_classifier(ep))

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_PATH / "classification_results.csv",
        index=False
    )

    print("\nClassification results saved.")