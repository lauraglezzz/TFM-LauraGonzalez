# =========================================================
# FILE:
# src/modeling/train_lightgbm_log.py
# =========================================================

from pathlib import Path
import pandas as pd
import numpy as np

from scipy.stats import pearsonr

from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    mean_squared_error,
    make_scorer,
)

import lightgbm as lgb
import joblib


# =====================================
# PATHS
# =====================================
PROCESSED_PATH = Path("data/processed")

RESULTS_PATH = Path("reports/results_log")
MODELS_PATH = Path("models_log")
PREDICTIONS_PATH = Path("reports/predictions")

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)


# =====================================
# PEARSON SCORER
# =====================================
def pearson_r_score(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


pearson_scorer = make_scorer(
    pearson_r_score,
    greater_is_better=True
)


# =====================================
# TRAIN + EVALUATE
# =====================================
def train_and_evaluate_log(csv_filename: str):

    endpoint = csv_filename.replace("_rdkit.csv", "")

    print(f"\nTraining REAL regression for {endpoint}")

    csv_path = PROCESSED_PATH / csv_filename

    df = pd.read_csv(csv_path)

    # =====================================
    # FEATURES / TARGET
    # =====================================
    X = df.drop(columns=["activity"])
    y = df["activity"]

    # =====================================
    # SPLIT
    # =====================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=84,
    )

    # =====================================
    # MODEL
    # =====================================
    model = lgb.LGBMRegressor(
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
    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=128
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring=pearson_scorer,
        cv=rkf
    )

    pearson_cv = np.mean(cv_scores)

    # =====================================
    # TRAIN
    # =====================================
    model.fit(X_train, y_train)

    # =====================================
    # SAVE MODEL
    # =====================================
    model_name = csv_filename.replace(
        "_rdkit.csv",
        "_lightgbm.pkl"
    )

    joblib.dump(model, MODELS_PATH / model_name)

    # =====================================
    # PREDICT
    # =====================================
    y_pred_test = model.predict(X_test)

    # =====================================
    # SAVE PREDICTIONS
    # =====================================
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "prediction": y_pred_test
    })

    pred_df.to_csv(
        PREDICTIONS_PATH / f"{endpoint}_regression_predictions.csv",
        index=False
    )

    # =====================================
    # METRICS
    # =====================================
    pearson_test = pearsonr(
        y_test,
        y_pred_test
    )[0]

    mse_test = mean_squared_error(
        y_test,
        y_pred_test
    )

    # =====================================
    # RESULTS
    # =====================================
    results = {
        "endpoint": endpoint,
        "Pearson_r_CV": pearson_cv,
        "Pearson_r_test": pearson_test,
        "MSE_test": mse_test,
    }

    return results