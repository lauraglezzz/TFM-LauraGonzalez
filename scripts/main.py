import os
import sys
import argparse

# =========================================================
# PATH
# =========================================================
ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC_PATH = os.path.join(ROOT_PATH, "src")

sys.path.append(ROOT_PATH)
sys.path.append(SRC_PATH)

# ==============================
# DATA & MODELING
# ==============================

from src.data.build_dataset import build_all_datasets
from src.data.sanity import run_all_sanity_checks

from src.analysis.reproduce_table1 import reproduce_table1
from src.analysis.shap_postprocessing import run_full_analysis

# REGRESSION
from src.modeling.train_lightgbm import (
    train_and_evaluate
)

from src.modeling.train_lightgbm_log import (
    train_and_evaluate_log
)

# SHAP
from src.modeling.shap_analysis import run_all_shap
from src.analysis.shap_local_explanations import (
    run_all_local_shap
)

# CLASSIFICATION
from src.modeling.train_lightgbm_classifier import (
    run_all_classifiers
)

from src.analysis.shap_local_classifier import (
    run_all_local_shap_classifier
)

# ==============================
# LLM
# ==============================

from llm.core.explanation_generator import (
    run_llm_explanations
)

from src.analysis.generate_local_llm_explanations import (
    run_local_llm
)

# ==============================
# LLM EXPERIMENTS
# ==============================

from llm.experiments.run_llm_model_comparison import (
    run_model_comparison
)

from llm.experiments.run_llm_prompt_experiments import (
    run_prompt_experiments
)

# ==============================
# LLM EVALUATION
# ==============================

from llm.evaluation.run_llm_judge import (
    run_llm_judge
)

from src.analysis.evaluate_llm_explanations import (
    run_evaluation
)

from src.analysis.llm_as_a_judge_results import (
    run_judge_analysis
)

# ==============================
# ENDPOINTS
# ==============================

ENDPOINTS = [
    "ADME_HLM_rdkit.csv",
    "ADME_hPPB_rdkit.csv",
    "ADME_MDR1_ER_rdkit.csv",
    "ADME_RLM_rdkit.csv",
    "ADME_rPPB_rdkit.csv",
    "ADME_Sol_rdkit.csv",
]

# =========================================================
# TRAINING — REAL SCALE REGRESSION
# =========================================================

def run_training_pipeline():

    print("\n==============================")
    print("Training REAL SCALE regression")
    print("==============================")

    results = []

    for csv_file in ENDPOINTS:

        print(f"\nTraining {csv_file}...")

        res = train_and_evaluate(csv_file)

        results.append(res)

    import pandas as pd

    os.makedirs(
        "reports/results",
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        "reports/results/final_results_real.csv",
        index=False
    )

    print("\nSaved REAL SCALE results")


# =========================================================
# TRAINING — LOG SCALE REGRESSION
# =========================================================

def run_training_pipeline_log():

    print("\n==============================")
    print("Training LOG SCALE regression")
    print("==============================")

    results = []

    for csv_file in ENDPOINTS:

        print(f"\nTraining {csv_file}...")

        res = train_and_evaluate_log(csv_file)

        results.append(res)

    import pandas as pd

    os.makedirs(
        "reports/results_log",
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        "reports/results_log/final_results_log.csv",
        index=False
    )

    print("\nSaved LOG SCALE results")


# =========================================================
# TRAINING — CLASSIFICATION
# =========================================================

def run_training_pipeline_classifier():

    print("\n==============================")
    print("Training CLASSIFICATION models")
    print("==============================")

    run_all_classifiers()

    print("\nSaved classification results")


# =========================================================
# FULL PIPELINE — REAL SCALE
# =========================================================

def run_full_pipeline_real():

    print("\n========== REAL PIPELINE ==========\n")

    build_all_datasets()

    run_all_sanity_checks()

    reproduce_table1()

    # REAL regression
    run_training_pipeline()

    # SHAP
    run_all_shap()

    run_all_local_shap()

    # LLM
    run_local_llm()

    print("\n========== REAL PIPELINE COMPLETE ==========\n")


# =========================================================
# FULL PIPELINE — LOG SCALE
# =========================================================

def run_full_pipeline_log():

    print("\n========== LOG PIPELINE ==========\n")

    build_all_datasets()

    run_all_sanity_checks()

    reproduce_table1()

    # LOG regression
    run_training_pipeline_log()

    print("\n========== LOG PIPELINE COMPLETE ==========\n")


# =========================================================
# FULL PIPELINE — CLASSIFICATION
# =========================================================

def run_full_pipeline_classifier():

    print("\n========== CLASSIFICATION PIPELINE ==========\n")

    # Classifier
    run_training_pipeline_classifier()

    # SHAP classifier
    run_all_local_shap_classifier()

    print("\n========== CLASSIFICATION COMPLETE ==========\n")


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="ADME ML + LLM Pipeline"
    )

    # -------------------------
    # Regression
    # -------------------------
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train REAL SCALE regression"
    )

    parser.add_argument(
        "--train_log",
        action="store_true",
        help="Train LOG SCALE regression"
    )

    # -------------------------
    # Classification
    # -------------------------
    parser.add_argument(
        "--train_clf",
        action="store_true"
    )

    # -------------------------
    # SHAP
    # -------------------------
    parser.add_argument(
        "--shap",
        action="store_true"
    )

    parser.add_argument(
        "--shap_local",
        action="store_true"
    )

    parser.add_argument(
        "--shap_local_clf",
        action="store_true"
    )

    # -------------------------
    # LLM
    # -------------------------
    parser.add_argument(
        "--llm_gen",
        action="store_true"
    )

    parser.add_argument(
        "--llm_local",
        action="store_true"
    )

    # -------------------------
    # Full pipelines
    # -------------------------
    parser.add_argument(
        "--all_real",
        action="store_true"
    )

    parser.add_argument(
        "--all_log",
        action="store_true"
    )

    parser.add_argument(
        "--all_clf",
        action="store_true"
    )

    args = parser.parse_args()

    # =====================================================
    # FULL PIPELINES
    # =====================================================

    if args.all_real:

        run_full_pipeline_real()

        return

    if args.all_log:

        run_full_pipeline_log()

        return

    if args.all_clf:

        run_full_pipeline_classifier()

        return

    # =====================================================
    # REGRESSION
    # =====================================================

    if args.train:

        run_training_pipeline()

    if args.train_log:

        run_training_pipeline_log()

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if args.train_clf:

        run_training_pipeline_classifier()

    # =====================================================
    # SHAP
    # =====================================================

    if args.shap:

        run_all_shap()

    if args.shap_local:

        run_all_local_shap()

    if args.shap_local_clf:

        run_all_local_shap_classifier()

    # =====================================================
    # LLM
    # =====================================================

    if args.llm_gen:

        run_llm_explanations()

    if args.llm_local:

        run_local_llm()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    main()