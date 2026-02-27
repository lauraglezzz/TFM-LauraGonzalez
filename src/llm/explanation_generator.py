import os
import numpy as np
import pandas as pd

from .prompt_builder import build_prompt
from .llm_client import query_llm


# ==============================
# ENDPOINTS
# ==============================

ENDPOINTS = [
    "ADME_HLM",
    "ADME_hPPB",
    "ADME_MDR1_ER",
    "ADME_RLM",
    "ADME_rPPB",
    "ADME_Sol",
]


# ==============================
# CONFIG
# ==============================

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_STRATEGY = "biomedical"
DEFAULT_TOPK = 3

MAX_SAMPLES = 5 


# ==============================
# MAIN ORCHESTRATOR
# ==============================

def run_llm_explanations(
    model=DEFAULT_MODEL,
    strategy=DEFAULT_STRATEGY,
    top_k=DEFAULT_TOPK,
    max_samples=MAX_SAMPLES,
):

    print("\n==============================")
    print("Generating LLM explanations")
    print("==============================")

    os.makedirs("reports/results/llm", exist_ok=True)

    for endpoint in ENDPOINTS:

        print(f"\nProcessing {endpoint}")

        shap_path = f"reports/shap/{endpoint}/data/shap_values.npy"
        importance_path = f"reports/shap/{endpoint}/data/shap_importance.csv"

        if not os.path.exists(shap_path):
            print(f"SHAP values not found for {endpoint}")
            continue

        shap_values = np.load(shap_path)
        importance = pd.read_csv(importance_path)

        feature_names = importance["feature"].tolist()

        explanations = generate_endpoint_explanations(
            endpoint,
            shap_values,
            feature_names,
            model,
            strategy,
            top_k,
            max_samples,
        )

        output_file = f"reports/results/llm/{endpoint}_{strategy}_{model}.csv"
        explanations.to_csv(output_file, index=False)

        print(f"Saved explanations → {output_file}")


# ==============================
# PER ENDPOINT
# ==============================

def generate_endpoint_explanations(
    endpoint,
    shap_values,
    feature_names,
    model,
    strategy,
    top_k,
    max_samples,
):

    rows = []

    n_samples = min(len(shap_values), max_samples)

    for i in range(n_samples):

        vals = shap_values[i]

        # top features
        idx = np.argsort(np.abs(vals))[::-1][:top_k]

        top_features = [feature_names[j] for j in idx]
        top_vals = [vals[j] for j in idx]

        prompt = build_prompt(
            features=top_features,
            shap_values=top_vals,
            strategy=strategy,
        )

        explanation = query_llm(prompt, model=model)

        rows.append(
            {
                "endpoint": endpoint,
                "sample_id": i,
                "model": model,
                "strategy": strategy,
                "top_k": top_k,
                "features": str(top_features),
                "shap_values": str(top_vals),
                "prompt": prompt,
                "explanation": explanation,
            }
        )

        print(f"Sample {i} done")

    return pd.DataFrame(rows)