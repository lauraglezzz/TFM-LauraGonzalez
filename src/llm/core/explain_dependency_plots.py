import os
import numpy as np
import pandas as pd

from llm.core.llm_client import query_llm


# ==============================
# CONFIG
# ==============================

ENDPOINTS = [
    "ADME_HLM",
    "ADME_hPPB",
    "ADME_MDR1_ER",
    "ADME_RLM",
    "ADME_rPPB",
    "ADME_Sol",
]

MODEL = "gpt-4.1-mini"

# cuántos descriptors explicar
N_FEATURES = 5

# cuántos puntos mostrar al LLM
N_POINTS = 20

OUTPUT_DIR = "reports/results/llm/dependency_explanations"


PROMPT_TEMPLATE = """
You are a biomedical scientist interpreting machine learning explanations.

Below is the relationship between a molecular descriptor and the predicted drug property.

Descriptor: {feature}

Observed values (descriptor value -> contribution to prediction):

{data}

Explain the relationship between this descriptor and the predicted drug property.

Focus on chemical or biological interpretation.
Do not mention SHAP or machine learning.
"""


# ==============================
# SAMPLE DEPENDENCY DATA
# ==============================

def sample_dependency(feature_values, shap_values):

    df = pd.DataFrame({
        "feature": feature_values,
        "shap": shap_values
    })

    # ordenar por valor del descriptor
    df = df.sort_values("feature")

    # muestreo uniforme a lo largo del rango
    idx = np.linspace(
        0,
        len(df) - 1,
        N_POINTS,
        dtype=int
    )

    sampled = df.iloc[idx]

    rows = []

    for _, r in sampled.iterrows():

        rows.append(
            f"{r['feature']:.3f} -> {r['shap']:.3f}"
        )

    return "\n".join(rows)


# ==============================
# MAIN
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for endpoint in ENDPOINTS:

        print("\n======================")
        print("Endpoint:", endpoint)
        print("======================")

        shap_path = f"reports/shap/{endpoint}/data/shap_values.npy"
        importance_path = f"reports/shap/{endpoint}/data/shap_importance.csv"

        dataset_path = f"data/processed/{endpoint}_rdkit.csv"

        if not os.path.exists(shap_path):

            print("SHAP values not found")
            continue

        shap_values = np.load(shap_path)

        importance = pd.read_csv(importance_path)

        df = pd.read_csv(dataset_path)

        # top descriptors
        features = importance["feature"].tolist()[:N_FEATURES]

        rows = []

        for feature in features:

            print("Explaining:", feature)

            feature_idx = df.columns.get_loc(feature)

            shap_feature = shap_values[:, feature_idx]

            # alinear tamaños
            feature_values = df[feature].values[:len(shap_feature)]

            sampled = sample_dependency(
                feature_values,
                shap_feature
            )

            prompt = PROMPT_TEMPLATE.format(
                feature=feature,
                data=sampled
            )

            explanation = query_llm(
                prompt,
                model=MODEL
            )

            rows.append({

                "endpoint": endpoint,
                "feature": feature,
                "prompt": prompt,
                "explanation": explanation

            })

        output_file = f"{OUTPUT_DIR}/{endpoint}_dependency_explanations.csv"

        pd.DataFrame(rows).to_csv(output_file, index=False)

        print("Saved:", output_file)


if __name__ == "__main__":
    main()