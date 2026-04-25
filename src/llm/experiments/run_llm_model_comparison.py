import os
from llm.core.explanation_generator import run_llm_explanations


# ==============================
# CONFIG
# ==============================

STRATEGY = "biomedical"
TOP_K = 3

MODELS = [
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "llama3",
    "mistral",
    "phi3",
    "gemma"
]

OUTPUT_DIR = "reports/results/llm/model_comparison"


# ==============================
# RUN
# ==============================

def run_model_comparison():

    print("\n==============================")
    print("LLM MODEL COMPARISON")
    print("==============================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model in MODELS:

        print("\n====================")
        print("MODEL:", model)
        print("====================")

        run_llm_explanations(
            strategy=STRATEGY,
            top_k=TOP_K,
            model=model,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    run_model_comparison()