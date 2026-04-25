import os
from llm.core.explanation_generator import run_llm_explanations


STRATEGIES = ["basic", "biomedical", "reasoning"]
TOPK = [2, 3, 5]
MODELS = ["gpt-4.1-mini"]

OUTPUT_DIR = "reports/results/llm/prompt_experiments"


def run_prompt_experiments():

    print("\n==============================")
    print("LLM PROMPT EXPERIMENTS")
    print("==============================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for strategy in STRATEGIES:
        for top_k in TOPK:
            for model in MODELS:

                print("\n====================")
                print(strategy, top_k, model)
                print("====================")

                run_llm_explanations(
                    strategy=strategy,
                    top_k=top_k,
                    model=model,
                    output_dir=OUTPUT_DIR
                )


if __name__ == "__main__":
    run_prompt_experiments()