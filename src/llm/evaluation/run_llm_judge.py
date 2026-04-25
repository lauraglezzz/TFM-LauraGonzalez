import os
import random
import pandas as pd
from itertools import combinations

from llm.core.llm_client import query_llm


RESULTS_DIR = "reports/results/llm/model_comparison"
OUTPUT_DIR = "reports/results/llm/judge"
OUTPUT_FILE = f"{OUTPUT_DIR}/judge_results.csv"

JUDGE_MODEL = "gpt-4o-mini"
N_SAMPLES = 100


PROMPT_TEMPLATE = """
You are a biomedical expert evaluating explanations of machine learning predictions for drug properties.

Evaluate the explanations based on:

1. Scientific correctness
2. Faithfulness to the molecular descriptors
3. Specificity
4. Clarity for biomedical researchers

Explanation A:
{exp_a}

Explanation B:
{exp_b}

Which explanation is better?

Respond ONLY with:

A
B
TIE
"""


def run_llm_judge():

    print("\n==============================")
    print("LLM AS A JUDGE")
    print("==============================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]

    data = {}

    for f in files:
        model = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(os.path.join(RESULTS_DIR, f))
        data.setdefault(model, []).append(df)

    for model in data:
        data[model] = pd.concat(data[model]).reset_index(drop=True)

    models = list(data.keys())

    print("\nModels detected:", models)

    rows = []

    pairs = list(combinations(models, 2))

    for m1, m2 in pairs:

        print(f"\nComparing {m1} vs {m2}")

        df1 = data[m1]
        df2 = data[m2]

        n = min(len(df1), len(df2))
        indices = random.sample(range(n), min(N_SAMPLES, n))

        for i in indices:

            endpoint = df1.loc[i, "endpoint"]

            exp_a = df1.loc[i, "explanation"]
            exp_b = df2.loc[i, "explanation"]

            prompt = PROMPT_TEMPLATE.format(exp_a=exp_a, exp_b=exp_b)

            result = query_llm(prompt, model=JUDGE_MODEL).strip().upper()

            if "A" in result:
                winner = m1
            elif "B" in result:
                winner = m2
            else:
                winner = "tie"

            rows.append({
                "model_a": m1,
                "model_b": m2,
                "endpoint": endpoint,
                "winner": winner
            })

            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_llm_judge()