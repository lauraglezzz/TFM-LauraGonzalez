import os
import re
import random
import pandas as pd
from itertools import combinations

from src.llm.core.llm_client import query_llm


RESULTS_DIR = "reports/results/llm/model_comparison"
OUTPUT_DIR = "reports/results/llm/judge"
OUTPUT_FILE = f"{OUTPUT_DIR}/judge_results.csv"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
N_SAMPLES = 100


PROMPT_TEMPLATE = """You are a biomedical expert evaluating explanations of machine learning predictions for drug properties.

Evaluate the two explanations below based on:
1. Scientific correctness
2. Faithfulness to the molecular descriptors
3. Specificity
4. Clarity for biomedical researchers

--- EXPLANATION 1 (label: FIRST) ---
{exp_a}

--- EXPLANATION 2 (label: SECOND) ---
{exp_b}

Your response must be exactly one of these three tokens and nothing else:
FIRST
SECOND
TIE

Do not write any other text, punctuation, or explanation. Just the single token."""


def parse_result(raw, m1, m2):
    cleaned = raw.strip().upper()
    # Match exactly FIRST, SECOND, or TIE on their own line or as the whole response
    if re.search(r'\bFIRST\b', cleaned):
        return m1
    if re.search(r'\bSECOND\b', cleaned):
        return m2
    return "tie"


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

    # Load existing results to resume
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        rows = existing.to_dict("records")
        print(f"\nResuming from {len(rows)} existing comparisons.")
    else:
        rows = []

    pairs = list(combinations(models, 2))

    for m1, m2 in pairs:

        already_done = sum(1 for r in rows if r["model_a"] == m1 and r["model_b"] == m2)
        if already_done >= N_SAMPLES:
            print(f"\nSkipping {m1} vs {m2} (already complete: {already_done})")
            continue

        print(f"\nComparing {m1} vs {m2} (done so far: {already_done})")

        df1 = data[m1]
        df2 = data[m2]

        n = min(len(df1), len(df2))
        random.seed(42)
        all_indices = random.sample(range(n), min(N_SAMPLES, n))
        indices = all_indices[already_done:]

        for i in indices:

            endpoint = df1.loc[i, "endpoint"]

            exp_a = df1.loc[i, "explanation"]
            exp_b = df2.loc[i, "explanation"]

            prompt = PROMPT_TEMPLATE.format(exp_a=exp_a, exp_b=exp_b)

            raw = query_llm(prompt, model=JUDGE_MODEL)
            winner = parse_result(raw, m1, m2)

            rows.append({
                "model_a": m1,
                "model_b": m2,
                "endpoint": endpoint,
                "winner": winner,
                "raw_response": raw.strip()[:100],
            })

            pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)

        completed = sum(1 for r in rows if r["model_a"] == m1 and r["model_b"] == m2)
        ties = sum(1 for r in rows if r["model_a"] == m1 and r["model_b"] == m2 and r["winner"] == "tie")
        print(f"  Done: {completed} comparisons, {ties} ties ({100*ties//max(completed,1)}%)")

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_llm_judge()