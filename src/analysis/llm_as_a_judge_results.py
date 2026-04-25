import os
import pandas as pd


INPUT_FILE = "reports/results/llm/judge/judge_results.csv"

OUTPUT_GLOBAL = "reports/results/analysis/llm_model_ranking.csv"
OUTPUT_ENDPOINT = "reports/results/analysis/llm_model_ranking_by_endpoint.csv"


def compute_win_rates(df):

    models = set(df["model_a"]).union(set(df["model_b"]))

    results = []

    for model in models:

        wins = 0
        losses = 0
        ties = 0

        for _, row in df.iterrows():

            if row["winner"] == "tie":

                if row["model_a"] == model or row["model_b"] == model:
                    ties += 1

            elif row["winner"] == model:
                wins += 1

            elif row["model_a"] == model or row["model_b"] == model:
                losses += 1

        total = wins + losses + ties

        if total == 0:
            continue

        results.append({
            "model": model,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / total
        })

    df_results = pd.DataFrame(results)

    df_results = df_results.sort_values("win_rate", ascending=False)

    return df_results


def compute_win_rates_by_endpoint(df):

    endpoints = df["endpoint"].unique()

    all_rows = []

    for endpoint in endpoints:

        df_ep = df[df["endpoint"] == endpoint]

        ranking = compute_win_rates(df_ep)

        ranking["endpoint"] = endpoint

        all_rows.append(ranking)

    df_all = pd.concat(all_rows)

    return df_all


def run_judge_analysis():

    print("\n==============================")
    print("Evaluating LLM explanations")
    print("==============================")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = pd.read_csv(INPUT_FILE)

    print("\nLoaded comparisons:", len(df))

    os.makedirs("reports/results/analysis", exist_ok=True)

    # ==========================
    # Global ranking
    # ==========================

    print("\nComputing global ranking...")

    ranking_global = compute_win_rates(df)

    ranking_global.to_csv(OUTPUT_GLOBAL, index=False)

    print("\nGlobal model ranking:")
    print(ranking_global)

    print("\nSaved to:")
    print(OUTPUT_GLOBAL)

    # ==========================
    # Ranking by endpoint
    # ==========================

    print("\nComputing ranking by endpoint...")

    ranking_endpoint = compute_win_rates_by_endpoint(df)

    ranking_endpoint.to_csv(OUTPUT_ENDPOINT, index=False)

    print("\nSaved endpoint ranking to:")
    print(OUTPUT_ENDPOINT)


if __name__ == "__main__":
    run_judge_analysis()