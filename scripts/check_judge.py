import pandas as pd

df = pd.read_csv("reports/results/llm/judge/judge_results.csv")
print(f"Completadas: {len(df)} / 1500")
print(f"Ties: {df['winner'].eq('tie').sum()} ({100*df['winner'].eq('tie').mean():.1f}%)")
print()
print(df.groupby(["model_a", "model_b"]).size().to_string())
