# src/analysis/reproduce_table1.py

from pathlib import Path
import pandas as pd


PROCESSED_PATH = Path("data/processed")


def reproduce_table1():
    """
    Reproduce Table 1 statistics from processed datasets.
    """

    files = {
        "Sol": "ADME_Sol_rdkit.csv",
        "MDR1": "ADME_MDR1_ER_rdkit.csv",
        "rPPB": "ADME_rPPB_rdkit.csv",
        "hPPB": "ADME_hPPB_rdkit.csv",
        "RLM": "ADME_RLM_rdkit.csv",
        "HLM": "ADME_HLM_rdkit.csv",
    }

    results = []

    for endpoint, filename in files.items():
        csv_path = PROCESSED_PATH / filename
        df = pd.read_csv(csv_path)

        y = df["activity"]

        stats = {
            "Endpoint": endpoint,
            "N": len(y),
            "Mean": y.mean(),
            "Std": y.std(),
            "Min": y.min(),
            "25%": y.quantile(0.25),
            "50%": y.quantile(0.50),
            "75%": y.quantile(0.75),
            "Max": y.max(),
        }

        results.append(stats)

    table1 = pd.DataFrame(results)

    print("\nReproduced Table 1:\n")
    print(table1)

    return table1
