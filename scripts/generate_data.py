"""Write the synthetic dataset and its ground truth to data/.

Usage:  python scripts/generate_data.py
Outputs:
  data/synthetic_weekly.csv   observable data (date, spend_*, sales)
  data/true_params.json       the parameters the model should recover
  data/true_components.csv    per-week decomposition of sales (for validation)
"""

import json
from pathlib import Path

from mmm_sandbox.data import generate_synthetic_data

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    df, truth = generate_synthetic_data()
    DATA_DIR.mkdir(exist_ok=True)

    df.to_csv(DATA_DIR / "synthetic_weekly.csv", index=False)
    truth["components"].to_csv(DATA_DIR / "true_components.csv", index=False)
    with open(DATA_DIR / "true_params.json", "w") as f:
        json.dump(truth["params"], f, indent=2)

    print(f"Wrote {len(df)} weeks to {DATA_DIR}/")
    print(df.describe().round(0).T[["mean", "min", "max"]])
    print("\nShare of total sales by component:")
    comp = truth["components"].drop(columns="date")
    print((comp.sum() / comp.sum().sum()).round(3).to_string())


if __name__ == "__main__":
    main()
