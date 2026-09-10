"""
Print the analysis tables from the saved posterior. No sampling.

Run from the repo root after scripts/fit_model.py:

    python scripts/report_analysis.py

Shows the same numbers the Streamlit app renders: sales decomposition,
channel ROI with the synthetic truth beside it, marginal ROI, and the
model-fit summary.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm_sandbox import analysis as A  # noqa: E402
from mmm_sandbox.posterior import load_posterior  # noqa: E402


def main() -> None:
    idata = load_posterior("artifacts/posterior.nc")
    spend = pd.read_csv("data/synthetic_weekly.csv", parse_dates=["date"])
    true_components = pd.read_csv("data/true_components.csv")
    draws = A.extract_draws(idata)

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    pd.set_option("display.width", 140)

    print(f"Posterior: {draws.n_draws} draws, {len(draws.dates)} weeks, channels {draws.channels}\n")

    print("Sales decomposition, totals over the period (P10 / P50 / P90, share of expected sales)")
    print(A.decomposition_table(draws, spend).to_string(index=False))

    print("\nChannel ROI = incremental sales / spend (posterior P10 / P50 / P90 vs synthetic truth)")
    print(A.channel_roi(draws, spend, true_components).to_string(index=False))

    print("\nMarginal ROI = sales from one more weekly dollar at the channel's average weekly spend")
    print(A.marginal_roi(draws, spend).to_string(index=False))

    _, stats = A.fit_summary(idata)
    print("\nModel fit (posterior predictive vs actual, in dollars)")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
