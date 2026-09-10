"""
Print the budget-optimiser results from the saved posterior. No sampling.

Run from the repo root after scripts/fit_model.py:

    python scripts/report_optimizer.py

Optimises the current average weekly budget under two objectives (posterior
mean and P10), compares each with the current and equal splits, and shows
the uncertainty at the chosen spend per channel.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm_sandbox import analysis as A  # noqa: E402
from mmm_sandbox import optimizer as O  # noqa: E402
from mmm_sandbox.posterior import load_posterior  # noqa: E402


def main() -> None:
    idata = load_posterior("artifacts/posterior.nc")
    spend = pd.read_csv("data/synthetic_weekly.csv", parse_dates=["date"])
    draws = A.extract_draws(idata)

    current = np.array([spend[f"spend_{c}"].mean() for c in draws.channels])
    budget = current.sum()
    lower, upper = O.default_bounds(spend, draws.channels)

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    pd.set_option("display.width", 160)
    print(f"Weekly budget: ${budget:,.0f} (current average). Upper bounds = max observed weekly spend per channel.\n")

    for objective in ["mean", 0.10]:
        r = O.optimize_budget(draws, budget, lower, upper, objective=objective, n_starts=12, seed=1)
        n_agree = int((r.start_values >= r.value * (1 - 1e-3)).sum())
        label = "posterior mean" if objective == "mean" else f"P{int(objective * 100)}"
        print(f"=== Objective: {label} of weekly media sales")
        print(f"multi-start: {n_agree}/{r.n_starts} starts within 0.1% of best; converged={r.converged}; "
              f"allocation spread among them ${r.spread:,.0f}")
        alloc = pd.DataFrame({"channel": r.channels, "current": current, "optimized": r.spend, "change": r.spend - current})
        print(alloc.to_string(index=False))
        cmp = O.compare_allocations(
            draws, {"current": current, "equal split": np.full(len(current), budget / len(current)), "optimized": r.spend}, reference="current"
        )
        print(cmp.to_string(index=False))
        print("uncertainty at optimized spend:")
        print(O.spend_uncertainty(draws, r.spend, spend).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
