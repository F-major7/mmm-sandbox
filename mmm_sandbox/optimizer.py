"""
Fixed-budget reallocation across channels, using posterior draws.

Question answered: given a total weekly budget and per-channel spend limits,
which split maximises predicted weekly sales?

How it works
------------
* Predicted sales for an allocation are the *steady-state* response from
  ``analysis.py``: with normalised adstock, constant weekly spend x is
  carried into exactly x, so each channel contributes beta * hill(x).
  Baseline, trend and seasonality do not depend on spend and are left out;
  only the media part moves.
* That prediction is computed for every posterior draw, giving a
  distribution of sales for each candidate allocation. The optimiser scores
  an allocation with one number from that distribution:

    - ``objective="mean"``   the posterior mean (the expected-value answer)
    - ``objective=0.10``     a quantile, e.g. P10 (the risk-averse answer:
                             "which split does well even if the uncertain
                             channels turn out weak?")

  The quantile is taken of *total* sales per draw, i.e. sum the four
  channels inside each draw first, then take the quantile across draws.
  Taking per-channel quantiles and adding them would assume every channel
  is unlucky in the same scenario and overstate the risk.
* The search is scipy's SLSQP with an equality constraint (spend sums to
  budget) and box bounds. Variables are spend as a fraction of budget and
  the objective is sales per budget dollar, so everything the solver sees
  is order 1 and its finite-difference gradients are accurate.
* The objective is not concave: an S-shaped channel (Hill slope > 1) has an
  accelerating region, so a local optimum is possible. ``optimize_budget``
  therefore runs from several starting allocations and returns the best,
  and reports whether the starts agree.
* A quantile of 1000 draws is not smooth: it has a kink wherever two draws
  swap order, which at our scale is every hundred dollars or so. SLSQP's
  finite-difference gradients stall on those kinks. For quantile objectives
  each SLSQP endpoint is therefore polished with Nelder-Mead, a
  derivative-free method that steps over kinks. The mean objective is
  smooth and needs no polish.

Default bounds keep each channel at or below its maximum *observed* weekly
spend, so the optimiser never leans on the part of a response curve the
data never saw. That is a hard cap, not a judgement about the curve; the
uncertainty at the chosen spend is reported separately as a P10-P90 band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mmm_sandbox.analysis import Draws, hill_over_draws, summarize


# --------------------------------------------------------------------------- #
# Predicted sales for an allocation
# --------------------------------------------------------------------------- #

def media_sales_per_draw(draws: Draws, spend: np.ndarray) -> np.ndarray:
    """
    Steady-state weekly media sales for one allocation, per draw: ``(D,)``.

    ``spend`` is one weekly dollar amount per channel, in ``draws.channels``
    order. Each channel's contribution is beta * hill(spend); the four are
    summed *within* each draw.
    """
    x = np.asarray(spend, dtype=float)
    total = np.zeros(draws.n_draws)
    for c in range(len(draws.channels)):
        total += draws.beta[:, c] * hill_over_draws(np.array([x[c]]), draws.K[:, c], draws.S[:, c])[:, 0]
    return total


def score(sales_per_draw: np.ndarray, objective) -> float:
    """
    Collapse a distribution of sales to the single number being maximised.

    ``objective`` is ``"mean"`` or a quantile level in (0, 1).
    """
    if objective == "mean":
        return float(sales_per_draw.mean())
    q = float(objective)
    if not 0.0 < q < 1.0:
        raise ValueError(f"objective must be 'mean' or a quantile in (0, 1); got {objective!r}")
    return float(np.quantile(sales_per_draw, q))


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #

@dataclass
class OptimizationResult:
    channels: list[str]
    objective: object
    budget: float
    spend: np.ndarray                 # best allocation found, dollars per week
    value: float                      # objective at that allocation
    sales_p10: float
    sales_p50: float
    sales_p90: float
    n_starts: int
    start_values: np.ndarray          # objective reached from each start
    start_spends: np.ndarray          # (n_starts, C) allocation reached from each start
    converged: bool                   # every start's objective is within tol of the best
    spread: float = field(default=0.0)  # max allocation distance (dollars) among starts within tol of the best

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"channel": self.channels, "spend": self.spend})


def default_bounds(spend_history: pd.DataFrame, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Lower bound 0, upper bound = max observed weekly spend per channel.

    The upper bound keeps the optimiser inside the range the model was fit
    on. Callers can raise it deliberately; the response curve past it is an
    extrapolation and the P10-P90 band there says how much of one.
    """
    lower = np.zeros(len(channels))
    upper = np.array([spend_history[f"spend_{c}"].max() for c in channels], dtype=float)
    return lower, upper


def _random_starts(budget: float, lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    ``n`` feasible-ish starting allocations: equal split first, then random
    Dirichlet splits of the budget clipped to the bounds. SLSQP repairs any
    small constraint violation from clipping on its first step.
    """
    C = len(lower)
    starts = [np.full(C, budget / C)]
    for _ in range(n - 1):
        starts.append(budget * rng.dirichlet(np.ones(C)))
    return np.clip(np.array(starts), lower, upper)


def _polish(objective_fn, x0: np.ndarray, budget: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Derivative-free refinement for non-smooth (quantile) objectives.

    Nelder-Mead over the first C-1 channels; the last channel takes the
    remainder of the budget so the sum constraint holds by construction,
    and out-of-bounds points are given a huge penalty.
    """
    C = len(x0)

    def unpack(z):
        return np.append(z, budget - z.sum())

    def penalised(z):
        x = unpack(z)
        if np.any(x < lower - 1e-9) or np.any(x > upper + 1e-9):
            return 1e12
        return -objective_fn(x)

    # Simplex edges of 1% of budget: big enough to step over kinks.
    step = 0.01 * budget
    simplex = np.vstack([x0[:-1]] + [x0[:-1] + step * np.eye(C - 1)[i] for i in range(C - 1)])
    res = minimize(
        penalised,
        x0[:-1],
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1.0, "fatol": 1.0, "maxiter": 4000},
    )
    x = unpack(res.x)
    return x if -res.fun > objective_fn(x0) else x0  # never make it worse


def optimize_budget(
    draws: Draws,
    budget: float,
    lower: np.ndarray,
    upper: np.ndarray,
    objective="mean",
    n_starts: int = 8,
    seed: int = 0,
    tol: float = 1e-3,
) -> OptimizationResult:
    """
    Maximise ``score(media_sales_per_draw(spend), objective)`` subject to
    ``sum(spend) == budget`` and ``lower <= spend <= upper``.

    Runs SLSQP from ``n_starts`` starting points (equal split plus random
    Dirichlet splits), polishes each with Nelder-Mead when the objective is
    a quantile, and keeps the best.

    ``converged`` is True when every start reached an objective within
    ``tol`` (relative) of the best. ``spread`` is the largest distance, in
    dollars, between the allocations of those starts: a large spread with
    ``converged=True`` means the optimum is a plateau, i.e. several splits
    are equally good, which is itself useful to know.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not lower.sum() <= budget <= upper.sum():
        raise ValueError(f"budget {budget:,.0f} is outside the feasible range [{lower.sum():,.0f}, {upper.sum():,.0f}]")

    def objective_fn(x):
        return score(media_sales_per_draw(draws, x), objective)

    # Work in fractions of budget so the solver sees numbers of order 1.
    def negative_objective(w):
        return -objective_fn(w * budget) / budget

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = list(zip(lower / budget, upper / budget))

    rng = np.random.default_rng(seed)
    starts = _random_starts(budget, lower, upper, n_starts, rng)
    finals, values = [], []
    for x0 in starts:
        res = minimize(
            negative_objective,
            x0 / budget,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500, "eps": 1e-6},
        )
        x = np.clip(res.x, lower / budget, upper / budget) * budget
        if objective != "mean":
            x = _polish(objective_fn, x, budget, lower, upper)
        finals.append(x)
        values.append(objective_fn(x))
    finals = np.array(finals)
    values = np.array(values)

    best = int(np.argmax(values))
    within_tol = values >= values[best] * (1 - tol)
    spend = finals[best]
    sales = media_sales_per_draw(draws, spend)
    q = summarize(sales)
    return OptimizationResult(
        channels=draws.channels,
        objective=objective,
        budget=budget,
        spend=spend,
        value=float(values[best]),
        sales_p10=float(q["p10"]),
        sales_p50=float(q["p50"]),
        sales_p90=float(q["p90"]),
        n_starts=n_starts,
        start_values=values,
        start_spends=finals,
        converged=bool(within_tol.all()),
        spread=float(np.abs(finals[within_tol] - spend).max()),
    )


# --------------------------------------------------------------------------- #
# Comparisons and uncertainty at the chosen spend
# --------------------------------------------------------------------------- #

def compare_allocations(draws: Draws, allocations: dict[str, np.ndarray], reference: str) -> pd.DataFrame:
    """
    Predicted weekly media sales for several allocations, with the lift of
    each over ``reference`` computed *within draws* (so the lift has its own
    P10 / P50 / P90 that respects the shared uncertainty).
    """
    sales = {name: media_sales_per_draw(draws, x) for name, x in allocations.items()}
    ref = sales[reference]
    rows = []
    for name, s in sales.items():
        q = summarize(s)
        lift = summarize(s - ref)
        rows.append(
            {
                "allocation": name,
                "total_spend": float(np.sum(allocations[name])),
                "sales_p10": q["p10"],
                "sales_p50": q["p50"],
                "sales_p90": q["p90"],
                "lift_p10": lift["p10"],
                "lift_p50": lift["p50"],
                "lift_p90": lift["p90"],
            }
        )
    return pd.DataFrame(rows)


def spend_uncertainty(draws: Draws, spend: np.ndarray, spend_history: pd.DataFrame) -> pd.DataFrame:
    """
    For each channel at the given weekly spend: where that spend sits
    relative to the observed range, and the P10-P90 band on that channel's
    predicted contribution, in dollars and as a fraction of the median.

    This is how extrapolation shows up: a channel pushed past the spend
    levels the data covers has a wide band, and the number is reported
    rather than hidden behind a cutoff.
    """
    rows = []
    for c, name in enumerate(draws.channels):
        hist = spend_history[f"spend_{name}"]
        contrib = draws.beta[:, c] * hill_over_draws(np.array([spend[c]]), draws.K[:, c], draws.S[:, c])[:, 0]
        q = summarize(contrib)
        rows.append(
            {
                "channel": name,
                "spend": float(spend[c]),
                "share_of_max_observed": float(spend[c] / hist.max()),
                "pct_of_weeks_at_or_above": float((hist >= spend[c]).mean()),
                "contrib_p10": q["p10"],
                "contrib_p50": q["p50"],
                "contrib_p90": q["p90"],
                "band_width": q["p90"] - q["p10"],
                "band_over_p50": (q["p90"] - q["p10"]) / q["p50"] if q["p50"] > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
