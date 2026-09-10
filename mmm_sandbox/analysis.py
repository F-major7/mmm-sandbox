"""
Everything the app shows, computed from saved posterior draws with numpy.

This module never imports PyMC. It takes the posterior saved by
``scripts/fit_model.py`` (loaded via ``mmm_sandbox.posterior``) plus the
weekly spend table, and produces:

* channel contributions per week, per draw, in dollars
* a sales decomposition (baseline, trend, seasonality, channels)
* channel ROI as a distribution (P10 / P50 / P90)
* marginal ROI: the return on the *next* dollar at current spend
* response curves: expected weekly sales vs constant weekly spend
* a model-fit summary: posterior predictive vs actual

Every function works on a ``Draws`` object, which is the posterior flattened
to plain numpy arrays with the library's scaling already undone. Working in
dollars end to end means every number here can be read off directly.

The arithmetic is deliberately the same equations as ``mmm_sandbox.transforms``,
vectorised over posterior draws so the whole module runs in well under a
second; tests check the vectorised versions against the reference loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from mmm_sandbox.posterior import posterior_params_in_dollars

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


# --------------------------------------------------------------------------- #
# Posterior draws as plain arrays
# --------------------------------------------------------------------------- #

@dataclass
class Draws:
    """
    The posterior, flattened to ``n_draws`` rows, in dollars.

    Shapes: ``(D,)`` per draw, ``(D, C)`` per draw and channel,
    ``(D, T)`` per draw and week. ``C`` follows ``channels``; ``T`` follows
    ``dates``.
    """

    channels: list[str]
    dates: pd.DatetimeIndex
    l_max: int
    alpha: np.ndarray        # (D, C) carryover
    K: np.ndarray            # (D, C) half-saturation, weekly dollars
    S: np.ndarray            # (D, C) Hill slope
    beta: np.ndarray         # (D, C) max weekly contribution, dollars
    baseline: np.ndarray     # (D,)   intercept, dollars per week
    trend: np.ndarray        # (D, T) trend contribution, dollars
    seasonality: np.ndarray  # (D, T) seasonal contribution, dollars
    sigma: np.ndarray        # (D,)   noise SD, dollars
    target_scale: float      # max weekly sales used by the library

    @property
    def n_draws(self) -> int:
        return self.alpha.shape[0]


def _flat(da) -> np.ndarray:
    """Merge (chain, draw, ...) into (draws, ...)."""
    v = np.asarray(da.values)
    return v.reshape(-1, *v.shape[2:])


def extract_draws(idata: az.InferenceData) -> Draws:
    """Flatten a saved posterior into a ``Draws`` object in dollars."""
    import json

    params = posterior_params_in_dollars(idata)
    post = idata.posterior
    scale = float(idata.constant_data["target_scale"])
    meta = json.loads(idata.attrs.get("mmm_sandbox", "{}"))
    return Draws(
        channels=[str(c) for c in params["channel"].values],
        dates=pd.DatetimeIndex(post["date"].values),
        l_max=int(meta.get("l_max", 8)),
        alpha=_flat(params["alpha"]),
        K=_flat(params["K"]),
        S=_flat(params["S"]),
        beta=_flat(params["beta"]),
        baseline=_flat(post["intercept_contribution"]) * scale,
        trend=_flat(post["control_contribution"]).sum(axis=-1) * scale,
        seasonality=_flat(post["yearly_seasonality_contribution"]) * scale,
        sigma=_flat(post["y_sigma"]) * scale,
        target_scale=scale,
    )


# --------------------------------------------------------------------------- #
# Transforms vectorised over draws (same equations as transforms.py)
# --------------------------------------------------------------------------- #

def adstock_over_draws(x: np.ndarray, alpha: np.ndarray, l_max: int) -> np.ndarray:
    """
    Normalised geometric adstock of one spend series for many alphas at once.

    ``x`` is ``(T,)``, ``alpha`` is ``(D,)``; returns ``(D, T)``. Identical to
    ``transforms.geometric_adstock(x, a, l_max, normalize=True)`` for each
    ``a`` in ``alpha``; the loop here is over lags (8) instead of weeks (104)
    so numpy does the heavy lifting.
    """
    x = np.asarray(x, dtype=float)
    lags = np.arange(l_max)
    weights = alpha[:, None] ** lags[None, :]           # (D, l_max)
    weights = weights / weights.sum(axis=1, keepdims=True)
    out = np.zeros((alpha.shape[0], x.shape[0]))
    for lag in range(l_max):
        # Week t receives weight[lag] * spend from week t - lag.
        out[:, lag:] += weights[:, lag : lag + 1] * x[None, : x.shape[0] - lag]
    return out


def hill_over_draws(x: np.ndarray, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Hill saturation x**S / (K**S + x**S) with ``K`` and ``S`` per draw.

    ``x`` is ``(D, T)`` or ``(T,)``; ``K`` and ``S`` are ``(D,)``. Returns
    ``(D, T)`` in [0, 1].
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    K = K[:, None]
    S = S[:, None]
    return x**S / (K**S + x**S)


# --------------------------------------------------------------------------- #
# Contributions and decomposition
# --------------------------------------------------------------------------- #

def channel_contributions(draws: Draws, spend: pd.DataFrame) -> np.ndarray:
    """
    Weekly contribution of each channel in dollars, per draw: ``(D, T, C)``.

    contribution = beta * hill(adstock(spend)). ``spend`` must have one
    ``spend_<channel>`` column per channel in ``draws.channels``.
    """
    D, T, C = draws.n_draws, len(spend), len(draws.channels)
    out = np.empty((D, T, C))
    for c, name in enumerate(draws.channels):
        x = spend[f"spend_{name}"].to_numpy(dtype=float)
        effective = adstock_over_draws(x, draws.alpha[:, c], draws.l_max)
        out[:, :, c] = draws.beta[:, c : c + 1] * hill_over_draws(effective, draws.K[:, c], draws.S[:, c])
    return out


def decomposition(draws: Draws, spend: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Sales split into its parts, each ``(D, T)`` in dollars.

    Keys: ``baseline``, ``trend``, ``seasonality``, then one per channel.
    The parts sum to the model's expected sales for each draw.
    """
    contrib = channel_contributions(draws, spend)
    parts = {
        "baseline": np.broadcast_to(draws.baseline[:, None], draws.trend.shape).copy(),
        "trend": draws.trend,
        "seasonality": draws.seasonality,
    }
    for c, name in enumerate(draws.channels):
        parts[name] = contrib[:, :, c]
    return parts


def expected_sales(draws: Draws, spend: pd.DataFrame) -> np.ndarray:
    """Model's expected weekly sales (no noise) per draw: ``(D, T)``."""
    return sum(decomposition(draws, spend).values())


def summarize(values: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """P10 / P50 / P90 and mean along ``axis`` (default: across draws)."""
    out = {name: np.quantile(values, q, axis=axis) for name, q in QUANTILES.items()}
    out["mean"] = values.mean(axis=axis)
    return out


def decomposition_table(draws: Draws, spend: pd.DataFrame) -> pd.DataFrame:
    """
    Total contribution of each component over the whole period, with
    P10 / P50 / P90 across draws and the share of total expected sales.
    """
    parts = decomposition(draws, spend)
    total = sum(parts.values()).sum(axis=1)  # (D,)
    rows = []
    for name, series in parts.items():
        s = series.sum(axis=1)  # (D,)
        q = summarize(s)
        rows.append(
            {
                "component": name,
                "p10": q["p10"],
                "p50": q["p50"],
                "p90": q["p90"],
                "share_p50": float(np.median(s / total)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #

def channel_roi(draws: Draws, spend: pd.DataFrame, true_components: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    ROI per channel = total incremental sales / total spend, as a posterior
    distribution summarised by P10 / P50 / P90.

    "Incremental" is relative to zero spend; since hill(0) = 0 that is just
    the channel's contribution. When ``true_components`` (from the data
    generator) is given, a ``true_roi`` column is added so the recovery can
    be judged on the business metric, not only on parameters.
    """
    contrib = channel_contributions(draws, spend)          # (D, T, C)
    total_spend = np.array([spend[f"spend_{c}"].sum() for c in draws.channels])
    roi = contrib.sum(axis=1) / total_spend[None, :]        # (D, C)
    q = summarize(roi)
    table = pd.DataFrame(
        {
            "channel": draws.channels,
            "total_spend": total_spend,
            "roi_p10": q["p10"],
            "roi_p50": q["p50"],
            "roi_p90": q["p90"],
            "roi_mean": q["mean"],
        }
    )
    if true_components is not None:
        table["true_roi"] = [
            true_components[f"contribution_{c}"].sum() / spend[f"spend_{c}"].sum() for c in draws.channels
        ]
        table["true_inside_p10_p90"] = (table["true_roi"] >= table["roi_p10"]) & (table["true_roi"] <= table["roi_p90"])
    return table


def marginal_roi(draws: Draws, spend: pd.DataFrame, at_spend: np.ndarray | None = None) -> pd.DataFrame:
    """
    Sales gained from one more dollar per week, per channel, at a given
    steady weekly spend (default: each channel's average weekly spend).

    This is the derivative of the response curve beta * hill(x):

        d/dx = beta * S * K**S * x**(S-1) / (K**S + x**S)**2

    Average ROI (``channel_roi``) says what a channel *has* returned;
    marginal ROI says what the *next* dollar returns, which is what a budget
    optimiser trades on. A saturated channel can have high average ROI and
    low marginal ROI at the same time.
    """
    if at_spend is None:
        at_spend = np.array([spend[f"spend_{c}"].mean() for c in draws.channels])
    x = np.asarray(at_spend, dtype=float)[None, :]  # (1, C)
    K, S, beta = draws.K, draws.S, draws.beta
    slope = beta * S * K**S * x ** (S - 1) / (K**S + x**S) ** 2  # (D, C)
    q = summarize(slope)
    return pd.DataFrame(
        {
            "channel": draws.channels,
            "at_weekly_spend": x.ravel(),
            "mroi_p10": q["p10"],
            "mroi_p50": q["p50"],
            "mroi_p90": q["p90"],
        }
    )


# --------------------------------------------------------------------------- #
# Response curves
# --------------------------------------------------------------------------- #

def response_curve(draws: Draws, channel: str, spend_grid: np.ndarray) -> dict[str, np.ndarray]:
    """
    Expected weekly sales contribution vs a *constant* weekly spend.

    With normalised adstock, a constant weekly spend x is carried over into
    exactly x (the weights sum to 1), so the steady-state curve is simply
    beta * hill(x). One curve per draw; returned as P10 / P50 / P90 / mean
    arrays over ``spend_grid`` plus the grid itself.
    """
    c = draws.channels.index(channel)
    curves = draws.beta[:, c : c + 1] * hill_over_draws(spend_grid, draws.K[:, c], draws.S[:, c])  # (D, G)
    out = summarize(curves)
    out["spend"] = np.asarray(spend_grid, dtype=float)
    return out


def default_spend_grid(spend: pd.DataFrame, channel: str, n: int = 100, extend: float = 1.5) -> np.ndarray:
    """0 to ``extend`` x the channel's max observed weekly spend."""
    return np.linspace(0.0, extend * spend[f"spend_{channel}"].max(), n)


# --------------------------------------------------------------------------- #
# Model fit
# --------------------------------------------------------------------------- #

def fit_summary(idata: az.InferenceData, hdi_prob: float = 0.94) -> tuple[pd.DataFrame, dict]:
    """
    Posterior predictive vs actual sales, in dollars, week by week, plus
    in-sample RMSE and R-squared.

    The posterior predictive includes the noise term, so its interval is
    the range of *sales* the model would expect to see, not just the mean.
    """
    scale = float(idata.constant_data["target_scale"])
    pp = idata.posterior_predictive["y"] * scale
    obs = idata.observed_data["y"].values * scale
    hdi = az.hdi(pp, hdi_prob=hdi_prob)["y"].values
    mean = pp.mean(("chain", "draw")).values
    table = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(idata.observed_data["date"].values),
            "actual": obs,
            "predicted_mean": mean,
            "predicted_low": hdi[:, 0],
            "predicted_high": hdi[:, 1],
        }
    )
    resid = obs - mean
    stats = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": float(1 - np.sum(resid**2) / np.sum((obs - obs.mean()) ** 2)),
        "coverage": float(np.mean((obs >= hdi[:, 0]) & (obs <= hdi[:, 1]))),
        "hdi_prob": hdi_prob,
    }
    return table, stats
