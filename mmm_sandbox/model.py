"""
Bayesian media-mix model: build, fit, save, load, and read back parameters.

This module wraps PyMC-Marketing's ``MMM`` class so that every modelling
decision (transform settings, priors, sampler settings, what gets saved) is
written down in one place with a reason next to it.

The model, in one equation per week t:

    sales_t / max(sales) = intercept
                         + sum over channels c of  beta_c * hill(adstock(spend_c / max(spend_c)))
                         + gamma_trend * t_scaled
                         + yearly seasonality (Fourier terms)
                         + noise

Two scaling facts drive everything below and are worth memorising:

* PyMC-Marketing divides each channel's spend by that channel's *maximum*
  weekly spend, and divides sales by the *maximum* weekly sales, before the
  model sees them. So inside the model, spend and sales both live on a
  0-to-1 axis. Priors are stated on that axis.
* Consequently the fitted half-saturation ``kappa`` is a fraction of max
  spend, and ``beta`` is a fraction of max sales. ``posterior_params_in_dollars``
  multiplies the scale factors back in so the rest of the project can work
  in dollars. ``alpha`` (carryover) and ``slope`` (curve shape) are unitless
  and need no conversion.

Nothing here samples at import time. Only ``fit_mmm`` runs MCMC, and only the
offline script ``scripts/fit_model.py`` calls it. Reading the saved posterior
back lives in ``mmm_sandbox.posterior`` (no PyMC import) so the Streamlit app
never touches this module.
"""

from __future__ import annotations

import json
import warnings

import arviz as az
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, HillSaturation
from pymc_marketing.mmm.transformers import hill_function
from pymc_marketing.mmm.multidimensional import MMM

from mmm_sandbox.data import CHANNELS
from mmm_sandbox.posterior import load_posterior, posterior_params_in_dollars  # noqa: F401  (re-exported)

# --------------------------------------------------------------------------- #
# Structural choices
# --------------------------------------------------------------------------- #

# Carryover window in weeks. Matches the data generator so the model is
# correctly specified. On real data the rule is: pick l_max so that
# alpha**l_max is small for the slowest-decaying channel. At alpha = 0.70,
# 8 weeks leaves 5.8% of the carryover outside the window; 12 weeks leaves 1.4%.
L_MAX = 8

# Number of yearly Fourier harmonics. 2 harmonics = 4 sin/cos terms, enough to
# capture one smooth annual cycle plus a mild asymmetry, without giving the
# model enough flexibility to absorb channel effects into "seasonality".
YEARLY_SEASONALITY = 2

# Column names as the model sees them. Spend columns keep the "spend_" prefix
# from the CSV; the trend control is a column we add in ``prepare_features``.
DATE_COL = "date"
TARGET_COL = "sales"
TREND_COL = "t"
SPEND_COLS = [f"spend_{c}" for c in CHANNELS]


# Floor added to scaled spend inside the Hill function. On the 0-to-1 axis
# 1e-6 is one millionth of a channel's max weekly spend (well under a dime).
#
# Why it exists: the Hill curve x**S / (K**S + x**S) has an infinite slope at
# x = 0 whenever S < 1, so its gradient is undefined there. TV has one week
# where adstocked spend is *exactly* zero (an 8-week gap at the end of the
# series), and the first fit attempt died with a NaN gradient the moment a
# chain explored S < 1 for TV. The floor keeps every gradient finite and
# changes the fitted curve by nothing measurable.
HILL_FLOOR = 1e-6


class FlooredHillSaturation(HillSaturation):
    """Library Hill saturation with ``HILL_FLOOR`` added to spend (see above)."""

    lookup_name = "hill_floored"

    def function(self, x, slope, kappa, beta, *, dim=None):
        return beta * hill_function(x + HILL_FLOOR, slope, kappa)


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #

def make_priors() -> dict[str, Prior]:
    """
    Return the full prior configuration for the model.

    Every prior is stated on the *scaled* axis (spend / max spend,
    sales / max sales). Each was chosen from the scale of the axis alone,
    not from the true parameters used to simulate the data, so the
    parameter-recovery table is a genuine test.

    ``dims="channel"`` means one independent copy of the prior per channel.
    """
    return {
        # ---- Media transform parameters (one per channel) ------------------
        #
        # alpha: geometric carryover, in [0, 1]. Beta(2, 2) has median 0.5 and
        # 95% mass on 0.09 to 0.91: symmetric and deliberately uninformed.
        # The library default Beta(1, 3) has median 0.21 and would pull a
        # slow-decaying channel like TV downward.
        "adstock_alpha": Prior("Beta", alpha=2, beta=2, dims="channel"),
        #
        # kappa: half-saturation point as a fraction of max weekly spend.
        # Median 0.5 = midpoint of the observed spend range; sigma 0.75 gives
        # a 95% interval of 0.12 to 2.2, i.e. from "saturates at a tenth of
        # peak spend" to "half-saturation at twice peak spend". LogNormal
        # rather than HalfNormal because kappa -> 0 (every dollar fully
        # saturated) is degenerate with the intercept.
        "saturation_kappa": Prior("LogNormal", mu=float(np.log(0.5)), sigma=0.75, dims="channel"),
        #
        # slope: Hill curve shape. S < 1 is concave from the first dollar,
        # S > 1 is S-shaped. Median exactly 1 so neither regime is favoured;
        # sigma 0.5 allows a factor of ~2.7 either way (95%: 0.37 to 2.7).
        "saturation_slope": Prior("LogNormal", mu=0.0, sigma=0.5, dims="channel"),
        #
        # beta: a channel's maximum possible contribution, as a fraction of
        # peak weekly sales (Hill saturates at 1, so beta is the ceiling).
        # HalfNormal(0.5) puts 95% below 1.1: one channel should not be able
        # to add more than the best week ever observed.
        "saturation_beta": Prior("HalfNormal", sigma=0.5, dims="channel"),
        #
        # ---- Everything that is not media ----------------------------------
        #
        # intercept: baseline sales with zero media, as a fraction of peak
        # sales. Must lie in (0, 1); Normal(0.5, 0.5) covers that range
        # loosely and lets the data decide.
        "intercept": Prior("Normal", mu=0.5, sigma=0.5),
        #
        # gamma_control: coefficient on the trend column t in [0, 1]. Equals
        # the total drift over the whole series as a fraction of peak sales.
        # Normal(0, 0.5): a 2-year drift beyond +/-100% of peak is implausible.
        "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
        #
        # gamma_fourier: amplitude of each seasonal harmonic as a fraction of
        # peak sales. Laplace(0, 0.1) keeps 95% within +/-0.3 and shrinks
        # unneeded harmonics toward zero.
        "gamma_fourier": Prior("Laplace", mu=0, b=0.1, dims="fourier_mode"),
        #
        # likelihood: Gaussian noise on scaled sales. HalfNormal(0.1) on the
        # noise SD puts 95% below 0.2 = 20% of peak sales, which is generous;
        # the library default sigma=2 would allow noise larger than sales.
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.1), dims="date"),
    }


# --------------------------------------------------------------------------- #
# Build and fit
# --------------------------------------------------------------------------- #

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the weekly dataframe into model inputs X and target y.

    Adds a linear trend column ``t`` scaled to [0, 1]. Control columns are
    *not* scaled by the library, so scaling it ourselves keeps its
    coefficient on the same 0-to-1 footing as everything else.
    """
    X = df[[DATE_COL, *SPEND_COLS]].copy()
    X[DATE_COL] = pd.to_datetime(X[DATE_COL])
    n = len(X)
    X[TREND_COL] = np.arange(n) / (n - 1)
    y = df[TARGET_COL].astype(float)
    return X, y


def build_mmm(l_max: int = L_MAX, yearly_seasonality: int = YEARLY_SEASONALITY) -> MMM:
    """
    Construct the (unfitted) PyMC-Marketing model.

    * ``GeometricAdstock(normalize=True)`` matches ``mmm_sandbox.transforms``
      and the data generator: carryover redistributes spend across weeks
      rather than inflating it, so kappa stays interpretable in spend units.
    * ``FlooredHillSaturation`` is the library's Hill curve, the same
      x**S / (K**S + x**S) as our own ``hill_saturation`` (the library calls
      K ``kappa`` and S ``slope``), plus a tiny floor on spend so the
      gradient exists at zero-spend weeks.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return MMM(
            date_column=DATE_COL,
            channel_columns=SPEND_COLS,
            target_column=TARGET_COL,
            adstock=GeometricAdstock(l_max=l_max, normalize=True),
            saturation=FlooredHillSaturation(),
            control_columns=[TREND_COL],
            yearly_seasonality=yearly_seasonality,
            model_config=make_priors(),
        )


def fit_mmm(
    mmm: MMM,
    X: pd.DataFrame,
    y: pd.Series,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
    """
    Run NUTS and attach a posterior-predictive sample for the training weeks.

    ``target_accept=0.9`` (library default 0.8) takes smaller leapfrog steps,
    which costs time but reduces divergences in the saturation parameters,
    whose posterior can be sharply curved when kappa and beta trade off.
    """
    mmm.fit(
        X,
        y,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        progressbar=False,
    )
    # Posterior predictive on the training weeks, for the "model fit" plot.
    # Values come back on the scaled sales axis (fraction of max sales).
    mmm.sample_posterior_predictive(X, extend_idata=True, combined=False, random_seed=seed)
    return mmm.idata


# --------------------------------------------------------------------------- #
# Diagnostics and recovery
# --------------------------------------------------------------------------- #

# The variables whose convergence we care about. Deterministic quantities
# (contributions) are functions of these, so checking these is sufficient.
FREE_PARAMS = [
    "adstock_alpha",
    "saturation_kappa",
    "saturation_slope",
    "saturation_beta",
    "intercept_contribution",
    "gamma_control",
    "gamma_fourier",
    "y_sigma",
]


def fit_diagnostics(idata: az.InferenceData) -> dict:
    """
    Summarise MCMC health in three numbers plus the sampler settings.

    * r-hat compares between-chain to within-chain variance; 1.00 means the
      chains agree. Anything above 1.01 means at least one chain is stuck.
    * ESS (effective sample size) is how many *independent* draws the
      correlated chain is worth. Below ~400 the tails are poorly estimated.
    * Divergences are numerical failures of the integrator; even a handful
      indicate a region of the posterior the sampler cannot explore, so the
      target is zero.
    """
    summary = az.summary(idata, var_names=FREE_PARAMS, round_to=6)
    post = idata.posterior
    return {
        "n_chains": int(post.sizes["chain"]),
        "n_draws_per_chain": int(post.sizes["draw"]),
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "n_divergences": int(idata.sample_stats["diverging"].sum()),
        "sampling_time_seconds": float(idata.sample_stats.attrs.get("sampling_time", float("nan"))),
    }


def parameter_recovery_table(idata: az.InferenceData, true_params: dict, hdi_prob: float = 0.94) -> pd.DataFrame:
    """
    Compare true simulation parameters with the posterior, one row per
    (channel, parameter).

    Columns: true, post_mean, hdi_low, hdi_high, inside_hdi. ``inside_hdi``
    is the pass/fail we care about: the truth should sit inside the 94%
    interval for most rows. A few misses out of 16 are expected by chance;
    a systematic miss on one parameter is a modelling problem.
    """
    params = posterior_params_in_dollars(idata)
    hdi = az.hdi(params, hdi_prob=hdi_prob)
    rows = []
    for channel in params["channel"].values:
        for name in ["alpha", "K", "S", "beta"]:
            true = float(true_params[channel][name])
            draws = params[name].sel(channel=channel)
            lo, hi = (float(v) for v in hdi[name].sel(channel=channel).values)
            rows.append(
                {
                    "channel": channel,
                    "param": name,
                    "true": true,
                    "post_mean": float(draws.mean()),
                    "hdi_low": lo,
                    "hdi_high": hi,
                    "inside_hdi": bool(lo <= true <= hi),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Save and load
# --------------------------------------------------------------------------- #

# Groups and variables kept in the saved artifact. Everything the app needs
# and nothing that only PyMC would use.
_KEEP_POSTERIOR_VARS = [
    *FREE_PARAMS,
    "channel_contribution",           # date x channel, scaled sales axis
    "control_contribution",           # date x control (the trend)
    "yearly_seasonality_contribution",  # date
]


def thin_posterior(idata: az.InferenceData, keep_draws: int = 1000) -> az.InferenceData:
    """
    Keep every k-th draw so ``keep_draws`` remain across all chains.

    Why: 4 chains x 1000 draws with the per-week deterministic
    contributions is tens of MB; 1000 draws is plenty for P10/P50/P90
    summaries and keeps the artifact small enough to commit and to load
    on Streamlit Community Cloud. The chain dimension is preserved so
    r-hat can still be recomputed from the file if anyone wants to.
    """
    post = idata.posterior
    total = post.sizes["chain"] * post.sizes["draw"]
    step = max(1, total // keep_draws)
    thin = {"draw": slice(None, None, step)}
    groups = {
        "posterior": post[_KEEP_POSTERIOR_VARS].isel(thin),
        "sample_stats": idata.sample_stats[["diverging"]].isel(thin),
        "constant_data": idata.constant_data,
        "observed_data": idata.observed_data,
    }
    if "posterior_predictive" in idata.groups():
        groups["posterior_predictive"] = idata.posterior_predictive.isel(thin)
    return az.InferenceData(**groups)


def save_posterior(idata: az.InferenceData, path: str, metadata: dict | None = None) -> None:
    """Write the (thinned) InferenceData to NetCDF, with our metadata in attrs."""
    meta = {
        "channels": CHANNELS,
        "spend_columns": SPEND_COLS,
        "l_max": L_MAX,
        "adstock_normalize": True,
        "yearly_seasonality": YEARLY_SEASONALITY,
        **(metadata or {}),
    }
    idata.attrs["mmm_sandbox"] = json.dumps(meta)
    idata.to_netcdf(path)
