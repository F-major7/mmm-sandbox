"""Synthetic weekly advertiser data with *known* ground truth.

Why synthetic? With real data you never know the true channel effects, so you
cannot tell whether the model is right. Here we generate sales from a known
formula and keep the parameters, so later we can check that the Bayesian model
recovers them (parameter recovery is the honest way to validate an MMM).

The data-generating process, per week t and channel c:

    effective_spend[c, t] = geometric_adstock(spend[c], alpha[c], l_max)[t]
    contribution[c, t]    = beta[c] * hill_saturation(effective_spend[c, t], K[c], S[c])
    sales[t] = baseline + trend[t] + seasonality[t] + sum_c contribution[c, t] + noise[t]

Everything uses the from-scratch transforms in mmm_sandbox.transforms, with
normalize=True so effective spend stays in dollars and K is a dollar amount.
"""

import numpy as np
import pandas as pd

from mmm_sandbox.transforms import geometric_adstock, hill_saturation

# ---------------------------------------------------------------------------
# Ground-truth parameters. These are the numbers the model must recover.
#
#   alpha : weekly retention of the ad effect (TV lingers, Search is immediate)
#   K     : weekly spend (in $) at which the channel reaches half its max effect
#   S     : Hill slope (< 1 concave from the first dollar, > 1 S-shaped)
#   beta  : maximum weekly sales contribution (in $) at infinite spend
#
# Implied ROI at spend = K is (beta / 2) / K:
#   TV 1.33x, Search 2.0x, Social 1.5x, Display 1.0x.  Display is deliberately
#   the weakest channel so a budget optimizer has something to cut.
# ---------------------------------------------------------------------------
CHANNELS = ["tv", "search", "social", "display"]

TRUE_PARAMS = {
    "tv":      {"alpha": 0.70, "K": 30_000.0, "S": 1.8, "beta": 80_000.0},
    "search":  {"alpha": 0.10, "K": 15_000.0, "S": 0.9, "beta": 60_000.0},
    "social":  {"alpha": 0.40, "K": 15_000.0, "S": 1.2, "beta": 45_000.0},
    "display": {"alpha": 0.30, "K": 10_000.0, "S": 1.0, "beta": 20_000.0},
}

L_MAX = 8            # carryover truncated after 8 weeks (shared across channels)
BASELINE = 200_000.0 # weekly sales with zero advertising, at t = 0
TREND_TOTAL = 0.15   # +15% organic growth over the full period
SEASON_AMPLITUDE = 0.10  # +/-10% of baseline, peaking in early December
NOISE_SD = 6_000.0   # observation noise, ~3% of baseline


def _spend_patterns(n_weeks, rng):
    """Weekly spend per channel, designed so channels are NOT collinear.

    Collinear spend (every channel up and down together) is the classic reason
    an MMM cannot separate channel effects. Each channel here has a different
    temporal shape so the model has something to work with:

      tv      : bursty "flights" - a few weeks on, several weeks off
      search  : steady always-on spend with mild noise
      social  : ramps up over the two years (a growing channel)
      display : noisy, moderate, with occasional dark weeks
    """
    spend = {}

    # TV: on/off flights of 3-5 weeks with 5-8 week gaps, random amplitude.
    tv = np.zeros(n_weeks)
    week = rng.integers(0, 4)
    while week < n_weeks:
        flight_len = rng.integers(3, 6)
        amplitude = rng.uniform(35_000, 65_000)
        tv[week : week + flight_len] = amplitude * rng.uniform(0.85, 1.15, size=min(flight_len, n_weeks - week))
        week += flight_len + rng.integers(5, 9)
    spend["tv"] = tv

    # Search: steady, always-on, around $18k with small weekly noise and no trend.
    # Keeping it flat means social is the only channel correlated with time,
    # so any trend confounding in the model has exactly one known source.
    spend["search"] = 18_000 * rng.uniform(0.85, 1.15, size=n_weeks)

    # Social: ramps from ~$5k to ~$28k with noise.
    spend["social"] = np.linspace(5_000, 28_000, n_weeks) * rng.uniform(0.80, 1.20, size=n_weeks)

    # Display: noisy around $9k, with ~10% of weeks switched off entirely.
    display = 9_000 * rng.uniform(0.5, 1.5, size=n_weeks)
    display[rng.random(n_weeks) < 0.10] = 0.0
    spend["display"] = display

    return {c: np.round(spend[c], 2) for c in CHANNELS}


def generate_synthetic_data(n_weeks=104, seed=42, start_date="2024-01-01"):
    """Generate weekly spend + sales for four channels with known ground truth.

    Returns
    -------
    df : pd.DataFrame
        The *observable* dataset: `date`, one `spend_<channel>` column per
        channel, and `sales`. This is what the model gets to see.
    truth : dict
        Everything the model does NOT get to see:
          - "params": TRUE_PARAMS plus l_max, baseline, noise_sd
          - "components": DataFrame with baseline, trend, seasonality, one
            `contribution_<channel>` column per channel, and noise, such that
            they sum exactly to `sales`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    dates = pd.date_range(start_date, periods=n_weeks, freq="W-MON")

    spend = _spend_patterns(n_weeks, rng)

    # Media contributions: adstock -> saturation -> scale by beta.
    contributions = {}
    for c in CHANNELS:
        p = TRUE_PARAMS[c]
        effective = geometric_adstock(spend[c], p["alpha"], L_MAX, normalize=True)
        contributions[c] = p["beta"] * hill_saturation(effective, p["K"], p["S"])

    # Non-media components.
    trend = BASELINE * TREND_TOTAL * t / (n_weeks - 1)
    # Yearly cosine peaking at week ~49 (early December): phase-shift a 52-week cycle.
    seasonality = BASELINE * SEASON_AMPLITUDE * np.cos(2 * np.pi * (t - 49) / 52)
    noise = rng.normal(0, NOISE_SD, size=n_weeks)

    sales = BASELINE + trend + seasonality + sum(contributions.values()) + noise

    df = pd.DataFrame({"date": dates})
    for c in CHANNELS:
        df[f"spend_{c}"] = spend[c]
    df["sales"] = np.round(sales, 2)

    components = pd.DataFrame({"date": dates, "baseline": BASELINE, "trend": trend, "seasonality": seasonality})
    for c in CHANNELS:
        components[f"contribution_{c}"] = contributions[c]
    # Fold the rounding of `sales` into noise so the components sum *exactly*.
    components["noise"] = df["sales"].to_numpy() - (
        BASELINE + trend + seasonality + sum(contributions.values())
    )

    params = {
        "channels": CHANNELS,
        "l_max": L_MAX,
        "baseline": BASELINE,
        "trend_total": TREND_TOTAL,
        "season_amplitude": SEASON_AMPLITUDE,
        "noise_sd": NOISE_SD,
        "seed": seed,
        "channel_params": TRUE_PARAMS,
    }
    return df, {"params": params, "components": components}
