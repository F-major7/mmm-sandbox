"""
Tests for mmm_sandbox.optimizer.

Toy posteriors with hand-picked parameters make the right answer knowable:
constraints must hold exactly, the optimum must beat naive splits, money
must flow to the obviously better channel, and the mean and P10 objectives
must disagree when one channel is high-mean but uncertain. The last group
runs against the real saved posterior and is skipped if it is absent.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mmm_sandbox import analysis as A
from mmm_sandbox import optimizer as O

ARTIFACT = "artifacts/posterior.nc"
needs_artifact = pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run scripts/fit_model.py first")


def make_draws(K, S, beta, n_draws=1, channels=("a", "b")):
    """A Draws object with identical parameters in every draw unless arrays are given."""
    C = len(channels)

    def tile(v):
        v = np.asarray(v, dtype=float)
        return np.tile(v, (n_draws, 1)) if v.ndim == 1 else v

    T = 5
    return A.Draws(
        channels=list(channels),
        dates=pd.date_range("2024-01-01", periods=T, freq="W-MON"),
        l_max=4,
        alpha=tile(np.full(C, 0.5)),
        K=tile(K),
        S=tile(S),
        beta=tile(beta),
        baseline=np.zeros(n_draws),
        trend=np.zeros((n_draws, T)),
        seasonality=np.zeros((n_draws, T)),
        sigma=np.zeros(n_draws),
        target_scale=1.0,
    )


# --------------------------------------------------------------------------- #
# Scoring semantics
# --------------------------------------------------------------------------- #

def test_quantile_is_of_total_within_draw_not_sum_of_channel_quantiles():
    """
    Two channels, two draws, perfectly anti-correlated: when a is strong b
    is weak and vice versa. Total is 100 in every draw, so P10 of total is
    100; the sum of per-channel P10s would be 40.
    """
    draws = make_draws(K=[1.0, 1.0], S=[1.0, 1.0], beta=np.array([[160.0, 40.0], [40.0, 160.0]]), n_draws=2)
    spend = np.array([1.0, 1.0])  # hill(1) = 0.5 with K = 1 -> contributions are beta / 2
    sales = O.media_sales_per_draw(draws, spend)
    np.testing.assert_allclose(sales, [100.0, 100.0])
    assert O.score(sales, 0.10) == pytest.approx(100.0)


def test_score_rejects_bad_objective():
    with pytest.raises(ValueError):
        O.score(np.ones(5), 1.5)


# --------------------------------------------------------------------------- #
# Constraints and comparisons
# --------------------------------------------------------------------------- #

@pytest.fixture
def two_channel():
    # Channel a: strong and concave; channel b: weak. Both saturate around 1000.
    return make_draws(K=[1000.0, 1000.0], S=[1.0, 1.0], beta=[5000.0, 1000.0])


def test_budget_and_bounds_hold(two_channel):
    lower = np.array([100.0, 100.0])
    upper = np.array([1500.0, 1500.0])
    r = O.optimize_budget(two_channel, budget=2000.0, lower=lower, upper=upper, n_starts=4)
    assert r.spend.sum() == pytest.approx(2000.0, abs=1e-6)
    assert np.all(r.spend >= lower - 1e-6) and np.all(r.spend <= upper + 1e-6)


def test_infeasible_budget_raises(two_channel):
    with pytest.raises(ValueError):
        O.optimize_budget(two_channel, budget=5000.0, lower=np.zeros(2), upper=np.array([1000.0, 1000.0]))


def test_optimum_beats_equal_and_current(two_channel):
    budget = 2000.0
    r = O.optimize_budget(two_channel, budget, lower=np.zeros(2), upper=np.full(2, budget), n_starts=4)
    current = np.array([500.0, 1500.0])  # deliberately bad: most money in the weak channel
    table = O.compare_allocations(two_channel, {"current": current, "equal": np.full(2, 1000.0), "optimized": r.spend}, reference="current")
    opt = table.set_index("allocation").loc["optimized"]
    eq = table.set_index("allocation").loc["equal"]
    assert opt["sales_p50"] > eq["sales_p50"] > table.set_index("allocation").loc["current", "sales_p50"]
    assert opt["lift_p50"] > 0
    # Marginal returns should be equal at an interior optimum: 5000/(1000+a)^2 == 1000/(1000+b)^2
    a, b = r.spend
    assert 5000.0 / (1000.0 + a) ** 2 == pytest.approx(1000.0 / (1000.0 + b) ** 2, rel=1e-3)


def test_money_goes_to_better_channel_until_bound_binds():
    # Channel a is 10x better than b everywhere; cap a at 800 of a 1000 budget.
    draws = make_draws(K=[1000.0, 1000.0], S=[1.0, 1.0], beta=[10_000.0, 1000.0])
    r = O.optimize_budget(draws, budget=1000.0, lower=np.zeros(2), upper=np.array([800.0, 1000.0]), n_starts=4)
    assert r.spend[0] == pytest.approx(800.0, abs=1e-3)
    assert r.spend[1] == pytest.approx(200.0, abs=1e-3)


def test_multistart_agrees_on_smooth_objective(two_channel):
    r = O.optimize_budget(two_channel, budget=2000.0, lower=np.zeros(2), upper=np.full(2, 2000.0), n_starts=6, seed=3)
    assert r.converged
    assert r.spread < 1.0  # all starts within a dollar of each other


# --------------------------------------------------------------------------- #
# Mean vs P10 objective
# --------------------------------------------------------------------------- #

def test_mean_and_p10_objectives_disagree_when_one_channel_is_uncertain():
    """
    Channel a: high mean, very uncertain (beta is 0 in 30% of draws).
    Channel b: lower mean, certain. Mean objective favours a; P10 favours b.
    """
    rng = np.random.default_rng(0)
    D = 500
    beta_a = np.where(rng.random(D) < 0.3, 0.0, 3000.0)   # mean 2100, P10 = 0
    beta_b = np.full(D, 1500.0)                           # mean 1500, P10 = 1500
    draws = make_draws(K=[500.0, 500.0], S=[1.0, 1.0], beta=np.column_stack([beta_a, beta_b]), n_draws=D)
    budget, lo, hi = 1000.0, np.zeros(2), np.full(2, 1000.0)
    r_mean = O.optimize_budget(draws, budget, lo, hi, objective="mean", n_starts=4)
    r_p10 = O.optimize_budget(draws, budget, lo, hi, objective=0.10, n_starts=4)
    assert r_mean.spend[0] > r_mean.spend[1]      # mean objective backs the risky channel
    assert r_p10.spend[1] > r_p10.spend[0]        # P10 objective backs the safe one
    assert r_p10.sales_p10 >= r_mean.sales_p10    # and is at least as good on its own metric


# --------------------------------------------------------------------------- #
# Uncertainty report
# --------------------------------------------------------------------------- #

def test_spend_uncertainty_reports_position_and_band():
    D = 200
    rng = np.random.default_rng(1)
    beta = np.column_stack([rng.uniform(900, 1100, D), rng.uniform(100, 1900, D)])  # b is far more uncertain
    draws = make_draws(K=[500.0, 500.0], S=[1.0, 1.0], beta=beta, n_draws=D)
    history = pd.DataFrame({"spend_a": [100.0, 400.0, 1000.0], "spend_b": [100.0, 400.0, 1000.0]})
    table = O.spend_uncertainty(draws, np.array([800.0, 800.0]), history).set_index("channel")
    assert table.loc["a", "share_of_max_observed"] == pytest.approx(0.8)
    assert table.loc["a", "pct_of_weeks_at_or_above"] == pytest.approx(1 / 3)
    assert table.loc["b", "band_over_p50"] > 5 * table.loc["a", "band_over_p50"]
    assert (table["contrib_p10"] <= table["contrib_p50"]).all() and (table["contrib_p50"] <= table["contrib_p90"]).all()


# --------------------------------------------------------------------------- #
# Against the real posterior
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real():
    from mmm_sandbox.posterior import load_posterior

    idata = load_posterior(ARTIFACT)
    spend = pd.read_csv("data/synthetic_weekly.csv", parse_dates=["date"])
    draws = A.extract_draws(idata)
    current = np.array([spend[f"spend_{c}"].mean() for c in draws.channels])
    return draws, spend, current


@needs_artifact
def test_real_mean_optimum_converges_and_beats_current(real):
    draws, spend, current = real
    lower, upper = O.default_bounds(spend, draws.channels)
    r = O.optimize_budget(draws, current.sum(), lower, upper, objective="mean", n_starts=8, seed=1)
    assert r.converged and r.spread < 1.0
    assert r.spend.sum() == pytest.approx(current.sum(), abs=1e-3)
    assert np.all(r.spend <= upper + 1e-6)
    table = O.compare_allocations(draws, {"current": current, "optimized": r.spend}, reference="current").set_index("allocation")
    assert table.loc["optimized", "lift_p50"] > 0
    assert table.loc["optimized", "lift_p10"] >= 0  # mean-optimal split does not lose even at P10


@needs_artifact
def test_real_p10_optimum_is_stable_across_seeds(real):
    """The P10 optimum is a plateau; the best value must agree across seeds even if allocations wobble."""
    draws, spend, current = real
    lower, upper = O.default_bounds(spend, draws.channels)
    r1 = O.optimize_budget(draws, current.sum(), lower, upper, objective=0.10, n_starts=8, seed=1)
    r2 = O.optimize_budget(draws, current.sum(), lower, upper, objective=0.10, n_starts=8, seed=7)
    assert abs(r1.value - r2.value) / r1.value < 1e-3
    assert np.abs(r1.spend - r2.spend).max() < 2000.0
