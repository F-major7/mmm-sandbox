"""
Tests for mmm_sandbox.analysis.

Two kinds:
* pure-numpy checks that the draw-vectorised transforms equal the reference
  implementations in mmm_sandbox.transforms, and that the derived quantities
  behave (ordering, monotonicity, known values);
* checks against the real saved posterior in artifacts/, including that our
  contributions reproduce PyMC-Marketing's own, and that the whole analysis
  is fast enough for the app. These are skipped if the artifact is absent.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from mmm_sandbox import analysis as A
from mmm_sandbox.transforms import geometric_adstock, hill_saturation

ARTIFACT = "artifacts/posterior.nc"
needs_artifact = pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="run scripts/fit_model.py first")


# --------------------------------------------------------------------------- #
# Vectorised transforms == reference loops
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("l_max", [1, 4, 8])
def test_adstock_over_draws_matches_reference(l_max):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1000, size=30)
    x[5:9] = 0.0
    alphas = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    got = A.adstock_over_draws(x, alphas, l_max)
    for i, a in enumerate(alphas):
        np.testing.assert_allclose(got[i], geometric_adstock(x, a, l_max, normalize=True), rtol=1e-12, atol=1e-12)


def test_hill_over_draws_matches_reference():
    x = np.linspace(0, 5000, 50)
    K = np.array([100.0, 1000.0, 3000.0])
    S = np.array([0.5, 1.0, 2.5])
    got = A.hill_over_draws(x, K, S)
    for i in range(3):
        np.testing.assert_allclose(got[i], hill_saturation(x, K[i], S[i]), rtol=1e-12)


def test_constant_spend_is_fixed_point_of_normalised_adstock():
    """Steady state: constant weekly spend x is carried over into exactly x."""
    x = np.full(40, 1234.5)
    got = A.adstock_over_draws(x, np.array([0.3, 0.8]), l_max=8)
    np.testing.assert_allclose(got[:, 7:], 1234.5)  # after the first l_max-1 warm-up weeks


# --------------------------------------------------------------------------- #
# Synthetic Draws object with known values
# --------------------------------------------------------------------------- #

@pytest.fixture
def toy():
    """Two channels, three draws, ten weeks, hand-picked parameters."""
    D, T = 3, 10
    dates = pd.date_range("2024-01-01", periods=T, freq="W-MON")
    draws = A.Draws(
        channels=["a", "b"],
        dates=dates,
        l_max=4,
        alpha=np.array([[0.2, 0.6]] * D),
        K=np.array([[100.0, 500.0]] * D),
        S=np.array([[1.0, 2.0]] * D),
        beta=np.array([[1000.0, 2000.0]] * D),
        baseline=np.full(D, 5000.0),
        trend=np.zeros((D, T)),
        seasonality=np.zeros((D, T)),
        sigma=np.full(D, 50.0),
        target_scale=1.0,
    )
    spend = pd.DataFrame({"date": dates, "spend_a": np.full(T, 100.0), "spend_b": np.full(T, 500.0)})
    return draws, spend


def test_contributions_at_half_saturation(toy):
    draws, spend = toy
    contrib = A.channel_contributions(draws, spend)
    assert contrib.shape == (3, 10, 2)
    # Constant spend equal to K -> hill = 0.5 -> contribution = beta / 2 (after warm-up)
    np.testing.assert_allclose(contrib[:, 3:, 0], 500.0)
    np.testing.assert_allclose(contrib[:, 3:, 1], 1000.0)


def test_decomposition_sums_to_expected_sales(toy):
    draws, spend = toy
    parts = A.decomposition(draws, spend)
    assert set(parts) == {"baseline", "trend", "seasonality", "a", "b"}
    np.testing.assert_allclose(sum(parts.values()), A.expected_sales(draws, spend))
    table = A.decomposition_table(draws, spend)
    assert table["share_p50"].sum() == pytest.approx(1.0)


def test_roi_quantiles_ordered_and_true_column(toy):
    draws, spend = toy
    true = pd.DataFrame({"contribution_a": np.full(10, 400.0), "contribution_b": np.full(10, 900.0)})
    table = A.channel_roi(draws, spend, true)
    assert (table["roi_p10"] <= table["roi_p50"]).all() and (table["roi_p50"] <= table["roi_p90"]).all()
    assert table["true_roi"].tolist() == pytest.approx([4.0, 1.8])


def test_response_curve_shape_and_known_points(toy):
    draws, spend = toy
    grid = np.linspace(0, 2000, 201)
    curve = A.response_curve(draws, "b", grid)
    assert curve["p50"][0] == 0.0
    assert np.all(np.diff(curve["p50"]) >= 0)               # monotone
    assert np.interp(500.0, grid, curve["p50"]) == pytest.approx(1000.0)  # beta/2 at K
    assert curve["p90"][-1] <= 2000.0                       # never above beta


def test_marginal_roi_matches_finite_difference(toy):
    draws, spend = toy
    at = np.array([100.0, 500.0])
    table = A.marginal_roi(draws, spend, at_spend=at)
    h = 1e-3
    for c, name in enumerate(draws.channels):
        up = A.response_curve(draws, name, np.array([at[c] + h]))["p50"][0]
        dn = A.response_curve(draws, name, np.array([at[c] - h]))["p50"][0]
        assert table.loc[c, "mroi_p50"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)


# --------------------------------------------------------------------------- #
# Against the real artifact
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real():
    from mmm_sandbox.posterior import load_posterior

    idata = load_posterior(ARTIFACT)
    spend = pd.read_csv("data/synthetic_weekly.csv", parse_dates=["date"])
    return idata, A.extract_draws(idata), spend


@needs_artifact
def test_contributions_reproduce_library(real):
    """
    Our transforms on raw dollars must match PyMC-Marketing's contributions
    computed inside the model on the scaled axis. The only difference is the
    1e-6 floor inside the model's Hill function, worth a few dollars at most.
    """
    idata, draws, spend = real
    ours = A.channel_contributions(draws, spend)
    lib = A._flat(idata.posterior["channel_contribution"]) * draws.target_scale
    assert np.abs(ours - lib).max() < 10.0


@needs_artifact
def test_expected_sales_matches_posterior_predictive_mean(real):
    idata, draws, spend = real
    fit, stats = A.fit_summary(idata)
    ours = A.expected_sales(draws, spend).mean(axis=0)
    # Predictive mean carries Monte Carlo noise of sigma / sqrt(draws) ~ $200
    assert np.abs(ours - fit["predicted_mean"].to_numpy()).max() < 1500.0
    assert 0.8 <= stats["coverage"] <= 1.0
    assert stats["r2"] > 0.9


@needs_artifact
def test_full_analysis_is_fast(real):
    """The app recomputes these on every slider move; budget is well under 1 s."""
    idata, draws, spend = real
    t0 = time.perf_counter()
    A.channel_roi(draws, spend)
    A.marginal_roi(draws, spend)
    A.decomposition_table(draws, spend)
    for c in draws.channels:
        A.response_curve(draws, c, A.default_spend_grid(spend, c))
    assert time.perf_counter() - t0 < 0.5


def test_analysis_module_does_not_import_pymc():
    code = "import sys, mmm_sandbox.analysis; print(any(m.startswith('pymc') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
