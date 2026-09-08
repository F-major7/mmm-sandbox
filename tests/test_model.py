"""
Tests for mmm_sandbox.model and mmm_sandbox.posterior.

None of these run MCMC. They check that the model is wired the way the
docstrings claim (priors, variables, scaling), that the dollars conversion
is right, and that a posterior survives a save/load round trip. The actual
fit is validated by scripts/fit_model.py's diagnostics and recovery table.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm_sandbox.data import CHANNELS, generate_synthetic_data
from mmm_sandbox.model import (
    FREE_PARAMS,
    L_MAX,
    build_mmm,
    make_priors,
    parameter_recovery_table,
    prepare_features,
    save_posterior,
    thin_posterior,
)
from mmm_sandbox.posterior import load_posterior, posterior_params_in_dollars

SPEND_COLS = [f"spend_{c}" for c in CHANNELS]


# --------------------------------------------------------------------------- #
# Fake posterior: enough structure to exercise the conversion and I/O code
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_idata():
    """A tiny InferenceData shaped like a real fit, with known scale factors."""
    rng = np.random.default_rng(0)
    chains, draws, n_ch, n_dates = 2, 50, len(CHANNELS), 10
    dims_c = ("chain", "draw", "channel")
    coords = {"chain": [0, 1], "draw": np.arange(draws), "channel": SPEND_COLS, "date": np.arange(n_dates)}

    def rv(center, spread):
        return (center + spread * rng.normal(size=(chains, draws, n_ch)), dims_c)

    posterior = {
        "adstock_alpha": rv(0.5, 0.01),
        "saturation_kappa": rv(0.4, 0.01),
        "saturation_slope": rv(1.5, 0.01),
        "saturation_beta": rv(0.2, 0.01),
        "intercept_contribution": (0.5 + 0.01 * rng.normal(size=(chains, draws)), ("chain", "draw")),
        "gamma_control": (0.1 + 0.01 * rng.normal(size=(chains, draws, 1)), ("chain", "draw", "control")),
        "gamma_fourier": (0.01 * rng.normal(size=(chains, draws, 4)), ("chain", "draw", "fourier_mode")),
        "y_sigma": (0.02 + 0.001 * rng.normal(size=(chains, draws)), ("chain", "draw")),
        "channel_contribution": (rng.uniform(size=(chains, draws, n_dates, n_ch)), ("chain", "draw", "date", "channel")),
        "control_contribution": (rng.uniform(size=(chains, draws, n_dates, 1)), ("chain", "draw", "date", "control")),
        "yearly_seasonality_contribution": (rng.uniform(size=(chains, draws, n_dates)), ("chain", "draw", "date")),
    }
    post = xr.Dataset({k: xr.DataArray(v[0], dims=v[1]) for k, v in posterior.items()}, coords=coords)
    stats = xr.Dataset({"diverging": xr.DataArray(np.zeros((chains, draws), bool), dims=("chain", "draw"))})
    const = xr.Dataset(
        {
            "channel_scale": xr.DataArray([60_000.0, 20_000.0, 30_000.0, 10_000.0], dims="channel"),
            "target_scale": xr.DataArray(400_000.0),
        },
        coords={"channel": SPEND_COLS},
    )
    obs = xr.Dataset({"y": xr.DataArray(rng.uniform(size=n_dates), dims="date")})
    return az.InferenceData(posterior=post, sample_stats=stats, constant_data=const, observed_data=obs)


# --------------------------------------------------------------------------- #
# Priors: what the code says must be what the summary said
# --------------------------------------------------------------------------- #

def test_priors_are_the_stated_ones():
    p = make_priors()
    assert p["adstock_alpha"].distribution == "Beta" and p["adstock_alpha"].parameters == {"alpha": 2, "beta": 2}
    assert p["saturation_kappa"].distribution == "LogNormal"
    assert p["saturation_kappa"].parameters["mu"] == pytest.approx(np.log(0.5))
    assert p["saturation_kappa"].parameters["sigma"] == 0.75
    assert p["saturation_slope"].distribution == "LogNormal"
    assert p["saturation_slope"].parameters == {"mu": 0.0, "sigma": 0.5}
    assert p["saturation_beta"].distribution == "HalfNormal" and p["saturation_beta"].parameters == {"sigma": 0.5}


def test_media_priors_are_per_channel():
    p = make_priors()
    for name in ["adstock_alpha", "saturation_kappa", "saturation_slope", "saturation_beta"]:
        assert p[name].dims == ("channel",), name


# --------------------------------------------------------------------------- #
# Model wiring (builds the PyMC graph; does not sample)
# --------------------------------------------------------------------------- #

def test_prepare_features_adds_unit_trend():
    df, _ = generate_synthetic_data(n_weeks=20)
    X, y = prepare_features(df)
    assert list(X.columns) == ["date", *SPEND_COLS, "t"]
    assert X["t"].iloc[0] == 0.0 and X["t"].iloc[-1] == 1.0
    assert len(y) == 20


def test_build_mmm_has_expected_free_variables_and_scaling():
    df, _ = generate_synthetic_data(n_weeks=30)
    X, y = prepare_features(df)
    mmm = build_mmm()
    mmm.build_model(X, y)

    assert {v.name for v in mmm.model.free_RVs} == set(FREE_PARAMS)
    assert mmm.adstock.l_max == L_MAX
    assert mmm.adstock.normalize is True

    # Library scales each channel by its own max spend and sales by max sales.
    scales = mmm.get_scales_as_xarray()
    np.testing.assert_allclose(scales["channel_scale"].values, X[SPEND_COLS].max().values)
    assert float(scales["target_scale"]) == pytest.approx(y.max())


# --------------------------------------------------------------------------- #
# Dollars conversion and recovery table
# --------------------------------------------------------------------------- #

def test_posterior_params_in_dollars_undoes_scaling(fake_idata):
    params = posterior_params_in_dollars(fake_idata)
    assert list(params["channel"].values) == CHANNELS  # "spend_" prefix stripped
    post = fake_idata.posterior
    np.testing.assert_allclose(params["alpha"].values, post["adstock_alpha"].values)
    np.testing.assert_allclose(params["S"].values, post["saturation_slope"].values)
    np.testing.assert_allclose(params["K"].sel(channel="tv").values, post["saturation_kappa"].isel(channel=0).values * 60_000.0)
    np.testing.assert_allclose(params["beta"].values, post["saturation_beta"].values * 400_000.0)


def test_recovery_table_flags_inside_and_outside(fake_idata):
    # Truth chosen to sit at the fake posterior's center for tv, far away for display.
    true = {c: {"alpha": 0.5, "K": 0.4 * s, "S": 1.5, "beta": 0.2 * 400_000.0}
            for c, s in zip(CHANNELS, [60_000.0, 20_000.0, 30_000.0, 10_000.0])}
    true["display"]["alpha"] = 0.99
    table = parameter_recovery_table(fake_idata, true)
    assert len(table) == 4 * 4
    assert table.set_index(["channel", "param"]).loc[("tv", "K"), "inside_hdi"]
    assert not table.set_index(["channel", "param"]).loc[("display", "alpha"), "inside_hdi"]


# --------------------------------------------------------------------------- #
# Thinning and I/O
# --------------------------------------------------------------------------- #

def test_thin_keeps_requested_draws_and_chains(fake_idata):
    thin = thin_posterior(fake_idata, keep_draws=20)
    assert thin.posterior.sizes["chain"] == 2
    assert thin.posterior.sizes["chain"] * thin.posterior.sizes["draw"] == 20
    assert "channel_scale" in thin.constant_data
    assert "diverging" in thin.sample_stats


def test_save_and_load_round_trip(fake_idata, tmp_path):
    path = tmp_path / "posterior.nc"
    save_posterior(thin_posterior(fake_idata, keep_draws=20), str(path), metadata={"note": "test"})
    loaded = load_posterior(str(path))
    assert "mmm_sandbox" in loaded.attrs
    params_before = posterior_params_in_dollars(thin_posterior(fake_idata, keep_draws=20))
    params_after = posterior_params_in_dollars(loaded)
    xr.testing.assert_allclose(params_before, params_after)


def test_posterior_module_does_not_import_pymc():
    """The app-side loader must stay PyMC-free so Streamlit Cloud starts fast."""
    import subprocess, sys
    code = "import sys, mmm_sandbox.posterior; print(any(m.startswith('pymc') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_model_gradient_is_finite_at_zero_spend_with_concave_saturation():
    """
    Regression test for the first fit attempt, which crashed with a NaN
    gradient: the real data has a week where TV's adstocked spend is
    exactly zero, and the Hill curve's slope at zero is infinite when S < 1.
    """
    df = pd.read_csv("data/synthetic_weekly.csv")
    X, y = prepare_features(df)
    mmm = build_mmm()
    mmm.build_model(X, y)
    point = mmm.model.initial_point()
    point["saturation_slope_log__"] = np.log(np.full(len(CHANNELS), 0.8))  # S < 1 everywhere
    grad = mmm.model.compile_dlogp()(point)
    assert np.all(np.isfinite(grad))
