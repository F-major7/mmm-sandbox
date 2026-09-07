"""Tests for mmm_sandbox.transforms.

The key test is *equivalence*: our hand-written NumPy transforms must produce
the same numbers as PyMC-Marketing's tensor implementations. That is what lets
us use our simple versions for data generation and the optimizer while the
library's versions run inside the Bayesian model.
"""

import numpy as np
import pytest
import xarray as xr
from pymc_marketing.mmm import transformers as pmm_transformers

from mmm_sandbox.transforms import geometric_adstock, hill_saturation

# ---------------------------------------------------------------------------
# Helpers: PyMC-Marketing 0.19.x transforms are xtensor-based, so they need an
# xarray.DataArray with a named dimension rather than a plain NumPy array.
# ---------------------------------------------------------------------------


def library_adstock(x, alpha, l_max, normalize):
    x_da = xr.DataArray(np.asarray(x, dtype=float), dims=["date"])
    result = pmm_transformers.geometric_adstock(
        x_da, alpha=alpha, l_max=l_max, dim="date", normalize=normalize
    )
    return np.asarray(result.eval())


def library_hill(x, K, S):
    x_da = xr.DataArray(np.asarray(x, dtype=float), dims=["x"])
    # Library naming: kappa = our K (half-saturation), slope = our S.
    return np.asarray(pmm_transformers.hill_function(x_da, slope=S, kappa=K).eval())


# ---------------------------------------------------------------------------
# Geometric adstock: equivalence with the library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("l_max", [1, 4, 12])
@pytest.mark.parametrize("normalize", [False, True])
def test_adstock_matches_pymc_marketing(alpha, l_max, normalize):
    rng = np.random.default_rng(0)
    spend = rng.uniform(0, 1000, size=30)
    ours = geometric_adstock(spend, alpha, l_max, normalize=normalize)
    theirs = library_adstock(spend, alpha, l_max, normalize)
    np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-10)


def test_adstock_matches_library_when_series_shorter_than_kernel():
    # T=5 < l_max=12: the kernel is longer than the data, so the causal cut-off
    # (no spend before week 0) is exercised at every time step.
    spend = np.array([100.0, 0.0, 50.0, 0.0, 25.0])
    for normalize in (False, True):
        ours = geometric_adstock(spend, 0.6, 12, normalize=normalize)
        theirs = library_adstock(spend, 0.6, 12, normalize)
        np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Geometric adstock: hand-checkable behaviour and edge cases
# ---------------------------------------------------------------------------


def test_adstock_impulse_decays_geometrically():
    # A single 100 spend, un-normalized: 100, 50, 25, 12.5, then truncated.
    spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = geometric_adstock(spend, alpha=0.5, l_max=4, normalize=False)
    np.testing.assert_allclose(out, [100.0, 50.0, 25.0, 12.5, 0.0, 0.0])


def test_adstock_l_max_boundary_exactly_l_max_nonzero_entries():
    # An impulse at t=0 must affect exactly l_max weeks: indices 0..l_max-1
    # non-zero, index l_max exactly zero.
    l_max = 5
    spend = np.zeros(10)
    spend[0] = 1.0
    out = geometric_adstock(spend, alpha=0.8, l_max=l_max, normalize=False)
    assert np.all(out[:l_max] > 0)
    assert out[l_max] == 0.0
    assert np.all(out[l_max:] == 0.0)


def test_adstock_l_max_one_is_identity():
    # With a kernel of length 1 there is no carryover, whatever alpha is.
    spend = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    for alpha in (0.0, 0.5, 1.0):
        for normalize in (False, True):
            np.testing.assert_allclose(
                geometric_adstock(spend, alpha, l_max=1, normalize=normalize), spend
            )


def test_adstock_alpha_zero_is_identity():
    # alpha=0 means weights [1, 0, 0, ...]: this week's spend only.
    spend = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    np.testing.assert_allclose(geometric_adstock(spend, 0.0, l_max=8), spend)


def test_adstock_zero_spend_gives_zero():
    out = geometric_adstock(np.zeros(20), alpha=0.7, l_max=6)
    np.testing.assert_array_equal(out, np.zeros(20))


def test_adstock_single_time_point():
    # One week of data: no history to carry over, so only weight[0] applies.
    assert geometric_adstock([100.0], alpha=0.5, l_max=4, normalize=False)[0] == 100.0
    # Normalized: weight[0] = 1 / (1 + .5 + .25 + .125) = 1 / 1.875
    np.testing.assert_allclose(
        geometric_adstock([100.0], alpha=0.5, l_max=4, normalize=True), [100.0 / 1.875]
    )


def test_adstock_normalized_conserves_total_spend():
    # normalize=True spreads spend over time without inflating it, provided the
    # series is long enough for the whole tail to land inside it.
    spend = np.zeros(30)
    spend[:10] = 100.0
    out = geometric_adstock(spend, alpha=0.6, l_max=8, normalize=True)
    assert out.sum() == pytest.approx(spend.sum())


def test_adstock_output_is_non_negative_and_same_shape():
    rng = np.random.default_rng(1)
    spend = rng.uniform(0, 500, size=52)
    out = geometric_adstock(spend, alpha=0.4, l_max=10)
    assert out.shape == spend.shape
    assert np.all(out >= 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=-0.1, l_max=4),
        dict(alpha=1.5, l_max=4),
        dict(alpha=0.5, l_max=0),
        dict(alpha=0.5, l_max=2.5),
    ],
)
def test_adstock_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(5), **kwargs)


def test_adstock_rejects_2d_input():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones((5, 2)), alpha=0.5, l_max=4)


# ---------------------------------------------------------------------------
# Hill saturation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [0.5, 2.0, 300.0])
@pytest.mark.parametrize("S", [0.5, 1.0, 2.5])
def test_hill_matches_pymc_marketing(K, S):
    x = np.linspace(0, 5 * K, 50)
    # For non-integer slopes the compiled PyTensor graph evaluates x**S slightly
    # differently from NumPy (agreement ~1e-8 relative, not 1e-16), even though
    # both run in float64. Integer slopes match exactly. This is a numerical
    # detail of the library's graph, not a formula difference.
    np.testing.assert_allclose(hill_saturation(x, K, S), library_hill(x, K, S), rtol=1e-6)


def test_hill_zero_spend_gives_zero_response():
    assert hill_saturation(np.array([0.0]), K=100.0, S=1.5)[0] == 0.0


@pytest.mark.parametrize("S", [0.3, 1.0, 4.0])
def test_hill_half_saturation_at_K_for_any_slope(S):
    # By construction, response at x = K is exactly 0.5 regardless of S.
    assert hill_saturation(np.array([250.0]), K=250.0, S=S)[0] == pytest.approx(0.5)


def test_hill_is_monotone_increasing_and_bounded():
    x = np.linspace(0, 1e6, 1000)
    y = hill_saturation(x, K=1000.0, S=1.2)
    assert np.all(np.diff(y) >= 0)
    assert np.all((y >= 0) & (y < 1))


def test_hill_saturates_towards_one():
    # Far beyond K the curve flattens: extra spend buys almost nothing.
    assert hill_saturation(np.array([1e6]), K=10.0, S=2.0)[0] == pytest.approx(1.0, abs=1e-6)


def test_hill_single_value_scalar_input():
    # Scalars are accepted and returned as 0-d arrays.
    assert float(hill_saturation(2.0, K=2.0, S=3.0)) == pytest.approx(0.5)


@pytest.mark.parametrize("kwargs", [dict(K=0.0, S=1.0), dict(K=-1.0, S=1.0), dict(K=1.0, S=0.0)])
def test_hill_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        hill_saturation(np.array([1.0]), **kwargs)


def test_hill_rejects_negative_spend():
    with pytest.raises(ValueError):
        hill_saturation(np.array([-1.0]), K=1.0, S=1.0)
