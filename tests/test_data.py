"""Tests for the synthetic data generator."""

import json

import numpy as np
import pandas as pd
import pytest

from mmm_sandbox.data import CHANNELS, L_MAX, TRUE_PARAMS, generate_synthetic_data
from mmm_sandbox.transforms import geometric_adstock, hill_saturation


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic_data(n_weeks=104, seed=42)


def test_shape_and_columns(dataset):
    df, truth = dataset
    assert len(df) == 104
    assert list(df.columns) == ["date"] + [f"spend_{c}" for c in CHANNELS] + ["sales"]
    assert df["date"].is_monotonic_increasing
    assert (df["date"].diff().dropna() == pd.Timedelta(weeks=1)).all()


def test_deterministic_for_same_seed():
    df_a, _ = generate_synthetic_data(seed=7)
    df_b, _ = generate_synthetic_data(seed=7)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_different_seed_gives_different_data():
    df_a, _ = generate_synthetic_data(seed=1)
    df_b, _ = generate_synthetic_data(seed=2)
    assert not np.allclose(df_a["sales"], df_b["sales"])


def test_spend_non_negative_and_sales_positive(dataset):
    df, _ = dataset
    for c in CHANNELS:
        assert (df[f"spend_{c}"] >= 0).all()
        assert df[f"spend_{c}"].sum() > 0  # every channel is actually used
    assert (df["sales"] > 0).all()


def test_components_sum_exactly_to_sales(dataset):
    # The decomposition must be an identity, otherwise "true contribution"
    # would be a fiction and parameter-recovery checks would be meaningless.
    df, truth = dataset
    comp = truth["components"].drop(columns="date")
    np.testing.assert_allclose(comp.sum(axis=1), df["sales"], rtol=0, atol=1e-6)


def test_contributions_reproducible_from_spend_and_true_params(dataset):
    # Round trip: applying our transforms with the stored parameters to the
    # observable spend must reproduce the stored contributions exactly.
    df, truth = dataset
    for c in CHANNELS:
        p = truth["params"]["channel_params"][c]
        effective = geometric_adstock(df[f"spend_{c}"].to_numpy(), p["alpha"], L_MAX, normalize=True)
        expected = p["beta"] * hill_saturation(effective, p["K"], p["S"])
        np.testing.assert_allclose(truth["components"][f"contribution_{c}"], expected)


def test_channels_are_not_collinear(dataset):
    # Design goal: distinct spend patterns so the model can separate channels.
    df, _ = dataset
    corr = df[[f"spend_{c}" for c in CHANNELS]].corr().abs().to_numpy()
    off_diagonal = corr[~np.eye(len(CHANNELS), dtype=bool)]
    assert off_diagonal.max() < 0.5, f"max |corr| between channels = {off_diagonal.max():.2f}"


def test_media_is_a_meaningful_but_not_dominant_share_of_sales(dataset):
    # Sanity on scale: media should explain a realistic slice of sales
    # (typically 10-40% in real MMMs), not 0% and not 90%.
    df, truth = dataset
    comp = truth["components"]
    media_share = comp[[f"contribution_{c}" for c in CHANNELS]].sum().sum() / df["sales"].sum()
    assert 0.10 < media_share < 0.40, f"media share = {media_share:.2%}"


def test_true_params_are_json_serializable(dataset):
    _, truth = dataset
    json.dumps(truth["params"])  # raises if not serializable
    assert set(truth["params"]["channel_params"]) == set(TRUE_PARAMS)
