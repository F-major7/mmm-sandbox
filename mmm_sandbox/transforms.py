"""Media transforms implemented from scratch in plain NumPy.

Two ideas turn raw ad spend into the "effective" spend a media-mix model sees:

1. Adstock (carryover): an ad you paid for this week keeps working in later
   weeks. Geometric adstock says the effect decays by a constant factor
   `alpha` each week, so the effective spend at time t is a weighted sum of
   spend at t, t-1, t-2, ... with weights 1, alpha, alpha^2, ...

2. Saturation (diminishing returns): the tenth dollar buys less than the
   first. The Hill function maps effective spend to a 0..1 "response" that
   rises fast at first and flattens out as spend grows.

Both functions are deliberately written as explicit loops / one-line formulas
so every step can be read off the code. The PyMC-Marketing library has its
own (tensor-based) versions; tests/test_transforms.py proves ours produce the
same numbers.
"""

import numpy as np


def geometric_adstock(x, alpha, l_max, normalize=True):
    """Apply geometric (constant-rate) carryover to a single spend series.

    Parameters
    ----------
    x : array-like, shape (T,)
        Weekly spend for one channel, in time order.
    alpha : float in [0, 1]
        Retention rate. 0 means no carryover (this week's spend only);
        0.9 means 90% of last week's effect is still present this week.
    l_max : int >= 1
        Maximum number of weeks the effect lasts (kernel length). Effects
        older than l_max weeks are truncated to zero.
    normalize : bool, default True
        If True, weights are divided by their sum so adstock *redistributes*
        spend over time without inflating the total. If False, weights are
        the raw powers of alpha (matches PyMC-Marketing's default).

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked ("effective") spend.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D (one channel over time); got shape {x.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if not (isinstance(l_max, (int, np.integer)) and l_max >= 1):
        raise ValueError(f"l_max must be an integer >= 1; got {l_max!r}")

    # Weight for a lag of k weeks is alpha**k: 1, alpha, alpha^2, ..., alpha^(l_max-1).
    weights = np.array([alpha**lag for lag in range(l_max)])
    if normalize:
        weights = weights / weights.sum()

    # Effective spend at time t = sum over lags of weight[lag] * spend[t - lag].
    # Spend before the series starts is treated as zero (causal convolution).
    adstocked = np.zeros_like(x)
    for t in range(len(x)):
        for lag in range(l_max):
            if t - lag < 0:
                break  # no spend exists before the first week
            adstocked[t] += weights[lag] * x[t - lag]
    return adstocked


def hill_saturation(x, K, S):
    """Hill saturation curve: response = x^S / (K^S + x^S), bounded in [0, 1).

    Parameters
    ----------
    x : array-like, non-negative
        Effective (usually adstocked) spend.
    K : float > 0
        Half-saturation point: the spend at which response is exactly 0.5,
        for any S. Larger K means the channel takes longer to saturate.
    S : float > 0
        Slope / shape. S <= 1 gives concave diminishing returns from the
        first dollar; S > 1 gives an S-shaped curve with a slow start.

    Returns
    -------
    np.ndarray
        Saturated response in [0, 1). Multiply by a channel coefficient
        (beta) to get sales contribution.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("x must be non-negative (spend cannot be negative)")
    if K <= 0:
        raise ValueError(f"K must be > 0; got {K}")
    if S <= 0:
        raise ValueError(f"S must be > 0; got {S}")

    return x**S / (K**S + x**S)
