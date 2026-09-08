"""
Read a saved posterior and hand back parameters in dollars.

This is the only bridge between the offline fit (``model.py``, which needs
PyMC) and everything that runs at app time (``analysis.py``, ``optimizer.py``,
the Streamlit app). It imports ArviZ and xarray only, never PyMC, so the
deployed app stays light and starts fast.
"""

from __future__ import annotations

import arviz as az
import xarray as xr

# Parameter names as PyMC-Marketing stores them, mapped to the names used
# in this project (which match ``mmm_sandbox.transforms`` and the data
# generator's ``TRUE_PARAMS``).
MEDIA_PARAMS = ["alpha", "K", "S", "beta"]


def load_posterior(path: str) -> az.InferenceData:
    """Read a saved posterior from NetCDF."""
    return az.from_netcdf(path)


def posterior_params_in_dollars(idata: az.InferenceData) -> xr.Dataset:
    """
    Return the four media parameters per channel with the library's scaling
    undone.

    PyMC-Marketing fits on spend / max(spend) and sales / max(sales), so its
    ``saturation_kappa`` is a fraction of each channel's max weekly spend and
    ``saturation_beta`` is a fraction of max weekly sales. Multiplying the
    saved scale factors back in gives:

    * ``alpha`` -- carryover, unitless (unchanged)
    * ``K``     -- half-saturation in weekly dollars  = kappa * max spend
    * ``S``     -- Hill slope, unitless (unchanged)
    * ``beta``  -- max weekly contribution in dollars = beta * max sales

    Output dims are (chain, draw, channel); channel coords are the short
    names ``tv``, ``search``, ... with the ``spend_`` column prefix removed.
    Keeping this conversion in one function means no other module can get
    the units wrong.
    """
    post = idata.posterior
    const = idata.constant_data
    channel_scale = const["channel_scale"]  # dims: channel
    target_scale = float(const["target_scale"])
    out = xr.Dataset(
        {
            "alpha": post["adstock_alpha"],
            "K": post["saturation_kappa"] * channel_scale,
            "S": post["saturation_slope"],
            "beta": post["saturation_beta"] * target_scale,
        }
    )
    short_names = [str(c).removeprefix("spend_") for c in out["channel"].values]
    return out.assign_coords(channel=short_names)
