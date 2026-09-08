"""
Fit the Bayesian MMM offline and save everything the app needs.

Run from the repo root:

    python scripts/fit_model.py

Outputs (all in artifacts/):

    posterior.nc              thinned posterior + posterior predictive + scale factors
    diagnostics.json          r-hat, ESS, divergences, timing, sampler settings
    parameter_recovery.csv    true vs posterior for every (channel, parameter)

This is the only place MCMC runs. The Streamlit app reads posterior.nc and
does numpy arithmetic over the saved draws; it never samples.
"""

import json
import os
import sys
import time

import pandas as pd

# Allow `python scripts/fit_model.py` without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm_sandbox.model import (  # noqa: E402
    build_mmm,
    fit_diagnostics,
    fit_mmm,
    parameter_recovery_table,
    prepare_features,
    save_posterior,
    thin_posterior,
)

DATA_PATH = "data/synthetic_weekly.csv"
TRUE_PARAMS_PATH = "data/true_params.json"
ARTIFACT_DIR = "artifacts"

# Sampler settings. 4 chains x 1000 draws is the standard "enough to trust
# r-hat and ESS" configuration. target_accept=0.95 (library default 0.8) and
# 2000 tuning steps were needed because a first run at 0.9 / 1000 produced
# 136 divergences, concentrated where a channel's Hill slope is far below 1
# and the curve is almost flat (beta, K and the intercept then trade off
# along a ridge). Smaller steps let the sampler follow that ridge.
SAMPLER = {"draws": 1000, "tune": 2000, "chains": 4, "target_accept": 0.95, "seed": 42}
KEEP_DRAWS = 1000  # size of the thinned posterior that gets committed


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    with open(TRUE_PARAMS_PATH) as f:
        true_params = json.load(f)["channel_params"]

    X, y = prepare_features(df)
    mmm = build_mmm()

    print(f"Fitting: {SAMPLER}")
    t0 = time.time()
    idata = fit_mmm(mmm, X, y, **SAMPLER)
    wall = time.time() - t0
    print(f"Done in {wall/60:.1f} min")

    # --- Diagnostics on the FULL posterior (before thinning) ----------------
    diag = fit_diagnostics(idata)
    diag["wall_time_seconds"] = round(wall, 1)
    diag["sampler"] = SAMPLER
    print("\nDiagnostics")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # --- Parameter recovery -------------------------------------------------
    recovery = parameter_recovery_table(idata, true_params)
    n_inside = int(recovery["inside_hdi"].sum())
    diag["recovery_inside_94_hdi"] = f"{n_inside}/{len(recovery)}"
    print("\nParameter recovery (true vs posterior mean and 94% HDI)")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print(recovery.to_string(index=False))
    print(f"\n{n_inside}/{len(recovery)} true values inside the 94% HDI")

    # --- Save -----------------------------------------------------------------
    thinned = thin_posterior(idata, keep_draws=KEEP_DRAWS)
    posterior_path = os.path.join(ARTIFACT_DIR, "posterior.nc")
    save_posterior(thinned, posterior_path, metadata={"sampler": SAMPLER, "diagnostics": diag})
    recovery.to_csv(os.path.join(ARTIFACT_DIR, "parameter_recovery.csv"), index=False)
    with open(os.path.join(ARTIFACT_DIR, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    size_mb = os.path.getsize(posterior_path) / 1e6
    kept = thinned.posterior.sizes["chain"] * thinned.posterior.sizes["draw"]
    print(f"\nSaved {posterior_path}: {kept} draws, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
