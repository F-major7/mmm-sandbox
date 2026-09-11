"""
Generate the data file behind the portfolio site, straight from the model.

Run from the repo root after scripts/fit_model.py:

    python scripts/export_site_data.py

Writes:

    site/mmm-data.js     window.MMM = {...}; consumed by the page
    site/mmm-data.json   the same object, for tests and diffing

Every number the site shows comes from this file, and every number in this
file is computed here from artifacts/posterior.nc, data/*.csv and
data/true_params.json via mmm_sandbox.analysis and mmm_sandbox.optimizer.
Nothing is typed in by hand. The site's prose reads these fields, so a
refit followed by a re-export updates the page end to end.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmm_sandbox import analysis as A  # noqa: E402
from mmm_sandbox import optimizer as O  # noqa: E402
from mmm_sandbox.posterior import load_posterior  # noqa: E402

POSTERIOR = "artifacts/posterior.nc"
DIAGNOSTICS = "artifacts/diagnostics.json"
RECOVERY = "artifacts/parameter_recovery.csv"
SPEND_CSV = "data/synthetic_weekly.csv"
TRUE_COMPONENTS = "data/true_components.csv"
TRUE_PARAMS = "data/true_params.json"
OUT_DIR = "site"

# Presentation-only metadata. Colours and one-line pattern descriptions are
# the only hand-written strings; every number is computed below.
CHANNEL_META = {
    "tv": {"label": "TV", "color": "oklch(0.60 0.16 250)", "pattern": "Bursty on-off flights, 3-5 weeks on, 5-8 weeks dark"},
    "search": {"label": "Search", "color": "oklch(0.60 0.16 165)", "pattern": "Steady always-on, about $18k a week, no trend"},
    "social": {"label": "Social", "color": "oklch(0.62 0.16 50)", "pattern": "Ramping from about $5k to $28k over two years"},
    "display": {"label": "Display", "color": "oklch(0.58 0.16 320)", "pattern": "Noisy around $9k, about 10% of weeks switched off"},
}
RESPONSE_GRID_POINTS = 61
OPTIMIZER_STARTS = 12


def r(x, nd=0):
    """Round for JSON: ints when nd == 0, else floats."""
    return int(round(float(x))) if nd == 0 else round(float(x), nd)


def file_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def build() -> dict:
    idata = load_posterior(POSTERIOR)
    spend = pd.read_csv(SPEND_CSV, parse_dates=["date"])
    true_components = pd.read_csv(TRUE_COMPONENTS)
    true_params = json.load(open(TRUE_PARAMS))
    diag = json.load(open(DIAGNOSTICS))
    recovery = pd.read_csv(RECOVERY)
    draws = A.extract_draws(idata)
    channels = draws.channels
    spend_cols = [f"spend_{c}" for c in channels]

    # ---- Series and descriptive statistics -------------------------------
    fit, fit_stats = A.fit_summary(idata)
    sales = spend["sales"].to_numpy()
    n = len(spend)
    quarter = n // 8  # 13 weeks of a 104-week series
    series = {
        "dates": spend["date"].dt.strftime("%Y-%m-%d").tolist(),
        "sales": [r(v) for v in sales],
        "predicted": [r(v) for v in fit["predicted_mean"]],
        "predLo": [r(v) for v in fit["predicted_low"]],
        "predHi": [r(v) for v in fit["predicted_high"]],
        "spend": {c: [r(v) for v in spend[f"spend_{c}"]] for c in channels},
    }
    sales_stats = {
        "weeks": n,
        "start": series["dates"][0],
        "end": series["dates"][-1],
        "mean": r(sales.mean()),
        "min": r(sales.min()),
        "max": r(sales.max()),
        "earlyWeeks": quarter,
        "early": r(sales[:quarter].mean()),
        "late": r(sales[-quarter:].mean()),
        "peakDate": series["dates"][int(sales.argmax())],
        # Month of the seasonal component's peak (from the generator), not the peak sales week,
        # which trend and media can move.
        "seasonPeakDate": series["dates"][int(true_components["seasonality"].to_numpy().argmax())],
    }
    corr_matrix = spend[spend_cols].corr().to_numpy().copy()
    np.fill_diagonal(corr_matrix, 0.0)
    i, j = np.unravel_index(np.abs(corr_matrix).argmax(), corr_matrix.shape)
    corr = {
        "max": r(abs(corr_matrix[i, j]), 2),
        "pair": [channels[i], channels[j]],
        "pairs": [
            {"a": channels[a], "b": channels[b], "r": r(corr_matrix[a, b], 2)}
            for a in range(len(channels))
            for b in range(a + 1, len(channels))
        ],
    }

    # ---- Decomposition and shares ----------------------------------------
    dec = A.decomposition_table(draws, spend).set_index("component")
    total_sales = sales.sum()
    true_share = {c: true_components[f"contribution_{c}"].sum() / total_sales for c in channels}
    channel_rows = []
    for c in channels:
        channel_rows.append(
            {
                "key": c,
                **CHANNEL_META[c],
                "current": r(spend[f"spend_{c}"].mean()),
                "maxObserved": r(spend[f"spend_{c}"].max()),
                "totalSpend": r(spend[f"spend_{c}"].sum()),
                "share": r(dec.loc[c, "share_p50"] * 100, 1),
                "trueShare": r(true_share[c] * 100, 1),
            }
        )
    non_media = [
        {"key": "baseline", "label": "Baseline", "share": r(dec.loc["baseline", "share_p50"] * 100, 1)},
        {"key": "trend", "label": "Trend", "share": r(dec.loc["trend", "share_p50"] * 100, 1)},
    ]
    media_share = r(sum(dec.loc[c, "share_p50"] for c in channels) * 100, 1)
    true_media_share = r(sum(true_share.values()) * 100, 1)

    # ---- Parameter recovery ----------------------------------------------
    params = {c: {} for c in channels}
    for _, row in recovery.iterrows():
        nd = 3 if row["param"] in ("alpha", "S") else 0
        params[row["channel"]][row["param"]] = {
            "t": r(row["true"], nd),
            "p": r(row["post_mean"], nd),
            "lo": r(row["hdi_low"], nd),
            "hi": r(row["hdi_high"], nd),
            "inside": bool(row["inside_hdi"]),
        }
    n_inside = int(recovery["inside_hdi"].sum())

    # ---- ROI, marginal ROI, response curves --------------------------------
    roi_table = A.channel_roi(draws, spend, true_components).set_index("channel")
    mroi_table = A.marginal_roi(draws, spend).set_index("channel")
    roi = {
        c: {
            "p10": r(roi_table.loc[c, "roi_p10"], 2),
            "p50": r(roi_table.loc[c, "roi_p50"], 2),
            "p90": r(roi_table.loc[c, "roi_p90"], 2),
            "t": r(roi_table.loc[c, "true_roi"], 2),
            "inside": bool(roi_table.loc[c, "true_inside_p10_p90"]),
        }
        for c in channels
    }
    mroi = {
        c: {"p10": r(mroi_table.loc[c, "mroi_p10"], 2), "p50": r(mroi_table.loc[c, "mroi_p50"], 2), "p90": r(mroi_table.loc[c, "mroi_p90"], 2)}
        for c in channels
    }
    response = {}
    for c in channels:
        grid = A.default_spend_grid(spend, c, n=RESPONSE_GRID_POINTS, extend=1.5)
        curve = A.response_curve(draws, c, grid)
        response[c] = {
            "x": [r(v) for v in curve["spend"]],
            "p10": [r(v) for v in curve["p10"]],
            "p50": [r(v) for v in curve["p50"]],
            "p90": [r(v) for v in curve["p90"]],
        }

    # ---- Optimizer -------------------------------------------------------------
    current = np.array([spend[f"spend_{c}"].mean() for c in channels])
    budget = float(current.sum())
    lower, upper = O.default_bounds(spend, channels)
    alloc = {}
    for key, objective, label in [("ev", "mean", "Expected value"), ("risk", 0.10, "Risk-averse (P10)")]:
        res = O.optimize_budget(draws, budget, lower, upper, objective=objective, n_starts=OPTIMIZER_STARTS, seed=1)
        cmp = O.compare_allocations(
            draws,
            {"current": current, "equal": np.full(len(channels), budget / len(channels)), "optimized": res.spend},
            reference="current",
        ).set_index("allocation")
        unc = O.spend_uncertainty(draws, res.spend, spend).set_index("channel")
        agreeing = int((res.start_values >= res.value * (1 - 1e-3)).sum())
        alloc[key] = {
            "label": label,
            "objective": objective,
            **{c: r(res.spend[k]) for k, c in enumerate(channels)},
            "lift": r(cmp.loc["optimized", "lift_p50"]),
            "lo": r(cmp.loc["optimized", "lift_p10"]),
            "hi": r(cmp.loc["optimized", "lift_p90"]),
            "salesP10": r(cmp.loc["optimized", "sales_p10"]),
            "salesP50": r(cmp.loc["optimized", "sales_p50"]),
            "salesP90": r(cmp.loc["optimized", "sales_p90"]),
            "precise": bool(res.converged),
            "startsAgreeing": agreeing,
            "starts": res.n_starts,
            "spread": r(res.spread),
            "shareOfMax": {c: r(unc.loc[c, "share_of_max_observed"], 2) for c in channels},
            "bandOverP50": {c: r(unc.loc[c, "band_over_p50"], 2) for c in channels},
        }
        if key == "ev":
            equal_split = {
                "lift": r(cmp.loc["equal", "lift_p50"]),
                "lo": r(cmp.loc["equal", "lift_p10"]),
                "hi": r(cmp.loc["equal", "lift_p90"]),
            }
            current_sales = {"p10": r(cmp.loc["current", "sales_p10"]), "p50": r(cmp.loc["current", "sales_p50"]), "p90": r(cmp.loc["current", "sales_p90"])}

    sampler = diag["sampler"]
    total_draws = diag["n_chains"] * diag["n_draws_per_chain"]
    return {
        "generated": {
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": git_commit(),
            "posteriorSha": file_sha(POSTERIOR),
            "script": "scripts/export_site_data.py",
        },
        "repo": "https://github.com/F-major7/mmm-sandbox",
        "series": series,
        "salesStats": sales_stats,
        "corr": corr,
        "lMax": draws.l_max,
        "seed": true_params["seed"],
        "noiseSd": r(true_params["noise_sd"]),
        "seasonAmplitude": r(true_params["season_amplitude"] * 100),
        "trueParams": true_params["channel_params"],
        "channels": channel_rows,
        "nonMedia": non_media,
        "mediaShare": media_share,
        "trueMediaShare": true_media_share,
        "params": params,
        "recovery": {"inside": n_inside, "total": int(len(recovery)), "hdiProb": 94},
        "fit": {
            "r2": r(fit_stats["r2"], 3),
            "rmse": r(fit_stats["rmse"]),
            "trueNoise": r(true_params["noise_sd"]),
            "coverage": r(fit_stats["coverage"] * 100, 1),
            "hdiProb": r(fit_stats["hdi_prob"] * 100),
        },
        "sampler": {
            "rhat": r(diag["max_rhat"], 3),
            "divergences": diag["n_divergences"],
            "draws": total_draws,
            "pct": r(diag["n_divergences"] / total_draws * 100, 1),
            "chains": diag["n_chains"],
            "drawsPerChain": diag["n_draws_per_chain"],
            "tune": sampler["tune"],
            "targetAccept": sampler["target_accept"],
            "essBulk": r(diag["min_ess_bulk"]),
            "essTail": r(diag["min_ess_tail"]),
            "seconds": r(diag["sampling_time_seconds"]),
            "savedDraws": int(draws.n_draws),
        },
        "roi": roi,
        "mroi": mroi,
        "response": response,
        "budget": r(budget),
        "currentSales": current_sales,
        "alloc": alloc,
        "equalSplit": equal_split,
        "equalSplitDelta": equal_split["lift"],
    }


def main() -> None:
    data = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "mmm-data.json"), "w") as f:
        json.dump(data, f, indent=1)
    header = (
        "/* GENERATED by scripts/export_site_data.py from artifacts/posterior.nc and data/*.csv.\n"
        f"   Do not edit by hand. Generated {data['generated']['at']} at commit {data['generated']['commit']}. */\n"
    )
    with open(os.path.join(OUT_DIR, "mmm-data.js"), "w") as f:
        f.write(header + "window.MMM = " + json.dumps(data, separators=(",", ":")) + ";\n")
        f.write("window.dispatchEvent(new Event('mmm-ready'));\n")
    size = os.path.getsize(os.path.join(OUT_DIR, "mmm-data.js")) / 1024
    print(f"wrote {OUT_DIR}/mmm-data.js ({size:.0f} KB) and {OUT_DIR}/mmm-data.json")
    print(f"  corr max {data['corr']['max']} ({'/'.join(data['corr']['pair'])}); mean sales {data['salesStats']['mean']:,}; "
          f"recovery {data['recovery']['inside']}/{data['recovery']['total']}")
    for k, a in data["alloc"].items():
        print(f"  alloc {k}: " + ", ".join(f"{c} {a[c]:,}" for c in data["series"]["spend"]) + f"; lift {a['lift']:,} [{a['lo']:,}, {a['hi']:,}]; precise={a['precise']}")


if __name__ == "__main__":
    main()
