# MMM Sandbox

Bayesian media-mix modeling sandbox: geometric adstock and Hill saturation implemented from scratch,
a PyMC-Marketing model on synthetic multi-channel data, channel ROI posteriors, response curves,
and a fixed-budget optimizer, presented on a static site generated from the saved posterior.

**Live demo:** https://f-major7.github.io/mmm-sandbox/

_Work in progress. Full plain-English explanation of adstock, saturation, and why-Bayesian coming in the final README._

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Regenerating the site after a refit

```bash
python scripts/fit_model.py          # MCMC, writes artifacts/posterior.nc
python scripts/export_site_data.py   # every number on the site, from the posterior -> site/mmm-data.js
python scripts/build_site.py         # assembles docs/, which GitHub Pages serves
```
