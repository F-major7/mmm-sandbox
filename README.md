# MMM Sandbox

Bayesian media-mix modeling sandbox: geometric adstock and Hill saturation implemented from scratch,
a PyMC-Marketing model on synthetic multi-channel data, channel ROI posteriors, response curves,
and a fixed-budget optimizer, served through a Streamlit app.

_Work in progress. Full plain-English explanation of adstock, saturation, and why-Bayesian coming in the final README._

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```
