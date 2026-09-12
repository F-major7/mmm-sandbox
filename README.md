# MMM Sandbox

**Which ads are actually driving sales, and where should the next dollar go?**

A Bayesian media mix model that answers both halves of that question and states how confident it is in each. Built on synthetic data with a planted answer, so every claim it makes can be checked rather than taken on faith.

**Live demo: https://f-major7.github.io/mmm-sandbox/** (toggle *Simple* / *Technical* in the tab bar).

---

## What it is

The model takes two years of weekly ad spend across four channels (TV, Search, Social, Display) and the weekly sales that followed, and estimates how much each channel contributed. Every estimate is a range, not a point, because the difference between a strong finding and a coin flip is the thing a decision depends on.

It then turns those estimates into a budget recommendation: for the same total spend, how should next week's money be split? Two objectives are offered, the split that does best in a typical week and the split that does best in a bad one. A Bayesian model is what makes that choice available.

The data is synthetic by design. The business, the rules that turn spend into sales, and the true effect of every channel were all written down first and hidden from the model. That is the only way to know whether a media mix model is right, and it is a test worth seeing passed before trusting one with a real budget.

> Generator: `mmm_sandbox/data.py`. Sales = baseline + linear trend + yearly cosine + Σ β · hill(adstock(spend)) + N(0, $6,000). The 16 true parameters are in `data/true_params.json`, seed 42.

## Why it matters

Most marketing measurement explains the past: here is what each channel did last quarter. The decision lives in the gap between "TV had an ROI of 1.3" and "move $15,000 a week into TV", and that gap is where uncertainty matters most.

A concrete case from this data. TV has the *lowest* average ROI of the four channels, 1.31 sales dollars per dollar spent, against 1.45 to 1.50 for the others. A report that stopped there would trim TV. But average ROI describes dollars already spent; the decision is about the next dollar. On that measure TV returns 1.44 and Search returns 0.45, because Search is saturated and TV is not. The optimizer moves money *into* TV, and says by how much and with what confidence. Getting this backwards is the default mistake in the field.

> `channel_roi` is total contribution over total spend per posterior draw. `marginal_roi` is the derivative of the response curve at current spend, β·S·K^S·x^(S−1)/(K^S+x^S)², per draw. Both in `mmm_sandbox/analysis.py`.

## Who it is for

A media buyer setting next month's split, who needs a recommendation and a sense of how firm it is. An account lead who has to defend "Search returned 1.5 per dollar" when the client asks how sure that is, and needs to be able to say "between 0.25 and 4.25, and here is why". And a technical reviewer judging whether the model is sound, for whom the rest of this document is written.

## How the numbers are made

**Fit once, serve from the result.** Fitting a Bayesian model means running a sampler for a few minutes. So `scripts/fit_model.py` runs it once, saves the posterior to `artifacts/posterior.nc`, and everything downstream, ROI ranges, response curves, the optimizer, the website, is arithmetic over that saved file. Every interaction is instant and the fit is reproducible from a seed. This is the train/serve split production ML systems use, and a test fails if the serving code ever imports the sampler.

> PyMC-Marketing's multidimensional `MMM`, `GeometricAdstock(l_max=8, normalize=True)`, Hill saturation, NUTS with 4 chains × 1,000 draws after 2,000 tuning steps, target accept 0.95. Posterior thinned to 1,000 draws (5.8 MB). Full analysis over it runs in under 0.5 s, enforced in `tests/test_analysis.py`.

**Two rules turn spend into effect.** An ad keeps working after you pay for it, fading week by week: that is adstock, or carryover. And the tenth dollar in a week buys less than the first: that is saturation, modelled by a Hill curve. Both were written from scratch here and then proven numerically identical to the library's versions.

> `geometric_adstock` weights lag L by α^L, normalised to sum to one, so carryover redistributes spend rather than inflating it. `hill_saturation(x, K, S) = x^S / (K^S + x^S)`, K the half-saturation spend, S the shape (S < 1 concave throughout, S > 1 S-shaped). `mmm_sandbox/transforms.py`; equivalence to `pymc_marketing.mmm.transformers` tested at 1e-10 over 24 parameter combinations. The library fits on max-scaled spend and sales, so K and β come back as fractions and are converted to dollars in one place, `mmm_sandbox/posterior.py`.

**Priors are set from scale, not from the answer.** A Bayesian model needs a starting belief about each parameter. On synthetic data the true values are on disk, so the priors were derived from rules that hold regardless of what the data contains: the half-saturation prior is centred on the midpoint of the observed spend range, which is 0.5 by construction on a max-scaled axis, and the shape prior is centred on exactly 1, the boundary between concave and S-shaped. An earlier draft of these priors had been chosen with the true values in view; it was discarded and replaced with the rule-based set, which is wider and less flattering. The recovery result below is only a test because of that.

> Per channel, on the scaled axis: α ~ Beta(2, 2); K ~ LogNormal(log 0.5, 0.75); S ~ LogNormal(0, 0.5); β ~ HalfNormal(0.5), so no single channel can exceed peak weekly sales. `make_priors()` in `mmm_sandbox/model.py`, with the rule beside each.

### Validation

Three checks shaped the model as much as the code did.

**Channel independence.** The spend patterns are designed so no two channels move together; if TV and Search always rose in step, no model could tell which earned the sale. The first version of the generator quietly added a 10% upward drift to Search, contradicting the design, and that single line produced a Search/Social correlation of 0.33 that looked like a genuine finding. Removing it left the other three channels byte-identical and dropped the pair to 0.04. The largest remaining correlation is Search/Display at −0.16, noise from Display's dark weeks. The general point: a plausible result can be an artefact of the code, and the only defence is checking the design against what was actually generated.

**Sampler divergences.** A first fit produced 136 divergences; tightening the sampler brought that to 43, or 1.1% of draws, with posterior means moving under 2% between runs. Rather than declaring that small enough, the usual suspect was tested directly. TV's ceiling β and half-saturation K are correlated at 0.98, because spend never reaches the flat part of TV's curve, and that ridge is where funnels form. Projecting every draw onto it, the divergent draws spread from the 12th to the 94th percentile with no pile-up at either end. A rank-sum test across all 19 parameters found nothing beyond what chance produces. Conclusion: diffuse curvature in the Hill parameterisation rather than a single fixable cause. A rerun at target accept 0.99, and saving the full divergence flags, remain open.

> `artifacts/diagnostics.json`: max r-hat 1.007, min bulk ESS 1,208, min tail ESS 970, 43 divergences in 4,000 draws, 212 s.

**The risk-averse objective.** The safest allocation is scored by the 10th percentile of predicted sales, and there are two ways to compute that: the 10th percentile of *total* sales within each posterior draw, or each channel's 10th percentile summed. The second assumes every channel is unlucky in the same world and, on this data, reports $39,282 where the correct figure is $62,981. The code uses the first, locked in by a minimal test: two channels, two draws, perfectly anti-correlated, so the correct P10 is 100 and the wrong one is 40.

One smaller fix is worth a line. The first fit crashed because TV has a week of exactly zero carried-over spend, and the Hill curve's slope at zero is infinite when S < 1. A floor of one millionth of max spend inside the saturation function resolves it, with a regression test on the real data.

## What it found

Sixteen parameters, four per channel, each estimated as a 94% range. All sixteen true values fall inside.

| Channel | α true / est | K true / est | S true / est | β true / est |
|---|---|---|---|---|
| TV | 0.70 / 0.69 | $30.0k / $46.1k | 1.80 / 1.45 | $80.0k / $107.0k |
| Search | 0.10 / 0.36 | $15.0k / $14.4k | 0.90 / 1.02 | $60.0k / $55.7k |
| Social | 0.40 / 0.44 | $15.0k / $22.9k | 1.20 / 0.97 | $45.0k / $61.0k |
| Display | 0.30 / 0.58 | $10.0k / $12.2k | 1.00 / 1.02 | $20.0k / $35.1k |

Two rows deserve a note. TV's K and β both come back high together because they trade off along the ridge above: the data pins the curve where spend occurred and leaves the ceiling loose, so K's interval runs from $18.9k to $93.6k. Search's carryover is nearly unidentified, because flat spend carried over by any amount still looks flat. In both cases the model is reporting that the data cannot say, which is the correct answer.

ROI in sales dollars per dollar spent over the two years:

| Channel | Cautious (P10) | Typical (P50) | Optimistic (P90) | True |
|---|---|---|---|---|
| TV | 1.18 | 1.31 | 1.48 | 1.23 |
| Search | 0.25 | 1.50 | 4.25 | 1.81 |
| Social | 0.49 | 1.45 | 2.90 | 1.37 |
| Display | 0.49 | 1.48 | 3.70 | 1.07 |

Every true ROI sits inside its range. TV's is tight because on-off flighting gives the model contrast; Search's is wide because two years of near-flat spend give it almost none. The model fits weekly sales with R² 0.945 and a typical error of $5,933 against $6,000 of injected noise: it explains everything except what is unexplainable by construction.

**The recommendation.** Keep the weekly budget at $60,377 and move about $15.5k a week from Search, Social and Display into TV. Typical gain: $7,290 a week in sales, with a range of $81 to $12,597, so it does not lose even in the cautious case. An equal split would lose $3,520 a week, which shows the current plan is already sensible and the gain is real rather than a win over a straw man. The safest allocation moves further into TV (about $39.9k) and cuts Search harder (about $5.2k), accepting a lower typical gain of $5,769 for a better worst case. That split is reported as approximate: many nearby allocations score almost identically, so the direction is the finding and the last dollar is not.

> Objective: Σ β_c · hill(x_c) on every draw, scored by posterior mean or by within-draw P10. SLSQP, Σ x = budget, 0 ≤ x_c ≤ max observed weekly spend so no curve is extrapolated. 12 starts; all agree to the dollar for the mean objective. The quantile objective is non-smooth, so each start is polished with Nelder–Mead; 4 of 12 reach the best P10 within 0.1%, differing by at most $518 per channel. `mmm_sandbox/optimizer.py`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                              # 110 tests, ~10 s
```

To regenerate from scratch, in order:

```bash
python scripts/generate_data.py     # synthetic data + true parameters -> data/
python scripts/fit_model.py         # MCMC, ~4 min -> artifacts/
python scripts/report_analysis.py   # decomposition, ROI, marginal ROI, fit summary
python scripts/report_optimizer.py  # both objectives, multi-start agreement
python scripts/export_site_data.py  # every number on the site -> site/mmm-data.js
python scripts/build_site.py        # assembles docs/, served by GitHub Pages
```

The site's data file is generated, never hand-edited; a test fails if the page reads a field the exporter does not write, and the site footer names the commit and posterior it was built from.

**Layout.** `mmm_sandbox/` holds all logic: `transforms.py`, `data.py`, `model.py` (priors, fit, diagnostics, recovery), `posterior.py` (load and convert to dollars, no PyMC import), `analysis.py`, `optimizer.py`. `scripts/` are entry points, `tests/` mirror the modules, `site/` is the page source, `docs/` the built site. `artifacts/` and `data/` are committed so the repo is self-contained.

**Stack.** PyMC-Marketing 0.19 on PyMC 5, ArviZ, NumPy, pandas, SciPy, pytest. The site is a single page on a small React runtime with no build tooling.

## Scope

These are deliberate cuts, not oversights. The data is synthetic and single-geography, so the model has not met missing weeks, mid-series channel launches, or competitor activity. There is no cold-start handling; a new channel with no history gets its prior and nothing else. Carryover is truncated at eight weeks, matching the generator; on real data with TV's decay rate about 5.8% of the effect would fall outside, and the rule for choosing the window is documented in `model.py`. Channel effects are fixed over time. Priors are set from scale alone; in deployment the natural next step is calibrating them against a geo-lift experiment. The forecaster and geo-lift notebooks planned as stretch work were not built.

---

Every number in this document is exported from `artifacts/posterior.nc` and the committed data. A refit followed by the regeneration steps above updates the site and this document's source of truth together.
