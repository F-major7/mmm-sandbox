# MMM Sandbox

**Which of our ads are actually driving sales, and where should the next dollar go?**

That is the whole question. Every advertiser asks it, most measurement tools answer only the first half, and almost none of them tell you how sure they are. This project is a small, complete answer to both halves, built so that every claim it makes can be checked against a known truth.

**Live demo: https://f-major7.github.io/mmm-sandbox/** (switch between *Simple* and *Technical* reading modes in the tab bar).

---

## What it is

The project does two things.

First, it takes two years of weekly ad spend across four channels (TV, Search, Social, Display) and the weekly sales that followed, and works out how much each channel contributed. Not as a single number, but as a range, because a model that pretends to be certain is more dangerous than one that admits doubt. This is a media mix model, the same family of tools large advertisers use to allocate hundreds of millions in spend.

Second, it turns that into a decision. Given the same total budget, how should next week's money be split to earn the most sales? And it lets you choose your appetite for risk: the split that does best in a typical week is not the split that does best in a bad one, and a Bayesian model is what makes that choice available.

There is one device that makes the whole thing trustworthy. The data is synthetic. We invented the business, wrote the rules that turn spend into sales, and then hid those rules from the model. So every answer the model gives can be compared against the truth, rather than taken on faith. Real advertisers never get that luxury, which is exactly why you want to see a model pass this test before you let it near a real budget.

> For the technical reader: the generator is `mmm_sandbox/data.py`. Sales are baseline plus a linear trend plus a yearly cosine plus four channel contributions plus Gaussian noise (σ = $6,000), with each contribution equal to β · hill(adstock(spend)). The 16 true parameters are stored in `data/true_params.json`, seed 42.

## Why it matters

Most marketing measurement stops at explaining the past: here is what each channel did last quarter. That is useful, but it is not a decision. The gap between "TV had an ROI of 1.3" and "move $15,000 a week into TV" is where money is actually made or lost, and it is where uncertainty matters most.

Here is the concrete version. On this data, TV has the *lowest* average ROI of the four channels, about 1.31 sales dollars per dollar spent, against 1.45 to 1.50 for the others. A report that stopped there would tell you to trim TV. But average ROI describes the dollars already spent, and the decision is about the *next* dollar. On that measure TV returns 1.44 while Search returns 0.45, because Search is already saturated and TV is not. The optimizer therefore moves money *into* TV, and the model can say by how much and with what confidence. Getting this backwards is not a hypothetical mistake. It is the default mistake.

> The two quantities are `channel_roi` (total contribution / total spend, per posterior draw) and `marginal_roi` (the derivative of the response curve at current spend, β·S·K^S·x^(S−1)/(K^S+x^S)², per draw) in `mmm_sandbox/analysis.py`.

## Who it is for

A media buyer setting next month's split, who needs a recommendation rather than a dashboard, and needs to know whether "move $15k into TV" is a strong finding or a coin flip.

An account lead who has to put a number in front of a client. "Search returned 1.5 dollars per dollar" invites the question "how sure are you?", and the honest answer here is "somewhere between 0.25 and 4.25, and here is why the range is that wide". Being able to say that, and explain it, is the difference between a number that survives the meeting and one that does not.

And a CTO deciding whether the person who built this understands what they built. The rest of this document is written with that reader in mind too.

## How the numbers are made

The build has a shape worth understanding, because it is the same shape a production system would have.

**The model is fit once, offline, and served from a saved result.** Fitting a Bayesian model means running a sampler for a few minutes; on this data it takes about three and a half. Nothing a user touches should wait for that. So `scripts/fit_model.py` runs the sampler, saves the result to `artifacts/posterior.nc`, and everything downstream, the ROI ranges, the response curves, the optimizer, the website, is plain arithmetic over that saved file. Every interaction is instant, and the fit is reproducible from a seed. This is the train/serve split every production ML system relies on, and it is enforced here by a test that fails if the serving code ever imports the sampler.

> The fit uses PyMC-Marketing's multidimensional `MMM` with `GeometricAdstock(l_max=8, normalize=True)` and Hill saturation, NUTS with 4 chains × 1,000 draws after 2,000 tuning steps, target accept 0.95. The saved posterior is thinned to 1,000 draws (5.8 MB). Loading it and computing every quantity the site shows takes well under a second; `tests/test_analysis.py` enforces 0.5 s.

**Two rules turn spend into effect.** An ad keeps working after you pay for it: this week's TV spend still nudges sales next week and the week after, fading each time. That is *adstock*, or carryover. And the tenth dollar in a week buys less than the first: a channel saturates. That is the *Hill curve*. Every media mix model needs both, and both were written from scratch here before any library was touched, then proven numerically identical to the library's versions.

> `geometric_adstock(x, α, l_max, normalize=True)` weights lag L by α^L, normalised to sum to one, so carryover redistributes spend rather than inflating it; `hill_saturation(x, K, S) = x^S / (K^S + x^S)`, where K is the spend at half saturation and S the shape (S < 1 concave from the first dollar, S > 1 S-shaped). Both live in `mmm_sandbox/transforms.py`; `tests/test_transforms.py` checks them against `pymc_marketing.mmm.transformers` at 1e-10 across 24 parameter combinations. Because the library scales spend and sales by their maxima before fitting, K and β come back as fractions and are converted to dollars in exactly one place, `mmm_sandbox/posterior.py`.

**Priors are stated, and stated before the answer was looked at.** A Bayesian model needs a starting belief about each parameter. Those beliefs should come from the scale of the problem, never from peeking at the answer, and on synthetic data the answer is sitting right there in a JSON file. That makes the following worth telling in full.

### How we know it can be trusted

Building a model is the easy half. The stories below are how this one was validated, and they are the part of the project I would most want a technical reader to look at. The model, tests and site were built with an AI assistant working step by step under review; the catches below are what that review was for.

**The prior that almost knew too much.** Midway through setting up the fit, the true parameter values had already been printed to the terminal for a scaling check. The priors for K and S proposed shortly afterwards had K centred at 0.6 on the scaled axis, comfortably in the middle of the true values, which sat between 0.46 and 0.75. The stated reasoning was scale-based. The number was not. I asked directly whether the truth had been in view when the priors were chosen, and the honest answer was yes. So both priors were thrown out and re-derived from rules that anyone who has never seen the truth could check: K's prior median is the midpoint of the observed spend range, which is 0.5 by construction on a max-scaled axis regardless of what the data contains, and S's prior median is exactly 1, the boundary between concave and S-shaped, so neither regime is favoured. The re-derived priors are wider and less flattering than the originals. They are the ones in the code, with the rule written beside each.

> Final priors, on the max-scaled axis, all per channel: α ~ Beta(2, 2); K ~ LogNormal(log 0.5, 0.75), 95% interval 0.12 to 2.2; S ~ LogNormal(0, 0.5), 95% interval 0.37 to 2.7; β ~ HalfNormal(0.5), 95% below 1.1 so no channel can exceed peak weekly sales on its own. See `make_priors()` in `mmm_sandbox/model.py`. The recovery table below is a genuine test only because of this step.

**The collinearity that was not there.** The channel spend patterns were designed so that no two channels move together, because if TV and Search always rise in step, no model can tell which one earned the sale. After generating the data, the report showed a max pairwise spend correlation of 0.33 between Search and Social, explained as both channels growing over time. I noticed the design said Search was supposed to be flat. It was supposed to be. A 10% upward drift had been added to Search "for realism", undeclared, contradicting the plan. That one line was the entire cause of the 0.33. Removing it (a one-line change that left TV, Social and Display byte-identical, because the replacement consumes exactly the same random numbers) dropped the Search/Social correlation to 0.04. The largest remaining pair is now Search/Display at −0.16, which is noise from Display's dark weeks. The lesson is not about the number. It is that a plausible-sounding finding was about to be presented as a property of the problem when it was an artefact of the code.

**Forty-three divergences, investigated rather than waved through.** The sampler reports when it fails to explore part of the posterior; these failures are called divergences, and the first fit at default-ish settings produced 136 of them. Tightening the sampler (target accept 0.95, 2,000 tuning steps) brought that to 43, or 1.1% of draws, with the posterior means moving by under 2% between the two runs. Small enough to ignore? The usual cause is a specific geometry: TV's ceiling β and half-saturation K are correlated at 0.98, because spend never reaches the flat part of TV's curve, so only their ratio is well identified. That ridge is the obvious suspect, so it was tested directly. Projecting every draw onto the ridge, the divergent ones land between its 12th and 94th percentiles with no pile-up at either end, which is not what a funnel looks like. Social's K/β pair, a milder version of the same, showed nothing either. A rank-sum test of divergent against ordinary draws across all 19 parameters found three at the edge of significance, about what chance produces in 19 tries, and none involving K or β. The conclusion is diffuse curvature in the Hill parameterisation rather than a single fixable cause, which points to a higher target accept rather than a reparameterisation. One honest caveat: only 13 of the 43 divergent transitions survived thinning, so the investigation used those. Saving the full flags, and a rerun at 0.99, are the two open items on this project.

> Final diagnostics from `artifacts/diagnostics.json`: max r-hat 1.007, min bulk ESS 1,208, min tail ESS 970, 43 divergences in 4,000 draws, 212 s sampling.

Two smaller catches deserve a line each. The risk-averse optimizer scores an allocation by the 10th percentile of predicted sales, and there are two ways to compute that: take the 10th percentile of *total* sales within each posterior draw, or take each channel's 10th percentile separately and add them. The second is wrong, because it assumes every channel is unlucky in the same world, and on this data it reports $39,282 where the right answer is $62,981. The code uses the first, and a deliberately minimal test locks it in: two channels, two draws, perfectly anti-correlated, total of 100 in every draw, so the correct P10 is 100 and the wrong one is 40. And the very first fit attempt crashed outright, because TV has one week where carried-over spend is exactly zero and the Hill curve's slope at zero is infinite whenever S is below 1, which gives the sampler a NaN gradient. The fix is a floor of one millionth of max spend inside the saturation function, with a regression test that evaluates the gradient at S = 0.8 on the real data. Small, real, and the kind of thing an actual Bayesian fit throws at you.

## What it found

The model was asked sixteen questions with known answers, four parameters for each of four channels, and gave a range for each. All sixteen true values fall inside the 94% ranges.

| Channel | α true / recovered | K true / recovered | S true / recovered | β true / recovered |
|---|---|---|---|---|
| TV | 0.70 / 0.69 | $30.0k / $46.1k | 1.80 / 1.45 | $80.0k / $107.0k |
| Search | 0.10 / 0.36 | $15.0k / $14.4k | 0.90 / 1.02 | $60.0k / $55.7k |
| Social | 0.40 / 0.44 | $15.0k / $22.9k | 1.20 / 0.97 | $45.0k / $61.0k |
| Display | 0.30 / 0.58 | $10.0k / $12.2k | 1.00 / 1.02 | $20.0k / $35.1k |

Two rows are worth explaining rather than admiring. TV's K and β both come back high, together, because the two trade off along the ridge described above: the data pins down the curve where spend actually happened and leaves the ceiling loosely constrained, so the interval on K runs from $18.9k to $93.6k. Search's carryover α is nearly unidentified, because flat spend carried over by any amount looks like flat spend. Both are the model saying "the data cannot tell me this", which is the correct answer.

On the numbers a client would actually see, in sales dollars per dollar spent over the two years:

| Channel | Cautious (P10) | Typical (P50) | Optimistic (P90) | True |
|---|---|---|---|---|
| TV | 1.18 | 1.31 | 1.48 | 1.23 |
| Search | 0.25 | 1.50 | 4.25 | 1.81 |
| Social | 0.49 | 1.45 | 2.90 | 1.37 |
| Display | 0.49 | 1.48 | 3.70 | 1.07 |

Every true ROI sits inside its range. TV's range is tight because its on-off flighting gives the model contrast to learn from; Search's is wide because two years of near-flat $18k weeks give it almost none. The model fits the weekly sales series with an R² of 0.945 and a typical error of $5,933 against $6,000 of injected noise, which is to say it explains everything except the part that is unexplainable by construction.

And the recommendation, which is what you would actually tell the client. Keep the weekly budget at $60,377 and move about $15.5k a week out of Search, Social and Display into TV. In a typical week that earns an extra $7,290 in sales; across the full range of what the model considers possible the gain runs from $81 to $12,597, so it does not lose even in the cautious case. Splitting the budget equally instead would *lose* $3,520 a week, which matters because it shows the current split is already sensible and the optimizer's gain is a real improvement, not a win over a straw man. Ask for the safest allocation instead and the model moves further into TV (about $39.9k) and cuts Search harder (about $5.2k), accepting a lower typical gain of $5,769 in exchange for a better worst case. That answer is presented as approximate on purpose: many nearby splits score almost identically on the cautious objective, so the direction is the finding and the last dollar is not.

> Optimizer details: steady-state media sales Σ β_c · hill(x_c) on every draw, scored by the posterior mean or by P10 of the within-draw total; SLSQP with Σ x = budget and 0 ≤ x_c ≤ max observed weekly spend per channel, so it never leans on the extrapolated part of a curve; 12 random starts, all agreeing to the dollar for the mean objective. The quantile objective is non-smooth (a kink wherever two draws swap order) so each start is polished with Nelder–Mead; 4 of 12 starts reach the best P10 within 0.1%, differing by at most $518 per channel. Lift is computed within draws against the current allocation. `mmm_sandbox/optimizer.py`, `tests/test_optimizer.py`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                              # 110 tests, about 10 seconds
```

To regenerate everything from scratch, in order:

```bash
python scripts/generate_data.py     # synthetic data + true parameters -> data/
python scripts/fit_model.py         # MCMC, ~4 min -> artifacts/posterior.nc, diagnostics, recovery table
python scripts/report_analysis.py   # decomposition, ROI, marginal ROI, fit summary (prints)
python scripts/report_optimizer.py  # both objectives, multi-start agreement, uncertainty at the optimum
python scripts/export_site_data.py  # every number on the site, from the posterior -> site/mmm-data.js
python scripts/build_site.py        # assembles docs/, which GitHub Pages serves
```

The site's data file is generated, never hand-edited, and a test fails if the page reads a field the exporter does not write. The footer of the live site names the commit and posterior file it was built from.

**Layout.** `mmm_sandbox/` is the library and holds all the logic: `transforms.py` (adstock, Hill), `data.py` (generator), `model.py` (priors, fit, diagnostics, recovery), `posterior.py` (load the saved fit, convert to dollars; no PyMC import), `analysis.py` (contributions, ROI, curves), `optimizer.py`. `scripts/` are thin entry points. `tests/` mirror the modules. `site/` is the page source and generated data; `docs/` is the built static site. `artifacts/` and `data/` are committed so the repo is self-contained.

**Stack.** PyMC-Marketing 0.19 on PyMC 5, ArviZ, NumPy, pandas, SciPy, pytest. The site is a single page rendered by a small React runtime; no build tooling.

## What was deliberately left out

These are scope cuts, stated so they are not mistaken for oversights.

The data is synthetic and one geography. That is the point of the project, not a shortcut, but it means nothing here has met real-world problems like missing weeks, channel launches mid-series, or competitor activity. There is no cold-start story: a new channel with no spend history would get the prior and nothing else. Carryover is cut off at eight weeks; that matches the generator exactly, but on real data with TV's decay rate about 5.8% of the effect would fall outside the window, and the rule for choosing the window is written in `model.py`. Channel effects are fixed over time rather than time-varying. The priors are set from scale alone; in a real deployment the natural next step is to replace them with results from a geo-lift experiment, which is the calibration loop that turns an observational model into a defensible one. The forecaster and geo-lift notebooks planned as stretch work were not built. And the two open items from the divergence investigation, saving the full flags and a rerun at target accept 0.99, remain open.

---

Built by Parth, assisted by Fable. Every number in this document is exported from `artifacts/posterior.nc` and the committed data; if a refit changes them, the regeneration steps above change this document's source of truth, and the site, in one pass.
