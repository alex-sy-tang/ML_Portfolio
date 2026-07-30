# Return-Prediction Models: Testing, Combination, and a Data-Leakage Fix

## Context

Following `docs/factor_analysis.md` (which tested individual factors), this round
built additional return-prediction models beyond the original Ridge baseline,
tested each one the same rigorous way, and combined them via IC-weighted averaging.
Along the way, testing a model family flexible enough to expose it (RandomForest)
surfaced a genuine data-leakage bug in two factors from the previous round
(`vol_1M/6M/12M`, `skew_1M/6M/12M`) that had gone unnoticed under linear models. That
fix is documented here since it materially changes every number in this report — all
final tables below are **post-fix**, with the pre-fix numbers shown separately to
make clear what the leak was worth.

---

## 1. Models built and tested

| Model | What it is | How its signal differs from the others |
|---|---|---|
| **Ridge** (existing) | L2-regularized linear regression on all 25 factors (17 original + 8 from the factor-testing round) | Baseline |
| **ElasticNet** (`ElasticNetCV`) | L1+L2 blended linear regression, regularization strength chosen via 5-fold CV | Same linear mechanism and features as Ridge — expected to be highly correlated with it |
| **RandomForest** | 200 trees, max depth 6, same 25+ factors | Non-linear, captures feature interactions no linear model can represent |
| **Fama-French loadings** | Each stock's rolling 60-day loadings on Mkt-RF/SMB/HML/RMW/CMA (free via `pandas_datareader`'s Ken French library access), fed in as 5 more columns to the models above | Not a separate model — new *information* for the existing models |
| **Fama-French standalone** | `expected_return = Σ loading_f × historical_mean_factor_return_f` — the classic asset-pricing formula, using only the training period's factor averages | Not fit to our target at all; a structurally different mechanism from every ML model above, built specifically as an ensemble diversification candidate |

## 2. Testing methodology

Parallel to the factor-testing process, applied to model predictions instead of raw
factors:
- **Out-of-sample IC and decile spread** per model on the held-out test set (same
  `spearmanr`/`qcut` calls used throughout this project).
- **Sub-period stability**: test set split chronologically in half, IC computed on
  each half separately — a model whose IC flips sign between halves is far less
  trustworthy than one that's consistently the same sign.
- **Model-correlation matrix**: Pearson correlation between each pair of models'
  predictions — the model-level equivalent of the factor redundancy check. Low
  correlation between two models signals real diversification value; high
  correlation means one is redundant with the other regardless of its own IC.
- **IC-weighted combination**: combination weights come from a validation slice
  carved out of the training period specifically for this purpose (`X_val`/`y_val`,
  via `carve_validation_split()`) — never from the test set, so the final test-set
  IC is an honest measure of whether combining helped, not a combination quietly fit
  to the same data it's evaluated on. Each model's predictions are standardized
  (z-scored, using validation-set mean/std) before combining, since different model
  families don't share an output scale. Negative or undefined validation IC is
  floored at zero (excluded from the blend, not inverted).

## 3. A data-leakage bug found during model testing (and fixed)

`calc_volatility()`/`calc_skewness()` compute a rolling window over
`weekly_log_return` with `min_periods=1` and no shift. Since `.rolling(window=W)` is
inclusive of the current row, and `target` is a direct copy of `weekly_log_return`
(`create_target_variable()` sets `target = weekly_log_return`), the window at row
*t* included `weekly_log_return[t]` — which **is** `target[t]` — as one of its
inputs. For `vol_1M`/`skew_1M` (window = 4), the current row is 25% of the
statistic.

**Why RandomForest, specifically, blew this leak up into an IC of 0.21** while
Ridge/ElasticNet barely moved: a linear model applies *one global coefficient* to a
feature, uniformly across every row. `vol_1M`'s value is a mix of genuine backward-
looking information (3 real past weeks) and a small leaked component (~25% weight),
smoothed together inside a single standard-deviation statistic — a linear model
can't "reach into" that mix and selectively use only the leaked part, so the net
effect on its coefficient is modest. A decision tree does the opposite: it builds
many narrow, conditional rules (`vol_1M > threshold AND momentum_6M > 0 → predict
X`), and with 200 trees at depth 6 it has enough capacity to carve out rules that
closely track the deterministic mathematical relationship between an unusual
`target[t]` and the resulting unusual `vol_1M[t]`/`skew_6M[t]` value. It's closer to
reverse-engineering an equation than fitting a general trend — something only a
sufficiently flexible model family can do. Skewness compounded this: as a
third-moment statistic (cubed deviations), it's far more sensitive to a single
outlier value than a standard deviation is, so even `skew_6M` (window = 26, only
~3.8% weight by row-count) showed up as one of RandomForest's two most important
features, alongside `vol_1M`.

**The fix** — shift the series by one row before rolling, so the window only ever
sees strictly past values:
```python
# before
lambda x: x.rolling(window = window, min_periods = 1).std() * np.sqrt(window)
# after
lambda x: x.shift(1).rolling(window = window, min_periods = 1).std() * np.sqrt(window)
```
(identical change for `calc_skewness`, with `.skew()`)

### Impact of the fix

| Model | IC before fix | IC after fix |
|---|---|---|
| Ridge | 0.0250 | **0.0479** |
| ElasticNet | 0.0248 | **0.0516** |
| RandomForest | **0.2086** | 0.0353 |
| Fama-French standalone | 0.0248 | 0.0244 (unaffected, as expected — doesn't use vol/skew) |
| Combined (IC-weighted) | 0.1426 | **0.0541** |
| Main Ridge model, single-model eval (all 25 factors) | 0.0310 | **0.0661** |

Ridge and ElasticNet's IC roughly *doubled* after the fix — the leak wasn't free
signal being handed only to RandomForest, it was actively corrupting `vol`/`skew`
for every model, diluting their otherwise-genuine backward-looking information with
a noisy, leaked component. Fama-French standalone barely moved, which is the
expected sanity check (it never touches those two factors).

---

## 4. Final results (post-fix)

### Model comparison
| Model | IC | Decile spread | Sub-period IC (1st / 2nd half) |
|---|---|---|---|
| Ridge | 0.0479 | 0.0082 | 0.067 / 0.035 |
| ElasticNet | 0.0516 | 0.0093 | 0.081 / 0.030 |
| RandomForest | 0.0353 | 0.0052 | **0.101 / -0.016** |
| Fama-French standalone | 0.0244 | 0.0098 | 0.035 / 0.017 |
| **Combined (IC-weighted)** | **0.0541** | **0.0114** | — |

The combination now beats **every** individual model, including the best one
(ElasticNet) — before the fix, it underperformed RandomForest alone, since
RandomForest's number was inflated and the other three were redundant with each
other rather than with it.

**RandomForest's sub-period IC flips sign** (0.101 → -0.016). Post-fix, it's both
the weakest individual model and the least stable across time — worth taking at
face value rather than explaining away. There isn't currently good evidence that
the non-linear capacity is adding real value with these hyperparameters
(`max_depth=6`, 200 trees) once the leak is removed; revisiting hyperparameters is a
reasonable next step, not a reason to drop tree-based models outright.

### Model correlation matrix (test-set predictions)
| | Ridge | ElasticNet | RandomForest | FF standalone |
|---|---|---|---|---|
| **Ridge** | 1.00 | 0.97 | 0.47 | 0.13 |
| **ElasticNet** | 0.97 | 1.00 | 0.50 | 0.17 |
| **RandomForest** | 0.47 | 0.50 | 1.00 | 0.15 |
| **FF standalone** | 0.13 | 0.17 | 0.15 | 1.00 |

Ridge and ElasticNet at 0.97 are effectively the same signal — expected, since
they're two regularized variants of the same linear fit on identical features.
RandomForest sits at a genuinely different ~0.47-0.50 — real, if partial,
diversification. Fama-French standalone is the least correlated with everything
(0.13-0.17), confirming it as the most structurally independent signal of the four,
despite having the most modest standalone IC — exactly the "different mechanism,
not just different features" reasoning that motivated building it.

### Combination weights (from validation-set IC, floored at zero)
Ridge 30.9%, ElasticNet 28.1%, RandomForest 21.5%, Fama-French standalone 19.5% — a
reasonably balanced blend, not dominated by any single model.

---

## 5. Key takeaways

1. **Testing multiple model families is what caught this bug.** A single linear
   model's evaluation gave no reason to suspect `vol_1M`/`skew_*` were leaking the
   target — the effect was too diluted for a linear coefficient to expose. Only
   testing something flexible enough to exploit the leak made it visible.
2. **A dramatic result from one model, everything else unchanged, is itself the
   signal to investigate** — not a result to report at face value. This is the same
   lesson from the original 17-stock DJIA IC of 0.85 earlier in this project.
3. **Post-fix, the honest picture is: Ridge and ElasticNet are the strongest and
   most similar; RandomForest is real but modest and unstable; Fama-French
   standalone is the weakest individually but the most independent — and combining
   all four genuinely beats any one of them alone.**

## 6. Caveats and next steps

- All results here come from a single chronological train/validation/test split, the
  same limitation noted in `docs/factor_analysis.md` — a full walk-forward backtest
  across many rebalance dates is still the more rigorous validation this needs
  before trusting it for real capital.
- RandomForest's hyperparameters (`max_depth=6`, `n_estimators=200`) were not tuned;
  its sub-period instability may improve or may not with different settings.
- The Fama-French *loadings-as-features* variant was evaluated implicitly (as part
  of the 25+ factor set every model already uses) rather than ablated on its own the
  way the standalone signal was — a dedicated ablation test (add/remove the 5
  loading columns specifically) would isolate their individual contribution, the
  same way Part 4's factor harness did for the other new factors.
