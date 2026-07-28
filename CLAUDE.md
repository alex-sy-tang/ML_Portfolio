# Task: Convert ML_Portfolio from Classifier to Return-Prediction/Ranking Model

**Repo:** `alex-sy-tang/ML_Portfolio`
**Primary files:** `notebooks/ML_Portfolio_Management.ipynb`, `functions.py`

## Context

The project currently trains a `LogisticRegression` classifier to predict whether a
stock's weekly return will beat the cross-sectional median (`target` is binary: 1 if
`weekly_log_return > median_weekly_return`, else 0). The weekly portfolio is built by
filtering to `prediction == 1` and equal-weighting the survivors.

The goal is to convert this into a return-prediction model: forecast each stock's
continuous weekly return, rank the universe by that forecast each week, and build the
portfolio from the top of the ranking instead of a classification threshold.

## Changes (in order)

### 1. Target variable
In `create_target_variable()`: stop binarizing against the cross-sectional median.
Set `target = weekly_log_return` directly. The median-comparison logic can be removed.

### 2. Model pipeline
Swap `LogisticRegression()` for a regressor. Start with `Ridge` to keep a direct,
comparable baseline against the old linear classifier. Keep the same `Pipeline`
structure: log-transform → drop columns → scale → model. (`RandomForestRegressor` or
`XGBRegressor` are reasonable next iterations after the baseline works.)

### 3. Evaluation
Replace `model.score()` (accuracy) with:
- **Spearman IC**: `scipy.stats.spearmanr(y_pred, y_test)` — correlation between
  predicted and realized returns
- **Top-vs-bottom decile spread**: mean actual return of the top predicted decile
  minus the bottom predicted decile

`model.score()` will still run (returns R² for a regressor) and can be printed
alongside IC, but IC is the metric that actually determines whether the ranking is
usable for portfolio construction.

### 4. Portfolio construction
In `functions.py`, rewrite `create_weekly_stock_portfolio()`:
- Sort by predicted return (descending)
- Take the top quantile (e.g. top 20%) as long positions
- Equal-weight within that group
- Leave short-side construction optional / off by default (long-only to start)

### 5. What NOT to touch
These are target-agnostic and should be left as-is:
- Feature engineering: RSI, Bollinger Bands, momentum, volatility, moving averages
- `time_aware_split()`
- Downstream portfolio-tracking functions: `update_stock_portfolio()`,
  `calculate_portfolio_metrics()`, `historical_performance()`
- `app.py` — unless the `stock_portfolio.csv` schema changes in a way it depends on
  (check column names after the rewrite)

## Constraints

- **Learning mode**: skeleton/plan first, no silent full rewrites — this mirrors how
  the crypto surveillance project is being built (Plan Mode, incremental review, no
  autopilot beyond boilerplate)
- Keep the same file structure and pipeline shape; this is a target/model/evaluation
  swap, not a rearchitecture
- Flag any place where the `stock_portfolio.csv` or `weekly_portfolio.csv` column
  schema changes, since `app.py` reads from those files directly

## Definition of done

- [ ] `target` is continuous weekly log return, not a binary outperform flag
- [ ] Pipeline uses a regressor instead of `LogisticRegression`
- [ ] Evaluation reports Spearman IC and decile spread instead of accuracy
- [ ] `create_weekly_stock_portfolio()` builds the portfolio from a return ranking,
      not a classification filter
- [ ] Feature engineering and downstream tracking functions unchanged
- [ ] `app.py` still runs against the new `stock_portfolio.csv` schema (or is updated
      if the schema changed)
