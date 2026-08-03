# ML Portfolio

A weekly return-prediction and ranking strategy for the S&P 500, with an automated
data/modeling pipeline and a Streamlit dashboard for tracking performance.

Each week, a Ridge regression model predicts every S&P 500 stock's forward weekly
return from ~30 engineered factors (momentum, volatility, liquidity, Fama-French
factor loadings, sector-relative signals, and more), ranks the universe by predicted
return, and builds an equal-weighted, long-only portfolio from the top 20 picks.
Realized performance is tracked over time and visualized in a dashboard alongside a
SPY benchmark.

**This is a learning/research project, not investment advice.** See
[Known limitations](#known-limitations) before reading too much into any of the
numbers it produces.

## What it does

1. **Data extraction** — S&P 500 constituent list (Wikipedia), historical daily
   prices (yfinance, with a stooq fallback), and a SPY benchmark series.
2. **Feature engineering** — moving averages, RSI, Bollinger Bands, momentum,
   volatility/skewness, liquidity (dollar volume, Amihud illiquidity), rolling market
   beta, Fama-French 5-factor rolling loadings, and sector-relative versions of RSI
   and momentum. Every factor was validated with a univariate-IC / redundancy /
   ablation / sub-period-stability test before being kept — see
   [`docs/factor_analysis.md`](docs/factor_analysis.md).
3. **Modeling** — a Ridge regression pipeline (log-transform → scale → predict) is
   the production model. ElasticNet, RandomForest, and a Fama-French standalone
   signal were also built and compared (IC, decile spread, sub-period stability,
   pairwise correlation) as a research exercise in combining models — see
   [`docs/model_analysis.md`](docs/model_analysis.md).
4. **Portfolio construction** — top 20 stocks by predicted return, equal-weighted,
   long-only. (A fixed count, not a percentage of the universe — see
   [Known limitations](#known-limitations).)
5. **Performance tracking** — `data/processed/historical_performance.csv` compounds
   the portfolio's realized weekly returns into a running total value, Sharpe,
   Sortino, CAGR, volatility, max drawdown, and beta vs. SPY.
6. **Walk-forward backtest** — where live tracking has gaps, a walk-forward backtest
   (train on strictly-prior data only, no look-ahead) fills them in with a
   simulation of the current strategy. Backtested weeks are tagged
   `source=backtest` in `historical_performance.csv`, distinct from
   `source=live`, and the dashboard visually distinguishes the two — a backtested
   stretch is never presented as if it were real tracked performance.
7. **Dashboard** — a Streamlit app (`app.py`) showing a per-symbol candlestick chart,
   the portfolio's equity curve vs. SPY, risk metrics, a drawdown chart, and the
   current allocation.

## Project structure

```
ML_Portfolio/
├── app.py                     # Streamlit dashboard
├── functions.py                # Functions shared with app.py (portfolio construction, tracking, risk metrics)
├── notebooks/
│   └── ML_Portfolio_Management.ipynb   # The full pipeline: data → features → model → portfolio → tracking → backtest
├── data/
│   ├── raw/                    # holdings.csv, spy_price.csv (tracked); historical_price.csv (gitignored — regenerable, 100MB+)
│   └── processed/               # stock_portfolio.csv, weekly_portfolio.csv, historical_performance.csv (tracked);
│                                  # processed_historical_price.csv (gitignored — regenerable, 60MB+)
├── docs/                       # Factor analysis, model comparison, and strategy discussion notes
├── .github/workflows/           # Scheduled pipeline (see below)
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt
```

Run the notebook (`notebooks/ML_Portfolio_Management.ipynb`) top to bottom to
(re)fetch data, engineer features, fit the model, build the current week's
portfolio, and update `historical_performance.csv`. This can take a while — it
fetches ~500 symbols of price history and fits several models.

Then start the dashboard:

```bash
streamlit run app.py
```

## Automated pipeline (CI)

[`.github/workflows/workflow.yaml`](.github/workflows/workflow.yaml) runs the
notebook daily (17:30 UTC) and on manual dispatch, then commits and pushes the
updated output files (`historical_performance.csv`, `stock_portfolio.csv`,
`weekly_portfolio.csv`, `holdings.csv`, `spy_price.csv`) back to the repo. The raw
and processed price panels are deliberately excluded from that commit and from git
entirely — they're large, fully regenerable from yfinance, and gain nothing from
being version-controlled.

## Known limitations

- **Fama-French data lag**: Ken French's factor data library (one of the model's
  feature sources) lags real time by ~2 months, so the model's picks are only ever
  as current as that data allows — not a bug, a live constraint of a free data
  source.
- **Small sample size**: performance is tracked over a few dozen weeks so far.
  Sharpe, Sortino, and beta are numerically correct but statistically noisy at this
  sample size — treat them as illustrative, not validated.
- **No transaction costs** are modeled anywhere in the pipeline or backtest.
- **Long-only**: short-side portfolio construction is designed but not implemented.
- **Fixed top-20 selection**: not a percentage of the universe, chosen deliberately
  so the portfolio size doesn't silently balloon if the universe grows (this
  happened once, going from a 17-stock DJIA universe to the ~500-stock S&P 500).
- **Backtested vs. live data**: always check the `source` column in
  `historical_performance.csv` (and the dashboard's line style) before trusting a
  given week's number — some of the history is a walk-forward simulation, not a
  real tracked result.

## Further reading

- [`docs/factor_analysis.md`](docs/factor_analysis.md) — every factor's univariate
  IC, redundancy, ablation impact, and sub-period stability
- [`docs/model_analysis.md`](docs/model_analysis.md) — Ridge vs. ElasticNet vs.
  RandomForest vs. Fama-French, a data-leakage bug found and fixed along the way,
  and the IC-weighted combination result
- [`docs/return-prediction-models-discussion.md`](docs/return-prediction-models-discussion.md) —
  general notes on how return-prediction strategies are built and evaluated
