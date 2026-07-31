# Return Prediction Models: Discussion Notes

## 1. How a hedge fund uses return prediction models to generate profit

### Signal generation
The model predicts some version of future returns — direction, magnitude, or a ranking across assets — from features like:
- **Price/volume data**: momentum, mean reversion, microstructure signals (order book imbalance, bid-ask spread dynamics)
- **Fundamental data**: earnings surprises, valuation ratios, analyst revisions
- **Alternative data**: satellite imagery, credit card transactions, web traffic, sentiment
- **Cross-asset signals**: correlations, lead-lag relationships between related instruments

The output is usually a *cross-sectional* ranking (which assets will outperform others) rather than an absolute prediction, because relative predictions are more stable than absolute ones.

### Signal validation
- Out-of-sample / walk-forward testing to avoid overfitting to historical noise
- **Information Coefficient (IC)**: correlation between predicted and realized returns — even IC ≈ 0.05 can be profitable at scale
- Decay analysis: how fast predictive power fades, which determines holding period
- Turnover vs. transaction cost tradeoff

### Portfolio construction
Raw predictions are converted into position sizes via:
- Mean-variance or risk-parity optimization, using predicted returns as the expected-return input
- Constraints: sector/factor neutrality, position limits, leverage limits, turnover limits
- Factor risk control — neutralizing known factors (market beta, size, value, momentum) so the model's edge is genuine alpha, not repackaged beta
- Multi-model ensembles of weakly-correlated signals so noise cancels and the aggregate is more stable (methodologically similar to combining an Extended Isolation Forest and an LSTM Autoencoder with a diversity/correlation constraint)

### Execution
- Smart order routing / algorithmic execution (VWAP, TWAP, implementation shortfall) to preserve theoretical edge
- Timing matched to signal decay speed — fast signals need low latency, slower signals can be worked patiently

### Risk management and monitoring
- Drawdown controls, stop-losses, position limits
- Live IC / performance monitoring to catch signal decay or regime change
- Regime detection overlays to scale exposure down in unfavorable conditions

### Where the profit actually comes from
Rarely "we have the best model." Usually:
- Unique data/features (proprietary alt data, faster access)
- Better risk-adjusted position sizing, not just raw prediction accuracy
- Superior, cost-preserving execution
- Diversification across many weak, uncorrelated signals rather than one strong bet (Renaissance/Two Sigma-style philosophy)

---

## 2. Return prediction vs. price forecasting

**Price forecasting** predicts the actual future price level (e.g., "AAPL will be $215 in 30 days"). Problems:
- Prices are non-stationary (trend, drift, unit roots) — statistically messy
- Conflates direction and scale, making risk-adjusted attractiveness hard to read off directly
- Multi-step forecast errors compound

**Return prediction** predicts the *change* over a horizon, as a distribution or ranking (e.g., "AAPL's 5-day return will be top quintile of the S&P 500"). Advantages:
- Closer to stationary, so models generalize better
- Directly usable as the expected-return input to mean-variance optimization
- Naturally comparable across assets, enabling cross-sectional ranking

### Usage comparison

| | Price forecasting | Return prediction |
|---|---|---|
| Primary use | Rare standalone; occasionally in derivatives pricing checks or discretionary target-price research | Core input to systematic/quant portfolio construction |
| Output shape | Point estimate or price path | Distribution, ranking, or classification |
| Feeds into | A discretionary decision, or vol/derivatives models | Optimizers, risk models, position sizing directly |
| Failure mode | Wrong scale = wrong trade size even if direction is right | Even weak/noisy predictions can be profitable at scale with proper sizing |

Quant funds prefer return prediction because their downstream infrastructure (risk models, optimizers, factor-neutralization) consumes expected returns and a covariance matrix, not price paths. Price forecasting shows up more in discretionary/fundamental single-name research.

### Examples of return prediction models

**Classical / statistical**
- Linear factor models (Fama-French style)
- ARIMA / VAR on return series (often a baseline)
- OLS / Ridge / Lasso regressions (Lasso for feature selection among many candidate signals)

**Cross-sectional ranking models**
- Gradient boosted trees (XGBoost, LightGBM) — very popular for nonlinear interactions and robustness to noisy data
- Random forests as ensemble members

**Deep learning**
- LSTM / GRU networks on sequential price/volume/order-flow features
- Temporal Convolutional Networks (TCNs)
- Transformer-based models for multi-asset return prediction

**Market microstructure-specific**
- Order book imbalance models for very short-horizon return direction
- Kyle's lambda / price impact models (used alongside prediction, for execution)

**Ensemble approaches**
- Multi-model stacking with decorrelation requirements between models (structurally similar to the Spearman ρ diversity framework used in LOB anomaly detection), swapping the target from "is this anomalous" to "will this asset's return be relatively high or low"

---

## 3. Confirmed workflow: forecast → rank

1. **Forecast** expected return for every asset in the universe over a fixed horizon
2. **Rank** assets cross-sectionally by that forecast
3. **Go long** the top decile/quintile, **go short** the bottom decile/quintile (long/short equity)
4. **Size positions** using ranks or forecast magnitude, subject to risk constraints

### Why rank rather than use the raw forecast directly
- Model errors are often systematic across the whole universe (e.g., a bias that overpredicts everything in a bull run); ranking cancels that out since only relative order matters
- Cross-sectional information ("A will outperform B") is generally more reliable than absolute-level information ("A will return exactly 2.3%")
- Rank-based long/short construction is naturally market-neutral, stripping out broad market beta and isolating stock-picking alpha

### Nuances
- Rank-to-weight mapping isn't always linear — some funds z-score the forecast (so conviction matters, not just order) before feeding it to an optimizer
- The optimizer, not the raw ranking, typically does final position sizing — balancing expected return against risk (correlation, volatility, factor exposure)
- Ranking isn't universal: single-asset directional strategies (e.g., "will SPY be up or down tomorrow") have no cross-section to rank against, so they use threshold-based decisions instead

---

## 4. Converting the ML Stock Picker from classification to return prediction

Current project: binary classifier predicting whether a stock will outperform the cross-sectional median.
Target: regression/ranking model that forecasts each stock's return and ranks the universe by that forecast.

### Key changes

**Target variable**
- From: binary label `1` if stock return > cross-sectional median else `0`
- To: continuous forward return over the chosen horizon (e.g., 5-day, 20-day forward return), optionally cross-sectionally demeaned or z-scored per period so the target is comparable across time

**Loss function**
- From: binary cross-entropy
- To: MSE or MAE for direct return regression; alternatively a pairwise/listwise ranking loss (e.g., RankNet-style or Spearman-based objective) if the end goal is purely the ranking rather than the calibrated magnitude

**Model output layer**
- From: sigmoid → probability
- To: linear output → continuous return estimate (no activation, or scaled activation if returns are bounded/clipped)

**Evaluation metrics**
- From: accuracy, AUC-ROC, precision/recall
- To: Information Coefficient (Pearson or Spearman correlation between predicted and realized returns), rank correlation across the cross-section per period, decile spread (top decile mean return − bottom decile mean return), and IC stability/decay over time

**Portfolio construction / backtesting**
- From: trade signal = classifier output above a probability threshold
- To: rank all stocks each period by predicted return, form long top-quantile / short bottom-quantile portfolio, and backtest with position sizing (equal-weight or rank-weighted) plus transaction cost assumptions

**Feature set**
- Can largely be reused from the classifier — same predictive features generally work for both framings — but it's worth re-checking feature stationarity assumptions now that the target is continuous rather than a threshold

### Suggested next steps
1. Redefine the label pipeline to output continuous forward returns instead of the binary outperform flag
2. Swap the model head and loss function; keep the same feature pipeline as a first pass
3. Re-run evaluation using IC and decile-spread analysis instead of classification metrics
4. Rebuild the backtest to use rank-based long/short portfolio construction instead of a probability threshold
5. Optionally test a ranking-specific loss (pairwise/listwise) as a second iteration to compare against plain regression
