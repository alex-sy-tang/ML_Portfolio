# Factor Analysis: New Factors, Testing Methodology, and Results

## Context

Following the classifier-to-return-prediction conversion and the Alpha Vantage/FMP →
Wikipedia/yfinance data migration (which unlocked the full S&P 500 universe instead of
a 17-stock DJIA subset), this round of work added candidate factors beyond the
original 17 and built a reusable process to test whether each factor — new or
existing — actually earns its place in the model. All numbers below come from a real
run against the live 502-symbol S&P 500 dataset (`data/raw/historical_price.csv`,
2021-01-04 through the most recent trading day), using the same held-out,
time-aware test split (`time_aware_split()`) as the main model evaluation.

---

## 1. Existing factors (baseline, 17)

| Factor | Category | What it measures |
|---|---|---|
| `close`, `volume`, `vwap` | Price level | Raw price/volume/typical-price |
| `MA_200`, `MA_100`, `MA_50` | Trend | Distance from 200/100/50-day moving average (log-transformed vs. `close`) |
| `vol_1M`, `vol_6M`, `vol_12M` | Volatility | Rolling std of `weekly_log_return`, 4/26/52-week windows |
| `RSI_3`, `RSI_9`, `RSI_14` | Short-term reversal | Relative Strength Index, 3/9/14-day windows |
| `hband`, `lband` | Mean reversion | Bollinger Bands (20-day, 2 std), log-transformed vs. `close` |
| `momentum_12M`, `momentum_6M`, `momentum_1M` | Momentum | Price change over ~252/126/21 trading days |

## 2. New factors — implemented and tested this round (8)

| Factor | Category | Formula / construction | Rationale |
|---|---|---|---|
| `dollar_volume` | Liquidity | `close × volume` | How many dollars actually trade — raw share volume alone doesn't capture this |
| `amihud_illiquidity` | Liquidity | Rolling mean of `\|daily_return\| / dollar_volume`, 21-day window | Standard illiquidity measure (Amihud, 2002) — how much price moves per dollar traded |
| `beta_60d` | Risk | Rolling cov(stock, market-proxy) / var(market-proxy), 60-day window. Market proxy = equal-weight average daily return across the full universe (no new data fetch) | Systematic risk exposure; also infrastructure for a future market-neutral long-short book |
| `skew_1M`, `skew_6M`, `skew_12M` | Higher-moment risk | Rolling skewness of `weekly_log_return`, 4/26/52-week windows (mirrors `vol_1M/6M/12M`) | Captures return-distribution asymmetry ("lottery stock" vs. "crash risk" behavior) |
| `RSI_14_sector_relative` | Sector-neutral | `RSI_14` minus its GICS sector's mean that week | Isolates stock-specific signal from sector-wide moves |
| `momentum_6M_sector_relative` | Sector-neutral | `momentum_6M` minus its GICS sector's mean that week | Same idea, applied to momentum |

A fix was bundled in alongside these: `vol_1M/6M/12M` previously ran on the daily
panel, so their windows (4/26/52) meant trading days, not the weeks their names and
comments claimed. They (and the new skewness columns) now run after the weekly
Wed/Thu filter, so the windows are genuine weeks.

## 3. Planned, not yet implemented — fundamentals

Value (P/E, P/B, EV/EBITDA, dividend yield), quality (ROE, gross margin, debt/equity,
earnings stability), and size (market cap) factors are scoped but not built. Approach
confirmed: prototype with `yfinance`'s `Ticker.info` on a handful of symbols first
(fast, pre-computed fields, patchy reliability at scale), then build the production
version on SEC EDGAR's free `data.sec.gov` XBRL company-facts API (official, free
forever, no rate-limit risk, but requires a ticker→CIK mapping and computing the
ratios ourselves from raw filing data rather than pre-computed fields).

---

## 4. Testing process

Four tests, ordered cheapest/most-informative-first so weak candidates get filtered
out before the expensive step:

1. **Univariate IC** — Spearman correlation between the raw factor alone and
   `target`, on the held-out test set. Answers: does this factor, in isolation,
   carry any information about forward returns at all?
2. **Redundancy check** — correlation between the candidate factor and each of the
   17 existing factors, flagged above `|r| ≥ 0.7`. A "new" factor highly correlated
   with an existing one isn't adding independent information, whatever its own IC.
3. **Ablation test** — the real test. For each of the **17 existing factors**:
   refit Ridge with that one factor *removed* from the baseline and compare IC/decile
   spread against the full baseline (leave-one-out — does removing it hurt?). For
   each **new candidate**: refit Ridge with that one factor *added* to the baseline
   and compare against the baseline (add-one — does adding it help?).
4. **Sub-period stability** — the test set split chronologically into two halves,
   univariate IC computed on each half separately. A factor whose IC flips sign
   between halves is far less trustworthy than one that's the same sign throughout,
   even if the two average out to the same overall number.

Implementation: `univariate_ic()`, `redundancy_check()`, `build_ridge_pipeline()`,
`ablation_test()`, `sub_period_ic()` in the notebook's "Factor Testing" section,
reusing the exact same `spearmanr`/`qcut` calls as the main evaluation cell — no new
statistical machinery, just applied per-factor instead of to the model's blended
output.

---

## 5. Results

**Baseline (17 existing factors only):** IC = 0.0291, decile spread = 0.0018
**Full model (all 25 columns):** R² = 0.0002, IC = 0.0248, decile spread = 0.0122

| Factor | Type | Univariate IC | p-value | Redundancy | Ablation Δ IC | Ablation Δ decile spread | Sub-period IC (1st / 2nd half) |
|---|---|---|---|---|---|---|---|
| close | existing | -0.0089 | 0.174 | — | +0.0045 | +0.0013 | -0.018 / 0.000 |
| volume | existing | 0.0199 | 0.002 | — | +0.0001 | 0.0000 | 0.029 / 0.016 |
| vwap | existing | -0.0087 | 0.186 | — | +0.0045 | +0.0012 | -0.017 / 0.000 |
| MA_200 | existing | -0.0159 | 0.015 | — | -0.0001 | 0.0000 | -0.028 / -0.003 |
| MA_100 | existing | -0.0107 | 0.104 | — | **+0.0064** | +0.0011 | -0.022 / 0.002 |
| MA_50 | existing | -0.0089 | 0.175 | — | -0.0005 | 0.0000 | -0.019 / 0.002 |
| vol_1M | existing | 0.0247 | <0.001 | — | **-0.0119** | -0.0022 | 0.041 / 0.022 |
| vol_6M | existing | 0.0109 | 0.096 | — | -0.0009 | -0.0012 | 0.021 / 0.014 |
| vol_12M | existing | 0.0154 | 0.019 | — | -0.0001 | -0.0012 | 0.031 / 0.003 |
| RSI_3 | existing | -0.0323 | <0.001 | — | -0.0008 | +0.0007 | -0.079 / 0.011 |
| RSI_9 | existing | -0.0268 | <0.001 | — | +0.0001 | +0.0001 | -0.040 / -0.016 |
| RSI_14 | existing | -0.0201 | 0.002 | — | +0.0002 | +0.0002 | -0.019 / -0.023 |
| hband | existing | -0.0074 | 0.261 | — | **-0.0030** | +0.0016 | -0.015 / 0.002 |
| lband | existing | -0.0092 | 0.162 | — | +0.0043 | +0.0009 | -0.018 / 0.000 |
| momentum_12M | existing | 0.0405 | <0.001 | — | +0.0028 | +0.0014 | 0.049 / 0.040 |
| momentum_6M | existing | 0.0262 | <0.001 | — | 0.0000 | 0.0000 | 0.039 / 0.013 |
| momentum_1M | existing | -0.0174 | 0.008 | — | -0.0004 | -0.0003 | 0.004 / -0.036 |
| dollar_volume | new | 0.0117 | 0.074 | 0.705 | -0.0005 | -0.0001 | 0.019 / 0.012 |
| amihud_illiquidity | new | 0.0023 | 0.729 | 0.000 | 0.0000 | 0.0000 | 0.005 / 0.001 |
| beta_60d | new | 0.0162 | 0.013 | 0.000 | **+0.0016** | +0.0005 | 0.006 / 0.026 |
| skew_1M | new | -0.0182 | 0.005 | 0.000 | **-0.0215** | +0.0045 | 0.000 / -0.036 |
| skew_6M | new | 0.0219 | <0.001 | 0.000 | -0.0044 | +0.0049 | 0.055 / -0.008 |
| skew_12M | new | 0.0063 | 0.338 | 0.000 | -0.0083 | +0.0027 | 0.015 / 0.006 |
| RSI_14_sector_relative | new | -0.0026 | 0.697 | 0.877 | +0.0038 | +0.0011 | 0.017 / -0.021 |
| **momentum_6M_sector_relative** | new | **0.0514** | **<0.001** | 0.928 | **+0.0201** | +0.0012 | 0.077 / 0.030 |

Note on `ablation_delta_ic` sign: for **existing** factors it's `baseline_IC −
ablated_IC` (positive = removing this factor hurts, i.e. it's pulling weight);
for **new** factors it's `expanded_IC − baseline_IC` (positive = adding this factor
helps).

### Interpretation

**A finding before the factor findings**: the plan expected `RSI_14` to show up as
the dominant factor here, based on the earlier 17-stock DJIA result. It doesn't
(ablation Δ = 0.0002, negligible). That's not a bug in the harness — it's the same
effect already observed once this session: an apparently strong signal on a tiny,
non-diverse universe mostly evaporated once tested on the real, much larger S&P 500
universe. No single existing factor dominates now.

**Strong — clear keep:**
- **`momentum_6M_sector_relative`** — by a wide margin the best factor tested,
  existing or new. Highest univariate IC (0.051, p < 10⁻¹⁴), largest ablation
  improvement (+0.020), same-sign stable across both sub-periods. Correlated 0.93
  with raw `momentum_6M`, so it's a refinement rather than independent information —
  but it clearly outperforms the original on every metric, which answers the earlier
  open question directly: for momentum specifically, removing the sector-wide
  component makes the signal *stronger*, not an artifact of one.
- **`beta_60d`** — modest (IC 0.016, ablation +0.0016) but genuinely real, no
  redundancy with anything existing, consistent sign across sub-periods. Framed from
  the start as infrastructure for a future market-neutral long-short book more than a
  standalone alpha source — that framing holds up.

**Weak or harmful — recommend dropping:**
- **`skew_1M`, `skew_6M`, `skew_12M`** — every one shows a *negative* ablation delta
  (adding any of them makes the model worse), and their univariate ICs don't even
  agree in sign with each other. Exactly the outcome flagged as the thing to check
  before trusting skewness.
- **`dollar_volume`** — univariate IC not significant (p = 0.074), redundant (0.70)
  with something already in the baseline, no ablation benefit.
- **`amihud_illiquidity`** — univariate IC ≈ 0 (p = 0.73). Not redundant (genuinely
  new information), but that new information doesn't carry signal in this
  particular construction — a window-size question rather than necessarily a
  dead end.

**Mixed:**
- **`RSI_14_sector_relative`** — unlike momentum's version, sector-neutralizing RSI
  doesn't show the same improvement (univariate IC ≈ 0, not significant). Real
  asymmetry: sector-relative construction helped one factor and not the other.

**Bonus finding, not part of the original scope**: two *existing* factors are
actively hurting the model — **`vol_1M`** (ablation Δ = -0.0119, the single largest
negative effect of any factor tested) and **`hband`** (Δ = -0.0030). Worth
reconsidering their place in the baseline regardless of anything new.

---

## 6. Caveats and next steps

- All results come from a **single chronological train/test split**. A credible
  final result needs the walk-forward, multi-period backtest already recommended
  separately — one split can't fully distinguish a stable effect from a lucky one.
- Recommended action based on this evidence: drop the three skewness columns from
  the working feature set, keep `momentum_6M_sector_relative` and `beta_60d`,
  revisit `dollar_volume`/`amihud_illiquidity` (different windows, or drop), and
  reconsider `vol_1M`/`hband`'s place in the existing baseline. Not yet acted on —
  this document is the evidence, the pruning is a separate decision.
- Fundamentals factors remain the next concrete step: yfinance prototype first,
  SEC EDGAR for the production version.
