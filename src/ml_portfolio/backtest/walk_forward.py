from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_portfolio.config import DATA_DIR
from ml_portfolio.models.pipeline import build_full_ridge_pipeline


def _weekly_predictions(panel: pd.DataFrame, week) -> pd.DataFrame:
    """Fit on strictly-prior data, predict the given week's full cross-section
    (every symbol, not just the top-n) -- shared by walk_forward_backtest (which
    only wants the top-n) and signal_decay_by_horizon (which needs the whole
    ranking to compute IC correctly)."""
    train_df = panel[panel['date'] < week]
    predict_df = panel[panel['date'] == week]

    pipeline = build_full_ridge_pipeline()
    pipeline.fit(train_df.drop(columns=['target']), train_df['target'])
    y_pred = pipeline.predict(predict_df.drop(columns=['target']))

    return predict_df.assign(predicted_return=y_pred).sort_values('predicted_return', ascending=False)


def _weekly_picks(panel: pd.DataFrame, week, top_n: int) -> pd.DataFrame:
    return _weekly_predictions(panel, week).head(top_n)


def walk_forward_backtest(panel: pd.DataFrame, weeks: list, top_n: int = 20) -> pd.DataFrame:
    results = []
    prev_symbols = None
    for w in weeks:
        picks = _weekly_picks(panel, w, top_n)
        # picks['target'] is that week's already-realized weekly_log_return (known once
        # the week plays out) -- convert log return to simple return before averaging
        # across stocks, since only an asset's own log returns compound additively
        # across time, not across different assets at a single point in time.
        simple_returns = np.exp(picks['target']) - 1

        symbols = set(picks['symbol'])
        # Turnover: fraction of the book replaced since the previous week (standard
        # "portfolio turnover ratio" convention -- fully replacing all N names is
        # 100% turnover, not 200%, since a name entering and a name exiting are the
        # two halves of the same trade). First week has no prior portfolio to diff
        # against, so its turnover is NaN, not 0 -- 0 would wrongly claim "no
        # trading happened" rather than "not applicable."
        if prev_symbols is None:
            turnover = np.nan
        else:
            turnover = len(symbols - prev_symbols) / len(symbols)
        prev_symbols = symbols

        results.append({'date': w, 'daily_return': simple_returns.mean(), 'turnover': turnover})

    return pd.DataFrame(results)


def signal_decay_by_horizon(panel: pd.DataFrame, weeks: list, horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """How fast does the signal's predictive power fade beyond one week?

    For each week, fits on strictly-prior data (same no-look-ahead property as
    walk_forward_backtest) and predicts the full cross-section, then measures
    Spearman IC against realized returns held N weeks forward for each N in
    horizons. Directly answers whether a slower rebalance cadence would give up
    much signal -- if IC barely decays by horizon 4, weekly rebalancing may be
    trading away turnover/costs for little extra alpha; if it decays to ~0 by
    horizon 2, weekly is closer to necessary.

    Forward returns are computed separately from `panel` and only joined back in
    after prediction -- they must never reach the pipeline as a feature (they are
    look-ahead by construction, which is the whole point: we need to know what
    actually happened N weeks later to score the prediction against it).
    """
    panel = panel.sort_values(['symbol', 'date']).reset_index(drop=True)

    forward_returns = panel[['symbol', 'date']].copy()
    for h in horizons:
        # h rows forward per symbol on this already-weekly-filtered panel is ~h
        # weeks forward -- same convention the original weekly_log_return/target
        # column uses (there: 5 trading days forward on the daily panel), just at
        # multiple horizons instead of one, and measured in filtered weeks instead
        # of trading days since that's the panel available here.
        forward_returns[f'fwd_return_{h}w'] = panel.groupby('symbol')['close'].transform(
            lambda x: (x.shift(-h) - x) / x
        )

    results = []
    for w in weeks:
        predictions = _weekly_predictions(panel, w)
        week_fwd = forward_returns[forward_returns['date'] == w]
        merged = predictions[['symbol', 'predicted_return']].merge(week_fwd, on='symbol', how='inner')

        for h in horizons:
            valid = merged[['predicted_return', f'fwd_return_{h}w']].dropna()
            if len(valid) < 5:
                continue
            ic, _ = spearmanr(valid['predicted_return'], valid[f'fwd_return_{h}w'])
            results.append({'date': w, 'horizon_weeks': h, 'ic': ic})

    return pd.DataFrame(results)


def apply_transaction_costs(daily_return: pd.Series, turnover: pd.Series, cost_bps_per_side: float = 7.5) -> pd.Series:
    # cost_bps_per_side is charged on both halves of one unit of turnover -- the buy
    # for each name entering and the sell for each name exiting -- so total cost per
    # period is turnover * cost_bps_per_side * 2 (not turnover's own NaN first week,
    # which has no trade to cost).
    cost = turnover.fillna(0) * (cost_bps_per_side * 2) / 10000
    return daily_return - cost


def backfill_backtest_gap(processed_panel_path: Path, hist_perf_file_path: Path = None) -> None:
    # Idempotent: skips any week already present in historical_performance.csv, so
    # this is a no-op once the gap (historical_performance.csv had no data between
    # 2026-01-16 and 2026-07-24 -- this pipeline didn't exist yet, and the earlier
    # notebook-execution CI job had no step to persist its output) is filled. Safe
    # to run on every pipeline execution, notebook or scripted.
    hist_perf_file_path = hist_perf_file_path or DATA_DIR / 'processed' / 'historical_performance.csv'

    backtest_panel = pd.read_csv(processed_panel_path)
    backtest_panel['date'] = pd.to_datetime(backtest_panel['date'])

    gap_start, gap_end = pd.Timestamp('2026-01-16'), pd.Timestamp('2026-07-24')
    candidate_weeks = sorted(
        backtest_panel[(backtest_panel['date'] > gap_start) & (backtest_panel['date'] < gap_end)]['date'].unique()
    )

    if hist_perf_file_path.is_file():
        existing_dates = set(pd.to_datetime(pd.read_csv(hist_perf_file_path)['date']))
    else:
        existing_dates = set()
    gap_weeks = [w for w in candidate_weeks if w not in existing_dates]

    if not gap_weeks:
        return

    backtest_returns = walk_forward_backtest(backtest_panel, gap_weeks)

    hist = pd.read_csv(hist_perf_file_path) if hist_perf_file_path.is_file() else pd.DataFrame(columns=['date', 'daily_return'])
    hist['date'] = pd.to_datetime(hist['date'])
    if 'source' not in hist.columns:
        hist['source'] = pd.NA
    hist['source'] = hist['source'].fillna('live')

    backtest_returns_tagged = backtest_returns.copy()
    backtest_returns_tagged['date'] = pd.to_datetime(backtest_returns_tagged['date'])
    backtest_returns_tagged['source'] = 'backtest'

    combined = pd.concat(
        [hist[['date', 'daily_return', 'source']], backtest_returns_tagged[['date', 'daily_return', 'source']]],
        ignore_index=True,
    ).sort_values('date').reset_index(drop=True)
    assert combined['date'].is_unique, "duplicate dates after splicing the backtest into historical_performance.csv"

    init_value = 100000
    combined['total_value'] = 0.0
    combined['cumulative_return'] = 0.0
    combined.loc[0, ['daily_return', 'total_value', 'cumulative_return']] = [0.0, init_value, 0.0]
    for i in range(1, len(combined)):
        prev_value = combined.loc[i - 1, 'total_value']
        prev_cum = combined.loc[i - 1, 'cumulative_return']
        r = combined.loc[i, 'daily_return']
        combined.loc[i, 'total_value'] = prev_value * (1 + r)
        combined.loc[i, 'cumulative_return'] = (1 + r) * (1 + prev_cum) - 1

    combined = combined[['date', 'total_value', 'daily_return', 'cumulative_return', 'source']]
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
    combined.to_csv(hist_perf_file_path, index=False)
