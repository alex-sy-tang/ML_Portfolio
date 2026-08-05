from pathlib import Path

import numpy as np
import pandas as pd

from ml_portfolio.config import DATA_DIR
from ml_portfolio.models.pipeline import build_full_ridge_pipeline


def walk_forward_backtest(panel: pd.DataFrame, weeks: list, top_n: int = 20) -> pd.DataFrame:
    results = []
    for w in weeks:
        train_df = panel[panel['date'] < w]
        predict_df = panel[panel['date'] == w]

        pipeline = build_full_ridge_pipeline()
        pipeline.fit(train_df.drop(columns=['target']), train_df['target'])
        y_pred = pipeline.predict(predict_df.drop(columns=['target']))

        picks = predict_df.assign(predicted_return=y_pred).sort_values('predicted_return', ascending=False).head(top_n)
        # picks['target'] is that week's already-realized weekly_log_return (known once
        # the week plays out) -- convert log return to simple return before averaging
        # across stocks, since only an asset's own log returns compound additively
        # across time, not across different assets at a single point in time.
        simple_returns = np.exp(picks['target']) - 1
        results.append({'date': w, 'daily_return': simple_returns.mean()})

    return pd.DataFrame(results)


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
