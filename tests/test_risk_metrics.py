import numpy as np
import pandas as pd
import pytest

from functions import calculate_risk_metrics


def _make_hist_perf(daily_returns, dates=None, total_value_start=100000.0):
    n = len(daily_returns)
    if dates is None:
        dates = pd.date_range('2026-01-07', periods=n, freq='7D')
    total_value = [total_value_start]
    for r in daily_returns[1:]:
        total_value.append(total_value[-1] * (1 + r))
    return pd.DataFrame({'date': dates, 'daily_return': daily_returns, 'total_value': total_value})


def test_volatility_uses_weekly_not_daily_annualization():
    # Regression test for the annualization-period bug: using 252 (daily) instead of
    # 52 (weekly) inflated Sharpe/Sortino/Volatility by sqrt(252/52) ~= 2.2x once the
    # walk-forward backtest started writing weekly-return rows into this column.
    returns = [0.0, 0.01, -0.02, 0.015, -0.01, 0.02, -0.015]
    hist = _make_hist_perf(returns)
    spy = pd.DataFrame({'date': hist['date'], 'close': 100 * np.cumprod([1.0] * len(returns))})

    metrics = calculate_risk_metrics(hist, spy)

    # pandas Series.std() defaults to ddof=1 (sample std), matching what
    # calculate_risk_metrics uses internally -- not numpy's default ddof=0.
    expected_vol = pd.Series(returns).std() * np.sqrt(52)
    assert metrics['volatility'] == pytest.approx(expected_vol, rel=1e-6)
    not_expected_daily_vol = pd.Series(returns).std() * np.sqrt(252)
    assert metrics['volatility'] != pytest.approx(not_expected_daily_vol, rel=1e-2)


def test_sharpe_uses_weekly_not_daily_annualization():
    returns = [0.0, 0.01, -0.02, 0.015, -0.01, 0.02, -0.015]
    hist = _make_hist_perf(returns)
    spy = pd.DataFrame({'date': hist['date'], 'close': 100 * np.cumprod([1.0] * len(returns))})

    metrics = calculate_risk_metrics(hist, spy, risk_free_rate=0.0)

    returns_arr = pd.Series(returns)
    expected_sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(52)
    assert metrics['sharpe'] == pytest.approx(expected_sharpe, rel=1e-6)


def test_beta_pairs_against_five_trading_day_spy_return():
    # Regression test for the beta-pairing bug: each portfolio row is a full week's
    # return, so it must be paired against SPY's return over that same ~week window
    # (5 trading days), not SPY's single-day return.
    dates = pd.date_range('2026-01-07', periods=10, freq='7D')
    rng = np.random.default_rng(0)
    daily_returns = rng.normal(0, 0.02, size=10)
    hist = _make_hist_perf(list(daily_returns), dates=dates)

    spy_close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=10))
    spy = pd.DataFrame({'date': dates, 'close': spy_close})

    metrics = calculate_risk_metrics(hist, spy)

    expected_spy_return = pd.Series(spy_close).pct_change(periods=5)
    merged = pd.DataFrame({'date': dates, 'daily_return': daily_returns}).merge(
        pd.DataFrame({'date': dates, 'spy_return': expected_spy_return}), on='date'
    ).dropna()
    expected_beta = merged['daily_return'].cov(merged['spy_return']) / merged['spy_return'].var()

    assert metrics['beta'] == pytest.approx(expected_beta)

    # And explicitly not the single-day pairing the bug used before the fix.
    wrong_spy_return = pd.Series(spy_close).pct_change(periods=1)
    wrong_merged = pd.DataFrame({'date': dates, 'daily_return': daily_returns}).merge(
        pd.DataFrame({'date': dates, 'spy_return': wrong_spy_return}), on='date'
    ).dropna()
    wrong_beta = wrong_merged['daily_return'].cov(wrong_merged['spy_return']) / wrong_merged['spy_return'].var()
    assert metrics['beta'] != pytest.approx(wrong_beta)


def test_max_drawdown_is_non_positive_and_reflects_worst_peak_to_trough_drop():
    returns = [0.0, 0.1, -0.5, 0.2, 0.1]
    hist = _make_hist_perf(returns)
    spy = pd.DataFrame({'date': hist['date'], 'close': [100, 101, 102, 103, 104]})

    metrics = calculate_risk_metrics(hist, spy)

    assert metrics['max_drawdown'] <= 0
    running_max = hist['total_value'].cummax()
    expected = (hist['total_value'] / running_max - 1).min()
    assert metrics['max_drawdown'] == pytest.approx(expected)
