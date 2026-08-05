import pandas as pd
import pytest

import ml_portfolio.portfolio.construction as construction_module
import ml_portfolio.portfolio.tracking as tracking_module
from ml_portfolio.portfolio.construction import create_weekly_stock_portfolio
from ml_portfolio.portfolio.tracking import calculate_portfolio_metrics, update_stock_portfolio


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    # Both functions under test write a CSV to DATA_DIR/'processed'/... as a side
    # effect. Redirect DATA_DIR to a throwaway tmp_path for every test in this file
    # so a test can never overwrite the real data/processed/*.csv the live dashboard
    # and CI pipeline depend on.
    (tmp_path / 'processed').mkdir()
    monkeypatch.setattr(construction_module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(tracking_module, 'DATA_DIR', tmp_path)


def test_create_weekly_stock_portfolio_sorts_and_weights_top_n():
    symbols = pd.Series(['A', 'B', 'C', 'D'], name='symbol')
    preds = pd.DataFrame({'predicted_return': [0.01, 0.05, -0.02, 0.03]})

    result = create_weekly_stock_portfolio(symbols, preds, top_n=2)

    assert list(result['symbol']) == ['B', 'D']
    assert result['weight'].tolist() == [0.5, 0.5]
    assert result['weight'].sum() == pytest.approx(1.0)


def test_create_weekly_stock_portfolio_cutoff_below_universe_size():
    symbols = pd.Series(['A', 'B'], name='symbol')
    preds = pd.DataFrame({'predicted_return': [0.01, 0.02]})

    result = create_weekly_stock_portfolio(symbols, preds, top_n=20)

    assert len(result) == 2
    assert result['weight'].sum() == pytest.approx(1.0)


def test_create_weekly_stock_portfolio_rejects_short():
    symbols = pd.Series(['A'], name='symbol')
    preds = pd.DataFrame({'predicted_return': [0.01]})
    with pytest.raises(NotImplementedError):
        create_weekly_stock_portfolio(symbols, preds, long_only=False)


def test_update_stock_portfolio_matches_by_week_not_exact_date():
    # Selection date is a Monday; price data only exists for the Wednesday of the
    # same ISO week -- an exact-date match would find nothing, the week-based match
    # (the actual fix for the stale-selection/fresh-price bug) should still find it.
    df_stock_portfolio = pd.DataFrame({'symbol': ['AAA'], 'predicted_return': [0.01], 'weight': [1.0]})
    df_price = pd.DataFrame({
        'symbol': ['AAA'],
        'date': ['2026-06-03'],  # Wednesday
        'close': [100.0],
    })
    selection_date = '2026-06-01'  # Monday, same ISO week

    result = update_stock_portfolio(df_stock_portfolio, df_price, selection_date)

    assert len(result) == 1
    assert result['close'].iloc[0] == 100.0


def test_update_stock_portfolio_no_crash_when_close_already_present():
    # Regression test for a latent NameError found while consolidating this function:
    # df_weekly_portfolio was never assigned when 'close' was already a column on
    # df_stock_portfolio, so the next line crashed instead of returning.
    df_stock_portfolio = pd.DataFrame({
        'symbol': ['AAA'], 'close': [100.0], 'date': pd.to_datetime(['2026-06-03']),
    })
    df_price = pd.DataFrame({'symbol': ['AAA'], 'date': ['2026-06-03'], 'close': [100.0]})

    result = update_stock_portfolio(df_stock_portfolio, df_price, '2026-06-01')

    assert 'week_of_day' in result.columns
    assert len(result) == 1


def test_calculate_portfolio_metrics_computes_returns_from_price_sum():
    portfolio = pd.DataFrame({
        'date': ['2026-01-01', '2026-01-01', '2026-01-08', '2026-01-08'],
        'close': [10.0, 20.0, 11.0, 22.0],
    })

    result = calculate_portfolio_metrics(portfolio)

    assert list(result['total_value']) == [30.0, 33.0]
    assert result['daily_return'].iloc[0] == 0
    assert result['daily_return'].iloc[1] == pytest.approx(0.1)
    assert result['cumulative_return'].iloc[1] == pytest.approx(0.1)
