import numpy as np
import pandas as pd
import pytest

import ml_portfolio.backtest.walk_forward as wf_module


class _FakePipeline:
    """Records what it was trained on instead of actually fitting a model, so the
    test can check walk_forward_backtest's data-slicing property in isolation from
    the real Ridge pipeline (which needs a full feature set)."""

    def __init__(self, sink):
        self._sink = sink

    def fit(self, X, y):
        self._sink.append(X['date'].max())
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_walk_forward_backtest_never_trains_on_the_predicted_week(monkeypatch):
    dates = pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28'])
    rows = [
        {'symbol': symbol, 'date': d, 'target': 0.01 * (i + 1)}
        for i, d in enumerate(dates)
        for symbol in ['AAA', 'BBB', 'CCC']
    ]
    panel = pd.DataFrame(rows)

    seen_max_train_dates = []
    monkeypatch.setattr(
        wf_module, 'build_full_ridge_pipeline', lambda: _FakePipeline(seen_max_train_dates)
    )

    weeks = [dates[1], dates[2], dates[3]]
    wf_module.walk_forward_backtest(panel, weeks, top_n=2)

    assert len(seen_max_train_dates) == len(weeks)
    for max_train_date, predicted_week in zip(seen_max_train_dates, weeks):
        assert max_train_date < predicted_week, (
            f"training data included {max_train_date}, which is not strictly "
            f"before the predicted week {predicted_week} -- look-ahead leak"
        )


def test_walk_forward_backtest_expanding_window_grows_each_week(monkeypatch):
    dates = pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28'])
    rows = [
        {'symbol': symbol, 'date': d, 'target': 0.01}
        for d in dates
        for symbol in ['AAA', 'BBB', 'CCC']
    ]
    panel = pd.DataFrame(rows)

    train_sizes = []

    class _SizeRecordingPipeline:
        def fit(self, X, y):
            train_sizes.append(len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(
        wf_module, 'build_full_ridge_pipeline', lambda: _SizeRecordingPipeline()
    )

    weeks = [dates[1], dates[2], dates[3]]
    wf_module.walk_forward_backtest(panel, weeks, top_n=2)

    assert train_sizes == sorted(train_sizes), "training set should only grow (expanding window), never shrink"
    assert train_sizes[0] < train_sizes[-1]


class _ScriptedPipeline:
    """Returns pre-programmed per-symbol scores for each week, so the resulting
    top-n picks -- and therefore turnover -- are known exactly, instead of being at
    the mercy of a real model's hard-to-predict output."""

    def __init__(self, scores_by_week):
        self._scores_by_week = scores_by_week

    def fit(self, X, y):
        return self

    def predict(self, X):
        week = X['date'].iloc[0]
        scores = self._scores_by_week[week]
        return X['symbol'].map(scores).values


def test_walk_forward_backtest_turnover_matches_hand_computed_picks(monkeypatch):
    dates = pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28', '2026-02-04'])
    symbols = ['A', 'B', 'C', 'D']
    rows = [{'symbol': s, 'date': d, 'target': 0.01} for d in dates for s in symbols]
    panel = pd.DataFrame(rows)

    # Programmed (via rank order of scores, top_n=2) so the picks are:
    #   dates[1]: A, B  -- first backtested week, no prior portfolio -> turnover NaN
    #   dates[2]: A, B  -- unchanged from the week before -> turnover 0.0
    #   dates[3]: C, D  -- fully replaced -> turnover 1.0
    #   dates[4]: A, C  -- 1 of 2 names replaced (B out, A back in; D out, C stays) -> turnover 0.5
    scores_by_week = {
        dates[1]: {'A': 4, 'B': 3, 'C': 2, 'D': 1},
        dates[2]: {'A': 4, 'B': 3, 'C': 2, 'D': 1},
        dates[3]: {'A': 1, 'B': 2, 'C': 4, 'D': 3},
        dates[4]: {'A': 4, 'B': 1, 'C': 3, 'D': 2},
    }
    monkeypatch.setattr(wf_module, 'build_full_ridge_pipeline', lambda: _ScriptedPipeline(scores_by_week))

    weeks = [dates[1], dates[2], dates[3], dates[4]]
    result = wf_module.walk_forward_backtest(panel, weeks, top_n=2)

    assert np.isnan(result['turnover'].iloc[0])
    assert result['turnover'].iloc[1] == pytest.approx(0.0)
    assert result['turnover'].iloc[2] == pytest.approx(1.0)
    assert result['turnover'].iloc[3] == pytest.approx(0.5)


def test_signal_decay_never_leaks_forward_returns_as_features(monkeypatch):
    # The whole point of signal_decay_by_horizon is to score predictions against
    # future data the model must never see at fit/predict time. This is the
    # critical safety property: assert none of the fwd_return_* columns the
    # function computes internally ever reach the pipeline.
    dates = pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28'])
    rows = [
        {'symbol': s, 'date': d, 'close': 100.0 + i, 'target': 0.01}
        for i, d in enumerate(dates)
        for s in ['AAA', 'BBB']
    ]
    panel = pd.DataFrame(rows)

    seen_columns = []

    class _ColumnRecordingPipeline:
        def fit(self, X, y):
            seen_columns.append(set(X.columns))
            return self

        def predict(self, X):
            seen_columns.append(set(X.columns))
            return np.zeros(len(X))

    monkeypatch.setattr(wf_module, 'build_full_ridge_pipeline', lambda: _ColumnRecordingPipeline())

    wf_module.signal_decay_by_horizon(panel, [dates[1], dates[2]], horizons=(1, 2))

    assert seen_columns, "pipeline was never called"
    for cols in seen_columns:
        leaked = [c for c in cols if c.startswith('fwd_return_')]
        assert not leaked, f"forward-return columns leaked into the model as features: {leaked}"


def test_signal_decay_by_horizon_matches_hand_computed_ic(monkeypatch):
    # 5 symbols, not 3 -- signal_decay_by_horizon requires >=5 valid pairs before
    # computing an IC for a week/horizon (avoids a meaningless correlation on too
    # few points), which a smaller synthetic panel would silently fall short of.
    dates = pd.to_datetime(['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28'])
    symbols = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
    # w0=train-only, w1=evaluated week, w2/w3 supply the 1wk/2wk-forward prices.
    w2_prices = {'AAA': 105.0, 'BBB': 110.0, 'CCC': 115.0, 'DDD': 120.0, 'EEE': 125.0}
    w3_prices = {'AAA': 125.0, 'BBB': 120.0, 'CCC': 115.0, 'DDD': 110.0, 'EEE': 105.0}
    prices = {s: {dates[0]: 100.0, dates[1]: 100.0, dates[2]: w2_prices[s], dates[3]: w3_prices[s]} for s in symbols}
    rows = [
        {'symbol': s, 'date': d, 'close': prices[s][d], 'target': 0.0}
        for s in symbols for d in dates
    ]
    panel = pd.DataFrame(rows)

    # Scripted predicted_return ranking at w1: AAA < BBB < CCC < DDD < EEE.
    # Horizon 1 (w1->w2) realized-return ranking: AAA(5%) < BBB(10%) < CCC(15%) < DDD(20%) < EEE(25%)
    #   -- same order as predicted -> IC = +1.
    # Horizon 2 (w1->w3) realized-return ranking: EEE(5%) < DDD(10%) < CCC(15%) < BBB(20%) < AAA(25%)
    #   -- exactly reversed -> IC = -1.
    scores_at_w1 = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4, 'EEE': 5}

    class _ScriptedPipeline:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X['symbol'].map(scores_at_w1).values

    monkeypatch.setattr(wf_module, 'build_full_ridge_pipeline', lambda: _ScriptedPipeline())

    result = wf_module.signal_decay_by_horizon(panel, [dates[1]], horizons=(1, 2))

    ic_h1 = result[result['horizon_weeks'] == 1]['ic'].iloc[0]
    ic_h2 = result[result['horizon_weeks'] == 2]['ic'].iloc[0]
    assert ic_h1 == pytest.approx(1.0)
    assert ic_h2 == pytest.approx(-1.0)


def test_apply_transaction_costs_charges_both_sides_of_turnover():
    daily_return = pd.Series([0.05, 0.03, -0.02])
    turnover = pd.Series([np.nan, 1.0, 0.5])

    net = wf_module.apply_transaction_costs(daily_return, turnover, cost_bps_per_side=10)

    # First week: NaN turnover (no prior portfolio to have traded from) costs nothing.
    assert net.iloc[0] == pytest.approx(0.05)
    # Full replacement at 10bps/side is a 20bps round-trip cost (both the exiting
    # sells and the entering buys).
    assert net.iloc[1] == pytest.approx(0.03 - 0.0020)
    # Half the book replaced -> half the round-trip cost.
    assert net.iloc[2] == pytest.approx(-0.02 - 0.0010)
