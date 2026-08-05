import numpy as np
import pandas as pd

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
