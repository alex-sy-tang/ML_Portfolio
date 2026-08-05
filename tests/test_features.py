import pandas as pd
import pytest

from ml_portfolio.features.engineering import calc_skewness, calc_volatility


def test_calc_volatility_does_not_leak_current_row():
    # calc_volatility must shift(1) before rolling: the window at row t can only see
    # strictly past weekly_log_return values, never weekly_log_return[t] itself (which
    # is == target[t] -- a direct leakage path if the shift is ever removed).
    n = 10
    base_returns = [0.001, -0.002, 0.0015, -0.001, 0.002, 0.0, 0.001, -0.0015, 0.002, -0.001]

    df_normal = pd.DataFrame({'symbol': ['AAA'] * n, 'weekly_log_return': base_returns})
    df_outlier = df_normal.copy()
    df_outlier.loc[5, 'weekly_log_return'] = 5.0  # extreme outlier at row 5 only

    result_normal = calc_volatility('vol_test', window=3, df=df_normal.copy())
    result_outlier = calc_volatility('vol_test', window=3, df=df_outlier.copy())

    # Row 5's own volatility must be identical whether or not row 5 itself is an
    # outlier -- if the rolling window ever included the current row, this would differ.
    assert result_normal['vol_test'].iloc[5] == pytest.approx(result_outlier['vol_test'].iloc[5])
    # The outlier must still show up starting the very next row (proves the two
    # results aren't just coincidentally equal everywhere).
    assert result_normal['vol_test'].iloc[6] != pytest.approx(result_outlier['vol_test'].iloc[6])


def test_calc_skewness_does_not_leak_current_row():
    n = 10
    base_returns = [0.001, -0.002, 0.0015, -0.001, 0.002, 0.0, 0.001, -0.0015, 0.002, -0.001]

    df_normal = pd.DataFrame({'symbol': ['AAA'] * n, 'weekly_log_return': base_returns})
    df_outlier = df_normal.copy()
    df_outlier.loc[5, 'weekly_log_return'] = 5.0

    result_normal = calc_skewness('skew_test', window=4, df=df_normal.copy())
    result_outlier = calc_skewness('skew_test', window=4, df=df_outlier.copy())

    assert result_normal['skew_test'].iloc[5] == pytest.approx(result_outlier['skew_test'].iloc[5])
    assert result_normal['skew_test'].iloc[6] != pytest.approx(result_outlier['skew_test'].iloc[6])


def test_calc_volatility_respects_symbol_grouping():
    df = pd.DataFrame({
        'symbol': ['AAA'] * 5 + ['BBB'] * 5,
        'weekly_log_return': [0.01, 0.02, 0.01, 100.0, 0.01] + [0.01, 0.02, 0.01, 0.02, 0.01],
    })

    result = calc_volatility('vol_test', window=3, df=df)

    # BBB's volatility must never be affected by AAA's outlier -- rolling must be
    # scoped per symbol via groupby, not across the concatenated panel.
    bbb_vol = result[result['symbol'] == 'BBB']['vol_test']
    assert (bbb_vol.dropna() < 1.0).all()
