import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def create_panel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by = "date")


def remove_columns(cols: list, df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns = cols, axis = 1)
    except Exception as e:
        print(e)
    finally:
        return df


def calc_moving_avg(window: int, df: pd.DataFrame) -> pd.DataFrame:
    col_name = f'MA_{str(window)}'
    df[col_name] = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window = window, min_periods = 1).mean()
    )

    return df


def calc_moving_avgs(windows: list, df: pd.DataFrame) -> pd.DataFrame:
    for window in windows:
        df = calc_moving_avg(window, df)
    return df


def calc_volatility(col_name: str, window: int, df: pd.DataFrame) -> pd.DataFrame:
    # shift(1) before rolling: the window must only see strictly past weekly_log_return
    # values. Without the shift, the window at row t includes weekly_log_return[t]
    # itself, which is now exactly target[t] — direct target leakage into the feature.
    df[col_name] = df.groupby('symbol')['weekly_log_return'].transform(
        lambda x: x.shift(1).rolling(window = window, min_periods = 1).std() * np.sqrt(window)
    )

    return df


def calc_volatilties(volatility_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    for col_name, window in volatility_dict.items():
        df = calc_volatility(col_name, window, df)
    return df


def calculate_rsi(window, group):
    rsi = RSIIndicator(close = group['close'], window = window).rsi()
    return rsi


def calculate_rsis(windows: list, df: pd.DataFrame) -> pd.DataFrame:
    for window in windows:
        col_name = f'RSI_{str(window)}'
        group = df.groupby('symbol', group_keys = False)
        df[col_name] = group.apply(lambda x: calculate_rsi(window, x))

    return df


def calculate_bb(group, window = 20, window_dev = 2):
    return BollingerBands(close = group['close'], window = window, window_dev = window_dev)


def calculate_bbs(bands: list, df:pd.DataFrame) -> pd.DataFrame:
    group = df.groupby('symbol', group_keys = False)
    for band in bands:
        if band == "hband":
            df[band] = group.apply(lambda x: calculate_bb(x).bollinger_hband())
        elif band == "lband":
            df[band] = group.apply(lambda x: calculate_bb(x).bollinger_lband())

    return df


def calculate_momentum(window_months: int, df: pd.DataFrame, trading_days :int = 21) -> pd.DataFrame:
    window_days = window_months * trading_days
    col_name = f"momentum_{str(window_months)}M"
    df[col_name] = df.groupby('symbol')['close'].pct_change(periods = window_days)

    return df


def calculate_momentums(windows: list, df:pd.DataFrame) -> pd.DataFrame:
    for window in windows:
        df = calculate_momentum(window, df)
    return df


def calc_amihud_illiquidity(window: int, df: pd.DataFrame) -> pd.DataFrame:
    illiquidity = df['daily_return'].abs() / df['dollar_volume']
    df['amihud_illiquidity'] = illiquidity.groupby(df['symbol']).transform(
        lambda x: x.rolling(window = window, min_periods = 1).mean()
    )
    return df


def calc_rolling_beta(window: int, df: pd.DataFrame) -> pd.DataFrame:
    # Equal-weight average daily return across the whole universe each day — a market
    # proxy built from data we already have, no extra fetch (e.g. no new SPY call).
    market_return = df.groupby('date')['daily_return'].transform('mean')

    def _beta(group):
        mkt = market_return.loc[group.index]
        cov = group['daily_return'].rolling(window, min_periods = window).cov(mkt)
        var = mkt.rolling(window, min_periods = window).var()
        return cov / var

    df[f'beta_{window}d'] = df.groupby('symbol', group_keys = False).apply(_beta)
    return df


def calc_ff_loadings(window: int, df: pd.DataFrame, factors: list) -> pd.DataFrame:
    loading_cols = {f: f"loading_{f.replace('-', '_')}" for f in factors}
    for col in loading_cols.values():
        df[col] = np.nan

    for symbol, group in df.groupby('symbol'):
        valid = group.dropna(subset=['daily_return'] + factors)
        if len(valid) < window:
            continue
        X = sm.add_constant(valid[factors])
        rolling_params = RollingOLS(valid['daily_return'], X, window=window, min_nobs=window).fit().params
        for f in factors:
            df.loc[valid.index, loading_cols[f]] = rolling_params[f]
    return df


def wed_thurs_selector(df, date_col='date', stock_col='symbol'):

    df['year_week'] = df[date_col].dt.strftime('%Y-%U')
    df['day_of_week'] = df[date_col].dt.dayofweek


    wed_thu = df[df['day_of_week'].isin([2, 3])].copy()


    filtered = wed_thu.groupby([stock_col, 'year_week']).first().reset_index()

    return filtered.drop(columns=['day_of_week', 'year_week'])


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = wed_thurs_selector(df)
    return df.dropna(ignore_index = True)


def calc_skewness(col_name: str, window: int, df: pd.DataFrame) -> pd.DataFrame:
    # Same shift(1) fix as calc_volatility, same reason: exclude the current row so
    # the window can't see weekly_log_return[t] (== target[t]).
    df[col_name] = df.groupby('symbol')['weekly_log_return'].transform(
        lambda x: x.shift(1).rolling(window = window, min_periods = 1).skew()
    )
    return df


def calc_skewnesses(skewness_dict: dict, df: pd.DataFrame) -> pd.DataFrame:
    for col_name, window in skewness_dict.items():
        df = calc_skewness(col_name, window, df)
    return df


def save_processed_data(df: pd.DataFrame, directory):
    df.to_csv(directory/"processed"/"processed_historical_price.csv", index = False)
