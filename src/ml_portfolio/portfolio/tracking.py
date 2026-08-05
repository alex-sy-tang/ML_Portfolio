from datetime import datetime

import pandas as pd

from ml_portfolio.config import DATA_DIR


def update_stock_portfolio(df_stock_portfolio: pd.DataFrame, df_historical_price: pd.DataFrame, week_date) -> pd.DataFrame:
    df_historical_price = df_historical_price.copy()

    df_historical_price['date'] = pd.to_datetime(df_historical_price['date'])
    df_historical_price['week_of_year'] = df_historical_price['date'].dt.strftime('%Y-%U')
    # Price the portfolio using the SAME week its stock selection came from, not
    # whichever week is latest in the raw price panel. Those can silently diverge
    # once a feature source (e.g. Fama-French) lags behind the raw price fetch —
    # using the panel's own latest week would then mark a stale, unrebalanced
    # selection to market with a much later week's prices and misreport it as a
    # fresh decision (exactly what happened: the FF-lag cap left df_last_week stuck
    # at 2026-05-27 while historical_price.csv kept advancing, so the "current"
    # portfolio was being priced two months after it was actually selected).
    target_week = pd.Timestamp(week_date).strftime('%Y-%U')
    df_weekly_price = df_historical_price[df_historical_price['week_of_year'] == target_week]

    if 'close' not in df_stock_portfolio.columns:
        df_weekly_portfolio = pd.merge(df_stock_portfolio, df_weekly_price, on = 'symbol', how = 'left')
    else:
        df_weekly_portfolio = df_stock_portfolio
    df_weekly_portfolio['week_of_day'] = df_weekly_portfolio['date'].dt.weekday
    df_weekly_portfolio.to_csv(DATA_DIR/'processed'/'weekly_portfolio.csv')

    return df_weekly_portfolio.dropna()


def calculate_portfolio_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    s = portfolio.groupby("date")['close'].sum()
    df = pd.DataFrame(s, columns=['close']).reset_index()
    df['daily_return'] = df['close'].pct_change()
    df.loc[0, 'daily_return'] = 0
    df['cumulative_return'] = (df['close'] - df['close'][0]) / df['close'][0]

    df = df.rename(columns = {'close':'total_value'})

    return df


def historical_performance(hist_perf_file_path, df_weekly_perf, prev_date: datetime, start_date = "2026-01-12", init_value = 100000):
    if hist_perf_file_path.is_file():
        df = pd.read_csv(hist_perf_file_path)
    else:
        df = pd.DataFrame()
    df_daily_perf = df_weekly_perf[df_weekly_perf['date'] == pd.Timestamp(prev_date)]

    if len(df) == 0:
        df = pd.concat([df, df_daily_perf], axis = 0)
    else:
        # Compare as actual dates, not raw strings — str(prev_date) vs. the CSV's
        # plain date string silently stopped matching once prev_date became a
        # Timestamp (which stringifies with a time component, e.g. "2026-05-27
        # 00:00:00") instead of a bare date object, which would have kept
        # re-appending a "new" row for a date that's already there.
        if not (pd.to_datetime(df['date']) == pd.Timestamp(prev_date)).any():
            df = pd.concat([df, df_daily_perf], axis = 0)
        else:
            return df

    df = df.reset_index(drop = True)
    df['date'] = pd.to_datetime(df['date'])
    df.loc[0,'total_value'] = init_value
    # Rows genuinely produced by this live pipeline are tagged 'live', distinct from
    # any 'backtest'-tagged rows already in the file (see the Walk-Forward Backtest
    # section) -- newly appended rows have no source yet, so they default to 'live'.
    if 'source' not in df.columns:
        df['source'] = pd.NA
    df['source'] = df['source'].fillna('live')

    #Calculate the total value
    df['temp_total_value'] = df['total_value'].shift(1)
    df.loc[1:,'total_value'] = df['temp_total_value'] * (1 + df['daily_return'])

    #Calculate the cumulative return iteratively
    df['temp_return'] = df['cumulative_return'].shift(1)
    df.loc[1:,'cumulative_return'] = (1 + df['daily_return'])*(1 + df['temp_return']) - 1

    df = df.drop(columns = ['temp_return', 'temp_total_value'])

    df.to_csv(hist_perf_file_path, index = False)

    return df
