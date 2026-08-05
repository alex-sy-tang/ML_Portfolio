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
