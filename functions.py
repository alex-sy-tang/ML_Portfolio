# Import libraries
try: 
    import pandas as pd
    import ta
    import numpy as np
    import requests
    import sys, os
    from dotenv import load_dotenv
    from pathlib import Path
    from datetime import datetime, timedelta
    
    print("Successfully Imported all the libraries")

except Exception as e: 
    print(f"Import Error: {e}")
    raise

# Variables
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT/'data'

# Load data from csv file
def load_data(filename): 
    df = pd.read_csv(filename)
    return df

# Load the historical price of the selected symbol
def load_symbol_price(filename, symbol): 
    df = load_data(filename)
    symbol_df = df[(df["symbol"] == symbol)]

    return symbol_df

# Get the price data for the most recent past week
def get_last_week_data(filename):
    df = load_data(filename)
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    return df[df['date'] == latest_date]

def create_weekly_stock_portfolio(
    symbols: pd.Series,
    predicted_returns: pd.DataFrame,
    top_n: int = 20,
    long_only: bool = True,
) -> pd.DataFrame:
    if not long_only:
        raise NotImplementedError("Short-side portfolio construction is not implemented yet.")

    df_stock_portfolio = pd.concat(
        [symbols.reset_index(drop=True), predicted_returns.reset_index(drop=True)], axis=1
    )
    df_stock_portfolio = df_stock_portfolio.sort_values(by='predicted_return', ascending=False)

    # Fixed count rather than a quantile of the universe: a percentage-based cutoff
    # silently changed size when the universe grew from 17 DJIA stocks to ~500 S&P
    # 500 stocks (top 20% went from ~3 stocks to ~100), which is neither a legible
    # pie-chart slice count nor a very selective long book.
    cutoff = max(1, min(top_n, len(df_stock_portfolio)))
    df_stock_portfolio = df_stock_portfolio.head(cutoff)

    df_stock_portfolio['weight'] = 1 / len(df_stock_portfolio)
    df_stock_portfolio.to_csv(DATA_DIR/'processed'/'stock_portfolio.csv', index = False)
    return df_stock_portfolio.reset_index(drop = True)

def update_stock_portfolio(df_stock_portfolio: pd.DataFrame, df_historical_price: pd.DataFrame) -> pd.DataFrame:
    # Anchor to the data's own latest date, not wall-clock datetime.now() — same
    # staleness fix already applied to get_last_week_data() above. historical_price.csv
    # isn't guaranteed to have been refreshed today, so "today" can match zero rows.
    df_today_price = df_historical_price[df_historical_price['date'] == df_historical_price['date'].max()]
    if 'close' not in df_stock_portfolio.columns:
        df_stock_portfolio = pd.merge(df_stock_portfolio, df_today_price, on = 'symbol', how = 'left')
    return df_stock_portfolio.dropna()

def calculate_risk_metrics(hist_perf: pd.DataFrame, spy_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    hist_perf = hist_perf.copy()
    hist_perf['date'] = pd.to_datetime(hist_perf['date'])
    hist_perf = hist_perf.sort_values('date').reset_index(drop=True)

    # Each daily_return value is a single calendar day's return (see
    # calculate_portfolio_metrics()/historical_performance()) sampled once per
    # pipeline run, not one return per elapsed day — so it's annualized with the
    # standard 252 trading-day convention regardless of gaps between runs.
    returns = hist_perf['daily_return']
    trading_days_per_year = 252

    start_value, end_value = hist_perf['total_value'].iloc[0], hist_perf['total_value'].iloc[-1]
    years = (hist_perf['date'].iloc[-1] - hist_perf['date'].iloc[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(trading_days_per_year)

    period_rf = risk_free_rate / trading_days_per_year
    excess_returns = returns - period_rf
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(trading_days_per_year) if returns.std() > 0 else np.nan

    downside_std = returns[returns < 0].std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days_per_year) if downside_std > 0 else np.nan

    running_max = hist_perf['total_value'].cummax()
    max_drawdown = (hist_perf['total_value'] / running_max - 1).min()

    # Beta pairs each daily_return against SPY's own same-day return (SPY's full
    # daily series, inner-joined on date) rather than SPY's return since the
    # previous historical_performance.csv row — that row can be months earlier
    # whenever the pipeline skipped a run, which would otherwise pair a one-day
    # portfolio move against a multi-month SPY move.
    spy_df = spy_df.copy()
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df['spy_return'] = spy_df['close'].pct_change()
    merged = pd.merge(hist_perf[['date', 'daily_return']], spy_df[['date', 'spy_return']], on='date', how='inner')
    beta = (merged['daily_return'].cov(merged['spy_return']) / merged['spy_return'].var()
            if len(merged) >= 2 and merged['spy_return'].var() > 0 else np.nan)

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'volatility': ann_vol,
        'max_drawdown': max_drawdown,
        'beta': beta,
    }

