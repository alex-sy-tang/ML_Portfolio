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
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from ml_portfolio.config import PROJECT_ROOT, DATA_DIR

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

def calculate_risk_metrics(hist_perf: pd.DataFrame, spy_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    hist_perf = hist_perf.copy()
    hist_perf['date'] = pd.to_datetime(hist_perf['date'])
    hist_perf = hist_perf.sort_values('date').reset_index(drop=True)

    # Most rows now represent a full week's return, not a single trading day: the
    # walk-forward backtest rows (see that notebook section) are the portfolio's
    # realized weekly return, and weekly is also the design-intended live cadence
    # going forward (wed_thurs_selector, weekly rebalance). Annualizing with the
    # daily convention (252) — correct back when every row genuinely was one day's
    # return — overstated Sharpe/Sortino/Volatility by ~2.2x (sqrt(252/52)) once
    # the weekly backtest rows were added. A handful of legacy rows from early
    # manual testing really are single-day returns, so 52 is an approximation for
    # those specifically, but the right choice for the dataset's now-dominant,
    # intended cadence.
    returns = hist_perf['daily_return']
    periods_per_year = 52

    start_value, end_value = hist_perf['total_value'].iloc[0], hist_perf['total_value'].iloc[-1]
    years = (hist_perf['date'].iloc[-1] - hist_perf['date'].iloc[0]).days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(periods_per_year)

    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

    downside_std = returns[returns < 0].std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else np.nan

    running_max = hist_perf['total_value'].cummax()
    max_drawdown = (hist_perf['total_value'] / running_max - 1).min()

    # Beta pairs each row against SPY's own return over the same ~1-week window (5
    # trading days) ending on that date, rather than SPY's single-day return —
    # same weekly-cadence reasoning as the annualization above, and the same
    # approximation for the handful of single-day legacy rows.
    spy_df = spy_df.copy()
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df['spy_return'] = spy_df['close'].pct_change(periods=5)
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

