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
    df['year_week'] = df['date'].dt.strftime('%Y-%U')
    last_year_week = (datetime.now() - timedelta(weeks = 1)).strftime('%Y-%U')

    df_last_week = df[df['year_week'] == last_year_week]
    return df_last_week.drop(columns=['year_week'])

def create_weekly_stock_portfolio(
    symbols: pd.Series,
    predicted_returns: pd.DataFrame,
    top_quantile: float = 0.2,
    long_only: bool = True,
) -> pd.DataFrame:
    if not long_only:
        raise NotImplementedError("Short-side portfolio construction is not implemented yet.")

    df_stock_portfolio = pd.concat(
        [symbols.reset_index(drop=True), predicted_returns.reset_index(drop=True)], axis=1
    )
    df_stock_portfolio = df_stock_portfolio.sort_values(by='predicted_return', ascending=False)

    cutoff = max(1, int(len(df_stock_portfolio) * top_quantile))
    df_stock_portfolio = df_stock_portfolio.head(cutoff)

    df_stock_portfolio['weight'] = 1 / len(df_stock_portfolio)
    df_stock_portfolio.to_csv(DATA_DIR/'processed'/'stock_portfolio.csv', index = False)
    return df_stock_portfolio.reset_index(drop = True)

def update_stock_portfolio(df_stock_portfolio: pd.DataFrame, df_historical_price: pd.DataFrame) -> pd.DataFrame:
    df_today_price = df_historical_price[df_historical_price['date'] == datetime.now().strftime('%Y-%m-%d')]
    if 'close' not in df_stock_portfolio.columns: 
        df_stock_portfolio = pd.merge(df_stock_portfolio, df_today_price, on = 'symbol', how = 'left')
    return df_stock_portfolio.dropna()

