"""Production entrypoint for the weekly ML_Portfolio pipeline.

Replaces `jupyter execute notebooks/ML_Portfolio_Management.ipynb` in the GitHub
Actions workflow. Mirrors the notebook's pipeline stages exactly (data fetch ->
feature engineering -> model fit -> weekly portfolio -> performance tracking ->
walk-forward gap backfill), but as a real script with logging and error handling
instead of notebook-cell-execution semantics that can silently discard output.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas_datareader import data as pdr
from scipy.stats import spearmanr

from ml_portfolio.config import DATA_DIR, PROJECT_ROOT
from ml_portfolio.data.io import get_last_week_data, load_data
from ml_portfolio.data.prices import get_historical_prices
from ml_portfolio.data.universe import get_sp500_universe
from ml_portfolio.features.engineering import (
    calc_amihud_illiquidity,
    calc_ff_loadings,
    calc_moving_avgs,
    calc_rolling_beta,
    calc_skewnesses,
    calc_volatilties,
    calculate_bbs,
    calculate_momentums,
    calculate_rsis,
    create_panel_dataset,
    filter_data,
    remove_columns,
    save_processed_data,
)
from ml_portfolio.features.target import calculate_weekly_returns, create_target_variable
from ml_portfolio.models.pipeline import build_full_ridge_pipeline
from ml_portfolio.models.train import save_model, time_aware_split
from ml_portfolio.portfolio.construction import create_weekly_stock_portfolio
from ml_portfolio.portfolio.tracking import (
    calculate_portfolio_metrics,
    historical_performance,
    update_stock_portfolio,
)
from ml_portfolio.backtest.walk_forward import backfill_backtest_gap

log = logging.getLogger("run_weekly_pipeline")

FF_FACTOR_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
VOLATILITY_WINDOWS = {'vol_1M': 4, 'vol_6M': 26, 'vol_12M': 52}
SKEWNESS_WINDOWS = {'skew_1M': 4, 'skew_6M': 26, 'skew_12M': 52}


def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    log.info("Fetching S&P 500 universe from Wikipedia")
    df_holdings = get_sp500_universe()
    holdings_list = df_holdings['symbol'].to_list()
    log.info("%d S&P 500 symbols", len(holdings_list))

    log.info("Fetching historical prices for %d symbols (%s to %s)", len(holdings_list), start_date, end_date)
    df = get_historical_prices(holdings_list, start_date, end_date)

    fetched_symbols = df["symbol"].unique()
    df_holdings = df_holdings[df_holdings["symbol"].isin(fetched_symbols)].reset_index(drop=True)
    df_holdings.to_csv(DATA_DIR / "raw" / "holdings.csv", index=False)

    log.info("Fetching SPY benchmark series")
    get_historical_prices(['SPY'], start_date, end_date, save_path=DATA_DIR / "raw" / "spy_price.csv")

    return df


def engineer_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    log.info("Computing target variable and panel dataset")
    df = calculate_weekly_returns(df)
    df = create_target_variable(df)
    df = create_panel_dataset(df)
    df = remove_columns(["open", "high", "low", "change", "changePercent", "weekly_return"], df)

    log.info("Computing factor features")
    df = calc_moving_avgs([200, 100, 50], df)
    df = calculate_rsis([3, 9, 14], df)
    df = calculate_bbs(["hband", "lband"], df)
    df = calculate_momentums([12, 6, 1], df)

    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    df['dollar_volume'] = df['close'] * df['volume']
    df = calc_amihud_illiquidity(21, df)
    df = calc_rolling_beta(60, df)

    log.info("Fetching Fama-French factors and computing rolling loadings")
    ff_raw = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start_date, end_date)[0]
    ff_factors = (ff_raw / 100).reset_index()  # library reports factors in percent
    ff_factors.columns = ['date'] + list(ff_factors.columns[1:])
    ff_factors['date'] = pd.to_datetime(ff_factors['date'])
    df = pd.merge(df, ff_factors[['date'] + FF_FACTOR_NAMES], on='date', how='left')
    df = calc_ff_loadings(60, df, FF_FACTOR_NAMES)
    df = df.drop(columns=['daily_return'] + FF_FACTOR_NAMES)

    log.info("Applying Wed/Thu weekly filter")
    df = filter_data(df)
    df = calc_volatilties(VOLATILITY_WINDOWS, df)
    df = calc_skewnesses(SKEWNESS_WINDOWS, df)

    log.info("Computing sector-relative factors")
    df_sectors = load_data(DATA_DIR / "raw" / "holdings.csv")[['symbol', 'gics_sector']]
    df = pd.merge(df, df_sectors, on='symbol', how='left')
    df['RSI_14_sector_relative'] = df['RSI_14'] - df.groupby(['date', 'gics_sector'])['RSI_14'].transform('mean')
    df['momentum_6M_sector_relative'] = df['momentum_6M'] - df.groupby(['date', 'gics_sector'])['momentum_6M'].transform('mean')
    df = df.drop(columns=['gics_sector'])

    df = df.dropna(ignore_index=True)
    save_processed_data(df, DATA_DIR)
    log.info("Processed panel saved: %d rows, %d columns", *df.shape)
    return df


def train_model(df: pd.DataFrame):
    log.info("Fitting Ridge pipeline on time-aware train split")
    X_train, X_test, y_train, y_test = time_aware_split(df)
    ridge_model = build_full_ridge_pipeline().fit(X_train, y_train)
    save_model(ridge_model)

    r2 = ridge_model.score(X_test, y_test)
    y_pred_test = ridge_model.predict(X_test)
    ic, ic_pvalue = spearmanr(y_pred_test, y_test)
    log.info("Held-out test set: R2=%.4f  IC=%.4f (p=%.4f)", r2, ic, ic_pvalue)
    return ridge_model


def build_weekly_portfolio(ridge_model) -> pd.DataFrame:
    processed_path = DATA_DIR / 'processed' / 'processed_historical_price.csv'
    df_last_week = get_last_week_data(processed_path).reset_index(drop=True)
    X_last_week = df_last_week.drop(columns=['target'])

    y_pred = ridge_model.predict(X_last_week)
    df_pred = pd.DataFrame(y_pred, columns=['predicted_return'])

    df_stock_portfolio = create_weekly_stock_portfolio(df_last_week['symbol'], df_pred)
    log.info("Selected %d stocks for week of %s", len(df_stock_portfolio), df_last_week['date'].iloc[0])

    df_historical_price = load_data(DATA_DIR / 'raw' / 'historical_price.csv')
    df_weekly_portfolio = update_stock_portfolio(
        df_stock_portfolio, df_historical_price, df_last_week['date'].iloc[0]
    )
    return df_weekly_portfolio, df_last_week


def track_performance(df_weekly_portfolio: pd.DataFrame, df_last_week: pd.DataFrame) -> pd.DataFrame:
    df_weekly_perf = calculate_portfolio_metrics(df_weekly_portfolio)
    prev_date = df_last_week['date'].iloc[0]

    hist_perf_file_path = DATA_DIR / 'processed' / 'historical_performance.csv'
    df_hist_perf = historical_performance(hist_perf_file_path, df_weekly_perf, prev_date)
    log.info("historical_performance.csv now has %d rows", len(df_hist_perf))
    return df_hist_perf


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(PROJECT_ROOT / '.env')

    start_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        df = fetch_data(start_date, end_date)
        df = engineer_features(df, start_date, end_date)
        ridge_model = train_model(df)
        df_weekly_portfolio, df_last_week = build_weekly_portfolio(ridge_model)
        track_performance(df_weekly_portfolio, df_last_week)
        log.info("Backfilling any historical_performance.csv gap via walk-forward backtest")
        backfill_backtest_gap(DATA_DIR / 'processed' / 'processed_historical_price.csv')
    except Exception:
        log.exception("Weekly pipeline failed")
        raise

    log.info("Weekly pipeline completed successfully")


if __name__ == "__main__":
    main()
