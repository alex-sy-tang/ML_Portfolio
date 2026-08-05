from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from ml_portfolio.config import DATA_DIR


def get_price_history_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = pdr.DataReader(ticker, "stooq", start=start, end=end)
        df = df.reset_index().sort_values("Date")
        df["symbol"] = ticker
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def get_historical_prices(tickers: list, start: str, end: str, save_path: Path = None) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True, progress=False)

    frames = []
    failed = []
    for ticker in tickers:
        try:
            # group_by='ticker' always returns MultiIndex columns keyed by ticker,
            # even for a single-ticker list, so raw[ticker] is the correct selector
            # regardless of how many tickers were requested.
            sub = raw[ticker].dropna(how='all')
            if sub.empty:
                raise ValueError("no data returned")
        except Exception:
            failed.append(ticker)
            continue
        sub = sub.reset_index()
        sub['symbol'] = ticker
        sub = sub.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        frames.append(sub[['symbol','date','open','high','low','close','volume']])

    if failed:
        print(f"yfinance missing {len(failed)} symbols, retrying via stooq: {failed}")
        for ticker in failed:
            sub = get_price_history_stooq(ticker, start, end)
            if not sub.empty:
                frames.append(sub)
            else:
                print(f"{ticker}: no data from yfinance or stooq")

    df = pd.concat(frames, axis = 0, ignore_index = True)
    df['change'] = df['close'] - df['open']
    df['changePercent'] = (df['change'] / df['open']) * 100
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3  # typical-price approximation; no true intraday VWAP available for free
    df.to_csv(save_path or DATA_DIR/"raw"/"historical_price.csv", index = False)
    return df
