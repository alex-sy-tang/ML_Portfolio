import pandas as pd
import requests

from ml_portfolio.config import DATA_DIR


def get_sp500_universe() -> pd.DataFrame:
    # Wikipedia rejects requests with no User-Agent (403), so fetch the HTML ourselves
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ML_Portfolio research script)"}
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
    tables = pd.read_html(response.text)
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["symbol", "security", "gics_sector", "gics_sub_industry"]
    # Wikipedia uses dots for share classes (BRK.B); yfinance/stooq expect dashes (BRK-B)
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df.to_csv(DATA_DIR/"raw"/"holdings.csv", index = False)
    return df
