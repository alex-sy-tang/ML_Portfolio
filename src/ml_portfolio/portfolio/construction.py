import pandas as pd

from ml_portfolio.config import DATA_DIR


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
