import numpy as np
import pandas as pd


def calculate_weekly_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['weekly_return'] = df.groupby('symbol')['close'].transform(
        lambda x: (x.shift(-5) - x) / x
    )
    df = df.dropna(subset = ["weekly_return"])

    df["weekly_log_return"] = np.log(1 + df["weekly_return"])

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['target'] = df['weekly_log_return']
    return df
