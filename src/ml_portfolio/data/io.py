import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def get_last_week_data(filename):
    df = load_data(filename)
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    return df[df['date'] == latest_date]
