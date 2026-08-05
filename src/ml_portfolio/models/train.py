import joblib
import pandas as pd

from ml_portfolio.config import MODELS_DIR


def create_variables(df: pd.DataFrame) -> list:
    return [df.drop(columns = ["target"]), df["target"]]


def time_aware_split(df):
    symbols = df["symbol"].unique().tolist()

    X_train, X_test = pd.DataFrame(), pd.DataFrame()
    y_train, y_test = pd.Series(dtype='float64'), pd.Series(dtype='float64')
    for symbol in symbols:
        symbol_df = df[df["symbol"] == symbol]
        cut = int(len(symbol_df) * 0.8)
        Xs, ys = create_variables(symbol_df)
        Xs_train, Xs_test = Xs[:cut], Xs[cut:]
        ys_train, ys_test = ys[:cut], ys[cut:]

        X_train = pd.concat([X_train, Xs_train])
        X_test = pd.concat([X_test, Xs_test])
        y_train = pd.concat([y_train, ys_train])
        y_test = pd.concat([y_test, ys_test])

    return X_train, X_test, y_train, y_test


def save_model(model, name: str = "ridge_regression_model.sav") -> None:
    # One consistent artifact location, independent of the caller's cwd -- the
    # notebook previously saved "ridge_regression_model.sav" relative to wherever
    # the process happened to be run from (repo root vs. notebooks/), with no
    # versioning tied to training date/data.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / name)


def load_model(name: str = "ridge_regression_model.sav"):
    return joblib.load(MODELS_DIR / name)
