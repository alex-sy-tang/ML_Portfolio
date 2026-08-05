import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_portfolio.models.pipeline import build_ridge_pipeline


def univariate_ic(df: pd.DataFrame, factor_col: str, target_col: str = 'target'):
    valid = df[[factor_col, target_col]].dropna()
    return spearmanr(valid[factor_col], valid[target_col])


def redundancy_check(df: pd.DataFrame, factor_col: str, other_cols: list, threshold: float = 0.7):
    corrs = df[other_cols + [factor_col]].corr()[factor_col].drop(factor_col)
    return corrs[corrs.abs() >= threshold]


def ablation_test(X_train, X_test, y_train, y_test, feature_cols: list) -> dict:
    cols_needed = ['symbol', 'date', 'weekly_log_return', 'close'] + [c for c in feature_cols if c != 'close']
    pipeline = build_ridge_pipeline(feature_cols)
    pipeline.fit(X_train[cols_needed], y_train)

    preds = pipeline.predict(X_test[cols_needed])
    ic, ic_pvalue = spearmanr(preds, y_test)
    r2 = pipeline.score(X_test[cols_needed], y_test)

    tmp = pd.DataFrame({'y_pred': preds, 'y_test': y_test.values})
    tmp['decile'] = pd.qcut(tmp['y_pred'], 10, labels=False, duplicates='drop')
    decile_returns = tmp.groupby('decile')['y_test'].mean()
    decile_spread = decile_returns.iloc[-1] - decile_returns.iloc[0]

    return {'ic': ic, 'ic_pvalue': ic_pvalue, 'r2': r2, 'decile_spread': decile_spread}


def sub_period_ic(df: pd.DataFrame, factor_col: str, target_col: str = 'target', date_col: str = 'date', n_splits: int = 2):
    df_sorted = df.sort_values(date_col)
    chunks = np.array_split(df_sorted, n_splits)
    return [univariate_ic(chunk, factor_col, target_col)[0] for chunk in chunks]


def carve_validation_split(X_train: pd.DataFrame, y_train: pd.Series, val_fraction: float = 0.2):
    symbols = X_train['symbol'].unique().tolist()
    X_train_final, X_val = pd.DataFrame(), pd.DataFrame()
    y_train_final, y_val = pd.Series(dtype='float64'), pd.Series(dtype='float64')

    for symbol in symbols:
        mask = X_train['symbol'] == symbol
        Xs, ys = X_train[mask], y_train[mask]
        cut = int(len(Xs) * (1 - val_fraction))

        X_train_final = pd.concat([X_train_final, Xs.iloc[:cut]])
        X_val = pd.concat([X_val, Xs.iloc[cut:]])
        y_train_final = pd.concat([y_train_final, ys.iloc[:cut]])
        y_val = pd.concat([y_val, ys.iloc[cut:]])

    return X_train_final, X_val, y_train_final, y_val


def evaluate_predictions(preds, y_true) -> dict:
    ic, ic_pvalue = spearmanr(preds, y_true)
    # A model that zeroes out every feature (over-regularized) predicts the same
    # constant value for every row — IC is undefined and qcut can't form deciles.
    # Report that plainly (NaN) instead of crashing.
    if pd.isna(ic) or np.std(preds) == 0:
        return {'ic': np.nan, 'ic_pvalue': np.nan, 'decile_spread': np.nan}
    tmp = pd.DataFrame({'y_pred': preds, 'y_true': y_true.values})
    tmp['decile'] = pd.qcut(tmp['y_pred'], 10, labels=False, duplicates='drop')
    decile_returns = tmp.groupby('decile')['y_true'].mean()
    if len(decile_returns) < 2:
        return {'ic': ic, 'ic_pvalue': ic_pvalue, 'decile_spread': np.nan}
    decile_spread = decile_returns.iloc[-1] - decile_returns.iloc[0]
    return {'ic': ic, 'ic_pvalue': ic_pvalue, 'decile_spread': decile_spread}


def sub_period_ic_predictions(dates, preds, y_true, n_splits: int = 2):
    tmp = pd.DataFrame({'date': dates.values, 'y_pred': preds, 'y_true': y_true.values}).sort_values('date')
    chunks = np.array_split(tmp, n_splits)
    return [spearmanr(chunk['y_pred'], chunk['y_true'])[0] for chunk in chunks]


def _floored_ic(preds, y_true):
    ic, _ = spearmanr(preds, y_true)
    return 0.0 if pd.isna(ic) else max(ic, 0.0)


def standardize(preds, stats):
    mean, std = stats
    return (preds - mean) / std if std > 0 else preds - mean
