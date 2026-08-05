import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = np.log(X['close'] / X[feature])
        return X


def build_ridge_pipeline(feature_cols: list) -> Pipeline:
    log_candidates = ['MA_200', 'MA_100', 'MA_50', 'hband', 'lband']
    log_transform_features = [c for c in log_candidates if c in feature_cols]
    # 'close' is structurally required by LogTransformer (the normalizing denominator
    # for MA/hband/lband), so it always flows into the pipeline; if it isn't one of the
    # features actually being tested, drop it explicitly after the log-transform step.
    cols_to_drop = ['symbol', 'date', 'weekly_log_return']
    if 'close' not in feature_cols:
        cols_to_drop = cols_to_drop + ['close']
    col_dropper = ColumnTransformer(
        transformers=[('drop_cols', 'drop', cols_to_drop)],
        remainder='passthrough'
    )
    return Pipeline(steps=[
        ('log_transformers', LogTransformer(log_transform_features)),
        ('col_dropper', col_dropper),
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])


def build_model_pipeline(feature_cols: list, regressor) -> Pipeline:
    log_candidates = ['MA_200', 'MA_100', 'MA_50', 'hband', 'lband']
    log_transform_features = [c for c in log_candidates if c in feature_cols]
    cols_to_drop = ['symbol', 'date', 'weekly_log_return']
    if 'close' not in feature_cols:
        cols_to_drop = cols_to_drop + ['close']
    col_dropper = ColumnTransformer(
        transformers=[('drop_cols', 'drop', cols_to_drop)],
        remainder='passthrough'
    )
    return Pipeline(steps=[
        ('log_transformers', LogTransformer(log_transform_features)),
        ('col_dropper', col_dropper),
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])


def build_full_ridge_pipeline() -> Pipeline:
    log_transform_features = ['MA_200', 'MA_100', 'MA_50', 'hband', 'lband']
    cols_to_drop = ['symbol', 'date', 'weekly_log_return']
    col_dropper = ColumnTransformer(
        transformers=[('drop_cols', 'drop', cols_to_drop)],
        remainder='passthrough'
    )
    return Pipeline(steps=[
        ('log_transformers', LogTransformer(log_transform_features)),
        ('col_dropper', col_dropper),
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])


def fama_french_standalone_signal(X: pd.DataFrame, mean_factor_returns: pd.Series) -> np.ndarray:
    loading_cols = [f"loading_{f.replace('-', '_')}" for f in mean_factor_returns.index]
    return X[loading_cols].values @ mean_factor_returns.values
