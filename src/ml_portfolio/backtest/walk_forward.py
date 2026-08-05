import numpy as np
import pandas as pd

from ml_portfolio.models.pipeline import build_full_ridge_pipeline


def walk_forward_backtest(panel: pd.DataFrame, weeks: list, top_n: int = 20) -> pd.DataFrame:
    results = []
    for w in weeks:
        train_df = panel[panel['date'] < w]
        predict_df = panel[panel['date'] == w]

        pipeline = build_full_ridge_pipeline()
        pipeline.fit(train_df.drop(columns=['target']), train_df['target'])
        y_pred = pipeline.predict(predict_df.drop(columns=['target']))

        picks = predict_df.assign(predicted_return=y_pred).sort_values('predicted_return', ascending=False).head(top_n)
        # picks['target'] is that week's already-realized weekly_log_return (known once
        # the week plays out) -- convert log return to simple return before averaging
        # across stocks, since only an asset's own log returns compound additively
        # across time, not across different assets at a single point in time.
        simple_returns = np.exp(picks['target']) - 1
        results.append({'date': w, 'daily_return': simple_returns.mean()})

    return pd.DataFrame(results)
