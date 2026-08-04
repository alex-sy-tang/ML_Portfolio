import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from functions import *


processed_data_filename = DATA_DIR/'processed'/'processed_historical_price.csv'
raw_data_filename = DATA_DIR/'raw'/'historical_price.csv'





st.title('Stock Portfolio Dashboard')
st.header('Historical Price')




holdings_df = load_data(DATA_DIR/"raw"/"holdings.csv")
symbols_list = holdings_df["symbol"].tolist()
selected_symbol = st.selectbox("Symbols from S&P 500", symbols_list)


historical_price_df = load_data(raw_data_filename)

earliest_date_str = historical_price_df.iloc[-1, 1]
format_string = "%Y-%m-%d"
earliest_date_obj = datetime.strptime(earliest_date_str, format_string)

symbol_df = load_symbol_price(raw_data_filename, selected_symbol)

dates = symbol_df["date"]
open, high, low, close = symbol_df["open"], symbol_df["high"], symbol_df["low"], symbol_df["close"]


candlestick = go.Candlestick(x=dates,
                open = open,
                high = high,
                low = low,
                close = close)

layout = go.Layout(title=f'Candlestick Chart for {selected_symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Price'))

fig = go.Figure(data=[candlestick], layout=layout)
st.plotly_chart(fig)

st.subheader('Raw Data')
st.dataframe(symbol_df)


#Stock Portfolio:
st.divider()
st.header('📊 My Portfolio')

my_portfolio = load_data(DATA_DIR/"processed"/"weekly_portfolio.csv")
historical_performance = load_data(DATA_DIR/'processed'/'historical_performance.csv')
historical_performance['date'] = pd.to_datetime(historical_performance['date'])

spy_df = load_data(DATA_DIR/"raw"/"spy_price.csv")
spy_df['date'] = pd.to_datetime(spy_df['date'])
risk_metrics = calculate_risk_metrics(historical_performance, spy_df)

# SPY cumulative return re-anchored to 0 at the portfolio's own start date, so both
# lines are comparable on the same chart even though SPY's own history goes back further.
spy_window = spy_df[(spy_df['date'] >= historical_performance['date'].min()) &
                     (spy_df['date'] <= historical_performance['date'].max())].copy()
spy_window['cumulative_return'] = spy_window['close'] / spy_window['close'].iloc[0] - 1

equity_fig = go.Figure()

# Split the portfolio line by source (live vs. backtest) so a simulated stretch is
# never visually indistinguishable from genuinely tracked performance. Each segment
# starts one row early (where possible) so the line stays visually continuous across
# the live/backtest boundary instead of showing a gap.
sources = historical_performance['source'].tolist()
seg_bounds = [0] + [i for i in range(1, len(sources)) if sources[i] != sources[i - 1]] + [len(sources)]
first_of_kind = {'Portfolio (live)': True, 'Portfolio (backtest)': True}
for start, end in zip(seg_bounds[:-1], seg_bounds[1:]):
    seg = historical_performance.iloc[max(0, start - 1):end]
    label = 'Portfolio (backtest)' if sources[start] == 'backtest' else 'Portfolio (live)'
    equity_fig.add_trace(go.Scatter(
        x=seg['date'], y=seg['cumulative_return'], mode='lines+markers',
        name=label, legendgroup=label, showlegend=first_of_kind.get(label, True),
        line=dict(color='#1f77b4' if label == 'Portfolio (live)' else '#ff7f0e',
                   dash='solid' if label == 'Portfolio (live)' else 'dash'),
    ))
    first_of_kind[label] = False

equity_fig.add_trace(go.Scatter(x=spy_window['date'], y=spy_window['cumulative_return'],
                                 mode='lines', name='SPY', line=dict(dash='dot')))
equity_fig.update_layout(title='Portfolio vs. SPY — Cumulative Return', xaxis_title='Date', yaxis_title='Cumulative Return')
st.plotly_chart(equity_fig, use_container_width=True)
if (historical_performance['source'] == 'backtest').any():
    st.caption("Dashed orange segment is a walk-forward backtest simulation filling a "
               "tracking gap (see docs), not genuinely tracked performance.")


m1, m2, m3 = st.columns(3)
total_balance = round(historical_performance.loc[historical_performance.index[-1],"total_value"],2)
invested_capital = round(historical_performance.loc[0,"total_value"])
total_profit_loss = round(total_balance - invested_capital, 2)
delta = str(round((total_balance - invested_capital) / invested_capital, 2))

m1.metric("💰 Total Balance", total_balance)
m2.metric("📈 Total Profit/Loss", total_profit_loss, delta= f'{delta} since 2026-01-12')
m3.metric("🏦 Invested Capital", invested_capital)

r1, r2, r3 = st.columns(3)
r1.metric("📐 Sharpe", f"{risk_metrics['sharpe']:.2f}" if pd.notna(risk_metrics['sharpe']) else "N/A")
r2.metric("📉 Sortino", f"{risk_metrics['sortino']:.2f}" if pd.notna(risk_metrics['sortino']) else "N/A")
r3.metric("🌱 CAGR", f"{risk_metrics['cagr']*100:.2f}%" if pd.notna(risk_metrics['cagr']) else "N/A")

r4, r5, r6 = st.columns(3)
r4.metric("📊 Volatility", f"{risk_metrics['volatility']*100:.2f}%" if pd.notna(risk_metrics['volatility']) else "N/A")
r5.metric("🔻 Max Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%" if pd.notna(risk_metrics['max_drawdown']) else "N/A")
r6.metric("β Beta (vs SPY)", f"{risk_metrics['beta']:.2f}" if pd.notna(risk_metrics['beta']) else "N/A")
if (historical_performance['source'] == 'backtest').any():
    n_backtest = (historical_performance['source'] == 'backtest').sum()
    st.caption(f"{n_backtest} of {len(historical_performance)} weeks above are backtested, not live-tracked — "
               "these metrics mix a real ~8-week track record with a 19-week simulation.")

drawdown_fig = go.Figure()
running_max = historical_performance['total_value'].cummax()
drawdown = historical_performance['total_value'] / running_max - 1
drawdown_fig.add_trace(go.Scatter(x=historical_performance['date'], y=drawdown, mode='lines+markers',
                                   fill='tozeroy', name='Drawdown'))
drawdown_fig.update_layout(title='Portfolio Drawdown', xaxis_title='Date', yaxis_title='Drawdown')
st.plotly_chart(drawdown_fig, use_container_width=True)

st.divider()

# Real per-symbol weight, not an equal-weight approximation — matters once the
# portfolio construction stops being uniformly equal-weighted.
df_asset = my_portfolio.drop_duplicates(subset='symbol')[['symbol', 'weight']].rename(
    columns={'symbol': 'Ticker', 'weight': 'Allocation'}
)

fig = px.pie(
    df_asset, 
    values='Allocation', 
    names='Ticker', 
    title='Weekly Portfolio Allocation',
    hole=0.4, # Optional: makes it a donut chart like some modern designs
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig.update_layout(
    template="plotly_dark",
    legend_title="Stocks",
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# st.dataframe(my_portfolio)

# my_portfolio['symbol'].unique



