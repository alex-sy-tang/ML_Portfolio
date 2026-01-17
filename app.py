import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from functions import *


PROJECT_ROOT = Path('/Users/shengyaotang/Desktop/Quant Project/ML_Portfolio_Management')
DATA_DIR = PROJECT_ROOT/'data'
processed_data_filename = DATA_DIR/'processed'/'processed_historical_price.csv'
raw_data_filename = DATA_DIR/'raw'/'historical_price.csv'





st.title('Stock Portfolio Dashboard')
st.header('Historical Price')




holdings_df = load_data(DATA_DIR/"raw"/"holdings.csv")
symbols_list = holdings_df["symbol"].tolist()
selected_symbol = st.selectbox("Symbols from DJIA", symbols_list)


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
st.line_chart(historical_performance, x = 'date', y = 'cumulative_return')


m1, m2, m3 = st.columns(3)
total_balance = round(historical_performance.loc[historical_performance.index[-1],"total_value"],2)
invested_capital = round(historical_performance.loc[0,"total_value"])
total_profit_loss = round(total_balance - invested_capital, 2)
delta = str(round((total_balance - invested_capital) / invested_capital, 2))

m1.metric("💰 Total Balance", total_balance)
m2.metric("📈 Total Profit/Loss", total_profit_loss, delta= f'{delta} since 2026-01-12')
m3.metric("🏦 Invested Capital", invested_capital)

st.divider()

data = {
    "Ticker": list(my_portfolio['symbol'].unique()), 
    "Allocation": [1 / my_portfolio['symbol'].nunique()] * my_portfolio['symbol'].nunique()
}

df_asset = pd.DataFrame(data)

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



