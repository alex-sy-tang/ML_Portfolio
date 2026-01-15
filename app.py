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



# def get_most_recent_workday(today_date): 
#     if today_date.weekday() == 0: 
#         diff = 3
#     elif today_date.weekday() == 6: 
#         diff = 2
#     else:
#         diff = 1
    
#     last_weekday = today_date - timedelta(days = diff)
#     return last_weekday



st.title('Stock Portfolio Dashboard')
st.subheader('Historical Price')


# add_selectbox = st.sidebar.selectbox(
#     "Pages: ",
#     ("Historical Price", "Portfolio")
# )


holdings_df = load_data(DATA_DIR/"raw"/"holdings.csv")
symbols_list = holdings_df["symbol"].tolist()
selected_symbol = st.selectbox("Symbols", symbols_list)


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
st.dataframe(symbol_df)


#Stock Portfolio: 
st.divider()
st.subheader('My Portfolio')

my_portfolio = load_data(DATA_DIR/"processed"/"weekly_portfolio.csv")
historical_performance = load_data(DATA_DIR/'processed'/'historical_performance.csv')
st.line_chart(historical_performance, x = 'date', y = 'cumulative_return')
st.dataframe(my_portfolio)
# st.dataframe(historical_performance)ååå

# symbol_df['symbol'].unique()

