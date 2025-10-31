import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from core.strategy import VWAPMeanReversion
from core.backtester import EventDrivenBacktester
from core.risk import RiskEngine

st.set_page_config(layout='wide', page_title='XAUUSD Prop-Safe Backtest')
st.title('XAUUSD Prop-Safe Streamlit MVP v2')

# Sidebar parameters
st.sidebar.header('Strategy & Risk')
vwap_window = st.sidebar.slider('VWAP window (bars)', 5, 120, 20)
band_std = st.sidebar.slider('Band std dev', 0.5, 3.0, 1.0)
per_trade_pct = st.sidebar.number_input('Per-trade risk %', 0.0001, 0.05, 0.003, format='%.4f')
start_equity = st.sidebar.number_input('Start equity', value=10000.0)

@st.cache_data
def fetch_data():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)
    data = yf.download('XAUUSD=X', start=start, end=end, interval='1m', progress=False)
    if data.empty:
        st.warning('No live data fetched, please try later')
        return None
    data = data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})[['open','high','low','close','volume']]
    data.index.name = 'datetime'
    return data

df = fetch_data()
if df is None or df.empty:
    st.stop()

st.write(f'Data points: {len(df)}')

strategy = VWAPMeanReversion(vwap_window=vwap_window, band_std=band_std)
risk = RiskEngine(start_equity=start_equity, per_trade_pct=per_trade_pct)

st.header('Backtest')
if st.button('Run backtest'):
    backtester = EventDrivenBacktester(df, strategy, risk)
    trades_df, equity_series = backtester.run()
    st.subheader('Equity Curve')
    st.line_chart(equity_series)
    st.subheader('Trades')
    st.dataframe(trades_df if not trades_df.empty else 'No trades generated')
    st.subheader('Risk Summary')
    st.write({'start_equity': risk.start_equity, 'end_equity': risk.equity})
