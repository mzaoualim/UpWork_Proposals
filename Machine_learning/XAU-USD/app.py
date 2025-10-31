import streamlit as st
import pandas as pd
from core.strategy import VWAPMeanReversion
from core.backtester import EventDrivenBacktester
from core.risk import RiskEngine

st.set_page_config(layout='wide', page_title='XAUUSD 15-min Backtest')
st.title('XAUUSD 15-min VWAP Mean-Reversion Backtest')

# Sidebar inputs
st.sidebar.header('Strategy & Risk')
vwap_window = st.sidebar.slider('VWAP window (bars)', 5, 120, 20)
band_std = st.sidebar.slider('Band std dev', 0.5, 3.0, 1.0)
per_trade_pct = st.sidebar.number_input('Per-trade risk %', 0.0001, 0.05, 0.003, format='%.4f')
start_equity = st.sidebar.number_input('Start equity', value=10000.0)

# Load CSV
@st.cache_data
def load_csv():
    df = pd.read_csv('Machine_learning/XAU-USD/data/XAU_15m_data.csv', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    # Fill missing values if any
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].fillna(method='ffill')
    return df

df = load_csv()
st.write(f"Data points loaded: {len(df)}")
st.dataframe(df.tail(5))

# Initialize strategy and risk engine
strategy = VWAPMeanReversion(vwap_window=vwap_window, band_std=band_std)
risk = RiskEngine(start_equity=start_equity, per_trade_pct=per_trade_pct)

# Run backtest
st.header('Backtest')
if st.button('Run Backtest'):
    backtester = EventDrivenBacktester(df, strategy, risk)
    trades_df, equity_series = backtester.run()
    st.subheader('Equity Curve')
    st.line_chart(equity_series)
    st.subheader('Trades')
    st.dataframe(trades_df if not trades_df.empty else 'No trades generated')
    st.subheader('Risk Summary')
    st.write({'start_equity': risk.start_equity, 'end_equity': risk.equity, 'daily_pnl': risk.daily_pnl})
