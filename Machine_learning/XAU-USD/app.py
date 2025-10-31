import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from core.strategy import VWAPMeanReversion
from core.backtester import EventDrivenBacktester
from core.risk import RiskEngine
from core.compliance import Compliance
from core.message_bus import MessageBus

st.set_page_config(layout='wide', page_title='XAUUSD Prop-Safe Simulator')
st.title('XAUUSD Prop-Safe Streamlit MVP')

st.sidebar.header('Data')
use_live = st.sidebar.checkbox('Use live data via yfinance (XAUUSD=X)', value=False)
start_date = st.sidebar.date_input('Start date', value=datetime(2025,10,20))
end_date = st.sidebar.date_input('End date', value=datetime(2025,10,20))

st.sidebar.header('Strategy & Risk')
vwap_window = st.sidebar.slider('VWAP window (bars)', 5, 120, 20)
band_std = st.sidebar.slider('Band std dev', 0.5, 3.0, 1.0)
per_trade_pct = st.sidebar.number_input('Per-trade risk %', 0.0001, 0.05, 0.003, format='%.4f')
start_equity = st.sidebar.number_input('Start equity', value=10000.0)

st.sidebar.header('Compliance')
news_ts = st.sidebar.time_input('Simulated news time (UTC)', value=datetime.utcnow().time())
news_enable = st.sidebar.checkbox('Enable news lock demo', value=True)

@st.cache_data
def load_sample():
    df = pd.read_csv('Machine_learning/XAU-USD/data/sample_xauusd_2025.csv', parse_dates=['datetime']).set_index('datetime')
    return df

@st.cache_data
def fetch_live(start, end):
    try:
        data = yf.download('XAUUSD=X', start=start, end=end, interval='1m', progress=False)
        if data.empty:
            return None
        data = data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})[['open','high','low','close','volume']]
        data.index.name = 'datetime'
        return data
    except Exception as e:
        st.warning(f'Live fetch failed: {e}')
        return None

if use_live:
    df = fetch_live(start_date.isoformat(), (end_date + timedelta(days=1)).isoformat()) or load_sample()
else:
    df = load_sample()

st.write(f'Data points: {len(df)}')

strategy = VWAPMeanReversion(vwap_window=vwap_window, band_std=band_std)
risk = RiskEngine(start_equity=start_equity, per_trade_pct=per_trade_pct)
compliance = Compliance(news_lock_seconds=15*60)
message_bus = MessageBus()

if news_enable:
    news_dt = datetime.combine(df.index[0].date(), news_ts)
    compliance.set_news(pd.Timestamp(news_dt))
    st.sidebar.write(f'Simulated news at {news_dt.isoformat()} UTC')

st.header('Backtest')
if st.button('Run backtest'):
    backtester = EventDrivenBacktester(df, strategy, risk)
    trades_df, equity_series = backtester.run()
    st.subheader('Equity Curve')
    st.line_chart(equity_series)
    st.subheader('Trades')
    st.dataframe(trades_df if not trades_df.empty else 'No trades generated')
    st.subheader('Risk Summary')
    st.write({'start_equity': risk.start_equity, 'end_equity': risk.equity, 'daily_pnl': risk.daily_pnl})

st.header('Message Bus (Idempotency Demo)')
with st.form('order_form'):
    coid = st.text_input('Client Order ID', value='order-123')
    valid_secs = st.number_input('Valid seconds', 1, 3600, 60)
    if st.form_submit_button('Send Order'):
        order = {'client_order_id': coid, 'valid_until': valid_secs, 'ts': time.time()}
        st.write(message_bus.send_order(order))

if st.button('Heartbeat'):
    st.write(message_bus.heartbeat())

st.markdown('---')
st.write('Compliance checks:')
now_demo = pd.Timestamp(datetime.utcnow())
st.write({'news_locked': compliance.is_locked(now_demo), 'flat_eod': compliance.is_flat_eod(now_demo)})
