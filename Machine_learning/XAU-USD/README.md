# XAUUSD Prop-Safe Streamlit MVP

Minimal Streamlit prototype demonstrating an event-driven VWAP mean-reversion strategy with:
- data ingestion (sample CSV + optional live fetch via yfinance)
- VWAP strategy
- simple event-driven backtester with spread + volatility-scaled slippage
- compliance rules (T-15/T+15 news lock, flat EOD)
- risk controls (per-trade %, daily caps)
- mock message bus showing idempotent handling of order IDs

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
