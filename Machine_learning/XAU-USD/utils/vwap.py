import pandas as pd

def vwap(df: pd.DataFrame, price_col: str = 'close', vol_col: str = 'volume', window: int = 20) -> pd.Series:
    tp = df[price_col]
    pv = tp * df[vol_col]
    rolling_pv = pv.rolling(window=window, min_periods=1).sum()
    rolling_v = df[vol_col].rolling(window=window, min_periods=1).sum()
    return rolling_pv / rolling_v
