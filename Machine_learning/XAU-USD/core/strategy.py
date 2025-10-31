import pandas as pd
import numpy as np
from utils.vwap import vwap

class VWAPMeanReversion:
    def __init__(self, vwap_window: int = 20, band_std: float = 1.0, slippage_pct: float = 0.0005):
        self.vwap_window = vwap_window
        self.band_std = band_std
        self.slippage_pct = slippage_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['vwap'] = vwap(df, window=self.vwap_window)
        df['typ_price'] = df['close']
        df['rolling_std'] = df['typ_price'].rolling(window=self.vwap_window, min_periods=1).std()
        df['upper'] = df['vwap'] + self.band_std * df['rolling_std']
        df['lower'] = df['vwap'] - self.band_std * df['rolling_std']

        df['signal'] = 0
        df.loc[df['close'] < df['lower'], 'signal'] = 1
        df.loc[df['close'] > df['upper'], 'signal'] = -1
        df['signal_shift'] = df['signal'].shift(1).fillna(0)
        df['order'] = 0
        df.loc[df['signal'] != df['signal_shift'], 'order'] = df['signal']
        return df
