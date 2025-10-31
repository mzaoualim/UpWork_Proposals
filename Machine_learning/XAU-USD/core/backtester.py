import pandas as pd
import numpy as np
from core.risk import RiskEngine

class EventDrivenBacktester:
    def __init__(self, df: pd.DataFrame, strategy, risk: RiskEngine, spread: float = 0.5, vol_slip_coeff: float = 0.0001):
        self.df = df.copy()
        self.strategy = strategy
        self.risk = risk
        self.spread = spread
        self.vol_slip_coeff = vol_slip_coeff
        self.trades = []
        self.equity_curve = []

    def model_slippage(self, row: pd.Series) -> float:
        vol = row.get('rolling_std', 0.0)
        return self.vol_slip_coeff * (vol if not np.isnan(vol) else 0)

    def run(self):
        df = self.strategy.generate_signals(self.df)
        equity = self.risk.equity
        self.equity_curve = []
        for idx, row in df.iterrows():
            order = row.get('order', 0)
            price = row['close']
            if order == 0:
                self.equity_curve.append(equity)
                continue
            size = self.risk.size_from_risk(price)
            slip = self.model_slippage(row)
            effective_trade_cost = (self.spread/2.0) + (self.spread/2.0) + (price * slip)
            pnl = - order * size * effective_trade_cost
            equity += pnl
            self.risk.update_pnl(pnl)
            self.trades.append({'ts': idx, 'order': order, 'price': price, 'size': size, 'pnl': pnl})
            self.equity_curve.append(equity)
        return pd.DataFrame(self.trades), pd.Series(self.equity_curve, index=df.index[:len(self.equity_curve)])
