class RiskEngine:
    def __init__(self, start_equity: float = 10000.0, per_trade_pct: float = 0.003, daily_cap_pct: float = -0.01):
        self.start_equity = start_equity
        self.equity = start_equity
        self.per_trade_pct = per_trade_pct
        self.daily_cap_pct = daily_cap_pct
        self.daily_pnl = 0.0

    def size_from_risk(self, price: float) -> float:
        risk_amount = self.per_trade_pct * self.equity
        return risk_amount / price

    def update_pnl(self, pnl: float):
        self.equity += pnl
        self.daily_pnl += pnl

    def daily_cap_reached(self) -> bool:
        return self.daily_pnl <= (self.daily_cap_pct * self.start_equity)

    def reset_daily(self):
        self.daily_pnl = 0.0
