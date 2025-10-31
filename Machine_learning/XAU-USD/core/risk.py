class RiskEngine:
    def __init__(self, start_equity: float = 10000.0, per_trade_pct: float = 0.003):
        self.start_equity = start_equity
        self.equity = start_equity
        self.per_trade_pct = per_trade_pct

    def size_from_risk(self, price: float) -> float:
        risk_amount = self.per_trade_pct * self.equity
        return risk_amount / price

    def update_pnl(self, pnl: float):
        self.equity += pnl
