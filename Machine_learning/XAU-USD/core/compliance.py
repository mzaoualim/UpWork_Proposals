import pandas as pd
from datetime import datetime

class Compliance:
    def __init__(self, news_lock_seconds: int = 15*60):
        self.news_lock_seconds = news_lock_seconds
        self.last_news_ts = None

    def set_news(self, ts: pd.Timestamp):
        self.last_news_ts = ts

    def is_locked(self, now: pd.Timestamp) -> bool:
        if self.last_news_ts is None:
            return False
        return abs((now - self.last_news_ts).total_seconds()) <= self.news_lock_seconds

    def is_flat_eod(self, now: pd.Timestamp, eod_hour: int = 23, eod_minute: int = 59) -> bool:
        eod = now.replace(hour=eod_hour, minute=eod_minute, second=0, microsecond=0)
        return now >= eod
