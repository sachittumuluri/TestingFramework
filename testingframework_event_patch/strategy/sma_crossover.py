"""Simple Moving Average crossover strategy."""

import pandas as pd
from strategy.base import Strategy, Signal


class SMACrossover(Strategy):
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self._fast = fast_period
        self._slow = slow_period

    @property
    def name(self) -> str:
        return f"SMA Crossover ({self._fast}/{self._slow})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        fast_sma = close.rolling(window=self._fast).mean()
        slow_sma = close.rolling(window=self._slow).mean()

        signals = pd.Series(Signal.HOLD, index=df.index)

        for i in range(1, len(df)):
            if pd.isna(fast_sma.iloc[i]) or pd.isna(slow_sma.iloc[i]):
                continue

            fast_above_now = fast_sma.iloc[i] > slow_sma.iloc[i]
            fast_above_prev = fast_sma.iloc[i - 1] > slow_sma.iloc[i - 1]

            if fast_above_now and not fast_above_prev:
                signals.iloc[i] = Signal.BUY
            elif not fast_above_now and fast_above_prev:
                signals.iloc[i] = Signal.SELL

        return signals
