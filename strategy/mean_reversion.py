"""Z-score mean reversion strategy (ported from Connor's BarBacktester)."""

from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategyState


class MeanReversion(Strategy):
    """
    Z-score mean reversion: long when price dips below the mean,
    short when it spikes above.

    Rules:
      - z = (close - rolling_mean) / rolling_std
      - Go long when z <= -entry_z
      - Go short when z >= +entry_z (if allow_short=True)
      - Exit to flat when |z| <= exit_z
      - Optional max_hold_bars to force exit (time stop)

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std. Default 20.
    entry_z : float
        Z-score threshold to enter a position. Default 1.5.
    exit_z : float
        Z-score threshold to exit. Must be < entry_z. Default 0.5.
    trade_size : int
        Number of shares per trade. Default 100.
    allow_short : bool
        Allow short positions. Default True.
    max_hold_bars : int or None
        Force exit after this many bars. None = no limit.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        trade_size: int = 100,
        allow_short: bool = True,
        max_hold_bars: Optional[int] = None,
    ):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if exit_z >= entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        if trade_size <= 0:
            raise ValueError("trade_size must be > 0")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.trade_size = int(trade_size)
        self.allow_short = allow_short
        self.max_hold_bars = max_hold_bars

    @property
    def name(self) -> str:
        short = "+short" if self.allow_short else "long-only"
        return f"Mean Reversion (z={self.entry_z}/{self.exit_z}, {short})"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        col = "Close" if "Close" in df.columns else "close"
        df["_mr_mean"] = df[col].rolling(self.lookback, min_periods=self.lookback).mean()
        df["_mr_std"] = df[col].rolling(self.lookback, min_periods=self.lookback).std(ddof=0)
        df["_mr_std"] = df["_mr_std"].where(df["_mr_std"] > 1e-12, np.nan)
        df["zscore"] = (df[col] - df["_mr_mean"]) / df["_mr_std"]
        return df

    def target_position(self, row: pd.Series, state: StrategyState) -> int:
        z = row.get("zscore", np.nan)
        if pd.isna(z):
            return int(state.current_position)

        # Time stop
        if (self.max_hold_bars is not None
                and state.current_position != 0
                and state.bars_held >= self.max_hold_bars):
            return 0

        pos = int(state.current_position)

        # Flat — look for entry
        if pos == 0:
            if z <= -self.entry_z:
                return self.trade_size
            if self.allow_short and z >= self.entry_z:
                return -self.trade_size
            return 0

        # Long — look for exit
        if pos > 0:
            if abs(z) <= self.exit_z or z >= self.entry_z:
                return 0
            return pos

        # Short — look for exit
        if abs(z) <= self.exit_z or z <= -self.entry_z:
            return 0
        return pos
