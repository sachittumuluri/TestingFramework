"""Z-score mean reversion strategy."""

from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategyState


class MeanReversion(Strategy):
    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        trade_size: int = 100,
        allow_short: bool = True,
        max_hold_bars: Optional[int] = None,
    ):
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
        df["_mr_mean"] = df["Close"].rolling(self.lookback, min_periods=self.lookback).mean()
        df["_mr_std"] = df["Close"].rolling(self.lookback, min_periods=self.lookback).std(ddof=0)
        df["_mr_std"] = df["_mr_std"].where(df["_mr_std"] > 1e-12, np.nan)
        df["zscore"] = (df["Close"] - df["_mr_mean"]) / df["_mr_std"]
        return df

    def target_position(self, row: pd.Series, state: StrategyState) -> int:
        z = row.get("zscore", np.nan)
        if pd.isna(z):
            return int(state.current_position)

        if (
            self.max_hold_bars is not None
            and state.current_position != 0
            and state.bars_held >= self.max_hold_bars
        ):
            return 0

        pos = int(state.current_position)

        if pos == 0:
            if z <= -self.entry_z:
                return self.trade_size
            if self.allow_short and z >= self.entry_z:
                return -self.trade_size
            return 0

        if pos > 0:
            if abs(z) <= self.exit_z or z >= self.entry_z:
                return 0
            return pos

        if abs(z) <= self.exit_z or z <= -self.entry_z:
            return 0
        return pos
