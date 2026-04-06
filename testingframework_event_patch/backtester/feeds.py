
"""Market-data feeds for historical and synthetic OHLCV replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import pandas as pd


def normalize_ohlcv(df: pd.DataFrame, keep_extra: bool = False) -> pd.DataFrame:
    """Normalize a DataFrame to the standard OHLCV schema."""
    if df is None or len(df) == 0:
        raise ValueError("OHLCV DataFrame cannot be empty")

    out = df.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    col_map = {c: c.capitalize() for c in out.columns if c.capitalize() in required}
    out = out.rename(columns=col_map)

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {missing}")

    keep_cols = list(out.columns) if keep_extra else required
    out = out[keep_cols].copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_localize(None)
    out.index.name = "Date"
    out = out.sort_index()

    for col in ["Open", "High", "Low", "Close"]:
        out[col] = out[col].astype(float)
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype(int)
    return out


@dataclass(frozen=True)
class Bar:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    row: pd.Series

    @classmethod
    def from_row(cls, symbol: str, timestamp: pd.Timestamp, row: pd.Series) -> "Bar":
        return cls(
            timestamp=pd.Timestamp(timestamp),
            symbol=symbol,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
            row=row,
        )


class DataFrameFeed(Iterable[Bar]):
    """Replay a standardized OHLCV DataFrame as bar events."""

    def __init__(self, df: pd.DataFrame, symbol: str = "ASSET"):
        self.df = normalize_ohlcv(df, keep_extra=True)
        self.symbol = symbol

    def __iter__(self) -> Iterator[Bar]:
        for timestamp, row in self.df.iterrows():
            yield Bar.from_row(self.symbol, timestamp, row)

    def next_timestamp(self, i: int) -> Optional[pd.Timestamp]:
        if i + 1 < len(self.df.index):
            return pd.Timestamp(self.df.index[i + 1])
        return None
