"""Abstract base class that all data providers must implement."""

from abc import ABC, abstractmethod
from datetime import date
import pandas as pd


class DataProvider(ABC):
    """
    Base interface for market data providers.

    Every provider returns a pandas DataFrame with a DatetimeIndex and columns:
        Open, High, Low, Close, Volume
    All prices are floats; Volume is int. Index is timezone-naive UTC.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this data source (e.g. 'Yahoo Finance')."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a US equity.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. 'AAPL').
        start : date
            Start date (inclusive).
        end : date
            End date (inclusive).
        interval : str
            Bar size — '1d', '1h', etc.  Support varies by provider.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume.
            Index: DatetimeIndex named 'Date'.
        """

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame matches the standard schema."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        # Rename columns if they come back lowercase
        col_map = {c: c.capitalize() for c in df.columns if c.capitalize() in required}
        df = df.rename(columns=col_map)

        df = df[required].copy()
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        # Normalize to midnight so daily bars from different providers
        # (e.g. Yahoo 05:00 vs Twelve Data 00:00) align for cross-validation
        df.index = df.index.normalize()
        df = df.sort_index()

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

        return df
