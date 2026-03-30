"""
Local CSV file provider — load OHLCV data from CSV files.

Useful for offline testing, custom datasets, or pre-downloaded data.
"""

from datetime import date
import pandas as pd

from data_layer.providers.base import DataProvider


class CsvFileProvider(DataProvider):
    """
    Loads OHLCV data from a local CSV file.

    Expected CSV format: a 'Date' column (or DatetimeIndex) plus
    Open, High, Low, Close, Volume columns (case-insensitive).

    Usage:
        provider = CsvFileProvider("path/to/data.csv")
    """

    def __init__(self, file_path: str):
        self._file_path = file_path

    @property
    def name(self) -> str:
        return f"CSV ({self._file_path})"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = pd.read_csv(self._file_path, parse_dates=True)

        # Try to find a date column to use as index
        date_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.set_index(date_cols[0])
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        df = df.loc[str(start):str(end)]

        if df.empty:
            raise ValueError(
                f"[{self.name}] No data for {symbol} in range {start} to {end}."
            )
        return self._normalize(df)
