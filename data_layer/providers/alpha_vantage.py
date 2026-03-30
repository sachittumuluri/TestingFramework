"""Alpha Vantage data provider (requires a free API key)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class AlphaVantageProvider(DataProvider):
    """
    Fetches OHLCV data from Alpha Vantage.

    Get a free key at https://www.alphavantage.co/support/#api-key
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Alpha Vantage"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise NotImplementedError(
                f"[{self.name}] Only '1d' interval is currently supported."
            )

        # Free tier only supports outputsize=compact (last 100 trading days).
        # Always use compact — if the requested range is older than ~5 months
        # ago it won't be available on the free tier.
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": self._api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        ts_key = "Time Series (Daily)"
        if ts_key not in payload:
            error_msg = (
                payload.get("Information")
                or payload.get("Note")
                or payload.get("Error Message")
                or str(payload)
            )
            raise ValueError(f"[{self.name}] API error for {symbol}: {error_msg}")

        raw = payload[ts_key]
        rows = {
            pd.Timestamp(dt): {
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["4. close"]),
                "Volume": int(vals["5. volume"]),
            }
            for dt, vals in raw.items()
        }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Filter to requested date range
        df = df.loc[str(start) : str(end)]

        if df.empty:
            raise ValueError(
                f"[{self.name}] No data in range {start} to {end} for {symbol}. "
                f"Free tier only provides the last ~100 trading days "
                f"(data available: {list(raw.keys())[-1]} to {list(raw.keys())[0]})."
            )
        return self._normalize(df)
