"""Twelve Data provider (free tier: 800 requests/day, 8 per minute)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class TwelveDataProvider(DataProvider):
    """
    Fetches OHLCV data from Twelve Data.

    Free key at https://twelvedata.com/account/api-keys
    """

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Twelve Data"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
        ) -> pd.DataFrame:
        # Twelve Data uses "1day" not "1d"
        interval_map = {"1d": "1day", "1h": "1h", "1w": "1week", "1m": "1min"}
        td_interval = interval_map.get(interval, interval)

        params = {
            "symbol": symbol,
            "interval": td_interval,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "outputsize": 5000,
            "apikey": self._api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "values" not in payload:
            error_msg = payload.get("message") or str(payload)
            raise ValueError(f"[{self.name}] API error for {symbol}: {error_msg}")

        rows = payload["values"]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        return self._normalize(df)
