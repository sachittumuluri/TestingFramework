"""Finnhub provider (free tier: 60 calls/min, ~1 year daily history)."""

from datetime import date, datetime
import time
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class FinnhubProvider(DataProvider):
    """
    Fetches OHLCV data from Finnhub.

    Free key at https://finnhub.io/register
    Free tier: 60 API calls/min, ~1 year of daily candle history.
    """

    BASE_URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Finnhub"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        resolution_map = {"1d": "D", "1h": "60", "1w": "W"}
        resolution = resolution_map.get(interval)
        if resolution is None:
            raise NotImplementedError(
                f"[{self.name}] Interval '{interval}' not supported. Use: {list(resolution_map.keys())}"
            )

        from_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.max.time()).timestamp())

        params = {
            "symbol": symbol.upper(),
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
            "token": self._api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("s") != "ok" or "t" not in data:
            raise ValueError(
                f"[{self.name}] No data for {symbol} ({start} to {end}). "
                f"Free tier only provides ~1 year of history."
            )

        df = pd.DataFrame({
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"],
        }, index=pd.to_datetime(data["t"], unit="s"))

        return self._normalize(df)
