"""Alpaca Markets provider (free tier: 200 calls/min, 5+ years daily history)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class AlpacaProvider(DataProvider):
    """
    Fetches OHLCV data from Alpaca Markets Data API.

    Free account at https://alpaca.markets (paper trading — no money needed).
    Free tier: IEX feed, 200 API calls/min, 5-6 years of daily history.
    """

    BASE_URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key

    @property
    def name(self) -> str:
        return "Alpaca"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        tf_map = {"1d": "1Day", "1h": "1Hour", "1w": "1Week"}
        timeframe = tf_map.get(interval)
        if timeframe is None:
            raise NotImplementedError(
                f"[{self.name}] Interval '{interval}' not supported."
            )

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

        all_bars = []
        page_token = None

        while True:
            params = {
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "adjustment": "all",
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token

            url = f"{self.BASE_URL}/{symbol.upper()}/bars"
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars", [])
            if not bars:
                break
            all_bars.extend(bars)

            page_token = data.get("next_page_token")
            if not page_token:
                break

        if not all_bars:
            raise ValueError(f"[{self.name}] No data for {symbol} ({start} to {end}).")

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t")
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
        })
        return self._normalize(df)
