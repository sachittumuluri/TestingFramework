"""Tiingo provider (free tier: 1000 requests/day, 20+ years of EOD data)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class TiingoProvider(DataProvider):
    """
    Fetches OHLCV data from Tiingo.

    Free key at https://api.tiingo.com/account/api/token
    """

    BASE_URL = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Tiingo"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise NotImplementedError(
                f"[{self.name}] Only '1d' interval is supported."
            )

        url = f"{self.BASE_URL}/{symbol}/prices"
        headers = {"Content-Type": "application/json"}
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "token": self._api_key,
        }
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            raise ValueError(f"[{self.name}] No data returned for {symbol}.")

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Tiingo returns adjOpen/adjHigh/etc for split-adjusted data
        if "adjClose" in df.columns:
            df = df.rename(columns={
                "adjOpen": "Open",
                "adjHigh": "High",
                "adjLow": "Low",
                "adjClose": "Close",
                "adjVolume": "Volume",
            })
        else:
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            })

        return self._normalize(df)
