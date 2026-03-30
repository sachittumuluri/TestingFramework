"""MarketStack provider (free tier: 100 requests/month, end-of-day data)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class MarketStackProvider(DataProvider):
    """
    Fetches OHLCV data from MarketStack.

    Free key at https://marketstack.com/signup/free
    Note: free tier uses HTTP only (no HTTPS).
    """

    BASE_URL = "http://api.marketstack.com/v1/eod"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "MarketStack"

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

        all_data = []
        offset = 0
        limit = 1000

        while True:
            params = {
                "access_key": self._api_key,
                "symbols": symbol,
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "limit": limit,
                "offset": offset,
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            if "error" in payload:
                raise ValueError(
                    f"[{self.name}] API error for {symbol}: "
                    f"{payload['error'].get('message', payload['error'])}"
                )

            data = payload.get("data", [])
            if not data:
                break

            all_data.extend(data)

            # Check if there are more pages
            pagination = payload.get("pagination", {})
            total = pagination.get("total", 0)
            if offset + limit >= total:
                break
            offset += limit

        if not all_data:
            raise ValueError(f"[{self.name}] No data returned for {symbol}.")

        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        return self._normalize(df)
