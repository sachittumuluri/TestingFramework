"""Financial Modeling Prep (FMP) provider (free tier: 250 requests/day)."""

from datetime import date
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class FMPProvider(DataProvider):
    """
    Fetches OHLCV data from Financial Modeling Prep.

    Free key at https://site.financialmodelingprep.com/developer/docs
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Financial Modeling Prep"

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

        url = f"{self.BASE_URL}/historical-price-eod/full"
        params = {
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "apikey": self._api_key,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if not payload:
            raise ValueError(f"[{self.name}] No data returned for {symbol}.")

        # The stable API returns a flat list of records
        if isinstance(payload, list):
            data = payload
        elif isinstance(payload, dict) and "historical" in payload:
            data = payload["historical"]
        else:
            error_msg = payload.get("Error Message") or str(payload)[:200]
            raise ValueError(f"[{self.name}] API error for {symbol}: {error_msg}")

        if not data:
            raise ValueError(f"[{self.name}] No data returned for {symbol} ({start} to {end}).")

        df = pd.DataFrame(data)
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
