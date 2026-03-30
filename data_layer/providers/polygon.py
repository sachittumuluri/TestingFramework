"""Polygon.io provider (free tier: 5 API calls/min, end-of-day data, 2yr history)."""

from datetime import date, timedelta
import time
import pandas as pd
import requests

from data_layer.providers.base import DataProvider


class PolygonProvider(DataProvider):
    """
    Fetches OHLCV data from Polygon.io.

    Free key at https://polygon.io/dashboard/signup

    Free tier limits:
        - 5 API calls per minute
        - End-of-day data only (15-min delay)
        - Up to 2 years of historical data
    """

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Polygon.io"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        interval_map = {
            "1d": (1, "day"),
            "1h": (1, "hour"),
            "1w": (1, "week"),
        }
        if interval not in interval_map:
            raise NotImplementedError(
                f"[{self.name}] Interval '{interval}' not supported."
            )
        multiplier, timespan = interval_map[interval]

        # Free tier limits history to ~2 years. Chunk into 1-year windows
        # to stay within limits and respect the 5 req/min rate limit.
        all_frames = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=365), end)

            url = (
                f"{self.BASE_URL}/{symbol.upper()}/range"
                f"/{multiplier}/{timespan}/{chunk_start.isoformat()}/{chunk_end.isoformat()}"
            )
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self._api_key,
            }
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 403:
                raise ValueError(
                    f"[{self.name}] 403 Forbidden for {symbol}. "
                    "The free tier may not cover this date range — "
                    "try a shorter/more recent window (last 2 years)."
                )
            resp.raise_for_status()
            payload = resp.json()

            if "results" in payload and payload.get("resultsCount", 0) > 0:
                all_frames.append(pd.DataFrame(payload["results"]))

            chunk_start = chunk_end + timedelta(days=1)

            # Respect rate limit (5/min) if we need more chunks
            if chunk_start < end:
                time.sleep(12)

        if not all_frames:
            raise ValueError(
                f"[{self.name}] No data returned for {symbol} ({start} to {end}). "
                "Free tier only provides ~2 years of history."
            )

        df = pd.concat(all_frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("Date")
        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        })
        return self._normalize(df)
