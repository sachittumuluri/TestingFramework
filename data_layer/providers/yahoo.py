"""Yahoo Finance data provider using the yfinance library."""

from datetime import date, timedelta
import pandas as pd
import yfinance as yf

from data_layer.providers.base import DataProvider


class YahooFinanceProvider(DataProvider):
    """Fetches OHLCV data from Yahoo Finance (free, no API key required)."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        # yfinance `end` is exclusive, so add one day
        df = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(
                f"[{self.name}] No data returned for {symbol} "
                f"({start} to {end}, interval={interval})"
            )
        return self._normalize(df)
