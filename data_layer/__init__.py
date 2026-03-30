"""
Data Layer — Multi-API data fetching and validation for the Trading Strategy Testing Framework.

Providers (10 total):
    - Yahoo Finance               (free, no key)
    - CSV File                    (local files, no key)
    - Alpha Vantage               (free key)
    - Twelve Data                 (free key)
    - Polygon.io                  (free key)
    - Tiingo                      (free key)
    - Financial Modeling Prep     (free key)
    - MarketStack                 (free key)
    - Finnhub                     (free key)
    - Alpaca                      (free key + secret)
"""

from data_layer.providers.base import DataProvider
from data_layer.providers.yahoo import YahooFinanceProvider
from data_layer.providers.alpha_vantage import AlphaVantageProvider
from data_layer.providers.twelve_data import TwelveDataProvider
from data_layer.providers.polygon import PolygonProvider
from data_layer.providers.tiingo import TiingoProvider
from data_layer.providers.stooq import CsvFileProvider
from data_layer.providers.fmp import FMPProvider
from data_layer.providers.marketstack import MarketStackProvider
from data_layer.providers.finnhub import FinnhubProvider
from data_layer.providers.alpaca import AlpacaProvider
from data_layer.validation import DataValidator, ValidationResult
from data_layer.data_layer import DataLayer

__all__ = [
    "DataProvider",
    "YahooFinanceProvider",
    "AlphaVantageProvider",
    "TwelveDataProvider",
    "PolygonProvider",
    "TiingoProvider",
    "CsvFileProvider",
    "FMPProvider",
    "MarketStackProvider",
    "FinnhubProvider",
    "AlpacaProvider",
    "DataValidator",
    "ValidationResult",
    "DataLayer",
]
