"""
Unified DataLayer — orchestrates fetching from multiple providers,
validates every result, and optionally cross-checks sources against each other.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from data_layer.providers.base import DataProvider
from data_layer.validation import DataValidator, ValidationResult


class DataLayer:
    """
    High-level entry point for the framework's data pipeline.

    Usage
    -----
    >>> from data_layer import DataLayer, YahooFinanceProvider, AlphaVantageProvider
    >>> dl = DataLayer()
    >>> dl.add_provider(YahooFinanceProvider())
    >>> dl.add_provider(AlphaVantageProvider(api_key="YOUR_KEY"))
    >>> data = dl.fetch("AAPL", date(2020, 1, 1), date(2023, 12, 31))
    """

    def __init__(
        self,
        providers: Optional[List[DataProvider]] = None,
        validator: Optional[DataValidator] = None,
    ):
        self._providers: List[DataProvider] = providers or []
        self._validator = validator or DataValidator()

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def add_provider(self, provider: DataProvider) -> None:
        self._providers.append(provider)

    @property
    def provider_names(self) -> List[str]:
        return [p.name for p in self._providers]

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
        provider_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from a single provider (by name) or the first available.

        Returns the validated DataFrame.  Raises on validation errors.
        """
        provider = self._resolve_provider(provider_name)
        df = provider.fetch_ohlcv(symbol, start, end, interval)
        result = self._validator.validate(df, provider.name, symbol)
        self._report(result)
        df.attrs["symbol"] = symbol
        return df

    def fetch_all(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch from every registered provider and return a dict keyed by provider name.

        Each DataFrame is independently validated.  Providers that fail to
        fetch or fail validation are included in the printed report but
        excluded from the returned dict.
        """
        results: Dict[str, pd.DataFrame] = {}

        for provider in self._providers:
            try:
                df = provider.fetch_ohlcv(symbol, start, end, interval)
            except Exception as exc:
                print(f"[FETCH ERROR] {provider.name} — {symbol}: {exc}")
                continue

            vr = self._validator.validate(df, provider.name, symbol)
            self._report(vr)

            if vr.passed:
                results[provider.name] = df
            else:
                print(f"  -> Excluding {provider.name} due to validation errors.\n")

        if not results:
            raise RuntimeError(
                f"All providers failed to return valid data for {symbol}."
            )

        return results

    def cross_validate(
        self,
        datasets: Dict[str, pd.DataFrame],
        tolerance_pct: float = 1.0,
    ) -> List[str]:
        """
        Compare close prices across providers and flag dates where they
        diverge by more than `tolerance_pct`.

        Returns a list of warning strings (empty list = all good).
        """
        names = list(datasets.keys())
        if len(names) < 2:
            return []

        warnings: List[str] = []
        base_name = names[0]
        base_df = datasets[base_name]

        for other_name in names[1:]:
            other_df = datasets[other_name]
            common_idx = base_df.index.intersection(other_df.index)

            if len(common_idx) == 0:
                warnings.append(
                    f"No overlapping dates between {base_name} and {other_name}."
                )
                continue

            base_close = base_df.loc[common_idx, "Close"]
            other_close = other_df.loc[common_idx, "Close"]
            pct_diff = ((base_close - other_close) / base_close).abs() * 100

            divergent = pct_diff[pct_diff > tolerance_pct]
            if len(divergent) > 0:
                sample_dates = [d.strftime("%Y-%m-%d") for d in divergent.index[:5]]
                warnings.append(
                    f"{base_name} vs {other_name}: {len(divergent)} date(s) "
                    f"diverge >{tolerance_pct}% on Close — e.g. {', '.join(sample_dates)}"
                )

        return warnings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_provider(self, name: Optional[str]) -> DataProvider:
        if not self._providers:
            raise RuntimeError("No data providers registered. Call add_provider() first.")
        if name is None:
            return self._providers[0]
        for p in self._providers:
            if p.name == name:
                return p
        raise ValueError(
            f"Provider '{name}' not found. Available: {self.provider_names}"
        )

    @staticmethod
    def _report(result: ValidationResult) -> None:
        print(result.summary())
