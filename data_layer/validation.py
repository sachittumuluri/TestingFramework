"""Data validation checks for OHLCV DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Container for all validation findings on a single DataFrame."""

    provider: str
    symbol: str
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.provider} — {self.symbol}"]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        return "\n".join(lines)


class DataValidator:
    """
    Runs a suite of validation checks on an OHLCV DataFrame.

    Checks performed:
        1. Missing values (NaN / null) in any column
        2. Timestamp integrity — sorted, no duplicates, expected trading-day frequency
        3. OHLCV sanity — High >= Low, all prices > 0, Volume >= 0
        4. Large gap detection — flags single-bar price moves > threshold
    """

    def __init__(self, max_gap_pct: float = 50.0, max_missing_pct: float = 5.0):
        """
        Parameters
        ----------
        max_gap_pct : float
            Flag bars where close-to-close % change exceeds this value.
        max_missing_pct : float
            Percentage of expected trading days that can be missing before
            it's flagged as an error (vs. a warning).
        """
        self.max_gap_pct = max_gap_pct
        self.max_missing_pct = max_missing_pct

    def validate(
        self, df: pd.DataFrame, provider: str, symbol: str
    ) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        result = ValidationResult(provider=provider, symbol=symbol)

        if df.empty:
            result.add_error("DataFrame is empty — no data to validate.")
            return result

        self._check_missing_values(df, result)
        self._check_timestamps(df, result)
        self._check_ohlcv_sanity(df, result)
        self._check_large_gaps(df, result)

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_missing_values(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Flag NaN / null values in any column."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls == 0:
            return

        detail = ", ".join(
            f"{col}: {cnt}" for col, cnt in null_counts.items() if cnt > 0
        )
        pct = total_nulls / (len(df) * len(df.columns)) * 100
        msg = f"{total_nulls} missing value(s) ({pct:.1f}% of cells) — {detail}"

        if pct > self.max_missing_pct:
            result.add_error(msg)
        else:
            result.add_warning(msg)

    def _check_timestamps(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Verify index is sorted, has no duplicates, and has reasonable frequency."""
        # Sorted
        if not df.index.is_monotonic_increasing:
            result.add_error("Timestamps are not sorted in ascending order.")

        # Duplicates
        dups = df.index.duplicated()
        if dups.any():
            n_dups = dups.sum()
            result.add_error(f"{n_dups} duplicate timestamp(s) found.")

        # Trading-day frequency check (for daily data)
        if len(df) < 2:
            return

        deltas = pd.Series(df.index).diff().dropna()
        median_delta = deltas.median()

        # If median gap is roughly 1 day, treat as daily data
        if pd.Timedelta(hours=12) < median_delta < pd.Timedelta(days=5):
            # Count gaps > 5 calendar days (more than a long weekend)
            big_gaps = deltas[deltas > pd.Timedelta(days=5)]
            if len(big_gaps) > 0:
                dates = [
                    df.index[i].strftime("%Y-%m-%d")
                    for i in big_gaps.index
                ]
                result.add_warning(
                    f"{len(big_gaps)} gap(s) > 5 calendar days: {', '.join(dates[:5])}"
                    + (" ..." if len(dates) > 5 else "")
                )

            # Check expected trading days vs actual rows
            total_calendar_days = (df.index[-1] - df.index[0]).days
            expected_trading_days = total_calendar_days * 252 / 365  # rough estimate
            if expected_trading_days > 0:
                coverage = len(df) / expected_trading_days * 100
                if coverage < (100 - self.max_missing_pct):
                    result.add_warning(
                        f"Only {len(df)} bars for ~{expected_trading_days:.0f} "
                        f"expected trading days ({coverage:.1f}% coverage)."
                    )

    def _check_ohlcv_sanity(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check that OHLCV values are physically reasonable."""
        price_cols = ["Open", "High", "Low", "Close"]

        # Negative or zero prices
        for col in price_cols:
            if col not in df.columns:
                result.add_error(f"Missing required column: {col}")
                continue
            bad = (df[col] <= 0).sum()
            if bad:
                result.add_error(f"{bad} non-positive value(s) in {col}.")

        # High >= Low
        if "High" in df.columns and "Low" in df.columns:
            violations = (df["High"] < df["Low"]).sum()
            if violations:
                result.add_error(f"{violations} bar(s) where High < Low.")

        # Negative volume
        if "Volume" in df.columns:
            neg_vol = (df["Volume"] < 0).sum()
            if neg_vol:
                result.add_error(f"{neg_vol} negative Volume value(s).")

    def _check_large_gaps(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Flag abnormally large single-bar price jumps."""
        if "Close" not in df.columns or len(df) < 2:
            return

        pct_change = df["Close"].pct_change().abs() * 100
        big_moves = pct_change[pct_change > self.max_gap_pct].dropna()

        if len(big_moves) > 0:
            dates = [idx.strftime("%Y-%m-%d") for idx in big_moves.index[:5]]
            result.add_warning(
                f"{len(big_moves)} bar(s) with >{self.max_gap_pct}% price move: "
                f"{', '.join(dates)}" + (" ..." if len(big_moves) > 5 else "")
            )
