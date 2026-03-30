"""
Helper functions for strategy development.

Inspired by Backtesting.py's lib module, these utilities simplify common
pattern-detection tasks inside Strategy.next() implementations.
"""

import math
from typing import Union

import numpy as np
import pandas as pd

Numeric = Union[int, float, np.integer, np.floating]
SeriesLike = Union[pd.Series, np.ndarray, list]


def _as_float(value) -> float:
    """Coerce a scalar (possibly numpy) to a plain Python float."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return float(value)


def _last_two(series) -> tuple:
    """Return the last two values of *series* as plain floats.

    *series* may be a number, pd.Series, np.ndarray, or list.
    A bare number is treated as a constant (same value for both positions).
    """
    if isinstance(series, (int, float, np.integer, np.floating)):
        v = _as_float(series)
        return v, v
    if isinstance(series, pd.Series):
        return _as_float(series.iloc[-2]), _as_float(series.iloc[-1])
    # numpy array or plain list
    arr = np.asarray(series)
    return _as_float(arr[-2]), _as_float(arr[-1])


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def crossover(series1: Union[Numeric, SeriesLike],
              series2: Union[Numeric, SeriesLike]) -> bool:
    """Return ``True`` if *series1* just crossed **above** *series2*.

    A crossover happens when, at the previous bar, series1 was <= series2
    and at the current bar series1 is > series2.

    Both arguments accept numbers (treated as constants), ``pd.Series``,
    ``np.ndarray``, or plain lists.
    """
    prev1, curr1 = _last_two(series1)
    prev2, curr2 = _last_two(series2)
    return prev1 <= prev2 and curr1 > curr2


def cross(series1: Union[Numeric, SeriesLike],
          series2: Union[Numeric, SeriesLike]) -> bool:
    """Return ``True`` if *series1* and *series2* just crossed in **either**
    direction (i.e. ``crossover(s1, s2) or crossover(s2, s1)``).
    """
    return crossover(series1, series2) or crossover(series2, series1)


def barssince(condition: Union[pd.Series, np.ndarray, list]) -> int:
    """Return the number of bars since *condition* was last ``True``.

    *condition* should be a boolean array-like (``pd.Series``, ``np.ndarray``,
    or list).  The search starts from the most recent bar and works backward.

    Returns ``math.inf`` if the condition has never been ``True``.
    """
    arr = np.asarray(condition, dtype=bool)
    if arr.size == 0:
        return math.inf
    # Find indices where condition is True
    true_indices = np.flatnonzero(arr)
    if true_indices.size == 0:
        return math.inf
    last_true = int(true_indices[-1])
    return (len(arr) - 1) - last_true


def quantile(series: Union[pd.Series, np.ndarray, list],
             q: float = None) -> float:
    """Quantile helper with two modes of operation.

    * **Rank mode** (``q is None``): return the quantile rank of the last
      value relative to all prior values (result in [0, 1]).
    * **Value mode** (``q`` in [0, 1]): return the value at the given
      quantile across the entire series.

    Parameters
    ----------
    series : array-like
        Numeric data (``pd.Series``, ``np.ndarray``, or list).
    q : float, optional
        If provided, must be between 0 and 1.  Switches from rank mode to
        value mode.
    """
    arr = np.asarray(series, dtype=float)

    if q is not None:
        # Value mode: return the q-th quantile of the full series.
        if not 0 <= q <= 1:
            raise ValueError(f"q must be between 0 and 1, got {q}")
        return float(np.nanquantile(arr, q))

    # Rank mode: quantile rank of last value vs all prior values.
    if arr.size < 2:
        return np.nan
    last = arr[-1]
    prior = arr[:-1]
    prior = prior[~np.isnan(prior)]
    if prior.size == 0:
        return np.nan
    return float(np.sum(prior <= last) / len(prior))
