
"""Backward-compatible import surface for the bar-based engine."""

from backtester.bar_engine import Backtester, BarBacktester
from backtester.models import BacktestConfig, BacktestResult
from backtester.portfolio import Portfolio

__all__ = ["Backtester", "BarBacktester", "BacktestConfig", "BacktestResult", "Portfolio"]
