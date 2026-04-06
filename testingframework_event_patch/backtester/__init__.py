
"""Backtester package exports."""

from backtester.engine import Backtester, BarBacktester, BacktestConfig, BacktestResult, Portfolio
from backtester.event_engine import EventDrivenBacktester
from backtester.execution import BarExecutionModel
from backtester.feeds import DataFrameFeed, normalize_ohlcv
from backtester.synthetic import (
    GANSource,
    GBMSource,
    BlockBootstrapSource,
    NoiseInjectionSource,
    RegimeSwitchingSource,
    make_oscillating,
    make_random_walk,
    make_trending,
    run_scenario_suite,
    run_validation_suite,
)

__all__ = [
    "Backtester",
    "BarBacktester",
    "BacktestConfig",
    "BacktestResult",
    "Portfolio",
    "EventDrivenBacktester",
    "BarExecutionModel",
    "DataFrameFeed",
    "normalize_ohlcv",
    "GANSource",
    "GBMSource",
    "BlockBootstrapSource",
    "NoiseInjectionSource",
    "RegimeSwitchingSource",
    "make_oscillating",
    "make_random_walk",
    "make_trending",
    "run_scenario_suite",
    "run_validation_suite",
]

# Optional legacy utilities from the existing repo.
try:
    from backtester.scorecard import generate_scorecard  # type: ignore
    __all__.append("generate_scorecard")
except Exception:
    pass

try:
    from backtester.optimize import optimize, plot_heatmap  # type: ignore
    __all__.extend(["optimize", "plot_heatmap"])
except Exception:
    pass

try:
    from backtester.distributions import generate_distribution_plots  # type: ignore
    __all__.append("generate_distribution_plots")
except Exception:
    pass
