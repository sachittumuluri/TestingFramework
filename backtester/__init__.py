from backtester.engine import Backtester, BacktestResult, BacktestConfig, Portfolio
from backtester.synthetic import (
    make_oscillating, make_trending, make_random_walk, run_validation_suite,
)
from backtester.scorecard import generate_scorecard
from backtester.optimize import optimize, plot_heatmap
from backtester.distributions import generate_distribution_plots

__all__ = [
    "Backtester", "BacktestResult", "BacktestConfig", "Portfolio",
    "make_oscillating", "make_trending", "make_random_walk", "run_validation_suite",
    "generate_scorecard",
    "optimize", "plot_heatmap",
    "generate_distribution_plots",
]
