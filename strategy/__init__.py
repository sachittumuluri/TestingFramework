"""
Strategy module — base interface and strategies for the Testing Framework.
"""

from strategy.base import Strategy, Signal, Trade, Fill, Order, StrategyState
from strategy.sma_crossover import SMACrossover
from strategy.mean_reversion import MeanReversion
from strategy.helpers import crossover, cross, barssince, quantile

# Registry of all available strategies
STRATEGIES = [SMACrossover, MeanReversion]

def list_strategies():
    """Print all available strategies."""
    print("Available strategies:")
    for cls in STRATEGIES:
        doc = cls.__doc__.strip().splitlines()[0] if cls.__doc__ else "No description"
        print(f"  - {cls.__name__}: {doc}")

__all__ = [
    "Strategy", "Signal", "Trade", "Fill", "Order", "StrategyState",
    "SMACrossover", "MeanReversion",
    "STRATEGIES", "list_strategies",
    "crossover", "cross", "barssince", "quantile",
]
