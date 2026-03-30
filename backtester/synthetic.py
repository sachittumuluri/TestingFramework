"""
Synthetic data generators for strategy validation.

These create known market conditions so you can verify a strategy
behaves as expected:
  - Oscillating: mean reversion should profit
  - Trending: momentum should profit, mean reversion should struggle
  - Random walk: no strategy should consistently profit after costs
"""

import numpy as np
import pandas as pd


def make_oscillating(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """
    Oscillating price path — reverts to a mean.
    Mean reversion strategies should do well here.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100.0 + 5.0 * np.sin(np.linspace(0, 40 * np.pi, n))
    close = base + rng.normal(0.0, 0.3, size=n)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.7, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.7, size=n)
    volume = rng.integers(5_000, 15_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def make_trending(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """
    Strong uptrend — momentum strategies should do well,
    mean reversion should struggle.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, 80.0, n) + rng.normal(0.0, 0.1, size=n)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.7, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.7, size=n)
    volume = rng.integers(5_000, 15_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def make_random_walk(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """
    Pure random walk — no strategy should consistently profit
    after transaction costs. If one does, it's likely overfit.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.7, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.7, size=n)
    volume = rng.integers(5_000, 15_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def run_validation_suite(strategy, config=None):
    """
    Run a strategy against all three synthetic datasets and print results.

    This is a sanity check: does the strategy behave as expected
    under known conditions?
    """
    from backtester.engine import Backtester, BacktestConfig

    cfg = config or BacktestConfig()
    bt = Backtester(config=cfg)

    datasets = {
        "Oscillating (mean-reverting)": make_oscillating(),
        "Trending (strong uptrend)": make_trending(),
        "Random Walk (no edge)": make_random_walk(),
    }

    print(f"Validation Suite: {strategy.name}")
    print("=" * 65)

    results = {}
    for name, data in datasets.items():
        result = bt.run(strategy, data)
        results[name] = result
        print(f"\n  {name}:")
        print(f"    Return: {result.total_return_pct:+.2f}%  |  "
              f"Sharpe: {result.sharpe_ratio:.2f}  |  "
              f"Trades: {result.num_trades}  |  "
              f"Win rate: {result.win_rate:.0f}%  |  "
              f"Max DD: {result.max_drawdown_pct:.2f}%")

    return results
