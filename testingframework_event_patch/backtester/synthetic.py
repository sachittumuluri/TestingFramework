
"""Synthetic data generators and scenario sources for simulation stress tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from backtester.feeds import normalize_ohlcv
from backtester.models import BacktestConfig, SimulationBatchResult


def _build_ohlcv_from_close(
    close: np.ndarray,
    rng: np.random.Generator,
    volume: Optional[np.ndarray] = None,
    start: str = "2020-01-01",
    extra: Optional[dict] = None,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="B")
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    spread = np.maximum(0.001, np.abs(close - open_))
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.8, size=len(close)) * np.maximum(spread, 0.25)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.8, size=len(close)) * np.maximum(spread, 0.25)

    if volume is None:
        volume = rng.integers(5_000, 25_000, size=len(close))

    data = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume.astype(int),
    }
    if extra:
        data.update(extra)
    return pd.DataFrame(data, index=idx)


def make_oscillating(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0 + 5.0 * np.sin(np.linspace(0, 40 * np.pi, n))
    close = base + rng.normal(0.0, 0.3, size=n)
    return _build_ohlcv_from_close(close, rng)


def make_trending(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.linspace(0.0, 80.0, n) + rng.normal(0.0, 0.5, size=n)
    return _build_ohlcv_from_close(close, rng)


def make_random_walk(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    return _build_ohlcv_from_close(close, rng)


@dataclass
class GBMSource:
    n_bars: int = 252
    start_price: float = 100.0
    drift: float = 0.08
    volatility: float = 0.20
    volume_level: int = 20_000
    name: str = "GBM Monte Carlo"
    symbol: str = "SIM"

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        shocks = rng.normal(
            loc=(self.drift - 0.5 * self.volatility ** 2) * dt,
            scale=self.volatility * np.sqrt(dt),
            size=self.n_bars,
        )
        close = self.start_price * np.exp(np.cumsum(shocks))
        volume = rng.integers(int(0.7 * self.volume_level), int(1.3 * self.volume_level), size=self.n_bars)
        return _build_ohlcv_from_close(close, rng, volume=volume)


@dataclass
class BlockBootstrapSource:
    historical_df: pd.DataFrame
    n_bars: Optional[int] = None
    block_size: int = 5
    name: str = "Block Bootstrap"
    symbol: str = "SIM"

    def __post_init__(self) -> None:
        self.historical_df = normalize_ohlcv(self.historical_df, keep_extra=True)
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = self.historical_df
        target_bars = self.n_bars or len(df)
        close = df["Close"].values
        rets = np.diff(np.log(close))
        if len(rets) < self.block_size:
            raise ValueError("historical_df is too short for the chosen block_size")

        vol = df["Volume"].iloc[1:].values
        ranges_up = (df["High"].iloc[1:].values - np.maximum(df["Open"].iloc[1:].values, df["Close"].iloc[1:].values))
        ranges_dn = (np.minimum(df["Open"].iloc[1:].values, df["Close"].iloc[1:].values) - df["Low"].iloc[1:].values)

        sampled_rets = []
        sampled_vol = []
        sampled_up = []
        sampled_dn = []
        while len(sampled_rets) < target_bars:
            start = rng.integers(0, len(rets) - self.block_size + 1)
            end = start + self.block_size
            sampled_rets.extend(rets[start:end])
            sampled_vol.extend(vol[start:end])
            sampled_up.extend(ranges_up[start:end])
            sampled_dn.extend(ranges_dn[start:end])

        sampled_rets = np.array(sampled_rets[:target_bars])
        sampled_vol = np.array(sampled_vol[:target_bars])
        sampled_up = np.array(sampled_up[:target_bars])
        sampled_dn = np.array(sampled_dn[:target_bars])

        start_price = float(df["Close"].iloc[0])
        close_path = start_price * np.exp(np.cumsum(sampled_rets))
        open_ = np.roll(close_path, 1)
        open_[0] = start_price
        high = np.maximum(open_, close_path) + np.abs(sampled_up)
        low = np.minimum(open_, close_path) - np.abs(sampled_dn)

        out = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close_path, "Volume": sampled_vol.astype(int)},
            index=pd.date_range(df.index[0], periods=target_bars, freq="B"),
        )
        return out


@dataclass
class RegimeSwitchingSource:
    n_bars: int = 252
    start_price: float = 100.0
    regimes: Optional[Sequence[dict]] = None
    transition_matrix: Optional[np.ndarray] = None
    initial_regime: int = 0
    name: str = "Regime Switching Monte Carlo"
    symbol: str = "SIM"

    def __post_init__(self) -> None:
        if self.regimes is None:
            self.regimes = (
                {"name": "bull", "drift": 0.18, "vol": 0.14},
                {"name": "sideways", "drift": 0.02, "vol": 0.08},
                {"name": "bear", "drift": -0.20, "vol": 0.24},
            )
        if self.transition_matrix is None:
            self.transition_matrix = np.array(
                [
                    [0.92, 0.06, 0.02],
                    [0.08, 0.84, 0.08],
                    [0.03, 0.07, 0.90],
                ],
                dtype=float,
            )
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=float)
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("transition_matrix must be square")
        if self.transition_matrix.shape[0] != len(self.regimes):
            raise ValueError("transition_matrix size must match number of regimes")

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        regime_idx = []
        current = int(self.initial_regime)
        for _ in range(self.n_bars):
            regime_idx.append(current)
            current = int(rng.choice(len(self.regimes), p=self.transition_matrix[current]))

        log_returns = np.empty(self.n_bars)
        regime_labels = []
        for i, ridx in enumerate(regime_idx):
            regime = self.regimes[ridx]
            regime_labels.append(regime.get("name", f"regime_{ridx}"))
            mu = regime["drift"]
            sigma = regime["vol"]
            log_returns[i] = rng.normal(
                loc=(mu - 0.5 * sigma ** 2) * dt,
                scale=sigma * np.sqrt(dt),
            )

        close = self.start_price * np.exp(np.cumsum(log_returns))
        volume = np.array([rng.integers(10_000, 40_000) for _ in range(self.n_bars)])
        return _build_ohlcv_from_close(close, rng, volume=volume, extra={"Regime": regime_labels})


@dataclass
class NoiseInjectionSource:
    base_df: pd.DataFrame
    price_noise_std: float = 0.003
    volume_noise_std: float = 0.10
    name: str = "Noise Injection"
    symbol: str = "SIM"

    def __post_init__(self) -> None:
        self.base_df = normalize_ohlcv(self.base_df, keep_extra=True)

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = self.base_df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * (1 + rng.normal(0.0, self.price_noise_std, size=len(df)))
        df["High"] = np.maximum(df["High"], df[["Open", "Close"]].max(axis=1))
        df["Low"] = np.minimum(df["Low"], df[["Open", "Close"]].min(axis=1))
        df["Volume"] = np.maximum(
            0,
            np.round(df["Volume"] * (1 + rng.normal(0.0, self.volume_noise_std, size=len(df)))),
        ).astype(int)
        return df


class GANSource:
    """
    Adapter for GAN-generated synthetic bars.

    Supply either:
    - a callable: generator(seed) -> DataFrame
    - or a sequence of pre-generated DataFrames.
    """

    def __init__(
        self,
        generator: Optional[Callable[[Optional[int]], pd.DataFrame]] = None,
        scenarios: Optional[Sequence[pd.DataFrame]] = None,
        name: str = "GAN Synthetic",
        symbol: str = "SIM",
    ):
        if generator is None and not scenarios:
            raise ValueError("Provide either a generator callable or scenarios")
        self.generator = generator
        self.scenarios = list(scenarios) if scenarios is not None else None
        self.name = name
        self.symbol = symbol

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        if self.generator is not None:
            df = self.generator(seed)
            return normalize_ohlcv(df, keep_extra=True)
        if self.scenarios is None:
            raise RuntimeError("No scenarios available")
        idx = 0 if seed is None else int(seed) % len(self.scenarios)
        return normalize_ohlcv(self.scenarios[idx], keep_extra=True)


def run_validation_suite(strategy, config: Optional[BacktestConfig] = None, engine: str = "bar"):
    """Run the strategy against three canonical synthetic datasets."""
    if engine == "bar":
        from backtester.bar_engine import BarBacktester as _Engine
    elif engine == "event":
        from backtester.event_engine import EventDrivenBacktester as _Engine
    else:
        raise ValueError("engine must be 'bar' or 'event'")

    cfg = config or BacktestConfig()
    tester = _Engine(config=cfg)

    datasets = {
        "Oscillating (mean-reverting)": make_oscillating(),
        "Trending (strong uptrend)": make_trending(),
        "Random Walk (no edge)": make_random_walk(),
    }

    print(f"Validation Suite [{engine}]: {strategy.name}")
    print("=" * 72)
    results = {}
    for name, data in datasets.items():
        result = tester.run(strategy, data)
        results[name] = result
        print(
            f"\n{name}:\n"
            f"  Return: {result.total_return_pct:+.2f}% | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Trades: {result.num_trades} | "
            f"Win rate: {result.win_rate:.0f}% | "
            f"Max DD: {result.max_drawdown_pct:.2f}%"
        )
    return results


def run_scenario_suite(
    strategy,
    scenario_source,
    n_scenarios: int = 100,
    config: Optional[BacktestConfig] = None,
    engine: str = "event",
) -> SimulationBatchResult:
    """Run repeated Monte Carlo / GAN / bootstrap scenarios."""
    cfg = config or BacktestConfig()
    if engine == "event":
        from backtester.event_engine import EventDrivenBacktester as _Engine
    elif engine == "bar":
        from backtester.bar_engine import BarBacktester as _Engine
    else:
        raise ValueError("engine must be 'bar' or 'event'")

    tester = _Engine(config=cfg)
    if not hasattr(tester, "run_scenarios"):
        # Bar engine doesn't have a native batch helper.
        import copy

        results = []
        for i in range(n_scenarios):
            df = scenario_source.generate(seed=i)
            result = tester.run(copy.deepcopy(strategy), df)
            result.metadata["scenario_id"] = i
            results.append(result)
        metrics_df = pd.DataFrame(
            [
                {
                    "scenario_id": r.metadata["scenario_id"],
                    "total_return_pct": r.total_return_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "num_trades": r.num_trades,
                    "turnover": r.turnover,
                }
                for r in results
            ]
        )
        summary = metrics_df.drop(columns=["scenario_id"]).describe().T
        return SimulationBatchResult(
            source_name=getattr(scenario_source, "name", scenario_source.__class__.__name__),
            results=results,
            metrics_frame=metrics_df,
            summary_frame=summary,
        )
    return tester.run_scenarios(strategy, scenario_source, n_scenarios=n_scenarios)
