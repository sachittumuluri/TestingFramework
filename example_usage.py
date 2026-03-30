"""
Quick start — run a strategy through the full pipeline.

Just run: python3 example_usage.py
"""

from datetime import date
from data_layer import DataLayer, YahooFinanceProvider
from strategy import SMACrossover
from backtester import (
    Backtester, BacktestConfig, generate_scorecard,
    optimize, plot_heatmap, run_validation_suite,
    generate_distribution_plots,
)

# 1. Get data (Yahoo Finance — no API key needed)
dl = DataLayer()
dl.add_provider(YahooFinanceProvider())

df = dl.fetch("SPY", date(2022, 1, 1), date(2025, 12, 31))
print(f"Data: {len(df)} bars\n")

# 2. Backtest
config = BacktestConfig(
    initial_capital=100_000,
    commission_per_order=1.00,
    slippage_bps=2.0,
)
result = Backtester(config).run(SMACrossover(10, 30), df)
print(result.summary())

# 3. Optimize (Optuna — smart Bayesian search)
print("\n" + "=" * 50)
best, params, results = optimize(
    SMACrossover, df, config,
    maximize="sharpe_ratio",
    method="optuna",
    n_trials=30,
    constraint=lambda p: p["fast_period"] < p["slow_period"],
    fast_period=[5, 10, 15, 20],
    slow_period=[20, 30, 40, 50, 60],
)
print(f"\nOptimal strategy: SMA Crossover ({params['fast_period']}/{params['slow_period']})")
print(f"Sharpe: {best.sharpe_ratio:.2f}  |  Return: {best.total_return_pct:+.1f}%")

# 4. Validate on synthetic data
print("\n" + "=" * 50)
run_validation_suite(SMACrossover(**params), config=config)

# 5. Generate scorecard with the optimized params
print()
generate_scorecard(
    strategy=SMACrossover(**params),
    df=df,
    config=config,
    output_path="scorecard.png",
)
