
"""Example usage for the refactored event-driven simulator."""

from datetime import date

from data_layer import DataLayer, YahooFinanceProvider
from strategy import MeanReversion, SMACrossover
from backtester import (
    BacktestConfig,
    BlockBootstrapSource,
    EventDrivenBacktester,
    GANSource,
    GBMSource,
    RegimeSwitchingSource,
    run_scenario_suite,
)

# Historical data
dl = DataLayer()
dl.add_provider(YahooFinanceProvider())
df = dl.fetch("SPY", date(2022, 1, 1), date(2025, 1, 1))

config = BacktestConfig(
    initial_capital=100_000,
    commission_per_order=1.0,
    slippage_bps=2.0,
    spread_bps=1.0,
)

# 1) Event-driven replay on ordinary OHLCV bars
engine = EventDrivenBacktester(config)
result = engine.run(SMACrossover(10, 30), df)
print(result.summary())

# 2) Monte Carlo stress tests
gbm_source = GBMSource(n_bars=252)
bootstrap_source = BlockBootstrapSource(df, n_bars=252, block_size=5)
regime_source = RegimeSwitchingSource(n_bars=252)

for source in (gbm_source, bootstrap_source, regime_source):
    batch = run_scenario_suite(MeanReversion(), source, n_scenarios=50, config=config, engine="event")
    print("\n" + batch.summary())

# 3) GAN integration (assumes you already trained a generator elsewhere)
def fake_gan_generator(seed=None):
    # Replace this with: your_model.sample(seed=seed) -> DataFrame(Open,High,Low,Close,Volume)
    return bootstrap_source.generate(seed=seed)

gan_source = GANSource(generator=fake_gan_generator)
gan_batch = run_scenario_suite(SMACrossover(10, 30), gan_source, n_scenarios=20, config=config, engine="event")
print("\n" + gan_batch.summary())
