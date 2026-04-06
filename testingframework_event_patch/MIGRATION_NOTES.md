
# Event-driven patch notes

This patch is designed to drop into the existing repository with minimal disruption.

What changed:
- `backtester/engine.py` is now a thin compatibility wrapper around `backtester/bar_engine.py`.
- Shared components moved into:
  - `backtester/models.py`
  - `backtester/portfolio.py`
  - `backtester/metrics.py`
  - `backtester/tracking.py`
- New event-driven simulator:
  - `backtester/events.py`
  - `backtester/execution.py`
  - `backtester/event_engine.py`
- Synthetic / simulated scenario sources expanded in `backtester/synthetic.py`.

Recommended adoption order:
1. Copy the new `backtester/*.py` files into the repo.
2. Keep the existing `strategy/` and `data_layer/` packages unchanged.
3. Run ordinary OHLCV replay with `EventDrivenBacktester` first.
4. Compare bar vs event results on the same strategies.
5. Then add Monte Carlo / GAN scenario batches.

Important behavior notes:
- Attached SL/TP bracket orders are supported as resting OCO exits.
- Child bracket orders become active on later bars, not within the same bar they were created.
- Stop-limit orders are intentionally not implemented yet.
- `GANSource` is an adapter: it accepts pre-generated DataFrames or a callable that returns them.
