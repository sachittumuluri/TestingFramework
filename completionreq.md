# Robust Testing Framework for Trading Strategies

## Overview

This project builds a shared, reusable testing framework that any member of the club can plug their trading strategy into and get back standardized, comparable results. Instead of everyone writing one-off backtesting scripts on QuantConnect with different date ranges and different metrics, this framework enforces consistency: same data, same evaluation pipeline, same output format.

The framework has three core pillars: Monte Carlo simulation for stress-testing, trade distribution analysis for evaluating probability of success, and multi-API backtesting for verifying results aren't dependent on a single data source. It targets US equities and is built entirely in Python.

---

## The Problem It Solves

Right now, everyone in the club tests their strategies independently. Someone backtests on QuantConnect using 2018–2022 data, someone else uses 2015–2020, and there's no way to compare results meaningfully. Each person writes their own testing code, picks their own metrics, and presents results that can't be evaluated against anyone else's.

This means overfitting goes undetected, survivorship bias creeps in, and there's no institutional knowledge that carries over semester to semester. When someone graduates, their testing code leaves with them.

This framework fixes that by giving the club a single, modular infrastructure where any strategy can be submitted, tested through a rigorous pipeline, and evaluated on a level playing field.

---

## Architecture

The framework is structured as a pipeline. A strategy goes in one end and a standardized report comes out the other.

### Strategy Interface

Every strategy must implement a common interface — a standard API that defines how the framework sends data to the strategy and how the strategy returns trading signals. This is what makes the system plug-and-play. Whether someone writes a simple moving average crossover or a complex ML model, it plugs into the same pipeline.

### Data Layer

The data layer abstracts away where market data comes from. It can pull US equities data from multiple API sources and serve it to the backtester in a consistent format. The key insight here is that if a strategy performs well on one data source but poorly on another, that's a red flag — the strategy might be fitting to quirks in the data rather than real market dynamics.

The data layer supports both historical OHLCV data for backtesting and can be extended to live feeds for paper trading.

### Backtesting Engine

The backtesting engine runs strategies against historical or synthetic data. It operates in two modes:

**Non-event-driven (bar-based):** All decisions happen at discrete time steps — daily bars, hourly bars, etc. This is simpler, faster to run, and sufficient for most swing trading and position trading strategies.

**Event-driven (simulated exchange):** A more granular mode that simulates an order book and processes events as they arrive. This is necessary for testing high-frequency or intraday strategies where execution timing and order flow matter.

Both modes include a configurable transaction cost model that accounts for commissions, slippage, and bid-ask spread. Transaction costs are a major source of real-world losses, so any strategy that only looks good without them isn't a real strategy.

### Monte Carlo Simulation

The Monte Carlo layer stress-tests strategies beyond what historical data alone can show. It generates synthetic data through several methods:

**Geometric Brownian Motion (GBM):** A fully stochastic process — pure randomness with drift and volatility parameters. No real strategy should perform well on GBM data. After transaction costs, the mean Sharpe ratio across many GBM runs should be negative. If a strategy shows positive performance on GBM, it's almost certainly overfitting to noise. This serves as a baseline sanity check.

**GAN-generated data:** Generative Adversarial Networks trained on real market data can produce synthetic price series that preserve the statistical properties of real markets (fat tails, volatility clustering, autocorrelation structure) while being entirely "new" data the strategy has never seen. This is a more rigorous test than GBM because the data is realistic but unseen.

**Noise injection:** A configurable noise layer that can be added on top of realistic data to test how gracefully a strategy degrades as conditions get noisier. A robust strategy should degrade smoothly; a brittle one will break suddenly.

For each Monte Carlo method, the framework runs a large number of simulations and computes distributional statistics: mean Sharpe ratio, variance of returns, drawdown distributions, and confidence intervals.

### Trade Distribution Analysis

After running a strategy through backtesting and Monte Carlo simulation, the framework computes a standardized set of performance metrics:

- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Win rate
- Profit & loss distribution
- Risk-adjusted return metrics

It also generates distribution visualizations — histograms and CDFs of trade outcomes — so users can visually inspect whether a strategy's return profile looks healthy or suspiciously concentrated.

The specific metrics can evolve over time. The important thing is that every strategy is evaluated on the same set, making cross-strategy comparison possible.

### Standardized Reporting

Every strategy that goes through the pipeline gets the same output format: a report with all metrics, distribution plots, Monte Carlo results, and cross-API validation flags. This means results are directly comparable across teams and across semesters.

---

## Validation — Testing the Testing Framework

A testing framework is only useful if it actually catches problems. To verify this, the project includes a suite of known strategies with predictable behavior:

**Mean reversion:** Should perform well on mean-reverting data and poorly on trending data. If the framework reports the opposite, something is broken in the pipeline.

**Arbitrage:** A strategy with a known mathematical edge. The framework should detect positive expected value and show that performance degrades predictably as transaction costs increase.

**HMM regime-switching:** A Hidden Markov Model that identifies market regimes (bullish, bearish, sideways) and adapts strategy accordingly — momentum in trending regimes, mean reversion in sideways markets. This tests whether the framework can handle adaptive, state-dependent strategies correctly.

If the framework produces sensible results for all three validation strategies across different data conditions, there's strong evidence it will produce trustworthy results for novel strategies too.

---

## Tech Stack

- **Language:** Python
- **Backtesting engines:** Backtrader / Zipline (evaluating both)
- **Scientific computing:** NumPy, SciPy, Pandas
- **Data:** US equities via multiple API sources (TBD), historical OHLCV
- **ML (for GAN + HMM components):** PyTorch or TensorFlow (TBD)

---

## Team

| Person | Responsibility |
|--------|---------------|
| Connor | Monte Carlo simulation — GBM, GANs, noise injection |
| Sachit | Backtesting engine, transaction costs, validation strategies (mean reversion, arbitrage, HMM) |
| Robbie | Framework architecture, strategy interface, pipeline orchestration, metrics engine, data layer, reporting |

---

## What Success Looks Like

The framework is successful if a new club member can take a trading strategy, plug it into the pipeline with minimal setup, and get back a comprehensive report that tells them whether their strategy is robust or overfitting — without writing any testing infrastructure themselves. The results should be reproducible, comparable to other strategies tested through the same framework, and carry over as institutional knowledge semester to semester.
