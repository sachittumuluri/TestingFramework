"""
Parameter optimization for backtesting strategies.

Two methods:
  - "grid": Exhaustive grid search — tries every combination. Best for small spaces.
  - "optuna": Bayesian optimization — learns from results, focuses on promising
    regions. Best for large spaces. Supports early pruning.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from backtester.engine import Backtester, BacktestConfig, BacktestResult
from strategy.base import Strategy


# ─────────────────────────────────────────────────────────────
# Worker function (must be top-level for pickling)
# ─────────────────────────────────────────────────────────────

def _run_single(args: Tuple) -> Tuple[Dict[str, Any], Optional[BacktestResult]]:
    strategy_class, params, df, config = args
    try:
        strategy = strategy_class(**params)
        bt = Backtester(config)
        result = bt.run(strategy, df)
        return params, result
    except Exception as exc:
        print(f"  [FAIL] params={params}: {exc}", file=sys.stderr)
        return params, None


# ─────────────────────────────────────────────────────────────
# optimize()
# ─────────────────────────────────────────────────────────────

def optimize(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    maximize: Union[str, Callable[[BacktestResult], float]] = "sharpe_ratio",
    constraint: Optional[Callable[[Dict[str, Any]], bool]] = None,
    method: str = "grid",
    n_trials: int = 100,
    max_workers: Optional[int] = None,
    **param_ranges: List[Any],
) -> Tuple[BacktestResult, Dict[str, Any], pd.DataFrame]:
    """
    Optimize strategy parameters.

    Parameters
    ----------
    strategy_class : Type[Strategy]
        The Strategy subclass to instantiate.
    df : pd.DataFrame
        OHLCV data to backtest against.
    config : BacktestConfig, optional
        Backtest configuration.
    maximize : str or callable
        Metric name (attribute of BacktestResult) or callable(result) -> float.
    constraint : callable, optional
        function(params_dict) -> bool. Return True to keep, False to skip.
    method : str
        "grid" for exhaustive search, "optuna" for Bayesian optimization.
    n_trials : int
        Number of trials for Optuna (ignored for grid). Default 100.
    max_workers : int, optional
        Parallel workers for grid search. Defaults to cpu_count - 1.
    **param_ranges
        Each value is a list of values to try.
        Example: fast_period=[5,10,15], slow_period=[20,30,40]

    Returns
    -------
    (best_result, best_params, all_results_df)
    """
    if not param_ranges:
        raise ValueError("No parameter ranges provided.")

    config = config or BacktestConfig()

    if isinstance(maximize, str):
        metric_name = maximize
        def score_fn(result: BacktestResult) -> float:
            return getattr(result, metric_name)
    else:
        score_fn = maximize
        metric_name = getattr(maximize, "__name__", "custom_metric")

    if method == "grid":
        return _optimize_grid(
            strategy_class, df, config, score_fn, metric_name,
            constraint, max_workers, param_ranges,
        )
    elif method == "optuna":
        return _optimize_optuna(
            strategy_class, df, config, score_fn, metric_name,
            constraint, n_trials, param_ranges,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'grid' or 'optuna'.")


# ─────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────

def _optimize_grid(
    strategy_class, df, config, score_fn, metric_name,
    constraint, max_workers, param_ranges,
):
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combos = [dict(zip(param_names, vals)) for vals in itertools.product(*param_values)]

    if constraint is not None:
        orig = len(all_combos)
        all_combos = [c for c in all_combos if constraint(c)]
        print(f"  Constraint filtered: {orig} -> {len(all_combos)} combos")

    total = len(all_combos)
    if total == 0:
        raise ValueError("No valid parameter combinations after constraints.")

    print(f"Optimizing {strategy_class.__name__} [grid] over {total} combinations")
    print(f"  Maximizing: {metric_name}")

    worker_args = [(strategy_class, params, df, config) for params in all_combos]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    results: List[Tuple[Dict, Optional[BacktestResult]]] = []
    start_time = time.time()

    if max_workers == 1 or total == 1:
        for i, args in enumerate(worker_args, 1):
            results.append(_run_single(args))
            _print_progress(i, total, start_time)
    else:
        print(f"  Workers: {max_workers}")
        with multiprocessing.Pool(processes=max_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_single, worker_args), 1):
                results.append(result)
                _print_progress(i, total, start_time)

    print()
    return _collect_results(results, score_fn, metric_name, start_time)


# ─────────────────────────────────────────────────────────────
# Optuna (Bayesian optimization)
# ─────────────────────────────────────────────────────────────

def _optimize_optuna(
    strategy_class, df, config, score_fn, metric_name,
    constraint, n_trials, param_ranges,
):
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for method='optuna'. Install it: pip install optuna"
        )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Optimizing {strategy_class.__name__} [optuna] for {n_trials} trials")
    print(f"  Maximizing: {metric_name}")
    print(f"  Parameters: {', '.join(f'{k}={v}' for k, v in param_ranges.items())}")

    # Store all results for the DataFrame
    all_results: List[Tuple[Dict, Optional[BacktestResult]]] = []
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, values in param_ranges.items():
            # Optuna picks from the categorical list of values
            params[name] = trial.suggest_categorical(name, values)

        # Apply constraint
        if constraint is not None and not constraint(params):
            raise optuna.TrialPruned()

        try:
            strategy = strategy_class(**params)
            bt = Backtester(config)
            result = bt.run(strategy, df)
            all_results.append((params, result))

            score = score_fn(result)
            _print_progress(len(all_results), n_trials, start_time)
            return score

        except Exception as exc:
            all_results.append((params, None))
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print()

    # Get best params from Optuna
    best_params = dict(study.best_params)

    # Run the best one more time to get the full result
    # (or find it in our stored results)
    best_result = None
    for params, result in all_results:
        if result is not None and params == best_params:
            best_result = result
            break

    if best_result is None:
        # Re-run best params
        strategy = strategy_class(**best_params)
        bt = Backtester(config)
        best_result = bt.run(strategy, df)

    # Build results DataFrame from all trials
    return _collect_results(all_results, score_fn, metric_name, start_time)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _collect_results(
    results: List[Tuple[Dict, Optional[BacktestResult]]],
    score_fn, metric_name, start_time,
):
    rows = []
    scored: List[Tuple[float, Dict, BacktestResult]] = []

    for params, bt_result in results:
        row = dict(params)
        if bt_result is not None:
            score = score_fn(bt_result)
            row["_score"] = score
            row["sharpe_ratio"] = bt_result.sharpe_ratio
            row["sortino_ratio"] = bt_result.sortino_ratio
            row["calmar_ratio"] = bt_result.calmar_ratio
            row["sqn"] = bt_result.sqn
            row["total_return_pct"] = bt_result.total_return_pct
            row["annual_return_pct"] = bt_result.annual_return_pct
            row["max_drawdown_pct"] = bt_result.max_drawdown_pct
            row["volatility_annual_pct"] = bt_result.volatility_annual_pct
            row["num_trades"] = bt_result.num_trades
            row["profit_factor"] = bt_result.profit_factor
            row["expectancy"] = bt_result.expectancy
            row["avg_trade_return_pct"] = bt_result.avg_trade_return_pct
            scored.append((score, params, bt_result))
        else:
            row["_score"] = float("-inf")
        rows.append(row)

    all_df = pd.DataFrame(rows)
    all_df.sort_values("_score", ascending=False, inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    if not scored:
        raise RuntimeError("All parameter combinations failed.")

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, best_result = scored[0]

    elapsed = time.time() - start_time
    print(f"Optimization complete in {elapsed:.1f}s")
    print(f"  Best {metric_name}: {best_score:.4f}")
    print(f"  Best params: {best_params}")

    return best_result, best_params, all_df


def _print_progress(current: int, total: int, start_time: float):
    elapsed = time.time() - start_time
    pct = current / total * 100
    if total > 1 and elapsed > 0:
        eta = elapsed / current * (total - current)
        print(f"\r  Progress: {current}/{total} ({pct:.0f}%) | "
              f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", end="", flush=True)


# ─────────────────────────────────────────────────────────────
# plot_heatmap()
# ─────────────────────────────────────────────────────────────

def plot_heatmap(
    all_results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe_ratio",
    title: Optional[str] = None,
    output_path: str = "heatmap.png",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdYlGn",
    annot: bool = True,
) -> str:
    """Generate a heatmap PNG showing metric values across two parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col in [param_x, param_y, metric]:
        if col not in all_results_df.columns:
            raise ValueError(f"'{col}' not found in results DataFrame.")

    pivot = all_results_df.pivot_table(
        index=param_y, columns=param_x, values=metric, aggfunc="mean"
    )

    try:
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot[sorted(pivot.columns)]
    except TypeError:
        pass

    fig, ax = plt.subplots(figsize=figsize)
    data = pivot.values
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    if title is None:
        title = f"{metric} by {param_x} vs {param_y}"
    ax.set_title(title)

    if annot:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = data[i, j]
                if np.isfinite(val):
                    val_range = np.nanmax(data) - np.nanmin(data)
                    if val_range > 0:
                        norm = (val - np.nanmin(data)) / val_range
                        text_color = "white" if norm < 0.3 or norm > 0.7 else "black"
                    else:
                        text_color = "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap saved to: {output_path}")
    return output_path
