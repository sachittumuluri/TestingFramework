import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch


def generate_sequences(generator, n_sequences, seq_len, noise_dim,
                       regime_label=0, device='cpu'):
    """
    Sample synthetic return sequences for a SPECIFIC regime.

    CHANGED from original: now accepts regime_label parameter.

    This is the core INFERENCE function — after training, this is how you
    create regime-specific synthetic data for backtesting, stress-testing, etc.

    Example usage:
        # Generate 1000 "Crash" sequences for VaR estimation
        crash_data = generate_sequences(gen, 1000, 50, 32, regime_label=3)

        # Generate 1000 "Bullish" sequences for optimistic scenario analysis
        bull_data = generate_sequences(gen, 1000, 50, 32, regime_label=0)

    Args:
        generator:     Trained Generator network (will be set to eval mode)
        n_sequences:   How many sequences to generate
        seq_len:       Length of each sequence (must match training)
        noise_dim:     Noise dimension (must match training)
        regime_label:  Integer label of the desired regime
                       (e.g., 0=Bullish, 3=Crash with default mapping)
        device:        torch device

    Returns:
        numpy array of shape (n_sequences, seq_len)
        Each row is one sequence of normalized returns for the specified regime.
    """
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(n_sequences, seq_len, noise_dim, device=device)
        labels = torch.full(
            (n_sequences,), regime_label, dtype=torch.long, device=device
        )

        fake = generator(noise, labels)

    generator.train()

    return fake.squeeze(-1).cpu().numpy()



def generate_all_regimes(generator, n_per_regime, seq_len, noise_dim,
                          num_regimes=4, regime_names=None, device='cpu'):
    """
    This produces a dictionary of regime → sequences, making it easy to
    compare across regimes in a single call.

    Args:
        generator:      Trained Generator network
        n_per_regime:   Sequences per regime
        seq_len:        Sequence length
        noise_dim:      Noise dimension
        num_regimes:    Number of regimes
        regime_names:   Dict {label_int: name_str} for display
        device:         torch device

    Returns:
        Dict {regime_label: numpy_array of shape (n_per_regime, seq_len)}
    """
    if regime_names is None:
        regime_names = {i: f"Regime_{i}" for i in range(num_regimes)}

    results = {}
    for label in range(num_regimes):
        sequences = generate_sequences(
            generator, n_per_regime, seq_len, noise_dim,
            regime_label=label, device=device
        )
        results[label] = sequences
        print(f"  Generated {n_per_regime} sequences for "
              f"{regime_names.get(label, f'Regime_{label}')}")

    return results


def compute_acf(series, max_lag=50):
    """
    Compute the sample autocorrelation function (ACF) of a 1D series.

    ACF(k) = Cov(x_t, x_{t+k}) / Var(x_t)
    """
    n = len(series)
    if n < max_lag + 1:
        max_lag = n - 1

    mean = np.mean(series)
    var = np.var(series)

    if var < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    centered = series - mean

    for k in range(max_lag + 1):
        acf[k] = np.sum(centered[:n - k] * centered[k:]) / (n * var)

    return acf


def compute_statistics(returns):
    """
    Compute summary statistics for a return series.
    """
    return {
        'mean':           float(np.mean(returns)),
        'std':            float(np.std(returns)),
        'skewness':       float(stats.skew(returns)),
        'kurtosis':       float(stats.kurtosis(returns)),
        'min':            float(np.min(returns)),
        'max':            float(np.max(returns)),
        'JB_stat':        float(stats.jarque_bera(returns).statistic),
        'JB_pvalue':      float(stats.jarque_bera(returns).pvalue),
    }


def compare_statistics(real_returns, fake_returns):
    """
    Print a side-by-side comparison table.
    """
    real_stats = compute_statistics(real_returns)
    fake_stats = compute_statistics(fake_returns)

    print(f"\n{'Statistic':<20} {'Real':>12} {'Generated':>12} {'Ratio':>10}")
    print("─" * 56)

    for key in real_stats:
        r = real_stats[key]
        f = fake_stats[key]
        ratio = f / r if abs(r) > 1e-10 else float('nan')
        print(f"  {key:<18} {r:>12.5f} {f:>12.5f} {ratio:>10.3f}")

    print()


def compare_regime_statistics(real_regime_data, fake_regime_data, regime_names=None):
    """
    This is the key quantitative test for conditional generation quality.
    For each regime, we check if the generated data matches the real data's
    statistical properties.

    What to look for:
      - Each regime should have ratios close to 1.0
      - Crash should have higher std, kurtosis than Bullish
      - Bullish should have positive mean, Crash should have negative mean

    Args:
        real_regime_data:  Dict {regime_label: 2D numpy array of real windows}
        fake_regime_data:  Dict {regime_label: 2D numpy array of fake windows}
        regime_names:      Dict {label: name} for display
    """
    if regime_names is None:
        regime_names = {k: f"Regime_{k}" for k in real_regime_data}

    for label in sorted(real_regime_data.keys()):
        if label not in fake_regime_data:
            continue

        name = regime_names.get(label, f"Regime_{label}")
        print(f"\n{'=' * 60}")
        print(f"  Regime: {name} (label={label})")
        print(f"{'=' * 60}")

        real_flat = real_regime_data[label].flatten()
        fake_flat = fake_regime_data[label].flatten()

        compare_statistics(real_flat, fake_flat)


def plot_evaluation(real_returns, fake_returns, title_prefix="", save_path=None):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{title_prefix}Real vs Generated Returns Comparison',
        fontsize=14, fontweight='bold', y=1.02
    )

    # Panel 1: Distribution
    ax = axes[0, 0]
    n_bins = 100
    combined = np.concatenate([real_returns, fake_returns])
    bin_edges = np.linspace(np.percentile(combined, 0.5),
                            np.percentile(combined, 99.5), n_bins)

    ax.hist(real_returns, bins=bin_edges, density=True, alpha=0.6,
            label='Real', color='steelblue')
    ax.hist(fake_returns, bins=bin_edges, density=True, alpha=0.6,
            label='Generated', color='coral')

    x_range = np.linspace(bin_edges[0], bin_edges[-1], 300)
    gaussian_pdf = stats.norm.pdf(x_range, loc=np.mean(real_returns),
                                  scale=np.std(real_returns))
    ax.plot(x_range, gaussian_pdf, 'k--', alpha=0.4, linewidth=1,
            label='Gaussian reference')

    ax.set_title('Return Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    # Panel 2: QQ-Plot
    ax = axes[0, 1]
    n_quantiles = min(1000, len(real_returns), len(fake_returns))
    probs = np.linspace(0.001, 0.999, n_quantiles)
    real_q = np.quantile(real_returns, probs)
    fake_q = np.quantile(fake_returns, probs)

    ax.scatter(real_q, fake_q, s=2, alpha=0.5, color='purple')
    q_min = min(real_q.min(), fake_q.min())
    q_max = max(real_q.max(), fake_q.max())
    ax.plot([q_min, q_max], [q_min, q_max], 'k--', alpha=0.5, linewidth=1)

    ax.set_title('QQ-Plot (Real vs Generated)')
    ax.set_xlabel('Real Quantiles')
    ax.set_ylabel('Generated Quantiles')
    ax.set_aspect('equal')

    # Panel 3: ACF of Returns
    ax = axes[1, 0]
    max_lag = min(40, len(real_returns) // 10)

    real_acf = compute_acf(real_returns, max_lag)
    fake_acf = compute_acf(fake_returns, max_lag)
    lags = np.arange(1, max_lag + 1)

    bar_width = 0.35
    ax.bar(lags - bar_width/2, real_acf[1:], width=bar_width,
           alpha=0.7, label='Real', color='steelblue')
    ax.bar(lags + bar_width/2, fake_acf[1:], width=bar_width,
           alpha=0.7, label='Generated', color='coral')

    n_eff = len(real_returns)
    ci = 1.96 / np.sqrt(n_eff)
    ax.axhline(y=ci, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=-ci, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_title('Autocorrelation of Returns')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.legend(fontsize=9)

    # Panel 4: ACF of Squared Returns (volatility clustering)
    ax = axes[1, 1]
    real_sq_acf = compute_acf(real_returns ** 2, max_lag)
    fake_sq_acf = compute_acf(fake_returns ** 2, max_lag)

    ax.bar(lags - bar_width/2, real_sq_acf[1:], width=bar_width,
           alpha=0.7, label='Real', color='steelblue')
    ax.bar(lags + bar_width/2, fake_sq_acf[1:], width=bar_width,
           alpha=0.7, label='Generated', color='coral')
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_title('ACF of Squared Returns  (Volatility Clustering)')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF of r²')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved → {save_path}")
    plt.show()


def plot_regime_comparison(fake_regime_data, regime_names=None, save_path=None):
    """

    This is the key VISUAL test for conditional generation:
      - Crash columns should show plunging cumulative returns
      - Bullish columns should show rising cumulative returns
      - Sideways columns should meander around zero
      - Bearish columns should drift downward

    Args:
        fake_regime_data: Dict {regime_label: 2D numpy array (n_sequences, seq_len)}
        regime_names:     Dict {label: name}
        save_path:        Optional path to save the figure
    """
    if regime_names is None:
        regime_names = {k: f"Regime_{k}" for k in fake_regime_data}

    n_regimes = len(fake_regime_data)
    n_samples = 3 

    fig, axes = plt.subplots(
        n_samples, n_regimes, figsize=(4 * n_regimes, 2.5 * n_samples)
    )
    if n_regimes == 1:
        axes = axes.reshape(-1, 1)
    if n_samples == 1:
        axes = axes.reshape(1, -1)


    colors = ['forestgreen', 'crimson', 'gray', 'darkred']

    for col, label in enumerate(sorted(fake_regime_data.keys())):
        data = fake_regime_data[label]
        name = regime_names.get(label, f"Regime_{label}")
        color = colors[col % len(colors)]

        for row in range(min(n_samples, len(data))):
            ax = axes[row, col]

            cumulative = np.cumsum(data[row])
            ax.plot(cumulative, color=color, linewidth=0.8)
            ax.axhline(y=0, color='black', linewidth=0.3, linestyle='--')
            ax.set_ylabel('Cum. return', fontsize=7)
            if row == 0:
                ax.set_title(name, fontweight='bold', fontsize=10)

        axes[-1, col].set_xlabel('Timestep')

    plt.suptitle('Generated Sequences by Regime (Cumulative Returns)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved → {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training metrics over epochs.  (UNCHANGED from original)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Wasserstein distance ---
    ax = axes[0]
    ax.plot(history['wasserstein_dist'], color='green', linewidth=0.8)
    ax.set_title('Estimated Wasserstein Distance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W-distance')
    ax.grid(True, alpha=0.3)

    # --- Losses ---
    ax = axes[1]
    ax.plot(history['critic_loss'], label='Critic', color='steelblue',
            linewidth=0.8, alpha=0.8)
    ax.plot(history['generator_loss'], label='Generator', color='coral',
            linewidth=0.8, alpha=0.8)
    ax.set_title('Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Gradient penalty ---
    ax = axes[2]
    ax.plot(history['gradient_penalty'], color='purple', linewidth=0.8)
    ax.set_title('Gradient Penalty')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('GP')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sample_sequences(real_windows, fake_windows, n_samples=4, save_path=None):
    """
    Side-by-side comparison of individual real and generated sequences.
    """
    n_samples = min(n_samples, len(real_windows), len(fake_windows))

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_samples):
        ax = axes[i, 0]
        ax.plot(real_windows[i], color='steelblue', linewidth=0.8)
        ax.axhline(y=0, color='black', linewidth=0.3)
        ax.set_ylabel('Return', fontsize=8)
        if i == 0:
            ax.set_title('Real', fontweight='bold')

        ax = axes[i, 1]
        ax.plot(fake_windows[i], color='coral', linewidth=0.8)
        ax.axhline(y=0, color='black', linewidth=0.3)
        if i == 0:
            ax.set_title('Generated', fontweight='bold')

    axes[-1, 0].set_xlabel('Timestep')
    axes[-1, 1].set_xlabel('Timestep')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()