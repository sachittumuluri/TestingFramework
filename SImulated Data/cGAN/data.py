import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEFAULT_REGIME_MAP = {
    'Bullish':  0,
    'Bearish':  1,
    'Sideways': 2,
    'Crash':    3,
}

DEFAULT_REGIME_NAMES = {v: k for k, v in DEFAULT_REGIME_MAP.items()}


class ReturnSequenceDataset(Dataset):

    def __init__(self, csv_path, seq_len=50, price_col='Close', stride=1,
                 regime_label=0, global_mu=None, global_sigma=None):
        """
        Args:
            csv_path:      Path to CSV file with at least a price column.
            seq_len:       Number of returns per window (T). 50 ≈ 2 months.
            price_col:     Column name containing prices (default: 'Close').
            stride:        Gap between consecutive window starts.
            regime_label:  Integer label for ALL windows from this file.
                           All windows in one CSV share the same regime.
            global_mu:     If provided, use this mean instead of per-file mean.
                           Set this for global normalization across files.
            global_sigma:  If provided, use this std instead of per-file std.
        """
        super().__init__()
        self.seq_len = seq_len
        self.regime_label = regime_label

        df = pd.read_csv(csv_path, parse_dates=True)

        if price_col not in df.columns:
            raise ValueError(
                f"Column '{price_col}' not found. Available: {list(df.columns)}"
            )

        prices = df[price_col].dropna().values.astype(np.float64)

        if len(prices) < seq_len + 1:
            raise ValueError(
                f"Not enough data: {len(prices)} prices for seq_len={seq_len}. "
                f"Need at least {seq_len + 1}."
            )


        self.log_returns = np.diff(np.log(prices))

        self.mu = global_mu if global_mu is not None else float(np.mean(self.log_returns))
        self.sigma = global_sigma if global_sigma is not None else float(np.std(self.log_returns))

        if self.sigma < 1e-10:
            raise ValueError(
                "Returns have ~zero standard deviation. "
                "Check your data — prices might be constant."
            )

        self.normalized = (self.log_returns - self.mu) / self.sigma
        self.windows = []
        n = len(self.normalized)

        for start in range(0, n - seq_len + 1, stride):
            window = self.normalized[start : start + seq_len]
            self.windows.append(window)

        self.windows = np.array(self.windows, dtype=np.float32)

        self.labels = np.full(len(self.windows), regime_label, dtype=np.int64)

        # --- Diagnostics ---
        print(f"[Data] {csv_path}: {len(prices):,} prices → "
              f"{len(self.log_returns):,} returns → "
              f"{len(self.windows):,} windows  "
              f"(regime={regime_label}, μ={self.mu:.6f}, σ={self.sigma:.6f})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns (window_tensor, label_tensor):
            window_tensor: shape (seq_len, 1)  — one window of normalized returns
            label_tensor:  scalar LongTensor   — regime label (integer)
        """
        window = self.windows[idx]
        label = self.labels[idx]
        return (
            torch.FloatTensor(window).unsqueeze(-1),
            torch.tensor(label, dtype=torch.long),
        )

    def denormalize(self, data):

        return data * self.sigma + self.mu


class MultiRegimeDataset:


    def __init__(self, csv_label_pairs, seq_len=50, price_col='Close',
                 stride=1, custom_strides=None, regime_map=None):
        """
        Args:
            csv_label_pairs: List of (csv_path, regime_name) tuples.
                             Example: [('bull.csv', 'Bullish'), ('crash.csv', 'Crash')]
            seq_len:         Window length.
            price_col:       Column containing prices.
            stride:          Window stride.
            regime_map:      Dict mapping regime names to integer labels.
                             If None, uses DEFAULT_REGIME_MAP.
        """
        self.regime_map = regime_map or DEFAULT_REGIME_MAP
        self.regime_names = {v: k for k, v in self.regime_map.items()}
        self.seq_len = seq_len

        print("\n[MultiRegime] Computing global normalization statistics...")
        all_log_returns = []
        for csv_path, _ in csv_label_pairs:
            df = pd.read_csv(csv_path, parse_dates=True)
            prices = df[price_col].dropna().values.astype(np.float64)
            log_rets = np.diff(np.log(prices))
            all_log_returns.append(log_rets)

        pooled_returns = np.concatenate(all_log_returns)
        self.global_mu = float(np.mean(pooled_returns))
        self.global_sigma = float(np.std(pooled_returns))

        print(f"[MultiRegime] Global stats: μ={self.global_mu:.6f}, "
              f"σ={self.global_sigma:.6f}")
        print(f"[MultiRegime] Total returns pooled: {len(pooled_returns):,}")


        self.datasets = []
        self.regime_datasets = {}

        for csv_path, regime_name in csv_label_pairs:
            if regime_name not in self.regime_map:
                raise ValueError(
                    f"Unknown regime '{regime_name}'. "
                    f"Available: {list(self.regime_map.keys())}"
                )

            regime_label = self.regime_map[regime_name]

            this_stride = custom_strides.get(csv_path, stride)

            ds = ReturnSequenceDataset(
                csv_path=csv_path,
                seq_len=seq_len,
                price_col=price_col,
                stride=this_stride,
                regime_label=regime_label,
                global_mu=self.global_mu,
                global_sigma=self.global_sigma,
            )
            self.datasets.append(ds)
            self.regime_datasets[regime_label] = ds

        self.combined_dataset = ConcatDataset(self.datasets)

        total = len(self.combined_dataset)
        print(f"\n[MultiRegime] Combined dataset: {total:,} total windows")
        for label, ds in self.regime_datasets.items():
            name = self.regime_names.get(label, f"Regime_{label}")
            print(f"  {name} (label={label}): {len(ds):,} windows")

    def denormalize(self, data):
        """Denormalize using global statistics."""
        return data * self.global_sigma + self.global_mu


def auto_label_regime(log_returns, thresholds=None):
    """
    Automatically assign a regime label to a series of log returns based
    on its mean return and volatility.

    This implements a simple quadrant-based classification:

                          Low Volatility    High Volatility
        Positive mean:    Bullish           (Bullish w/ vol)
        Negative mean:    Sideways          Bearish
        Very neg mean:    ---               Crash

    Useful when you have one long CSV and want to automatically segment
    it into regime-labeled windows.

    Args:
        log_returns:  1D numpy array of log returns (e.g., one window)
        thresholds:   Dict with keys:
                        'mean_pos':   above this → positive mean
                        'mean_neg':   below this → negative mean
                        'vol_high':   above this → elevated vol
                        'vol_crash':  above this → extreme vol
                      If None, uses sensible defaults for daily returns.

    Returns:
        String regime name (e.g., 'Bullish', 'Crash')
    """
    if thresholds is None:
        thresholds = {
            'mean_pos':  0.0002,
            'mean_neg': -0.0005,
            'vol_high':  0.015,
            'vol_crash': 0.025,
        }

    mean_ret = np.mean(log_returns)
    volatility = np.std(log_returns)

    # Crash: very high volatility AND negative returns
    if volatility > thresholds['vol_crash'] and mean_ret < thresholds['mean_neg']:
        return 'Crash'

    # Bearish: high volatility OR significantly negative returns
    if mean_ret < thresholds['mean_neg'] or (
        volatility > thresholds['vol_high'] and mean_ret < thresholds['mean_pos']
    ):
        return 'Bearish'

    # Bullish: positive returns with moderate volatility
    if mean_ret > thresholds['mean_pos']:
        return 'Bullish'

    # Sideways: everything else (low drift, low-to-moderate vol)
    return 'Sideways'


def label_from_filename(csv_path):
    """
    Extract regime label from a filename by matching keywords.

    Examples:
        'data/SPY_crash_2020.csv'  → 'Crash'
        'tech_bullish_run.csv'     → 'Bullish'
        'sideways_market.csv'      → 'Sideways'
        'bear_2008.csv'            → 'Bearish'

    Returns None if no keyword is found — use this as a fallback
    before manual labeling.
    """
    name = csv_path.lower()
    keyword_map = {
        'crash':    'Crash',
        'bull':     'Bullish',
        'bear':     'Bearish',
        'sideways': 'Sideways',
        'lateral':  'Sideways',
        'flat':     'Sideways',
    }
    for keyword, regime in keyword_map.items():
        if keyword in name:
            return regime
    return None


def create_regime_dataloader(csv_label_pairs, seq_len=50, batch_size=64,
                              stride=1, custom_strides=None, shuffle=True, regime_map=None,
                              price_col='Close'):
    """
    One-liner to build a DataLoader for multi-regime conditional training.

    Returns BOTH the DataLoader and the MultiRegimeDataset object, because
    we need the dataset later for:
      - Denormalization parameters (global_mu, global_sigma)
      - Per-regime datasets for evaluation
      - Regime name mappings

    Args:
        csv_label_pairs: List of (csv_path, regime_name) tuples
        seq_len:         Window length
        batch_size:      Batch size
        stride:          Window stride
        shuffle:         Shuffle the dataloader
        regime_map:      Custom regime name → integer mapping
        price_col:       Column name with prices

    Returns:
        dataloader:      DataLoader yielding (window_batch, label_batch) tuples
        multi_dataset:   MultiRegimeDataset object
    """
    custom_strides = custom_strides or {}

    multi_ds = MultiRegimeDataset(
        csv_label_pairs=csv_label_pairs,
        seq_len=seq_len,
        price_col=price_col,
        stride=stride,
        custom_strides=custom_strides,
        regime_map=regime_map,
    )

    loader = DataLoader(
        multi_ds.combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return loader, multi_ds



def generate_regime_csvs(output_dir=".", n_days=2000, seed=42):
    """
    Generate four synthetic CSV files, one per market regime, for testing.

    Each file simulates a price series with characteristics matching
    the named regime:
      - Bullish:  positive drift (+15% annual), moderate vol (15% annual)
      - Bearish:  negative drift (-10% annual), elevated vol (25% annual)
      - Sideways: near-zero drift (+2% annual), low vol (10% annual)
      - Crash:    extreme negative drift (-40% annual), extreme vol (45% annual),
                  frequent large negative jumps

    IMPORTANT: These are SYNTHETIC test data — not a replacement for real
    market data.  In production, use real OHLCV from your data vendor.

    Args:
        output_dir:  Directory to write CSV files.
        n_days:      Trading days per regime file.
        seed:        Random seed for reproducibility.

    Returns:
        List of (csv_path, regime_name) tuples — ready for create_regime_dataloader.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    regime_params = {
        'Bullish':  ( 0.15,  0.15, 0.02, 1.0),
        'Bearish':  (-0.10,  0.25, 0.04, 1.5),
        'Sideways': ( 0.02,  0.10, 0.01, 0.5),
        'Crash':    (-0.40,  0.45, 0.10, 3.0),
    }

    dt = 1 / 252
    pairs = []

    for regime_name, (drift, vol_annual, jump_prob, jump_scale) in regime_params.items():
        base_vol = vol_annual * np.sqrt(dt)
        kappa = 0.1     # Vol mean-reversion speed

        returns = np.zeros(n_days)
        vol = np.full(n_days, base_vol)

        for t in range(1, n_days):
            # --- Mean-reverting stochastic volatility (Ornstein-Uhlenbeck) ---
            vol[t] = (vol[t - 1]
                      + kappa * (base_vol - vol[t - 1])
                      + 0.02 * base_vol * rng.randn())
            vol[t] = max(vol[t], base_vol * 0.1)

            returns[t] = drift * dt + vol[t] * rng.randn()

            if rng.rand() < jump_prob:
                if regime_name == 'Crash':
                    jump_sign = rng.choice([-1, -1, -1, 1])
                else:
                    jump_sign = rng.choice([-1, 1])
                returns[t] += jump_sign * vol[t] * rng.exponential(jump_scale)

        prices = 100.0 * np.exp(np.cumsum(returns))

        # Build OHLCV DataFrame
        df = pd.DataFrame({
            'Date': pd.date_range('2000-01-01', periods=n_days, freq='B'),
            'Open': np.roll(prices, 1),
            'High': prices * (1 + np.abs(rng.randn(n_days)) * 0.005),
            'Low':  prices * (1 - np.abs(rng.randn(n_days)) * 0.005),
            'Close': prices,
            'Volume': rng.randint(1_000_000, 10_000_000, size=n_days),
        })
        df.iloc[0, df.columns.get_loc('Open')] = 100.0

        csv_path = os.path.join(output_dir, f"synthetic_{regime_name.lower()}.csv")
        df.to_csv(csv_path, index=False)
        pairs.append((csv_path, regime_name))

        print(f"[Data] Generated {regime_name} → {csv_path}  "
              f"({n_days:,} days, drift={drift:+.2f}, vol={vol_annual:.2f})")

    return pairs