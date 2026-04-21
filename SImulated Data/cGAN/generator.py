import torch
import numpy as np
from models import Generator
from data import DEFAULT_REGIME_MAP, DEFAULT_REGIME_NAMES

def load_generator(checkpoint_path, device=None):
    """
    Load a trained generator from a checkpoint.  ALL config is read
    from the checkpoint — no need to pass architecture params manually.

    Args:
        checkpoint_path:  Path to .pt checkpoint file

    Returns:
        generator:  Generator model in eval mode
        cfg:        Dict with everything needed for generation:
                      noise_dim, seq_len, device,
                      global_mu, global_sigma,
                      regime_map, regime_names
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Read saved config ──
    saved = checkpoint['config']

    generator = Generator(
        noise_dim=saved['noise_dim'],
        hidden_dim=saved['hidden_dim'],
        num_layers=saved['num_layers'],
        output_dim=saved.get('output_dim', 1),
        dropout=saved.get('dropout', 0.1),
        num_regimes=saved['num_regimes'],
        embed_dim=saved['embed_dim'],
    )

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    generator.to(device)

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"[Inference] Loaded generator from epoch {epoch}")
    print(f"[Inference] Architecture: noise_dim={saved['noise_dim']}, "
          f"hidden={saved['hidden_dim']}, layers={saved['num_layers']}, "
          f"regimes={saved['num_regimes']}, embed={saved['embed_dim']}")

    regime_map = checkpoint.get('regime_map', DEFAULT_REGIME_MAP)
    regime_names = {v: k for k, v in regime_map.items()} if regime_map else DEFAULT_REGIME_NAMES

    cfg = {
        'noise_dim':    saved['noise_dim'],
        'seq_len':      saved['seq_len'],
        'device':       device,
        'global_mu':    checkpoint.get('global_mu'),
        'global_sigma': checkpoint.get('global_sigma'),
        'regime_map':   regime_map,
        'regime_names': regime_names,
    }

    return generator, cfg


def generate_regime_data(generator, regime_label, n_sequences=100,
                         seq_len=50, noise_dim=32, device='cpu', **kwargs):
    """
    Generate synthetic NORMALIZED return sequences for a specific regime.

    Extra **kwargs are accepted so you can do:
        generate_regime_data(gen, regime_label=0, n_sequences=500, **cfg)
    and seq_len/noise_dim/device

    Returns:
        numpy array of shape (n_sequences, seq_len) — normalized returns
    """
    with torch.no_grad():
        noise = torch.randn(n_sequences, seq_len, noise_dim, device=device)
        labels = torch.full(
            (n_sequences,), regime_label, dtype=torch.long, device=device
        )
        fake = generator(noise, labels)

    return fake.squeeze(-1).cpu().numpy()


def denormalize(normalized_returns, global_mu, global_sigma):
    """Convert normalized returns back to actual log returns."""
    return normalized_returns * global_sigma + global_mu


def returns_to_prices(log_returns, initial_price=100.0):
    """Convert log returns to price paths."""
    if log_returns.ndim == 1:
        cum_returns = np.cumsum(log_returns)
        prices = initial_price * np.exp(cum_returns)
        return np.insert(prices, 0, initial_price)
    elif log_returns.ndim == 2:
        cum_returns = np.cumsum(log_returns, axis=1)
        prices = initial_price * np.exp(cum_returns)
        p0_col = np.full((prices.shape[0], 1), initial_price)
        return np.hstack([p0_col, prices])
    else:
        raise ValueError(f"Expected 1D or 2D array, got {log_returns.ndim}D")


def main():
    from scipy import stats

    config = {
        'checkpoint_path': 'checkpoints/checkpoint_epoch_800.pt',
    }

    generator, cfg = load_generator(config['checkpoint_path'])

    regime_map   = cfg['regime_map']
    regime_names = cfg['regime_names']
    print("Generating Data for all regimes")

    n_per_regime = 500
    all_regime_data = {}

    for name, label in regime_map.items():
        normalized = generate_regime_data(
            generator, regime_label=label, n_sequences=n_per_regime, **cfg
        )
        if cfg['global_mu'] is not None and cfg['global_sigma'] is not None:
            actual = denormalize(normalized, cfg['global_mu'], cfg['global_sigma'])
        else:
            actual = normalized

        all_regime_data[name] = actual

        flat = actual.flatten()
        print(f"\n  {name} (label={label}):")
        print(f"    Mean:     {np.mean(flat):.6f}")
        print(f"    Std:      {np.std(flat):.6f}")
        print(f"    Skew:     {float(stats.skew(flat)):.3f}")
        print(f"    Kurtosis: {float(stats.kurtosis(flat)):.2f}")

    print("Reproducibility Check")

    torch.manual_seed(42)
    batch_a = generate_regime_data(
        generator, regime_label=0, n_sequences=10, **cfg
    )

    torch.manual_seed(42)
    batch_b = generate_regime_data(
        generator, regime_label=0, n_sequences=10, **cfg
    )

    match = np.allclose(batch_a, batch_b)
    print(f"  Same seed → identical output: {match}")

    import pandas as pd

    for name, returns_data in all_regime_data.items():
        df = pd.DataFrame(
            returns_data,
            columns=[f'day_{t+1}' for t in range(cfg['seq_len'])],
        )
        csv_path = f'generated_{name.lower()}_returns.csv'
        df.to_csv(csv_path, index_label='scenario_id')
        print(f"  {name}: {len(df)} scenarios → {csv_path}")


if __name__ == '__main__':
    main()