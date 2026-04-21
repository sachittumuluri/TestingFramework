import torch
import numpy as np

from models import Generator, Discriminator
from data import (
    generate_regime_csvs,
    create_regime_dataloader,
    DEFAULT_REGIME_MAP,
    DEFAULT_REGIME_NAMES,
)
from train import train_gan
from utils import (
    generate_sequences,
    generate_all_regimes,
    compare_statistics,
    compare_regime_statistics,
    plot_evaluation,
    plot_training_history,
    plot_sample_sequences,
    plot_regime_comparison,
)


def main():

    config = {
        # --- Data ---
        'data_dir':    'regime_data',          # Directory for regime CSVs
        'price_col':   'Close',                # Column with prices
        'seq_len':     30,                     # Window length (T)
        'stride':      2,                      # Window stride
        'batch_size':  64,                     # Batch size

        # --- Model Architecture ---
        'noise_dim':   32,                     # Noise vector dimension per timestep
        'hidden_dim':  256,                    # LSTM hidden state size
        'num_layers':  2,                      # Stacked LSTM layers
        'dropout':     0.1,                    # Dropout rate
        'num_regimes': 4,                      # Number of market regimes
        'embed_dim':   64,                     # Regime embedding dimension

        # --- Training ---
        'n_epochs':    1000,                    # Training epochs
        'lr_g':        3e-4,                   # Generator learning rate
        'lr_d':        5e-5,                   # Critic learning rate
        'n_critic':    5,                      # Critic steps per generator step
        'lambda_gp':   10.0,                   # Gradient penalty weight

        # --- System ---
        'device':      'cuda' if torch.cuda.is_available() else 'cpu',
        'seed':        42,
    }

    regime_map = DEFAULT_REGIME_MAP
    regime_names = DEFAULT_REGIME_NAMES


    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['device'] == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])

    print("\n" + "=" * 60)
    print("  STEP 1: Preparing Multi-Regime Data")
    print("=" * 60)


    csv_label_pairs = [
    ('bullish_SPY_data.csv', 'Bullish'),
    ('bearish_SPY_data.csv', 'Bearish'),
    ('sideways_SPY_data.csv', 'Sideways'),
    ('crash_SPY_data.csv', 'Crash'),
    ]

    custom_strides = {
        'crash_SPY_data.csv': 1
    }

    dataloader, multi_dataset = create_regime_dataloader(
        csv_label_pairs=csv_label_pairs,
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        stride=config['stride'],
        custom_strides=custom_strides,
        regime_map=regime_map,
        price_col=config['price_col'],
    )

    print("\n" + "=" * 60)
    print("  STEP 2: Creating Conditional Models")
    print("=" * 60)

    generator = Generator(
        noise_dim=config['noise_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        output_dim=1,
        dropout=config['dropout'],
        num_regimes=config['num_regimes'],
        embed_dim=config['embed_dim'],
    )

    discriminator = Discriminator(
        input_dim=1,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_regimes=config['num_regimes'],
        embed_dim=config['embed_dim'],
    )

    print(f"\n  Generator:\n{generator}")
    print(f"\n  Discriminator:\n{discriminator}")

    print("\n" + "=" * 60)
    print("  STEP 3: Training Conditional WGAN-GP")
    print("=" * 60)

    history = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        n_epochs=config['n_epochs'],
        noise_dim=config['noise_dim'],
        lr_g=config['lr_g'],
        lr_d=config['lr_d'],
        n_critic=config['n_critic'],
        lambda_gp=config['lambda_gp'],
        device=config['device'],
        num_regimes=config['num_regimes'],
    )

    print("\n" + "=" * 60)
    print("  STEP 4: Evaluating Per-Regime Results")
    print("=" * 60)

    # --- Generate synthetic sequences for EACH regime ---
    n_eval = 200

    fake_regime_data = generate_all_regimes(
        generator, n_eval, config['seq_len'],
        config['noise_dim'], config['num_regimes'],
        regime_names, config['device']
    )

    # --- Collect real data per regime for comparison ---
    real_regime_data = {}
    for label, ds in multi_dataset.regime_datasets.items():
        real_regime_data[label] = ds.windows[:n_eval]

    # --- Per-regime statistical comparison ---
    print("\n  ── Per-Regime Statistical Comparison ──")
    compare_regime_statistics(real_regime_data, fake_regime_data, regime_names)

    # --- Per-regime 4-panel visual comparison ---
    for label in sorted(fake_regime_data.keys()):
        name = regime_names.get(label, f"Regime_{label}")
        if label in real_regime_data:
            real_flat = real_regime_data[label].flatten()
            fake_flat = fake_regime_data[label].flatten()
            plot_evaluation(real_flat, fake_flat, title_prefix=f"{name}: ")

    # --- Regime comparison: cumulative return paths ---
    plot_regime_comparison(fake_regime_data, regime_names)

    # --- Training curves ---
    plot_training_history(history)

    print("\n  Done.")

    # Generate 1000 crash scenarios for risk analysis
    crash_label = regime_map['Crash']
    crash_sequences = generate_sequences(
        generator, 1000, config['seq_len'],
        config['noise_dim'], regime_label=crash_label,
        device=config['device']
    )

    # Denormalize to get actual log returns
    crash_log_returns = multi_dataset.denormalize(crash_sequences.flatten())

    print(f"\n  Generated 1000 Crash sequences:")
    print(f"    Mean daily log return:  {np.mean(crash_log_returns):.6f}")
    print(f"    Daily volatility:       {np.std(crash_log_returns):.6f}")
    print(f"    Annualized vol:         {np.std(crash_log_returns) * np.sqrt(252):.4f}")
    print(f"    Worst single day:       {np.min(crash_log_returns):.6f}")
    print(f"    Excess kurtosis:        {float(np.mean((crash_log_returns - np.mean(crash_log_returns))**4) / np.std(crash_log_returns)**4 - 3):.2f}")


if __name__ == '__main__':
    main()