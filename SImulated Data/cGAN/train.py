import torch
import numpy as np
import os
from tqdm import tqdm


def compute_gradient_penalty(critic, real_data, fake_data, labels, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = (alpha * fake_data + (1.0 - alpha) * real_data).detach()
    interpolated.requires_grad_(True)

    critic_output = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients_flat = gradients.reshape(batch_size, -1)
    gradient_norm = gradients_flat.norm(2, dim=1)
    penalty = ((gradient_norm - 1.0) ** 2).mean()

    return penalty


def train_gan(
    generator,
    discriminator,
    dataloader,
    n_epochs=300,
    noise_dim=32,
    lr_g=1e-4,
    lr_d=1e-4,
    n_critic=5,
    lambda_gp=10.0,
    device='cpu',
    checkpoint_dir='checkpoints',
    log_interval=10,
    num_regimes=4,
    # ── NEW: pass these so the checkpoint is self-contained ──
    global_mu=None,
    global_sigma=None,
    regime_map=None,
    embed_dim=None,
):
    """
    Full CONDITIONAL WGAN-GP training loop.

    NEW params (for saving a self-contained checkpoint):
        global_mu:     Global mean of training log returns
        global_sigma:  Global std of training log returns
        regime_map:    Dict mapping regime names to integer labels
        embed_dim:     Regime embedding dimension (read from generator if None)
    """
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr_g, betas=(0.5, 0.9)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9)
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    sample_batch, sample_labels = next(iter(dataloader))
    seq_len = sample_batch.shape[1]

    # Read embed_dim from model if not explicitly passed
    if embed_dim is None:
        embed_dim = generator.embed_dim

    history = {
        'critic_loss': [],
        'generator_loss': [],
        'wasserstein_dist': [],
        'gradient_penalty': [],
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())

    print(f"\n{'=' * 60}")
    print(f"  Conditional WGAN-GP Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Device:           {device}")
    print(f"  Epochs:           {n_epochs}")
    print(f"  Batch size:       {sample_batch.shape[0]}")
    print(f"  Sequence length:  {seq_len}")
    print(f"  Noise dim:        {noise_dim}")
    print(f"  Hidden dim:       {generator.hidden_dim}")
    print(f"  LSTM layers:      {generator.num_layers}")
    print(f"  Num regimes:      {num_regimes}")
    print(f"  Embed dim (G):    {generator.embed_dim}")
    print(f"  Embed dim (D):    {discriminator.embed_dim}")
    print(f"  Critic steps:     {n_critic}")
    print(f"  λ_GP:             {lambda_gp}")
    print(f"  LR (G / D):       {lr_g} / {lr_d}")
    print(f"  Generator params: {n_params_g:,}")
    print(f"  Critic params:    {n_params_d:,}")
    print(f"{'=' * 60}\n")

    for epoch in range(n_epochs):

        epoch_d_losses = []
        epoch_g_losses = []
        epoch_w_dists = []
        epoch_gps = []

        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.to(device)
            labels = labels.to(device)
            batch_size_actual = real_data.size(0)

            # ═══ PHASE 1: Train Critic ═══
            for _ in range(n_critic):

                noise = torch.randn(
                    batch_size_actual, seq_len, noise_dim, device=device
                )

                with torch.no_grad():
                    fake_data = generator(noise, labels)

                score_real = discriminator(real_data, labels)
                score_fake = discriminator(fake_data, labels)

                gp = compute_gradient_penalty(
                    discriminator, real_data, fake_data, labels, device
                )

                d_loss = (
                    score_fake.mean()
                    - score_real.mean()
                    + lambda_gp * gp
                )

                w_dist = score_real.mean() - score_fake.mean()

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                epoch_d_losses.append(d_loss.item())
                epoch_w_dists.append(w_dist.item())
                epoch_gps.append(gp.item())

            # ═══ PHASE 2: Train Generator ═══
            noise = torch.randn(
                batch_size_actual, seq_len, noise_dim, device=device
            )

            fake_data = generator(noise, labels)
            score_fake = discriminator(fake_data, labels)

            g_loss = -score_fake.mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            epoch_g_losses.append(g_loss.item())

        # ── End of epoch ──
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)
        avg_w_dist = np.mean(epoch_w_dists)
        avg_gp = np.mean(epoch_gps)

        history['critic_loss'].append(avg_d_loss)
        history['generator_loss'].append(avg_g_loss)
        history['wasserstein_dist'].append(avg_w_dist)
        history['gradient_penalty'].append(avg_gp)

        if (epoch + 1) % log_interval == 0:
            print(
                f"  Epoch [{epoch+1:4d}/{n_epochs}]  "
                f"D_loss: {avg_d_loss:+.4f}  "
                f"G_loss: {avg_g_loss:+.4f}  "
                f"W_dist: {avg_w_dist:.4f}  "
                f"GP: {avg_gp:.4f}"
            )

        # ── Save checkpoints (CHANGED: now saves FULL config) ──
        save_interval = max(1, n_epochs // 5)
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == n_epochs:
            ckpt_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'history': history,
                # ── NEW: everything needed to reproduce generation ──
                'config': {
                    'noise_dim':   noise_dim,
                    'hidden_dim':  generator.hidden_dim,
                    'num_layers':  generator.num_layers,
                    'output_dim':  1,
                    'dropout':     generator.lstm.dropout if generator.num_layers > 1 else 0.0,
                    'num_regimes': num_regimes,
                    'embed_dim':   embed_dim,
                    'seq_len':     seq_len,
                },
                'global_mu':    global_mu,
                'global_sigma': global_sigma,
                'regime_map':   regime_map,
            }, ckpt_path)

    print(f"\n  Training complete.")
    print(f"  Final estimated W-distance: {history['wasserstein_dist'][-1]:.4f}")

    return history