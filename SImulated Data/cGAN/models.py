import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim=32, hidden_dim=128, num_layers=2,
                 output_dim=1, dropout=0.1, num_regimes=4, embed_dim=16):
        """
        Args:
            noise_dim:    Dimensionality of each timestep's noise vector.
                          32 is enough for univariate returns — the LSTM
                          does the heavy lifting of creating temporal structure.
            hidden_dim:   LSTM hidden state size.  128 = good balance of
                          capacity vs overfitting for typical financial data.
            num_layers:   Stacked LSTM layers.  Layer 1 learns basic patterns;
                          Layer 2 learns higher-order compositions.
            output_dim:   1 for univariate returns.  Set to N for N assets.
            dropout:      Applied BETWEEN LSTM layers (not within).
            num_regimes:  Number of distinct market regimes (e.g., 4).
            embed_dim:    Dimensionality of the regime embedding vector.
                          16 is sufficient for 4 regimes.  Rule of thumb:
                          embed_dim ≈ min(50, num_regimes // 2 + 1) for large K,
                          but for small K, 8-32 all work well.
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_regimes = num_regimes
        self.embed_dim = embed_dim
        self.regime_embedding = nn.Embedding(num_regimes, embed_dim)

        self.lstm = nn.LSTM(
            input_size=noise_dim + embed_dim,   # ← CHANGED
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, z, labels):
        """
        Transform noise into regime-specific synthetic returns.

        Args:
            z:      Noise tensor, shape (batch_size, seq_len, noise_dim)
                    Each z[b, t, :] is an i.i.d. sample from N(0, I).
            labels: Regime labels, shape (batch_size,)
                    Integer tensor with values in [0, num_regimes - 1].

        Returns:
            Synthetic normalized returns, shape (batch_size, seq_len, 1)

        Shape trace (B=64, T=50, noise_dim=32, embed_dim=16, hidden=128):
            z:              (64, 50, 32)
            regime_emb:     (64, 16)       ← one embedding per sample
            regime_expand:  (64, 50, 16)   ← same embedding at every timestep
            lstm_input:     (64, 50, 48)   ← noise + regime concatenated
            lstm_out:       (64, 50, 128)  ← hidden state at each timestep
            output:         (64, 50, 1)    ← one return per timestep
        """
        batch_size, seq_len, _ = z.shape

        regime_emb = self.regime_embedding(labels)
        regime_expanded = regime_emb.unsqueeze(1).expand(-1, seq_len, -1)

        lstm_input = torch.cat([z, regime_expanded], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        output = self.output_head(lstm_out)

        return output


class Discriminator(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2,
                 dropout=0.1, num_regimes=4, embed_dim=16):
        """
        Args:
            input_dim:    Features per timestep (1 for univariate returns).
            hidden_dim:   LSTM hidden size.  Should match or exceed generator's.
            num_layers:   LSTM depth.
            dropout:      Regularization.
            num_regimes:  Number of distinct market regimes.
            embed_dim:    Dimensionality of the regime embedding vector.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_regimes = num_regimes
        self.embed_dim = embed_dim
        self.regime_embedding = nn.Embedding(num_regimes, embed_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        combined_dim = hidden_dim + embed_dim
        self.score_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, 1),
        )

    def forward(self, x, labels):
        """
        Score a (sequence, regime) pair for realism.

        Args:
            x:      Return sequence, shape (batch_size, seq_len, 1)
            labels: Regime labels, shape (batch_size,)
                    Integer tensor with values in [0, num_regimes - 1].

        Returns:
            Realism scores, shape (batch_size, 1)
            Unbounded: higher = "more realistic for this regime".

        Shape trace (B=64, T=50, hidden=128, embed_dim=16):
            x:             (64, 50, 1)
            features:      (64, 50, 128)   ← after input projection
            lstm_out:      (64, 50, 128)   ← hidden state at each timestep
            h_n:           (2, 64, 128)    ← final hidden states, all layers
            last_hidden:   (64, 128)       ← top layer, final timestep
            regime_emb:    (64, 16)        ← regime embedding
            combined:      (64, 144)       ← h_T concatenated with regime
            score:         (64, 1)         ← one score per (sequence, regime) pair
        """

        features = self.input_projection(x)
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_hidden = h_n[-1]
        regime_emb = self.regime_embedding(labels)
        combined = torch.cat([last_hidden, regime_emb], dim=-1)
        score = self.score_head(combined)

        return score