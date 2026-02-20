"""
models/topk_sae.py

Top-K Sparse Autoencoder following Gao et al. (2024).

Architecture
------------
Encoder:
    z = ReLU(W_enc @ (x - b_dec) + b_enc)

Sparsity:
    z_topk = top-k(z)   (keep only the k largest values, zero the rest)

Decoder:
    x_hat = W_dec @ z_topk + b_dec

Loss (pure MSE — no L1 term needed because sparsity is structural):
    L = ||x - x_hat||_2^2

Key design decisions
--------------------
- b_dec is shared between encoder and decoder (tied bias), which is the
  standard formulation from Gao et al.
- Decoder columns are normalised to unit norm after each gradient step
  (via `remove_parallel_decoder_grads` + `normalise_decoder`).
  This prevents the trivially-low-loss solution of scaling up the decoder.
- We track the "dead feature" fraction (features that never fire in a
  window) for monitoring training health.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TopKSAEConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class SAEForwardOutput(NamedTuple):
    x_hat: torch.Tensor          # Reconstructed activations  [B, d_model]
    z_topk: torch.Tensor         # Sparse feature activations  [B, n_features]
    z_pre_topk: torch.Tensor     # Dense pre-sparsity activations (for analysis)
    feature_indices: torch.Tensor  # Indices of active features  [B, k]
    loss: torch.Tensor            # Scalar MSE loss


# ---------------------------------------------------------------------------
# Top-K SAE
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder (Gao et al., 2024).

    Parameters are initialised following the conventions in the paper:
    - W_enc ~ Kaiming uniform (or custom std)
    - W_dec columns are initialised from W_enc^T and immediately normalised
    - b_enc, b_dec start at zero
    """

    def __init__(self, cfg: TopKSAEConfig):
        super().__init__()
        self.cfg = cfg
        D, N, K = cfg.d_model, cfg.n_features, cfg.k

        # Encoder weight + bias
        self.W_enc = nn.Parameter(torch.empty(N, D))
        self.b_enc = nn.Parameter(torch.zeros(N))

        # Decoder weight (columns ≈ dictionary atoms)
        self.W_dec = nn.Parameter(torch.empty(D, N))

        # The tied pre-encoder bias b_dec is subtracted from x before encoding
        # and added back after decoding.
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Running activation norm estimate (for optional input normalisation)
        self.register_buffer(
            "running_norm", torch.ones(1), persistent=True
        )
        # EMA decay for running norm estimate
        self._norm_ema_decay = 0.99

        # Dead-feature tracking: count of steps since each feature last fired
        self.register_buffer(
            "feature_last_fired", torch.zeros(N, dtype=torch.long), persistent=False
        )
        self._step = 0

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        N, D = self.cfg.n_features, self.cfg.d_model
        nn.init.kaiming_uniform_(self.W_enc)
        # Initialise decoder as the transpose of the encoder, then normalise
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            self.normalise_decoder()

    # ------------------------------------------------------------------
    # Decoder normalisation (call after each optimiser step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalise_decoder(self):
        """
        Project decoder columns to unit norm.
        This is the standard constraint that prevents the trivial solution
        of making decoder columns very large and feature activations very small.
        """
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_decoder_grads(self):
        """
        Remove the component of the encoder gradient that is parallel to the
        decoder columns.  This is the 'constrained gradient' step from Gao et
        al. that keeps the encoder and decoder from co-adapting in a way that
        inflates feature norms.

        Must be called *before* the optimiser step (after loss.backward()).
        """
        if self.W_dec.grad is None:
            return
        # For each column j: grad_j -= (grad_j · W_dec_j) * W_dec_j
        W = self.W_dec.detach()  # [D, N]
        g = self.W_dec.grad       # [D, N]
        proj = (g * W).sum(dim=0, keepdim=True)  # [1, N]
        self.W_dec.grad.sub_(proj * W)

    # ------------------------------------------------------------------
    # Activation normalisation
    # ------------------------------------------------------------------

    def _update_running_norm(self, x: torch.Tensor):
        """Update the EMA of the RMS activation norm."""
        with torch.no_grad():
            rms = x.norm(dim=-1).mean()
            self.running_norm.mul_(self._norm_ema_decay).add_(
                rms * (1 - self._norm_ema_decay)
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode + sparsify.

        Returns
        -------
        z_topk   : [B, N] sparse feature activations (k non-zeros per row)
        z_pre    : [B, N] dense pre-sparsity activations
        """
        # Centre by b_dec (tied bias trick from Gao et al.)
        x_cent = x - self.b_dec.unsqueeze(0)  # [B, D]

        # Linear encoder
        z_pre = F.relu(x_cent @ self.W_enc.T + self.b_enc)  # [B, N]

        # Top-K mask
        K = self.cfg.k
        topk_vals, topk_idx = torch.topk(z_pre, K, dim=-1)  # [B, K]
        z_topk = torch.zeros_like(z_pre)
        z_topk.scatter_(1, topk_idx, topk_vals)

        return z_topk, z_pre, topk_idx

    def decode(self, z_topk: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Returns x_hat : [B, D]
        """
        return z_topk @ self.W_dec.T + self.b_dec.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = True,
    ) -> SAEForwardOutput:
        """
        Full forward pass.

        Parameters
        ----------
        x            : [B, d_model] activation tensor.
        compute_loss : If True, compute and return MSE loss.

        Returns
        -------
        SAEForwardOutput namedtuple.
        """
        if self.cfg.normalize_activations and self.training:
            self._update_running_norm(x)

        # Optionally scale to ~unit RMS
        norm_scale = self.running_norm if self.cfg.normalize_activations else torch.ones(1)

        # Make sure that norm_scale is on the same device as x
        norm_scale = norm_scale.to(x.device)
        x_norm = x / norm_scale.clamp(min=1e-8)

        z_topk, z_pre, topk_idx = self.encode(x_norm)
        x_hat_norm = self.decode(z_topk)

        # Un-normalise for loss computation (MSE in original scale)
        x_hat = x_hat_norm * norm_scale.clamp(min=1e-8)

        # Dead feature tracking
        if self.training:
            self._step += 1
            fired = topk_idx.unique()
            self.feature_last_fired[fired] = self._step

        loss = F.mse_loss(x_hat, x) if compute_loss else torch.zeros(1, device=x.device)

        return SAEForwardOutput(
            x_hat=x_hat,
            z_topk=z_topk,
            z_pre_topk=z_pre,
            feature_indices=topk_idx,
            loss=loss,
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dead_fraction(self, window: int = 1000) -> float:
        """Fraction of features that haven't fired in the last `window` steps."""
        if self._step == 0:
            return 0.0
        threshold = self._step - window
        return (self.feature_last_fired < threshold).float().mean().item()

    @torch.no_grad()
    def feature_frequencies(
        self, loader, n_batches: int = 50
    ) -> torch.Tensor:
        """
        Estimate the empirical firing frequency of each feature over
        `n_batches` batches from `loader`.

        Returns a [n_features] tensor of frequencies in [0, 1].
        """
        counts = torch.zeros(self.cfg.n_features, device=next(self.parameters()).device)
        total = 0
        self.eval()
        for i, x in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(next(self.parameters()).device)
            out = self(x, compute_loss=False)
            fired = (out.z_topk > 0).float().sum(dim=0)
            counts += fired
            total += x.shape[0]
        return counts / max(total, 1)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"d_model={c.d_model}, n_features={c.n_features}, k={c.k}, "
                f"layer={c.layer}, hook={c.hook_point}")
