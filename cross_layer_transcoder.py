"""
models/cross_layer_transcoder.py

Cross-Layer Transcoder (CLT) following Lindsey et al. (2024).

Architecture
------------
For each layer ℓ, CLT features read from the residual stream:

    a_ℓ = JumpReLU(W_enc_ℓ @ x_ℓ)          [N features]

The MLP output at layer ℓ is reconstructed from all preceding features:

    ŷ_ℓ = Σ_{ℓ'=1}^{ℓ}  W_dec_{ℓ'→ℓ} @ a_{ℓ'}

Loss
----
    L_MSE      = Σ_ℓ ||ŷ_ℓ - y_ℓ||²
    L_sparsity = λ · Σ_ℓ Σ_i tanh(c · ||W_dec_i_ℓ|| · a_i_ℓ)
    L_pre_act  = coeff · Σ_f ReLU(-h_f)        (optional, anti-dead-feature)
    L_total    = L_MSE + L_sparsity + L_pre_act

JumpReLU implementation
-----------------------
Forward:    JumpReLU(h, θ) = h  if h > θ  else 0
Gradient:   Straight-through estimator for the threshold.
            dL/dθ_f ≈ -dL/dz_f · δ_bandwidth(h_f - θ_f) · h_f
            where δ_bandwidth is a rectangular window of width `bandwidth`.
            Gradient flows through normally to all other params.

Normalisation
-------------
Each layer's residual stream input (x_ℓ) and MLP output target (y_ℓ) are
normalised by their estimated RMS norms, tracked as running EMA buffers.

Decoder initialisation
----------------------
U(-1 / (n_layers * d_model), +1 / (n_layers * d_model))   per the paper.

Encoder initialisation
----------------------
U(-1 / n_features, +1 / n_features)   per the paper.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CLTConfig


# ---------------------------------------------------------------------------
# JumpReLU with straight-through estimator
# ---------------------------------------------------------------------------

class JumpReLUFunction(torch.autograd.Function):
    """
    Custom autograd function implementing JumpReLU with the
    straight-through estimator (STE) for the learnable threshold.

    Forward:   output = pre_act  if pre_act > threshold  else 0
    Backward (w.r.t. pre_act): standard gradient (1 if active, 0 otherwise)
              but with STE bandwidth applied to the threshold neighbourhood.
    Backward (w.r.t. threshold): -grad_output * rect(pre_act - threshold, bw) * pre_act
    """

    @staticmethod
    def forward(
        ctx,
        pre_act: torch.Tensor,   # [B, N]
        log_threshold: torch.Tensor,  # [N]  (log-space for stability)
        bandwidth: float,
    ) -> torch.Tensor:
        threshold = log_threshold.exp()
        active = (pre_act > threshold.unsqueeze(0))
        output = pre_act * active
        ctx.save_for_backward(pre_act, threshold)
        ctx.bandwidth = bandwidth
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_act, threshold = ctx.saved_tensors
        bw = ctx.bandwidth

        # Gradient through pre_act: standard (straight-through for the gate)
        active = (pre_act > threshold.unsqueeze(0))
        grad_pre_act = grad_output * active.float()

        # Gradient through threshold: STE with rectangular window
        # δ_bw(h - θ) is 1/bw if |h - θ| < bw/2, else 0
        delta = (pre_act - threshold.unsqueeze(0)).abs() < (bw / 2)
        # d/d(log_θ) = d/dθ · dθ/d(log_θ) = d/dθ · θ
        # d/dθ = -grad_output · (1/bw) · h   (from the STE derivation)
        grad_log_threshold = -(
            grad_output * delta.float() * pre_act / bw
        ).sum(dim=0) * threshold  # chain rule for log

        return grad_pre_act, grad_log_threshold, None


def jumprelu(
    pre_act: torch.Tensor,
    log_threshold: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    return JumpReLUFunction.apply(pre_act, log_threshold, bandwidth)


# ---------------------------------------------------------------------------
# Single-layer CLT encoder
# ---------------------------------------------------------------------------

class CLTLayerEncoder(nn.Module):
    """
    Encoder for a single CLT layer ℓ.
    Reads from x_ℓ and produces feature activations a_ℓ.
    """

    def __init__(self, d_model: int, n_features: int, bandwidth: float,
                 init_log_threshold: float):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        # Store threshold in log-space for numerical stability
        self.log_threshold = nn.Parameter(
            torch.full((n_features,), init_log_threshold)
        )
        self.bandwidth = bandwidth
        self._init_weights(n_features)

    def _init_weights(self, n_features: int):
        lim = 1.0 / n_features
        nn.init.uniform_(self.W_enc, -lim, lim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, d_model]

        Returns
        -------
        a     : [B, n_features]  post-JumpReLU activations
        pre_a : [B, n_features]  pre-JumpReLU activations (for pre_act_loss)
        """
        pre_a = x @ self.W_enc.T + self.b_enc  # [B, N]
        a = jumprelu(pre_a, self.log_threshold, self.bandwidth)
        return a, pre_a


# ---------------------------------------------------------------------------
# Full Cross-Layer Transcoder
# ---------------------------------------------------------------------------

class CLTForwardOutput(NamedTuple):
    y_hat: Dict[int, torch.Tensor]          # {layer: [B, d_mlp]}  reconstructions
    activations: Dict[int, torch.Tensor]    # {layer: [B, n_features]}
    pre_activations: Dict[int, torch.Tensor]  # {layer: [B, n_features]}  (for loss)
    loss_mse: torch.Tensor
    loss_sparsity: torch.Tensor
    loss_pre_act: torch.Tensor
    loss_total: torch.Tensor


class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT).

    Parameters
    ----------
    cfg : CLTConfig

    Forward call signature
    ----------------------
    clt(
        residuals : {layer: Tensor[B, d_model]},
        mlp_outs  : {layer: Tensor[B, d_mlp]},
        sparsity_scale : float = 1.0,   # for ramp scheduling
    ) -> CLTForwardOutput
    """

    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L, D, N = cfg.n_layers, cfg.d_model, cfg.n_features

        # --- Encoders (one per layer) ---
        self.encoders = nn.ModuleList([
            CLTLayerEncoder(
                d_model=D,
                n_features=N,
                bandwidth=cfg.jumprelu_bandwidth,
                init_log_threshold=cfg.jumprelu_init_threshold,
            )
            for _ in range(L)
        ])

        # --- Decoders ---
        # W_dec[l_source][l_target] : Parameter [d_model, n_features]
        # Only l_target >= l_source is valid (upper triangular in layer space).
        # We store them in a flat ParameterList indexed by (l_src * L + l_tgt).
        self.decoders = nn.ParameterList()
        self._dec_idx = {}  # (l_src, l_tgt) -> flat index
        flat_idx = 0
        for l_src in range(L):
            for l_tgt in range(l_src, L):
                self.decoders.append(nn.Parameter(torch.empty(D, N)))
                self._dec_idx[(l_src, l_tgt)] = flat_idx
                flat_idx += 1

        # --- Normalisation buffers ---
        # Running RMS norms for residual stream inputs and MLP outputs
        if cfg.normalize_per_layer:
            self.register_buffer(
                "residual_norm", torch.ones(L), persistent=True
            )
            self.register_buffer(
                "mlp_out_norm", torch.ones(L), persistent=True
            )
            self._norm_ema = 0.99

        self._init_decoder_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_decoder_weights(self):
        L, D = self.cfg.n_layers, self.cfg.d_model
        lim = 1.0 / (L * D)
        for p in self.decoders:
            nn.init.uniform_(p, -lim, lim)

    def get_decoder(self, l_src: int, l_tgt: int) -> nn.Parameter:
        return self.decoders[self._dec_idx[(l_src, l_tgt)]]

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_norms(
        self,
        residuals: Dict[int, torch.Tensor],
        mlp_outs: Dict[int, torch.Tensor],
    ):
        decay = self._norm_ema
        for l in range(self.cfg.n_layers):
            if l in residuals:
                rms_r = residuals[l].norm(dim=-1).mean()
                self.residual_norm[l].mul_(decay).add_(rms_r * (1 - decay))
            if l in mlp_outs:
                rms_m = mlp_outs[l].norm(dim=-1).mean()
                self.mlp_out_norm[l].mul_(decay).add_(rms_m * (1 - decay))

    def _normalise_inputs(
        self,
        residuals: Dict[int, torch.Tensor],
        mlp_outs: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Return normalised copies."""
        if not self.cfg.normalize_per_layer:
            return residuals, mlp_outs
        norm_res = {
            l: v / self.residual_norm[l].clamp(min=1e-8)
            for l, v in residuals.items()
        }
        norm_mlp = {
            l: v / self.mlp_out_norm[l].clamp(min=1e-8)
            for l, v in mlp_outs.items()
        }
        return norm_res, norm_mlp

    # ------------------------------------------------------------------
    # Sparsity loss helpers
    # ------------------------------------------------------------------

    def _concat_decoder_norms(self, l_src: int) -> torch.Tensor:
        """
        Compute ||W_dec_i_{l_src}|| for all features i.

        W_dec_i_{l_src} is the concatenation across all l_tgt >= l_src
        of the i-th column of W_dec_{l_src → l_tgt}.

        Returns [n_features] tensor of norms.
        """
        L = self.cfg.n_layers
        # Collect columns for each feature i: list of [d_model] vectors
        cols = []
        for l_tgt in range(l_src, L):
            W = self.get_decoder(l_src, l_tgt)  # [d_model, n_features]
            cols.append(W)
        # Stack: [d_model * (L - l_src), n_features], then norm over dim 0
        cat = torch.cat(cols, dim=0)  # [d_model*(L-l_src), N]
        return cat.norm(dim=0)  # [N]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        residuals: Dict[int, torch.Tensor],
        mlp_outs: Dict[int, torch.Tensor],
        sparsity_scale: float = 1.0,
    ) -> CLTForwardOutput:
        """
        Parameters
        ----------
        residuals     : {layer_idx: [B, d_model]} — residual stream at each layer
        mlp_outs      : {layer_idx: [B, d_model]} — MLP output at each layer (targets)
        sparsity_scale: Multiplier for λ (used to implement the linear ramp schedule)
        """
        cfg = self.cfg
        device = next(self.parameters()).device

        # Update and apply normalisation
        if self.training and cfg.normalize_per_layer:
            self._update_norms(residuals, mlp_outs)
        norm_res, norm_mlp = self._normalise_inputs(residuals, mlp_outs)

        # ----- Encode all layers -----
        activations: Dict[int, torch.Tensor] = {}
        pre_activations: Dict[int, torch.Tensor] = {}
        for l in range(cfg.n_layers):
            if l not in norm_res:
                continue
            a, pre_a = self.encoders[l](norm_res[l])
            activations[l] = a
            pre_activations[l] = pre_a

        # ----- Decode: reconstruct ŷ_ℓ -----
        y_hat: Dict[int, torch.Tensor] = {}
        for l_tgt in range(cfg.n_layers):
            if l_tgt not in norm_mlp:
                continue
            contrib = None
            for l_src in range(l_tgt + 1):
                if l_src not in activations:
                    continue
                W = self.get_decoder(l_src, l_tgt)   # [d_model, n_features]
                term = activations[l_src] @ W.T       # [B, d_model]
                contrib = term if contrib is None else contrib + term
            y_hat[l_tgt] = contrib if contrib is not None else torch.zeros(
                next(iter(norm_mlp.values())).shape[0], cfg.d_model, device=device
            )

        # ----- MSE loss -----
        loss_mse = torch.zeros(1, device=device)
        for l in y_hat:
            if l in norm_mlp:
                # MSE in normalised space (targets are already normed)
                loss_mse = loss_mse + F.mse_loss(y_hat[l], norm_mlp[l])

        # ----- Sparsity loss -----
        loss_sparsity = torch.zeros(1, device=device)
        effective_lambda = cfg.sparsity_lambda * sparsity_scale
        for l_src in activations:
            dec_norms = self._concat_decoder_norms(l_src)  # [N]
            a = activations[l_src]                          # [B, N]
            # tanh(c · ||W_dec_i_ℓ|| · a_i_ℓ) summed over features and batch
            penalty = torch.tanh(cfg.sparsity_c * dec_norms.unsqueeze(0) * a)
            loss_sparsity = loss_sparsity + effective_lambda * penalty.sum(dim=-1).mean()

        # ----- Pre-activation loss (anti-dead-feature) -----
        loss_pre_act = torch.zeros(1, device=device)
        if cfg.pre_act_loss_coeff > 0:
            for pre_a in pre_activations.values():
                loss_pre_act = loss_pre_act + (
                    cfg.pre_act_loss_coeff * F.relu(-pre_a).sum(dim=-1).mean()
                )

        loss_total = loss_mse + loss_sparsity + loss_pre_act

        return CLTForwardOutput(
            y_hat=y_hat,
            activations=activations,
            pre_activations=pre_activations,
            loss_mse=loss_mse,
            loss_sparsity=loss_sparsity,
            loss_pre_act=loss_pre_act,
            loss_total=loss_total,
        )

    # ------------------------------------------------------------------
    # Convenience: average L0 sparsity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mean_l0(self, activations: Dict[int, torch.Tensor]) -> float:
        """Average number of active features per token per layer."""
        l0s = [
            (a > 0).float().sum(dim=-1).mean().item()
            for a in activations.values()
        ]
        return sum(l0s) / max(len(l0s), 1)

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"n_layers={c.n_layers}, d_model={c.d_model}, "
                f"n_features={c.n_features}, λ={c.sparsity_lambda}")
