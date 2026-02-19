"""
Centralized configuration dataclasses for the MoE-BERT interpretability pipeline.
All hyperparameters live here so experiments are fully reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> str:
    """
    MPS is happiest with float32; H200s can use bfloat16.
    """
    if device == "mps":
        return "float32"
    if device == "cuda":
        return "bfloat16"
    return "float32"


# ---------------------------------------------------------------------------
# Activation Buffer
# ---------------------------------------------------------------------------

@dataclass
class BufferConfig:
    """Configuration for the streaming activation buffer."""

    # How many activation vectors to hold in memory at once.
    # At d_model=768, float32: 1M vecs ≈ 3 GB.  Tune to your RAM budget.
    buffer_size: int = 500_000

    # Fraction of the buffer to fill before we start yielding batches.
    # Ensures good shuffle diversity.
    min_fill_fraction: float = 0.5

    # Number of times we re-shuffle the buffer per refill cycle.
    n_shuffle_passes: int = 2

    # Batch size yielded to the SAE / CLT trainer.
    batch_size: int = 4096

    # How many model forward passes to run to fill the buffer.
    # (sequences_per_fill * seq_len activations are collected per layer)
    sequences_per_fill: int = 256

    # If True, pin CPU memory for faster GPU transfers (disable on MPS).
    pin_memory: bool = True

    # Number of background worker threads for the buffer prefetch queue.
    # Set to 0 to disable async prefetching.
    prefetch_workers: int = 2

    # Maximum number of pre-filled batches to queue ahead of the trainer.
    prefetch_queue_size: int = 8

    # Device string ("cuda", "mps", "cpu").  Inferred automatically if None.
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _default_device()
        # MPS doesn't support pinned memory
        if self.device == "mps":
            self.pin_memory = False


# ---------------------------------------------------------------------------
# Top-K SAE
# ---------------------------------------------------------------------------

@dataclass
class TopKSAEConfig:
    """Configuration for a single Top-K Sparse Autoencoder."""

    # Dimension of the model activations being reconstructed.
    d_model: int = 768

    # Number of latent features (dictionary size).
    # Common choices: 4×, 8×, 16×, 32× d_model.
    n_features: int = 16_384  # 768 * ~21

    # Top-K sparsity: how many features are active per token.
    k: int = 64

    # Normalise input activations to unit variance before encoding.
    # Recommended: helps decouple the learning rate from activation scale.
    normalize_activations: bool = True

    # Which layer index this SAE is attached to (for bookkeeping).
    layer: int = 0

    # Name of the hook point, e.g. "residual" or "mlp_out".
    hook_point: str = "residual"

    # Dtype for model parameters.
    dtype: str = "float32"

    # Decoder weight initialisation std. None → use Kaiming uniform.
    decoder_init_std: Optional[float] = None


# ---------------------------------------------------------------------------
# Cross-Layer Transcoder
# ---------------------------------------------------------------------------

@dataclass
class CLTConfig:
    """Configuration for the Cross-Layer Transcoder."""

    # Number of transformer layers in the underlying model.
    n_layers: int = 22  # ModernBERT-base

    # Hidden dimension of the underlying model.
    d_model: int = 768

    # Number of CLT features per layer.
    n_features: int = 16_384

    # JumpReLU threshold initial value (log-space stored for numerical stability).
    jumprelu_init_threshold: float = 0.03

    # Straight-through bandwidth (bandwidth=1.0 per Lindsey et al., 2024).
    jumprelu_bandwidth: float = 1.0

    # Sparsity loss coefficient λ.
    sparsity_lambda: float = 2e-4

    # Sparsity loss shape parameter c.
    sparsity_c: float = 10.0

    # Pre-activation loss coefficient (set >0 only for large runs to prevent dead features).
    pre_act_loss_coeff: float = 0.0  # 3e-6 for large Haiku-style runs

    # Whether to separately normalize each layer's residual stream and MLP outputs.
    normalize_per_layer: bool = True

    # Dtype for model parameters.
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Shared training hyperparameters."""

    # Total number of SAE/CLT training steps.
    n_steps: int = 200_000

    # Peak learning rate.
    lr: float = 2e-4

    # Adam betas.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Gradient clipping norm.
    max_grad_norm: float = 1.0

    # Warmup steps for the LR scheduler.
    warmup_steps: int = 1_000

    # Log every N steps.
    log_every: int = 100

    # Save checkpoint every N steps.
    save_every: int = 5_000

    # Output directory for checkpoints and logs.
    output_dir: Path = Path("./checkpoints")

    # W&B project name (set to None to disable).
    wandb_project: Optional[str] = "moe-bert-interp"

    # Experiment name / run ID.
    run_name: str = "topk_sae_run_1"

    # Seed for reproducibility.
    seed: int = 42

    # Device.
    device: Optional[str] = None

    # Mixed-precision dtype ("float32", "bfloat16", "float16").
    dtype: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _default_device()
        if self.dtype is None:
            self.dtype = _default_dtype(self.device)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CLTTrainingConfig(TrainingConfig):
    """CLT-specific training config (extends shared TrainingConfig)."""

    # Sparsity ramp: lambda linearly increases from 0 → final value over all steps.
    sparsity_ramp_steps: Optional[int] = None  # None → use n_steps

    run_name: str = "clt_run_1"

    def __post_init__(self):
        super().__post_init__()
        if self.sparsity_ramp_steps is None:
            self.sparsity_ramp_steps = self.n_steps
