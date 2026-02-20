"""
training/sae_trainer.py

Training loop for the Top-K Sparse Autoencoder.

Features
--------
- Cosine LR schedule with linear warmup
- Decoder normalisation + parallel gradient removal after every step
- Dead-feature monitoring and optional "resampling" for dead features
- W&B logging (optional)
- MPS / CUDA / CPU compatible
- Checkpoint save / resume
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import TopKSAEConfig, TrainingConfig
from models.topk_sae import TopKSAE


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _make_schedule(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    return lr_lambda


# ---------------------------------------------------------------------------
# SAE Trainer
# ---------------------------------------------------------------------------

class SAETrainer:
    """
    Trains a TopKSAE on a stream of activations.

    Parameters
    ----------
    sae          : Initialised TopKSAE module.
    sae_cfg      : Config for the SAE (used for bookkeeping).
    train_cfg    : Training hyperparameters.
    activation_iter : Iterator that yields [B, d_model] tensors on the
                      appropriate device.  Use SingleLayerBuffer or any
                      custom iterable.
    """

    def __init__(
        self,
        sae: TopKSAE,
        sae_cfg: TopKSAEConfig,
        train_cfg: TrainingConfig,
        activation_iter: Iterable[torch.Tensor],
    ):
        self.sae = sae
        self.sae_cfg = sae_cfg
        self.cfg = train_cfg
        self.activation_iter = activation_iter

        self.device = torch.device(train_cfg.device)
        self.dtype = self._resolve_dtype(train_cfg.dtype)

        self.sae = self.sae.to(self.device, dtype=self.dtype)

        self.optimizer = optim.Adam(
            self.sae.parameters(),
            lr=train_cfg.lr,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_make_schedule(train_cfg.warmup_steps, train_cfg.n_steps),
        )

        self._step = 0
        self._run = None
        self._setup_wandb()

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        return {"float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16}[dtype_str]

    def _setup_wandb(self):
        if self.cfg.wandb_project is None:
            return
        try:
            import wandb
            self._run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config={
                    **vars(self.sae_cfg),
                    **vars(self.cfg),
                },
                reinit=True,
            )
        except ImportError:
            print("[SAETrainer] wandb not installed — skipping W&B logging.")

    def _log(self, metrics: dict):
        if self._run is not None:
            self._run.log(metrics, step=self._step)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _step_fn(self, x: torch.Tensor) -> dict:
        """One training step. Returns a dict of loggable metrics."""
        x = x.to(self.device, dtype=self.dtype)

        self.optimizer.zero_grad(set_to_none=True)
        out = self.sae(x, compute_loss=True)
        out.loss.backward()

        # Constrained gradient: remove component parallel to decoder columns
        self.sae.remove_parallel_decoder_grads()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Decoder re-normalisation
        self.sae.normalise_decoder()

        metrics = {
            "loss/mse": out.loss.item(),
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/mean_active_features": (out.z_topk > 0).float().sum(-1).mean().item(),
            "train/dead_fraction_1k": self.sae.dead_fraction(window=1000),
        }
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"[SAETrainer] Starting training: {self.cfg.n_steps} steps, "
              f"device={self.device}, dtype={self.dtype}")
        self.sae.train()
        data_iter: Iterator = iter(self.activation_iter)
        t0 = time.time()

        for step in range(self._step, self.cfg.n_steps):
            self._step = step

            try:
                x = next(data_iter)
            except StopIteration:
                print("[SAETrainer] Data iterator exhausted — stopping early.")
                break

            metrics = self._step_fn(x)

            if step % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:7d} | "
                    f"MSE {metrics['loss/mse']:.4f} | "
                    f"L0 {metrics['train/mean_active_features']:.1f} | "
                    f"dead {metrics['train/dead_fraction_1k']:.3f} | "
                    f"lr {metrics['train/lr']:.2e} | "
                    f"{elapsed:.1f}s"
                )
                self._log(metrics)
                t0 = time.time()

            if step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.cfg.n_steps, final=True)
        print("[SAETrainer] Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, final: bool = False):
        tag = "final" if final else f"step_{step:08d}"
        path = self.cfg.output_dir / f"sae_{tag}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "sae_cfg": vars(self.sae_cfg),
            "train_cfg": vars(self.cfg),
        }, path)
        print(f"[SAETrainer] Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._step = ckpt["step"]
        print(f"[SAETrainer] Resumed from step {self._step}")
