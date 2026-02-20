"""
training/clt_trainer.py

Training loop for the Cross-Layer Transcoder (CLT).

Key features
------------
- Linear λ-ramp: sparsity loss scales from 0 → λ over the full training run.
- Per-layer MSE tracking to detect collapsed layers early.
- Separate parameter groups so encoder and decoder LRs can be scaled independently.
- Checkpoint save / resume with full optimiser state.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import CLTConfig, CLTTrainingConfig
from models.cross_layer_transcoder import CrossLayerTranscoder


# ---------------------------------------------------------------------------
# LR schedule (shared with SAETrainer for consistency)
# ---------------------------------------------------------------------------

def _cosine_warmup_schedule(warmup: int, total: int):
    import math
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# CLT Trainer
# ---------------------------------------------------------------------------

class CLTTrainer:
    """
    Trains a CrossLayerTranscoder on a stream of (residuals, mlp_outs) batches.

    Parameters
    ----------
    clt         : Initialised CrossLayerTranscoder module.
    clt_cfg     : CLT configuration (used for bookkeeping).
    train_cfg   : CLT training hyperparameters.
    data_iter   : Iterator that yields dicts:
                  {
                    "residual": {layer: Tensor[B, d_model]},
                    "mlp_out":  {layer: Tensor[B, d_model]},
                  }
                  Use ActivationBuffer with hook_point="both".
    """

    def __init__(
        self,
        clt: CrossLayerTranscoder,
        clt_cfg: CLTConfig,
        train_cfg: CLTTrainingConfig,
        data_iter: Iterable[Dict],
    ):
        self.clt = clt
        self.clt_cfg = clt_cfg
        self.cfg = train_cfg
        self.data_iter = data_iter

        self.device = torch.device(train_cfg.device)
        self.dtype = self._resolve_dtype(train_cfg.dtype)

        self.clt = self.clt.to(self.device, dtype=self.dtype)

        # Separate parameter groups: encoders (log_threshold included) + decoders
        enc_params = list(self.clt.encoders.parameters())
        dec_params = list(self.clt.decoders)
        self.optimizer = optim.Adam(
            [
                {"params": enc_params, "lr": train_cfg.lr},
                {"params": dec_params, "lr": train_cfg.lr},
            ],
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_cosine_warmup_schedule(train_cfg.warmup_steps, train_cfg.n_steps),
        )

        self._step = 0
        self._ramp_steps = train_cfg.sparsity_ramp_steps or train_cfg.n_steps
        self._run = None
        self._setup_wandb()

    @staticmethod
    def _resolve_dtype(s: str) -> torch.dtype:
        return {"float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16}[s]

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _setup_wandb(self):
        if self.cfg.wandb_project is None:
            return
        try:
            import wandb
            self._run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config={**vars(self.clt_cfg), **vars(self.cfg)},
                reinit=True,
            )
        except ImportError:
            print("[CLTTrainer] wandb not installed — skipping W&B logging.")

    def _log(self, metrics: dict):
        if self._run is not None:
            self._run.log(metrics, step=self._step)

    # ------------------------------------------------------------------
    # Sparsity ramp
    # ------------------------------------------------------------------

    def _sparsity_scale(self) -> float:
        """Linear ramp from 0 → 1 over `_ramp_steps` steps."""
        return min(1.0, self._step / max(self._ramp_steps, 1))

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _step_fn(
        self,
        residuals: Dict[int, torch.Tensor],
        mlp_outs: Dict[int, torch.Tensor],
    ) -> dict:
        # Move to device
        residuals = {
            l: v.to(self.device, dtype=self.dtype)
            for l, v in residuals.items()
        }
        mlp_outs = {
            l: v.to(self.device, dtype=self.dtype)
            for l, v in mlp_outs.items()
        }

        self.optimizer.zero_grad(set_to_none=True)

        out = self.clt(residuals, mlp_outs, sparsity_scale=self._sparsity_scale())
        out.loss_total.backward()

        nn.utils.clip_grad_norm_(self.clt.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Per-layer MSE for monitoring
        per_layer_mse = {}
        for l, yhat in out.y_hat.items():
            if l in mlp_outs:
                # Normalised MSE (since both are in normalised space)
                per_layer_mse[f"mse/layer_{l:02d}"] = (
                    (yhat - mlp_outs[l] / self.clt.mlp_out_norm[l].clamp(1e-8)
                     ).pow(2).mean().item()
                )

        mean_l0 = self.clt.mean_l0(out.activations)

        metrics = {
            "loss/total": out.loss_total.item(),
            "loss/mse": out.loss_mse.item(),
            "loss/sparsity": out.loss_sparsity.item(),
            "loss/pre_act": out.loss_pre_act.item(),
            "train/mean_l0": mean_l0,
            "train/sparsity_scale": self._sparsity_scale(),
            "train/lr": self.scheduler.get_last_lr()[0],
            **per_layer_mse,
        }
        return metrics

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"[CLTTrainer] Starting training: {self.cfg.n_steps} steps, "
              f"device={self.device}, dtype={self.dtype}")
        self.clt.train()
        data_iter: Iterator = iter(self.data_iter)
        t0 = time.time()

        for step in range(self._step, self.cfg.n_steps):
            self._step = step

            try:
                batch = next(data_iter)
            except StopIteration:
                print("[CLTTrainer] Data iterator exhausted — stopping early.")
                break

            residuals = batch.get("residual", {})
            mlp_outs = batch.get("mlp_out", {})

            metrics = self._step_fn(residuals, mlp_outs)

            if step % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:7d} | "
                    f"total {metrics['loss/total']:.4f} | "
                    f"mse {metrics['loss/mse']:.4f} | "
                    f"spars {metrics['loss/sparsity']:.4f} | "
                    f"L0 {metrics['train/mean_l0']:.1f} | "
                    f"λ_scale {metrics['train/sparsity_scale']:.3f} | "
                    f"{elapsed:.1f}s"
                )
                self._log(metrics)
                t0 = time.time()

            if step > 0 and step % self.cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.cfg.n_steps, final=True)
        print("[CLTTrainer] Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, final: bool = False):
        tag = "final" if final else f"step_{step:08d}"
        path = self.cfg.output_dir / f"clt_{tag}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.clt.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "clt_cfg": vars(self.clt_cfg),
            "train_cfg": vars(self.cfg),
        }, path)
        print(f"[CLTTrainer] Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.clt.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._step = ckpt["step"]
        print(f"[CLTTrainer] Resumed from step {self._step}")
