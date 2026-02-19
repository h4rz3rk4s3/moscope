"""
tests/test_pipeline.py

Quick smoke tests for each major component.
Run with:  python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---- import path adjustment for running from repo root ----
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    TopKSAEConfig, CLTConfig, BufferConfig, TrainingConfig
)
from models.topk_sae import TopKSAE
from models.cross_layer_transcoder import (
    CrossLayerTranscoder, JumpReLUFunction
)


DEVICE = "mps"
DTYPE  = torch.float32


# ---------------------------------------------------------------------------
# Top-K SAE tests
# ---------------------------------------------------------------------------

class TestTopKSAE:

    def _make_sae(self, d=64, n=256, k=16):
        cfg = TopKSAEConfig(d_model=d, n_features=n, k=k)
        return TopKSAE(cfg).to(DEVICE).to(DTYPE)

    def test_output_shape(self):
        sae = self._make_sae()
        x = torch.randn(32, 64)
        out = sae(x)
        assert out.x_hat.shape == x.shape
        assert out.z_topk.shape == (32, 256)

    def test_sparsity_constraint(self):
        sae = self._make_sae(k=16)
        x = torch.randn(32, 64)
        out = sae(x, compute_loss=False)
        active_per_token = (out.z_topk > 0).sum(dim=-1)
        # Every token should have exactly k=16 active features
        assert (active_per_token == 16).all(), \
            f"Expected 16 active features, got {active_per_token.tolist()}"

    def test_loss_is_mse(self):
        sae = self._make_sae()
        x = torch.randn(8, 64)
        out = sae(x)
        expected_loss = ((out.x_hat - x) ** 2).mean()
        assert abs(out.loss.item() - expected_loss.item()) < 1e-5

    def test_decoder_normalisation(self):
        sae = self._make_sae()
        sae.normalise_decoder()
        norms = sae.W_dec.norm(dim=0)
        assert (norms - 1.0).abs().max().item() < 1e-5, \
            "Decoder columns should have unit norm after normalisation"

    def test_no_nan_in_forward(self):
        sae = self._make_sae()
        x = torch.randn(64, 64)
        out = sae(x)
        assert not torch.isnan(out.loss), "Loss should not be NaN"
        assert not torch.isnan(out.x_hat).any(), "x_hat should not contain NaN"

    def test_gradient_flows(self):
        sae = self._make_sae()
        x = torch.randn(8, 64)
        out = sae(x)
        out.loss.backward()
        assert sae.W_enc.grad is not None
        assert sae.b_enc.grad is not None

    def test_dead_fraction_tracking(self):
        sae = self._make_sae(n=256, k=4)
        sae.train()
        for _ in range(5):
            x = torch.randn(16, 64)
            sae(x)
        frac = sae.dead_fraction(window=100)
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# JumpReLU tests
# ---------------------------------------------------------------------------

class TestJumpReLU:

    def test_forward_thresholding(self):
        pre_act = torch.tensor([[0.5, 0.02, 0.1, -0.1]])
        log_theta = torch.log(torch.tensor([0.1, 0.1, 0.05, 0.05]))
        out = JumpReLUFunction.apply(pre_act, log_theta, 1.0)
        # 0.5 > 0.1 → 0.5; 0.02 < 0.1 → 0; 0.1 > 0.05 → 0.1; -0.1 < 0.05 → 0
        expected = torch.tensor([[0.5, 0.0, 0.1, 0.0]])
        assert torch.allclose(out, expected), f"Got {out}, expected {expected}"

    def test_gradient_flows(self):
        pre_act = torch.randn(4, 8, requires_grad=True)
        log_theta = torch.zeros(8, requires_grad=True)
        out = JumpReLUFunction.apply(pre_act, log_theta, 1.0)
        loss = out.sum()
        loss.backward()
        assert pre_act.grad is not None
        assert log_theta.grad is not None


# ---------------------------------------------------------------------------
# CLT tests
# ---------------------------------------------------------------------------

class TestCLT:

    def _make_clt(self, L=4, D=64, N=128):
        cfg = CLTConfig(
            n_layers=L, d_model=D, n_features=N,
            sparsity_lambda=1e-3, sparsity_c=5.0,
            normalize_per_layer=False,  # simplify for tests
        )
        return CrossLayerTranscoder(cfg).to(DEVICE)

    def _make_inputs(self, L=4, B=8, D=64):
        residuals = {l: torch.randn(B, D) for l in range(L)}
        mlp_outs  = {l: torch.randn(B, D) for l in range(L)}
        return residuals, mlp_outs

    def test_output_shapes(self):
        clt = self._make_clt(L=4, D=64, N=128)
        residuals, mlp_outs = self._make_inputs(L=4, B=8, D=64)
        out = clt(residuals, mlp_outs)
        for l in range(4):
            assert out.y_hat[l].shape == (8, 64), \
                f"y_hat[{l}] has wrong shape {out.y_hat[l].shape}"

    def test_cross_layer_contribution(self):
        """
        Features from layer 0 should contribute to reconstructions at all layers.
        Features from layer 3 should only contribute to layer 3.
        """
        clt = self._make_clt(L=4, D=64, N=128)
        residuals, mlp_outs = self._make_inputs(L=4, B=4, D=64)
        out = clt(residuals, mlp_outs)
        # Just verify that all target layers have non-zero reconstructions
        for l in range(4):
            assert out.y_hat[l].abs().sum() > 0, \
                f"y_hat[{l}] is all zeros — cross-layer contribution missing"

    def test_loss_components(self):
        clt = self._make_clt()
        residuals, mlp_outs = self._make_inputs()
        out = clt(residuals, mlp_outs)
        assert out.loss_mse >= 0
        assert out.loss_sparsity >= 0
        assert out.loss_total > 0

    def test_sparsity_ramp(self):
        clt = self._make_clt()
        residuals, mlp_outs = self._make_inputs()
        out_zero = clt(residuals, mlp_outs, sparsity_scale=0.0)
        out_full = clt(residuals, mlp_outs, sparsity_scale=1.0)
        # With scale=0, sparsity loss should be 0
        assert out_zero.loss_sparsity.item() == pytest.approx(0.0, abs=1e-6)
        # With scale=1, sparsity loss should be positive
        assert out_full.loss_sparsity.item() > 0

    def test_no_nan(self):
        clt = self._make_clt()
        residuals, mlp_outs = self._make_inputs()
        out = clt(residuals, mlp_outs)
        assert not torch.isnan(out.loss_total), "Total loss should not be NaN"

    def test_gradients_flow(self):
        clt = self._make_clt()
        residuals, mlp_outs = self._make_inputs()
        out = clt(residuals, mlp_outs)
        out.loss_total.backward()
        enc_grad = clt.encoders[0].W_enc.grad
        dec_grad = clt.decoders[0].grad
        assert enc_grad is not None, "Encoder grad should flow"
        assert dec_grad is not None, "Decoder grad should flow"

    def test_decoder_indexing(self):
        """Check (l_src, l_tgt) decoder lookup is consistent."""
        clt = self._make_clt(L=4)
        for l_src in range(4):
            for l_tgt in range(l_src, 4):
                W = clt.get_decoder(l_src, l_tgt)
                assert W.shape == (64, 128), \
                    f"W_dec[{l_src}→{l_tgt}] has wrong shape {W.shape}"

    def test_concat_decoder_norms(self):
        clt = self._make_clt(L=4, D=64, N=128)
        norms = clt._concat_decoder_norms(0)
        assert norms.shape == (128,)
        assert (norms >= 0).all()

    def test_mean_l0(self):
        clt = self._make_clt()
        residuals, mlp_outs = self._make_inputs()
        out = clt(residuals, mlp_outs)
        l0 = clt.mean_l0(out.activations)
        assert 0.0 <= l0 <= clt.cfg.n_features


# ---------------------------------------------------------------------------
# Buffer config tests
# ---------------------------------------------------------------------------

class TestBufferConfig:

    def test_mps_disables_pin_memory(self):
        cfg = BufferConfig(device="mps", pin_memory=True)
        assert cfg.pin_memory is False, \
            "pin_memory should be disabled on MPS"

    def test_cuda_keeps_pin_memory(self):
        cfg = BufferConfig(device="cuda", pin_memory=True)
        assert cfg.pin_memory is True

    def test_default_device_is_valid(self):
        cfg = BufferConfig()
        assert cfg.device in ("cuda", "mps", "cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
