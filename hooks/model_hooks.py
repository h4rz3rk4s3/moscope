"""
hooks/model_hooks.py

Forward-hook infrastructure for extracting activations from ModernBERT / MoE-BERT.

We support two hook points per layer:
  - "residual"  : the residual stream *after* the layer-norm, just before the MLP.
  - "mlp_out"   : the output of the MLP block (i.e. the addend to the residual stream).

The CLT needs both simultaneously; the per-layer SAE only needs "residual".

Usage
-----
    extractor = ActivationExtractor(model, layer_indices=[0, 1, 2, ...])
    with extractor:
        outputs = model(**inputs)
    residuals = extractor.get("residual")   # dict[int, Tensor]
    mlp_outs  = extractor.get("mlp_out")    # dict[int, Tensor]
    extractor.clear()
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ActivationExtractor:
    """
    Registers forward hooks on a HuggingFace-style transformer to collect
    intermediate activations.  Designed to be used as a context manager so
    hooks are always cleaned up.

    Supports ModernBERT (layers accessed via model.model.layers[i]) and an
    MoE variant where the MLP is replaced by a MoeMLP / mixture block.

    Parameters
    ----------
    model        : The transformer model.
    layer_indices: Which layers to hook (default: all).
    hook_points  : Which activations to capture, subset of {"residual","mlp_out"}.
    detach       : Whether to detach captured tensors from the computation graph.
                   Set False during CLT / SAE training when you need gradients
                   to flow back through the model (rare, but supported).
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        hook_points: Tuple[str, ...] = ("residual", "mlp_out"),
        detach: bool = True,
    ):
        self._model_ref = weakref.ref(model)
        self.hook_points = set(hook_points)
        self.detach = detach

        # Buffers: {layer_idx: Tensor | list[Tensor]}
        self._activations: Dict[str, Dict[int, torch.Tensor]] = {
            hp: {} for hp in hook_points
        }

        # Locate the layer list in the model.
        self._layers = self._find_layers(model)

        if layer_indices is None:
            layer_indices = list(range(len(self._layers)))
        self.layer_indices = layer_indices

        self._hooks: List = []

    # ------------------------------------------------------------------
    # Model introspection helpers
    # ------------------------------------------------------------------
    #TODO: Need to fix find_layers, as it doesn't catch layers, but breaks in the for-loop
    @staticmethod
    def _find_layers(model: nn.Module) -> nn.ModuleList:
        """
        Try common attribute paths for HuggingFace-style transformers.
        ModernBERT uses model.model.layers; fallback paths cover BERT, RoBERTa, etc.
        """
        candidates = [
            "model.layers",       # ModernBERT / Llama-style
            "encoder.layer",      # BERT / RoBERTa
            "transformer.h",      # GPT-2
            "model.encoder.layers",
        ]
        for path in candidates:
            obj = model
            try:
                for attr in path.split("."):
                    print
                    obj = getattr(obj, attr)
                    print(obj)
                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    return obj
            except AttributeError:
                continue
        raise ValueError(
            "Cannot locate transformer layers automatically.  "
            "Please subclass ActivationExtractor and override `_find_layers`."
        )

    @staticmethod
    def _find_mlp(layer_module: nn.Module) -> Optional[nn.Module]:
        """
        Return the MLP sub-module inside a transformer layer.
        Tries common attribute names.
        """
        for name in ("mlp"): #("mlp", "feed_forward", "ffn", "moe", "moe_block")
            if hasattr(layer_module, name):
                return getattr(layer_module, name)
        return None

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _make_residual_hook(self, layer_idx: int):
        """
        Pre-MLP residual stream hook.

        For ModernBERT the layer's forward signature is roughly:
            hidden_states, ...  = layer(hidden_states, ...)
        We capture hidden_states *at the input* of the MLP module, which equals
        the post-attention residual stream (after the attention layer-norm).
        """
        def hook(module, args, output):
            # `args[0]` is the hidden_states tensor entering the MLP.
            hidden = args[0] if isinstance(args, (tuple, list)) else args
            if self.detach:
                hidden = hidden.detach()
            self._activations["residual"][layer_idx] = hidden.clone()
        return hook

    def _make_mlp_out_hook(self, layer_idx: int):
        """
        Post-MLP output hook. Captures what gets *added* to the residual stream.
        """
        def hook(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if self.detach:
                out = out.detach()
            self._activations["mlp_out"][layer_idx] = out.clone()
        return hook

    def register_hooks(self):
        """Register all hooks on the model."""
        for idx in self.layer_indices:
            layer = self._layers[idx]
            mlp = self._find_mlp(layer)
            if mlp is None:
                raise ValueError(
                    f"Cannot find MLP sub-module in layer {idx}.  "
                    "Override `_find_mlp` for your architecture."
                )
            if "residual" in self.hook_points:
                h = mlp.register_forward_pre_hook(self._make_residual_hook(idx))
                self._hooks.append(h)
            if "mlp_out" in self.hook_points:
                h = mlp.register_forward_hook(self._make_mlp_out_hook(idx))
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        """Clear cached activations (call between batches to free memory)."""
        for hp in self._activations:
            self._activations[hp].clear()

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *_):
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Activation access
    # ------------------------------------------------------------------

    def get(self, hook_point: str, layer_idx: Optional[int] = None):
        """
        Return captured activations.

        Parameters
        ----------
        hook_point : "residual" or "mlp_out"
        layer_idx  : If given, return just that layer's tensor.
                     If None, return the full dict {layer_idx: tensor}.

        Tensors have shape [batch, seq_len, d_model].
        """
        buf = self._activations[hook_point]
        if layer_idx is not None:
            return buf[layer_idx]
        return dict(buf)

    def get_flat(self, hook_point: str) -> Dict[int, torch.Tensor]:
        """
        Like get() but each tensor is flattened to [batch*seq_len, d_model],
        ready for the activation buffer / SAE trainer.
        """
        out = {}
        for idx, tensor in self._activations[hook_point].items():
            B, T, D = tensor.shape
            out[idx] = tensor.reshape(B * T, D)
        return out


# ---------------------------------------------------------------------------
# Convenience: run model + extract activations in one call
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    model: nn.Module,
    inputs: dict,
    layer_indices: List[int],
    hook_points: Tuple[str, ...] = ("residual", "mlp_out"),
    device: Optional[str] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Run a single forward pass and return flat (tokens, d_model) activation tensors.

    Returns
    -------
    dict with keys "residual" and/or "mlp_out", each mapping layer_idx → tensor.
    """
    if device:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

    extractor = ActivationExtractor(model, layer_indices, hook_points, detach=True)
    with extractor:
        model(**inputs)

    return {hp: extractor.get_flat(hp) for hp in hook_points}
