"""
buffers/activation_buffer.py

Streaming activation buffer for training SAEs and CLTs.

Design goals
------------
1. **Memory-bounded** — holds at most `buffer_size` activation vectors in RAM.
   Old vectors are discarded once the buffer is full and a refill is triggered.
2. **Well-shuffled** — activations from different sequences, positions, and
   document types are interleaved so the SAE/CLT doesn't learn positional or
   local correlations.  Shuffling happens at two levels:
     a) Within a refill batch (random token draw from many sequences).
     b) In-buffer permutation before yielding batches.
3. **Async prefetch** — a background thread refills the buffer while the trainer
   consumes batches, hiding the model-forward latency.
4. **Device-aware** — uses pinned memory on CUDA (H200), skips it on MPS.

The buffer is intentionally kept CPU-side so it can store multiple gigabytes
without stressing the GPU/MPS VRAM.  Each yielded batch is moved to the target
device only at consumption time.

Usage
-----
    cfg = BufferConfig(buffer_size=500_000, batch_size=4096, device="cuda")
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        data_iter=text_iterator,       # yields strings or token dicts
        layer_indices=[0, 1, ..., 21],
        hook_point="residual",         # "residual" | "mlp_out" | "both"
        cfg=cfg,
    )
    for step, batch in enumerate(buffer):
        # batch: {"residual": {layer: Tensor[bs, d]}, ...}
        train_step(batch)
"""

from __future__ import annotations

import queue
import random
import threading
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import BufferConfig
from hooks.model_hooks import ActivationExtractor


# ---------------------------------------------------------------------------
# Helper: move a nested dict of tensors to a device
# ---------------------------------------------------------------------------

def _to_device(
    data: Dict[str, Dict[int, torch.Tensor]],
    device: str,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    out = {}
    for hp, layer_dict in data.items():
        out[hp] = {}
        for layer_idx, t in layer_dict.items():
            t = t.to(device=device, non_blocking=(device != "mps"))
            if dtype is not None:
                t = t.to(dtype=dtype)
            out[hp][layer_idx] = t
    return out


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


# ---------------------------------------------------------------------------
# Ring buffer (CPU-side, numpy-backed for fast random access)
# ---------------------------------------------------------------------------

class _RingBuffer:
    """
    Fixed-capacity ring buffer for float32 activation vectors.

    Stores activations for a *single* hook-point at a *single* layer.
    Multiple ring buffers are composed by ActivationBuffer.
    """

    def __init__(self, capacity: int, d_model: int, pin_memory: bool = False):
        self.capacity = capacity
        self.d_model = d_model
        # Allocate once; avoid repeated malloc.
        # We use numpy rather than torch so we can do fast in-place shuffles.
        self._data = np.zeros((capacity, d_model), dtype=np.float32)
        self._write_ptr = 0
        self._n_stored = 0

    def add(self, vecs: np.ndarray):
        """
        Add rows to the ring buffer, wrapping around if necessary.
        vecs : (N, d_model) float32 numpy array.
        """
        N = len(vecs)
        cap = self.capacity
        if N >= cap:
            # More data than buffer can hold; keep only the last `cap` rows
            vecs = vecs[-cap:]
            N = cap

        end = self._write_ptr + N
        if end <= cap:
            self._data[self._write_ptr:end] = vecs
        else:
            # Wrap
            first = cap - self._write_ptr
            self._data[self._write_ptr:] = vecs[:first]
            self._data[: N - first] = vecs[first:]

        self._write_ptr = end % cap
        self._n_stored = min(self._n_stored + N, cap)

    def shuffle(self, n_passes: int = 1):
        """In-place Fisher-Yates shuffle of the stored portion."""
        n = self._n_stored
        for _ in range(n_passes):
            idx = np.random.permutation(n)
            # Avoid full copy: shuffle in blocks to limit peak memory
            self._data[:n] = self._data[idx]

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Draw a random batch (with replacement if batch > n_stored)."""
        n = self._n_stored
        if n == 0:
            raise RuntimeError("Buffer is empty.")
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return self._data[idx].copy()

    def drain_batches(self, batch_size: int) -> Generator[np.ndarray, None, None]:
        """
        Yield non-overlapping batches from the stored data (after a shuffle).
        Used when we want to do a full pass over the buffer.
        """
        n = self._n_stored
        perm = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start: start + batch_size]
            yield self._data[idx].copy()

    @property
    def is_sufficiently_full(self) -> bool:
        return self._n_stored >= self.capacity * 0.5

    def __len__(self):
        return self._n_stored


# ---------------------------------------------------------------------------
# Multi-layer activation buffer
# ---------------------------------------------------------------------------

class ActivationBuffer:
    """
    Streaming multi-layer activation buffer.

    Parameters
    ----------
    model        : The frozen transformer model used to generate activations.
    tokenizer    : HuggingFace tokenizer.
    data_iter    : Iterable that yields either strings or pre-tokenised dicts.
    layer_indices: Which layers to collect activations for.
    hook_point   : "residual", "mlp_out", or "both".
    cfg          : BufferConfig instance.
    model_batch_size : Sequences per model forward pass.
    max_seq_len  : Maximum token length (sequences are padded/truncated here).
    dtype        : Dtype for yielded tensors (e.g. bfloat16 on H200).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        data_iter: Iterable,
        layer_indices: List[int],
        hook_point: str = "residual",
        cfg: Optional[BufferConfig] = None,
        model_batch_size: int = 32,
        max_seq_len: int = 512,
        dtype: str = "float32",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_iter = iter(data_iter)
        self.layer_indices = layer_indices
        self.cfg = cfg or BufferConfig()
        self.model_batch_size = model_batch_size
        self.max_seq_len = max_seq_len
        self._dtype = _dtype_from_str(dtype)
        self.device = self.cfg.device

        # Which hook points to collect
        if hook_point == "both":
            self._hook_points: Tuple[str, ...] = ("residual", "mlp_out")
        else:
            self._hook_points = (hook_point,)

        # Infer d_model
        self._d_model = self._infer_d_model()

        # One ring buffer per (hook_point, layer)
        cap = self.cfg.buffer_size
        pin = self.cfg.pin_memory
        self._rings: Dict[Tuple[str, int], _RingBuffer] = {
            (hp, l): _RingBuffer(cap, self._d_model, pin)
            for hp in self._hook_points
            for l in layer_indices
        }

        # Async prefetch queue
        self._queue: Optional[queue.Queue] = None
        self._refill_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.model.eval()

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    def _infer_d_model(self) -> int:
        cfg = getattr(self.model, "config", None)
        for attr in ("hidden_size", "d_model", "embed_dim"):
            if cfg is not None and hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError("Cannot infer d_model from model.config.  Pass it explicitly.")

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _get_next_batch_of_texts(self) -> Optional[List[str]]:
        texts = []
        for _ in range(self.model_batch_size):
            try:
                item = next(self.data_iter)
                texts.append(item if isinstance(item, str) else str(item))
            except StopIteration:
                break
        return texts if texts else None

    def _tokenize(self, texts: List[str]) -> dict:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Core refill logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refill(self):
        """
        Run enough model forward passes to fill the buffer to capacity,
        shuffling between passes for diversity.
        """
        seqs_processed = 0
        target_seqs = self.cfg.sequences_per_fill

        extractor = ActivationExtractor(
            self.model,
            self.layer_indices,
            self._hook_points,
            detach=True,
        )

        # Device for model inputs
        model_device = next(self.model.parameters()).device

        while seqs_processed < target_seqs:
            texts = self._get_next_batch_of_texts()
            if texts is None:
                break  # Dataset exhausted

            inputs = self._tokenize(texts)
            attention_mask = inputs.get("attention_mask", None)

            with extractor:
                self.model(**inputs)

            # Extract flat activations and push to rings
            for hp in self._hook_points:
                flat = extractor.get_flat(hp)  # {layer: (B*T, D)}
                for layer_idx, acts in flat.items():
                    # Filter padding positions
                    if attention_mask is not None:
                        mask = attention_mask.reshape(-1).bool().cpu()
                        acts = acts.cpu()[mask]
                    else:
                        acts = acts.cpu()

                    # Convert to float32 for storage (bfloat16 → f32 is cheap)
                    acts_np = acts.float().numpy()
                    self._rings[(hp, layer_idx)].add(acts_np)

            extractor.clear()
            seqs_processed += len(texts)

        # Shuffle all rings
        for ring in self._rings.values():
            ring.shuffle(n_passes=self.cfg.n_shuffle_passes)

    # ------------------------------------------------------------------
    # Async prefetch
    # ------------------------------------------------------------------

    def _prefetch_worker(self):
        """Background thread: keeps the queue full."""
        while not self._stop_event.is_set():
            try:
                # Check if any ring needs a refill
                any_low = any(
                    not ring.is_sufficiently_full
                    for ring in self._rings.values()
                )
                if any_low:
                    self._refill()

                batch = self._sample_batch()
                # Block until there's space, but check stop_event regularly
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                # Put exception on queue so main thread sees it
                self._queue.put(e)
                break

    def _sample_batch(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """Sample one training batch from all rings."""
        bs = self.cfg.batch_size
        result: Dict[str, Dict[int, torch.Tensor]] = {hp: {} for hp in self._hook_points}
        for (hp, layer_idx), ring in self._rings.items():
            arr = ring.sample_batch(bs)
            t = torch.from_numpy(arr)
            if self.cfg.pin_memory:
                t = t.pin_memory()
            result[hp][layer_idx] = t
        return result

    def start_async(self):
        if self._queue is not None:
            return  # Already running
        self._queue = queue.Queue(maxsize=self.cfg.prefetch_queue_size)
        self._stop_event.clear()
        # Initial fill so the queue has data immediately
        self._refill()
        if self.cfg.prefetch_workers > 0:
            self._refill_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self._refill_thread.start()

    def stop_async(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._refill_thread is not None:
            self._refill_thread.join(timeout=10)
            self._refill_thread = None
        self._queue = None

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Dict[int, torch.Tensor]]]:
        """
        Yields batches of activations.

        Each batch is a dict:
            {
              "residual": {0: Tensor[bs, d], 1: Tensor[bs, d], ...},
              "mlp_out":  {0: Tensor[bs, d], 1: Tensor[bs, d], ...},
            }
        Tensors are on `self.device` with `self._dtype`.
        """
        self.start_async()
        try:
            while True:
                if self.cfg.prefetch_workers > 0:
                    item = self._queue.get()
                    if isinstance(item, Exception):
                        raise item
                else:
                    # Synchronous fallback
                    any_low = any(
                        not ring.is_sufficiently_full
                        for ring in self._rings.values()
                    )
                    if any_low:
                        self._refill()
                    item = self._sample_batch()

                yield _to_device(item, self.device, dtype=self._dtype)
        finally:
            self.stop_async()

    def __del__(self):
        self.stop_async()


# ---------------------------------------------------------------------------
# Convenience: single-layer buffer (wraps ActivationBuffer for SAE training)
# ---------------------------------------------------------------------------

class SingleLayerBuffer:
    """
    Thin wrapper around ActivationBuffer that yields flat 2-D tensors
    (batch, d_model) for a single layer + hook_point.

    Perfect for per-layer SAE training.
    """

    def __init__(self, multi_buffer: "ActivationBuffer", layer_idx: int,
                 hook_point: str = "residual"):
        self._buf = multi_buffer
        self._layer = layer_idx
        self._hp = hook_point

    def __iter__(self) -> Iterator[torch.Tensor]:
        for batch_dict in self._buf:
            yield batch_dict[self._hp][self._layer]
