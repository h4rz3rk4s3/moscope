"""
run_clt.py

Train a Cross-Layer Transcoder (CLT) across all layers of MoE-BERT / ModernBERT.

The CLT reads residual stream activations at each layer and learns to jointly
reconstruct all MLP outputs, with cross-layer feature propagation.

Quick-start
-----------
    python run_clt.py \
        --model_name answerdotai/ModernBERT-base \
        --n_features 16384 \
        --n_steps 200000 \
        --sparsity_lambda 2e-4 \
        --output_dir ./checkpoints/clt \
        --wandb_project moe-bert-interp

On Apple MPS:
    python run_clt.py --device mps --dtype float32 ...

On H200 (multi-GPU with torchrun):
    torchrun --nproc_per_node=8 run_clt.py --device cuda --dtype bfloat16 ...
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from config import (
    BufferConfig,
    CLTConfig,
    CLTTrainingConfig,
)

from models.cross_layer_transcoder import CrossLayerTranscoder
from buffers.activation_buffer import (
    ActivationBuffer
)

from trainers.clt_trainer import CLTTrainer



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Cross-Layer Transcoder training")
    p.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    p.add_argument("--n_features", type=int, default=16_384)
    p.add_argument("--n_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sparsity_lambda", type=float, default=2e-4)
    p.add_argument("--sparsity_c", type=float, default=10.0)
    p.add_argument("--pre_act_loss_coeff", type=float, default=0.0)
    p.add_argument("--jumprelu_threshold", type=float, default=0.03)
    p.add_argument("--jumprelu_bandwidth", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--output_dir", default="./checkpoints/clt")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default="clt_run")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--dataset_text_field", default="text")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- Resolve device / dtype ----
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if args.dtype is None:
        dtype = "bfloat16" if device == "cuda" else "float32"
    else:
        dtype = args.dtype

    print(f"[run_clt] device={device}, dtype={dtype}")

    # ---- Load model ----
    print(f"[run_clt] Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    layer_indices = list(range(n_layers))
    print(f"[run_clt] n_layers={n_layers}, d_model={d_model}")

    # ---- Dataset ----
    print(f"[run_clt] Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, "wikitext-103-raw-v1",
                      split=args.dataset_split, streaming=True)
    text_iter = (
        row[args.dataset_text_field]
        for row in ds
        if len(row[args.dataset_text_field].strip()) > 50
    )

    # ---- Configs ----
    buf_cfg = BufferConfig(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
    )

    clt_cfg = CLTConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_features=args.n_features,
        jumprelu_init_threshold=args.jumprelu_threshold,
        jumprelu_bandwidth=args.jumprelu_bandwidth,
        sparsity_lambda=args.sparsity_lambda,
        sparsity_c=args.sparsity_c,
        pre_act_loss_coeff=args.pre_act_loss_coeff,
        normalize_per_layer=True,
        dtype=dtype,
    )

    train_cfg = CLTTrainingConfig(
        n_steps=args.n_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    # ---- Activation buffer (collects both residual stream + MLP outputs) ----
    buf = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        data_iter=text_iter,
        layer_indices=layer_indices,
        hook_point="both",       # need residual + mlp_out for CLT
        cfg=buf_cfg,
        dtype=dtype,
    )

    # ---- CLT ----
    clt = CrossLayerTranscoder(clt_cfg)

    # ---- Trainer ----
    trainer = CLTTrainer(
        clt=clt,
        clt_cfg=clt_cfg,
        train_cfg=train_cfg,
        data_iter=buf,
    )
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()
    print(f"\n[run_clt] Done.  Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
