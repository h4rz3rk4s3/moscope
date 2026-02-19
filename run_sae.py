"""
run_sae.py

Train a Top-K SAE on a single layer of MoE-BERT / ModernBERT.

Quick-start
-----------
    python run_sae.py \
        --model_name answerdotai/ModernBERT-base \
        --layer 6 \
        --hook_point residual \
        --n_features 16384 \
        --k 64 \
        --n_steps 100000 \
        --batch_size 4096 \
        --output_dir ./checkpoints/sae_layer6

On Apple MPS:
    python run_sae.py --device mps --dtype float32 ...

On H200:
    torchrun --nproc_per_node=1 run_sae.py --device cuda --dtype bfloat16 ...
"""

import argparse
import random
from itertools import islice

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer

from config import (
    BufferConfig,
    TopKSAEConfig,
    TrainingConfig,
)

from models.topk_sae import TopKSAE
from buffers.activation_buffer import (
    ActivationBuffer,
    SingleLayerBuffer
)

from trainers.sae_trainer import SAETrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Top-K SAE training")
    p.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--hook_point", default="residual",
                   choices=["residual", "mlp_out"])
    p.add_argument("--n_features", type=int, default=16_384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--output_dir", default="./checkpoints/sae")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default="topk_sae")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="hatecheck",
                   help="HuggingFace dataset name for training text")
    #p.add_argument("--dataset_split", default="train")
    #p.add_argument("--dataset_text_field", default="text")
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

    print(f"[run_sae] device={device}, dtype={dtype}")

    # ---- Load model + tokenizer ----
    print(f"[run_sae] Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.hidden_size

    # ---- Dataset ----
    print(f"[run_sae] Loading dataset {args.dataset}...")
    df_data = pd.read_csv(f"data/{args.dataset}")
    df_train, df_test = train_test_split(df_data, test_size=0.2, stratify="is_toxic")
    # ds = load_dataset(args.dataset, "wikitext-103-raw-v1",
    #                   split=args.dataset_split, streaming=True)
    train_ds = Dataset.from_pandas(df_train)
    test_ds = Dataset.from_pandas(df_test)
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

    sae_cfg = TopKSAEConfig(
        d_model=d_model,
        n_features=args.n_features,
        k=args.k,
        layer=args.layer,
        hook_point=args.hook_point,
        dtype=dtype,
    )

    train_cfg = TrainingConfig(
        n_steps=args.n_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    # ---- Activation buffer ----
    multi_buf = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        data_iter=text_iter,
        layer_indices=[args.layer],
        hook_point=args.hook_point,
        cfg=buf_cfg,
        dtype=dtype,
    )
    single_buf = SingleLayerBuffer(multi_buf, args.layer, args.hook_point)

    # ---- SAE ----
    sae = TopKSAE(sae_cfg)

    # ---- Trainer ----
    trainer = SAETrainer(
        sae=sae,
        sae_cfg=sae_cfg,
        train_cfg=train_cfg,
        activation_iter=single_buf,
    )
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()

    print(f"\n[run_sae] Done.  Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
