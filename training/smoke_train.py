#!/usr/bin/env python3
"""
Small single-GPU smoke test for OpenMythos pretraining on real FineWeb-Edu data.

This is NOT a real pretraining run. It exists to verify the full pipeline end
to end on a single consumer GPU (e.g. RTX 4090, 24 GB): tokenizer → streaming
dataset → forward/backward → optimizer step → checkpoint save/reload. Success
criteria are "runs N steps without error, loss is finite, a checkpoint is
written and reloads", not convergence.

It reuses the dataset / LR schedule / checkpoint helpers from the cluster
training script (3b_fine_web_edu.py) so the exercised code paths match the
real run as closely as possible.

Differences from 3b_fine_web_edu.py and why:
  * mythos_1b instead of mythos_3b — the 3B variant does not fit a single 24 GB
    card for training.
  * Single-GPU with bf16-native parameters (no FSDP, no fp32 master copy). With
    the gpt-oss tokenizer's ~200k vocab, mythos_1b is ~1.41B params. fp32
    params+grads+Adam(m,v) is ~22.5 GB, which leaves no room for activations on
    24 GB and OOMs; bf16-native halves that to ~11.3 GB and fits comfortably.
    (Running the model bf16-native relies on the RecurrentBlock ACT bookkeeping
    keeping h's dtype — see the dtype fix in open_mythos/main.py.)
  * Tiny budget — short seq_len, micro_batch 1, few recurrent loops, a few dozen
    steps — so it finishes in minutes.

Run (defaults are a ~1 minute single-GPU smoke test):
    python training/smoke_train.py

Every hyperparameter is overridable from the CLI, so experiments need no file
edits. Examples:
    # sweep recurrent depth
    python training/smoke_train.py --n-loops 8 --ckpt-dir ckpt_loops8
    # longer context, more steps
    python training/smoke_train.py --seq-len 1024 --max-steps 100
    python training/smoke_train.py --help     # full list
"""

import os
import time
import argparse
import importlib.util

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from open_mythos import OpenMythos, variants
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# Reuse helpers from the cluster script.
#
# Its filename starts with a digit ("3b_..."), which is not a valid module
# identifier, so it cannot be imported with a plain `import`. Load it by path.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "mythos_pretrain", os.path.join(_HERE, "3b_fine_web_edu.py")
)
_pretrain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pretrain)

FineWebEduDataset = _pretrain.FineWebEduDataset
get_lr = _pretrain.get_lr
save_checkpoint = _pretrain.save_checkpoint
load_checkpoint = _pretrain.load_checkpoint
_list_ckpts = _pretrain._list_ckpts


# Variant factories exposed by open_mythos.variants (mythos_1b, mythos_3b, ...).
# Only mythos_1b (and smaller custom configs) fit a single 24 GB card for
# training; larger variants are listed for convenience but will OOM.
_VARIANTS = sorted(n for n in dir(variants) if n.startswith("mythos_"))


def build_parser() -> argparse.ArgumentParser:
    """CLI for the smoke test. Defaults reproduce the ~1 minute baseline run."""
    p = argparse.ArgumentParser(
        description="Single-GPU smoke test / experiment harness for OpenMythos "
        "pretraining on streaming FineWeb-Edu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", choices=_VARIANTS, default="mythos_1b",
                   help="model config factory from open_mythos.variants")
    p.add_argument("--seq-len", type=int, default=512,
                   help="context length; activations scale with this")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="global batch tokens = micro_batch * grad_accum * seq_len")
    p.add_argument("--n-loops", type=int, default=4,
                   help="recurrent loop depth (compute/memory scale ~linearly)")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--log-every", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--ckpt-dir", default="checkpoints_smoke")
    p.add_argument("--dataset-subset", default="sample-10BT",
                   help="FineWeb-Edu config name (e.g. sample-10BT, default)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker processes")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto",
                   help="param dtype; auto = bf16 if supported else fp32")
    p.add_argument("--seed", type=int, default=0,
                   help="torch manual seed for reproducible comparisons")
    return p


def resolve_dtype(choice: str) -> torch.dtype:
    """Map --dtype to a torch dtype, honoring hardware bf16 support for 'auto'."""
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp32":
        return torch.float32
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16_ok else torch.float32


def main(args: argparse.Namespace) -> None:
    """Run a short single-GPU smoke train on streaming FineWeb-Edu."""
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_dtype = resolve_dtype(args.dtype)
    logger.info(
        f"Device: {device} | param dtype: {param_dtype} | seed: {args.seed}"
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    logger.info(f"Tokenizer: gpt-oss-20b | vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Model (selected variant, overridden seq_len / loop depth)
    # ------------------------------------------------------------------
    cfg = getattr(variants, args.variant)()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.max_loop_iters = args.n_loops

    model = OpenMythos(cfg).to(device=device, dtype=param_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.variant} | params: {n_params:,} ({n_params / 1e9:.2f}B)")

    global_batch_tok = args.micro_batch * args.grad_accum * args.seq_len
    logger.info(
        f"seq_len={args.seq_len} | micro_batch={args.micro_batch} "
        f"| grad_accum={args.grad_accum} | n_loops={args.n_loops} "
        f"| global_batch_tokens={global_batch_tok:,} | max_steps={args.max_steps}"
    )

    # fused AdamW needs CUDA; states inherit the param dtype.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Streaming dataset (real FineWeb-Edu, pulled on demand)
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(
        encoding, args.seq_len, args.dataset_subset, rank=0, world_size=1
    )
    loader = DataLoader(
        dataset, batch_size=args.micro_batch,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    last_loss = float("nan")

    for step in range(1, args.max_steps + 1):
        cur_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            # Loss in fp32: cross-entropy over a ~200k-class vocab loses too much
            # precision in low precision.
            loss = nn.functional.cross_entropy(
                logits.float().view(-1, vocab_size), y.view(-1)
            )
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss_accum

        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * args.log_every / dt
            logger.info(
                f"step {step:3d}/{args.max_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec:,.0f} tok/s"
            )
            t0 = time.perf_counter()

        if step % args.ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, args.ckpt_dir,
                ddp=False, master=True,
            )

    # ------------------------------------------------------------------
    # Verify the smoke test actually exercised the pipeline.
    # ------------------------------------------------------------------
    assert torch.isfinite(torch.tensor(last_loss)), f"non-finite loss: {last_loss}"

    ckpts = _list_ckpts(args.ckpt_dir)
    assert ckpts, f"no checkpoint written to {args.ckpt_dir}/"
    # Reload the latest checkpoint into a fresh model/optimizer to confirm the
    # save/load round-trip works (the part most likely to silently break).
    fresh = OpenMythos(cfg).to(device=device, dtype=param_dtype)
    fresh_opt = torch.optim.AdamW(
        fresh.parameters(), lr=args.lr, fused=(device == "cuda")
    )
    resumed_step = load_checkpoint(fresh, fresh_opt, ckpts[-1], ddp=False)
    logger.success(f"Checkpoint round-trip OK ({ckpts[-1]}, step {resumed_step})")

    if device == "cuda":
        logger.info(
            f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        )
    logger.success(f"Smoke test passed. Final loss: {last_loss:.4f}")


if __name__ == "__main__":
    main(build_parser().parse_args())
