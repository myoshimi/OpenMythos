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

Run:
    python training/smoke_train.py
"""

import os
import time
import importlib.util

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from open_mythos import OpenMythos
from open_mythos.variants import mythos_1b
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


# ---------------------------------------------------------------------------
# Smoke-test hyperparameters — all deliberately tiny. Bump these to scale up.
# ---------------------------------------------------------------------------
SEQ_LEN = 512        # short context to keep activations small
MICRO_BATCH = 1
GRAD_ACCUM = 4       # global batch = 1 * 4 * 512 = 2048 tokens/step
N_LOOPS = 4          # recurrent depth (mythos_1b default is 16; reduced for memory/speed)
MAX_STEPS = 40
WARMUP_STEPS = 5
LOG_EVERY = 2
CKPT_EVERY = 20
LR = 3e-4
WD = 0.1
CKPT_DIR = "checkpoints_smoke"
DATASET_SUBSET = "sample-10BT"


def main() -> None:
    """Run a short single-GPU smoke train on streaming FineWeb-Edu."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # bf16-native params to fit ~1.41B params + AdamW on a 24 GB card. fp16 is
    # avoided (no GradScaler here); on non-bf16 GPUs fall back to fp32.
    param_dtype = torch.bfloat16 if bf16_ok else torch.float32
    logger.info(f"Device: {device} | param dtype: {param_dtype}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    logger.info(f"Tokenizer: gpt-oss-20b | vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Model (mythos_1b, shrunk loop depth, bf16-native to fit 24 GB)
    # ------------------------------------------------------------------
    cfg = mythos_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS

    model = OpenMythos(cfg).to(device=device, dtype=param_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,} ({n_params / 1e9:.2f}B)")

    global_batch_tok = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
    logger.info(
        f"seq_len={SEQ_LEN} | micro_batch={MICRO_BATCH} | grad_accum={GRAD_ACCUM} "
        f"| n_loops={N_LOOPS} | global_batch_tokens={global_batch_tok:,} "
        f"| max_steps={MAX_STEPS}"
    )

    # fused AdamW needs CUDA; states inherit the bf16 param dtype.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Streaming dataset (real FineWeb-Edu, pulled on demand)
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(
        encoding, SEQ_LEN, DATASET_SUBSET, rank=0, world_size=1
    )
    loader = DataLoader(
        dataset, batch_size=MICRO_BATCH, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    last_loss = float("nan")

    for step in range(1, MAX_STEPS + 1):
        cur_lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LR, LR * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(GRAD_ACCUM):
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
            loss = loss / GRAD_ACCUM
            loss.backward()
            loss_accum += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss_accum

        if step % LOG_EVERY == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * LOG_EVERY / dt
            logger.info(
                f"step {step:3d}/{MAX_STEPS} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec:,.0f} tok/s"
            )
            t0 = time.perf_counter()

        if step % CKPT_EVERY == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, CKPT_DIR,
                ddp=False, master=True,
            )

    # ------------------------------------------------------------------
    # Verify the smoke test actually exercised the pipeline.
    # ------------------------------------------------------------------
    assert torch.isfinite(torch.tensor(last_loss)), f"non-finite loss: {last_loss}"

    ckpts = _list_ckpts(CKPT_DIR)
    assert ckpts, f"no checkpoint written to {CKPT_DIR}/"
    # Reload the latest checkpoint into a fresh model/optimizer to confirm the
    # save/load round-trip works (the part most likely to silently break).
    fresh = OpenMythos(cfg).to(device=device, dtype=param_dtype)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=LR, fused=(device == "cuda"))
    resumed_step = load_checkpoint(fresh, fresh_opt, ckpts[-1], ddp=False)
    logger.success(
        f"Checkpoint round-trip OK ({ckpts[-1]}, step {resumed_step})"
    )

    if device == "cuda":
        logger.info(
            f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        )
    logger.success(f"Smoke test passed. Final loss: {last_loss:.4f}")


if __name__ == "__main__":
    main()
