#!/usr/bin/env python3
"""
A string-in / string-out demo: character-level addition with OpenMythos.

The model reads a prompt string like ``"68+47="`` one character at a time and
must write the answer string ``"115"``. This is the most legible version of the
depth-bound reasoning theme: adding numbers needs carries to propagate from the
lowest digit upward, which is an inherently sequential computation — more digits
means more sequential steps, i.e. more "thinking". As with the permutation task,
we compare OpenMythos (which loops one block) against a parameter-matched vanilla
Transformer, and sweep OpenMythos's loop count to see whether "thinking longer"
helps.

Implementation notes
--------------------
- Vocabulary is just the characters ``0-9 + = _`` (``_`` = padding), so input and
  output are literally readable strings.
- The answer is produced **least-significant digit first** (the digits of the sum
  are reversed in the target), the standard trick that makes addition learnable
  left-to-right: each answer digit depends only on lower digits already emitted
  plus the carry. The end-of-run demo reverses it back so you see normal numbers.
- All samples in a batch share a digit count so batches are uniform-shaped (no
  padding needed during training); evaluation uses separate fixed-digit sets.

Run:
    python training/string_add.py                          # defaults (GPU if available)
    python training/string_add.py --dim 512 --steps 6000
    python training/string_add.py --help
"""

import os
import sys
import time
import argparse
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse config / baseline / helpers from reasoning_compare.py (which itself
# reuses tests/small_benchmark.py). Filenames are valid module names here.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "reasoning_compare", os.path.join(_HERE, "reasoning_compare.py")
)
_rc = importlib.util.module_from_spec(_spec)
sys.modules["reasoning_compare"] = _rc
_spec.loader.exec_module(_rc)

build_cfg = _rc.build_cfg
BaselineTransformer = _rc.BaselineTransformer
resolve_dtype = _rc.resolve_dtype
match_baseline_layers = _rc.match_baseline_layers

from open_mythos import OpenMythos  # noqa: E402


# ---------------------------------------------------------------------------
# Character-level addition task
# ---------------------------------------------------------------------------


class AdditionTask:
    """Generate ``a+b=`` prompts and reversed-sum answers as character ids.

    Vocabulary: digits 0-9, '+', '=', and '_' (pad, unused during training).
    """

    PAD = "_"
    VOCAB = list("0123456789+=" + PAD)

    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(self.VOCAB)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.VOCAB)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def prompt_len(self, digits: int) -> int:
        return 2 * digits + 2          # "DDD+DDD="

    def answer_len(self, digits: int) -> int:
        return digits + 1              # sum of two D-digit numbers has ≤ D+1 digits

    def make(self, batch_size: int, digits: int, device, generator):
        """Return (input_ids, targets, answer_len) for next-char prediction.

        Each sequence is ``"a+b=" + reverse(sum)``. Loss is scored only on the
        last ``answer_len`` target positions (the answer characters).
        """
        hi = 10 ** digits
        a = torch.randint(0, hi, (batch_size,), device=device, generator=generator)
        b = torch.randint(0, hi, (batch_size,), device=device, generator=generator)
        s = a + b
        seqs = []
        for ai, bi, si in zip(a.tolist(), b.tolist(), s.tolist()):
            ans = f"{si:0{digits + 1}d}"[::-1]          # reversed, fixed width
            seqs.append(self.encode(f"{ai:0{digits}d}+{bi:0{digits}d}=" + ans))
        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        return ids[:, :-1], ids[:, 1:], self.answer_len(digits)

    def format_example(self, digits: int, a: int, b: int, model_answer_rev: str) -> str:
        """Human-readable 'a + b = sum (model: ...)' line, un-reversing the answer."""
        true_sum = a + b
        model_sum = model_answer_rev[::-1].lstrip("0") or "0"
        ok = (model_sum == str(true_sum))
        mark = "OK " if ok else "X  "
        return (f"  {mark} {a:>{digits}} + {b:>{digits}} = {true_sum:<{digits + 1}}"
                f"   model: {model_sum}")


# ---------------------------------------------------------------------------
# Eval + generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy(model, task, digits, device, n_eval, batch_size, generator, n_loops=None):
    """Fraction of problems whose ENTIRE answer string is correct."""
    model.eval()
    correct = total = done = 0
    while done < n_eval:
        b = min(batch_size, n_eval - done)
        x, y, ans_len = task.make(b, digits, device, generator)
        logits = model(x, n_loops=n_loops) if isinstance(model, OpenMythos) else model(x)
        pred = logits[:, -ans_len:, :].argmax(-1)
        tgt = y[:, -ans_len:]
        correct += (pred == tgt).all(dim=1).sum().item()
        total += b
        done += b
    return correct / max(1, total)


@torch.no_grad()
def show_examples(model, task, digits, device, n, generator, n_loops=None):
    """Greedily decode a few problems and print readable 'a+b=sum' lines."""
    model.eval()
    plen = task.prompt_len(digits)
    ans_len = task.answer_len(digits)
    x, _, _ = task.make(n, digits, device, generator)
    # Recover a, b from the prompt portion of x (x is the full seq minus last char).
    lines = []
    cur = x[:, :plen].clone()           # just the "a+b=" prompt
    prompts = [task.decode(cur[i]) for i in range(n)]
    for _ in range(ans_len):
        logits = model(cur, n_loops=n_loops) if isinstance(model, OpenMythos) else model(cur)
        nxt = logits[:, -1:, :].argmax(-1)
        cur = torch.cat([cur, nxt], dim=1)
    gen_rev = [task.decode(cur[i, plen:plen + ans_len]) for i in range(n)]
    for p, g in zip(prompts, gen_rev):
        a_str, rest = p.split("+")
        b_str = rest.rstrip("=")
        lines.append(task.format_example(digits, int(a_str), int(b_str), g))
    return lines


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(model, task, args, device, gen, label):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    digits_choices = list(range(1, args.train_digits + 1))
    is_mythos = isinstance(model, OpenMythos)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        digits = digits_choices[step % len(digits_choices)]
        x, y, ans_len = task.make(args.batch_size, digits, device, gen)
        opt.zero_grad()
        if is_mythos:
            nl = int(torch.randint(
                args.train_loops_min, args.train_loops_max + 1, (1,),
                generator=gen, device=device,
            ).item())
            logits = model(x, n_loops=nl)
        else:
            logits = model(x)
        # Score only the answer characters (the last ans_len positions).
        loss = F.cross_entropy(
            logits[:, -ans_len:, :].reshape(-1, task.vocab_size).float(),
            y[:, -ans_len:].reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(f"  [{label}] step {step:>5}/{args.steps} | loss {loss.item():.4f} "
                  f"| {step / dt:.1f} steps/s")
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Character-level addition: OpenMythos vs vanilla Transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # model size (consumed by reasoning_compare.build_cfg)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--prelude-layers", type=int, default=1)
    p.add_argument("--coda-layers", type=int, default=1)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--expert-dim", type=int, default=0, help="0 = use --dim")
    p.add_argument("--lora-rank", type=int, default=8)
    # task / training
    p.add_argument("--train-digits", type=int, default=4,
                   help="train on additions of up to this many digits")
    p.add_argument("--eval-digits", default="3,4,6,8",
                   help="comma-separated digit counts to evaluate (some unseen/longer)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-loops-min", type=int, default=1)
    p.add_argument("--train-loops-max", type=int, default=8)
    p.add_argument("--baseline-layers", default="auto")
    p.add_argument("--depth-sweep", default="1,2,4,8,16")
    p.add_argument("--eval-samples", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    return p


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    task = AdditionTask()
    eval_digits = [int(s) for s in args.eval_digits.split(",") if s.strip()]
    max_digits = max(eval_digits + [args.train_digits])
    max_len = task.prompt_len(max_digits) + task.answer_len(max_digits)
    print(f"[task] char-level addition | vocab={task.vocab_size} "
          f"| train ≤{args.train_digits} digits | eval digits {eval_digits}")

    cfg = build_cfg(task.vocab_size, max_len, args)
    param_dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    mythos = OpenMythos(cfg).to(device=device, dtype=param_dtype)
    n_m = sum(p.numel() for p in mythos.parameters())

    baseline_layers = (match_baseline_layers(cfg, n_m, device)
                       if args.baseline_layers == "auto" else int(args.baseline_layers))
    torch.manual_seed(args.seed)
    baseline = BaselineTransformer(cfg, n_layers=baseline_layers).to(
        device=device, dtype=param_dtype)
    n_b = sum(p.numel() for p in baseline.parameters())
    print(f"[setup] dtype={param_dtype} | OpenMythos {n_m/1e6:.1f}M "
          f"(loops {args.train_loops_min}..{args.train_loops_max}) | "
          f"Baseline {n_b/1e6:.1f}M ({baseline_layers} layers)")

    print("\n[train] OpenMythos")
    gen.manual_seed(args.seed)
    train_one(mythos, task, args, device, gen, "mythos")
    print("\n[train] Baseline")
    gen.manual_seed(args.seed)
    train_one(baseline, task, args, device, gen, "base")

    # Accuracy by digit count
    print("\n" + "=" * 60)
    print("Whole-answer accuracy (% additions solved exactly)")
    print("=" * 60)
    print(f"  {'digits':>8} {'OpenMythos':>12} {'Baseline':>12}   note")
    eg = torch.Generator(device=device).manual_seed(args.seed + 1)
    for d in eval_digits:
        eg.manual_seed(args.seed + 1)
        am = accuracy(mythos, task, d, device, args.eval_samples, args.batch_size,
                      eg, n_loops=cfg.max_loop_iters)
        eg.manual_seed(args.seed + 1)
        ab = accuracy(baseline, task, d, device, args.eval_samples, args.batch_size, eg)
        note = "trained" if d <= args.train_digits else "longer (unseen)"
        print(f"  {d:>8} {am:>11.1%} {ab:>11.1%}   {note}")

    # Depth-extrapolation sweep
    sweep = sorted({int(s) for s in args.depth_sweep.split(",") if s.strip()})
    print("\n" + "=" * 60)
    print(f"Depth sweep — OpenMythos accuracy by n_loops (trained at {cfg.max_loop_iters})")
    print("=" * 60)
    print("  n_loops " + "".join(f"{('d=' + str(d)):>9}" for d in eval_digits))
    for nl in sweep:
        row = f"  {nl:>7} "
        for d in eval_digits:
            eg.manual_seed(args.seed + 1)
            row += f"{accuracy(mythos, task, d, device, args.eval_samples, args.batch_size, eg, n_loops=nl):>8.1%} "
        print(row + ("  ←trained" if nl == cfg.max_loop_iters else ""))

    # The actual string-in / string-out demo
    print("\n" + "=" * 60)
    print(f"String in → string out (OpenMythos, {cfg.max_loop_iters} loops)")
    print("=" * 60)
    demo_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    for d in sorted({min(args.train_digits, max(eval_digits)), max(eval_digits)}):
        print(f"  -- {d}-digit --")
        for line in show_examples(mythos, task, d, device, 5, demo_gen,
                                  n_loops=cfg.max_loop_iters):
            print(line)

    if device.type == "cuda":
        print(f"\n[mem] peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main(build_parser().parse_args())
