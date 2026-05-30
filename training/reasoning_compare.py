#!/usr/bin/env python3
"""
OpenMythos vs. a same-size vanilla Transformer on a depth-bound reasoning task.

Why this task
-------------
Plain language modelling (TinyStories etc.) does not need iterative depth, so a
looped model has no advantage there. This script instead uses the **permutation
composition** word problem (the S_k state-tracking task), which is the canonical
example of a problem that *requires* sequential computation:

    Given a sequence of permutations pi_1, pi_2, ..., pi_n of k items, output the
    running composition pi_1, then pi_1 o pi_2, ... at every step.

Computing step t needs the result of step t-1 (composition is non-commutative),
so the answer cannot be "shortcut" — it takes a number of sequential reasoning
steps that grows with the sequence length. A fixed-depth Transformer has a hard
ceiling on how many sequential steps it can represent; OpenMythos reuses one
block in a loop, so it can (in principle) keep composing by looping more — and
at inference it can loop *more times than it was trained with* to handle longer
chains (depth extrapolation).

Non-expert analogy: shuffle a deck one move at a time and track where a card
ends up. You cannot skip moves — you have to follow every shuffle in order.
"Thinking longer" (more loops) lets the model follow longer shuffle sequences.

What it compares
----------------
Both models share the same tiny config (dim=128) and are parameter-matched
(baseline depth = prelude + 1 recurrent block + coda), trained on the SAME
batches. We report **exact-match accuracy** of the final composition (intuitive:
"% of puzzles solved"), at the trained lengths and at longer, unseen lengths,
plus a depth-extrapolation sweep for OpenMythos.

Run:
    python training/reasoning_compare.py                       # defaults (GPU if available)
    python training/reasoning_compare.py --k 3 --steps 4000    # easier task, more steps
    python training/reasoning_compare.py --help
"""

import os
import sys
import time
import argparse
import importlib.util
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse the parameter-matched baseline + tiny config from tests/small_benchmark.py.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "small_benchmark", os.path.join(_HERE, "..", "tests", "small_benchmark.py")
)
_bench = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass inside the module can resolve __module__.
sys.modules["small_benchmark"] = _bench
_spec.loader.exec_module(_bench)

BaselineTransformer = _bench.BaselineTransformer
build_tiny_cfg = _bench.build_tiny_cfg

from open_mythos import OpenMythos  # noqa: E402


# ---------------------------------------------------------------------------
# Permutation-composition task
# ---------------------------------------------------------------------------


class PermutationComposition:
    """S_k word problem: vocab = the k! permutations, target = running composition.

    Each permutation is a tuple p of length k with p[i] = image of i. Composition
    is applied left to right: state_t = pi_t o state_{t-1}, where (a o b)[i] = a[b[i]].
    Token ids index the sorted list of all k! permutations, so input and target
    share one vocabulary of size k!.
    """

    def __init__(self, k: int):
        self.k = k
        self.perms = list(permutations(range(k)))  # k! tuples, sorted
        self.vocab_size = len(self.perms)
        self.index = {p: i for i, p in enumerate(self.perms)}
        # Composition table: compose[a, b] = id of (perm_a o perm_b).
        table = torch.empty(self.vocab_size, self.vocab_size, dtype=torch.long)
        for a, pa in enumerate(self.perms):
            for b, pb in enumerate(self.perms):
                composed = tuple(pa[pb[i]] for i in range(k))
                table[a, b] = self.index[composed]
        self.compose = table  # (V, V) lookup

    def batch(self, batch_size: int, seq_len: int, device, generator):
        """Return (input_ids, targets), both (batch_size, seq_len) long tensors.

        input_ids[b] is a random sequence of permutation tokens; targets[b, t] is
        the running composition of input_ids[b, 0..t] (the answer the model must
        produce at position t).
        """
        ids = torch.randint(
            0, self.vocab_size, (batch_size, seq_len),
            device=device, generator=generator,
        )
        targets = torch.empty_like(ids)
        state = ids[:, 0].clone()
        targets[:, 0] = state
        compose = self.compose.to(device)
        for t in range(1, seq_len):
            # state = pi_t o state  → compose[pi_t, state]
            state = compose[ids[:, t], state]
            targets[:, t] = state
        return ids, targets


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy(model, task, seq_len, device, n_eval, batch_size, generator, n_loops=None):
    """Exact-match accuracy of the FINAL running composition over n_eval samples.

    Final-position accuracy is the intuitive "did it solve the whole puzzle"
    metric. The model predicts the running composition at every position; we
    score the last one (the full composition of all seq_len permutations).
    """
    model.eval()
    correct = 0
    total = 0
    done = 0
    while done < n_eval:
        b = min(batch_size, n_eval - done)
        ids, targets = task.batch(b, seq_len, device, generator)
        logits = model(ids, n_loops=n_loops) if isinstance(model, OpenMythos) else model(ids)
        pred = logits[:, -1, :].argmax(-1)
        correct += (pred == targets[:, -1]).sum().item()
        total += b
        done += b
    return correct / max(1, total)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(model, task, args, device, gen, label):
    """Train a single model on mixed-length permutation sequences.

    For OpenMythos, the recurrent loop count is randomized per step in
    [train_loops_min, train_loops_max] so the model learns to produce the right
    answer at a *range* of thinking depths — the prerequisite for using extra
    loops at test time (depth extrapolation). The baseline has no loops.
    """
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    lengths = list(range(args.train_min_len, args.train_max_len + 1))
    is_mythos = isinstance(model, OpenMythos)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        # Sample one length per step so every batch is uniform-shaped (no padding).
        seq_len = lengths[step % len(lengths)]
        ids, targets = task.batch(args.batch_size, seq_len, device, gen)
        opt.zero_grad()
        if is_mythos:
            nl = int(torch.randint(
                args.train_loops_min, args.train_loops_max + 1, (1,), generator=gen,
                device=device,
            ).item())
            logits = model(ids, n_loops=nl)
        else:
            logits = model(ids)
        loss = F.cross_entropy(
            logits.reshape(-1, task.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(
                f"  [{label}] step {step:>5}/{args.steps} | loss {loss.item():.4f} "
                f"| {step / dt:.1f} steps/s"
            )
    return model


def match_baseline_layers(cfg, target_params: int, device) -> int:
    """Pick the baseline depth whose param count is closest to `target_params`.

    Keeps the comparison honest: the baseline gets at least as many parameters
    as OpenMythos, so any OpenMythos advantage comes from reusing weights for
    more effective depth, not from being the bigger model.
    """
    best_n, best_gap = 1, None
    for n in range(1, 13):
        p = sum(
            t.numel() for t in BaselineTransformer(cfg, n_layers=n).parameters()
        )
        gap = abs(p - target_params)
        if best_gap is None or gap < best_gap:
            best_n, best_gap = n, gap
        if p > target_params:
            break
    return best_n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenMythos vs vanilla Transformer on the S_k permutation-"
        "composition reasoning task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=5,
                   help="permutation size; vocab = k! (k=5 → 120, the classic hard case)")
    p.add_argument("--train-min-len", type=int, default=3)
    p.add_argument("--train-max-len", type=int, default=12,
                   help="train on chains of this many permutations and shorter")
    p.add_argument("--eval-lens", default="6,12,18,24,32",
                   help="comma-separated chain lengths to evaluate (some unseen/longer)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-loops-min", type=int, default=1,
                   help="min recurrent loops sampled per training step (OpenMythos)")
    p.add_argument("--train-loops-max", type=int, default=8,
                   help="max recurrent loops sampled per step; also cfg.max_loop_iters")
    p.add_argument("--baseline-layers", default="auto",
                   help="'auto' = param-match OpenMythos, or an integer layer count")
    p.add_argument("--depth-sweep", default="1,2,4,8,16,32",
                   help="n_loops values for OpenMythos depth-extrapolation eval")
    p.add_argument("--eval-samples", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    task = PermutationComposition(args.k)
    eval_lens = [int(s) for s in args.eval_lens.split(",") if s.strip()]
    max_len = max(eval_lens + [args.train_max_len])
    print(
        f"[task] S_{args.k} permutation composition | vocab={task.vocab_size} "
        f"| train lengths {args.train_min_len}..{args.train_max_len} "
        f"| eval lengths {eval_lens}"
    )
    print(f"[setup] device={device} | steps={args.steps} | batch={args.batch_size}")

    # Shared tiny config; seq_len budget must cover the longest eval length.
    cfg = build_tiny_cfg(task.vocab_size, max_len)
    # max_loop_iters bounds the LoRA depth adapters / loop embeddings; set it to
    # the largest depth used in training. Test-time loops beyond this are clamped
    # internally (depth extrapolation).
    cfg.max_loop_iters = args.train_loops_max

    torch.manual_seed(args.seed)
    mythos = OpenMythos(cfg).to(device)
    n_m = sum(p.numel() for p in mythos.parameters())

    if args.baseline_layers == "auto":
        baseline_layers = match_baseline_layers(cfg, n_m, device)
    else:
        baseline_layers = int(args.baseline_layers)
    torch.manual_seed(args.seed)
    baseline = BaselineTransformer(cfg, n_layers=baseline_layers).to(device)
    n_b = sum(p.numel() for p in baseline.parameters())
    print(
        f"[setup] OpenMythos params={n_m:,} "
        f"(trains at {args.train_loops_min}..{args.train_loops_max} loops) | "
        f"Baseline params={n_b:,} ({baseline_layers} fixed layers)"
    )

    # Same data stream for both: re-seed the generator before each model so they
    # see identical batches.
    print("\n[train] OpenMythos")
    gen.manual_seed(args.seed)
    train_one(mythos, task, args, device, gen, "mythos")
    print("\n[train] Baseline")
    gen.manual_seed(args.seed)
    train_one(baseline, task, args, device, gen, "base")

    # ------------------------------------------------------------------
    # Accuracy vs. chain length (intuitive "% solved")
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Exact-match accuracy of the full composition (% puzzles solved)")
    print("=" * 64)
    print(f"  {'chain length':>12} {'OpenMythos':>12} {'Baseline':>12}   note")
    eval_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    for L in eval_lens:
        eval_gen.manual_seed(args.seed + 1)  # same eval set for both models
        acc_m = accuracy(mythos, task, L, device, args.eval_samples,
                         args.batch_size, eval_gen, n_loops=cfg.max_loop_iters)
        eval_gen.manual_seed(args.seed + 1)
        acc_b = accuracy(baseline, task, L, device, args.eval_samples,
                         args.batch_size, eval_gen)
        note = "trained" if L <= args.train_max_len else "longer (unseen)"
        print(f"  {L:>12} {acc_m:>11.1%} {acc_b:>11.1%}   {note}")

    # ------------------------------------------------------------------
    # Depth extrapolation: OpenMythos accuracy as a function of n_loops.
    # ------------------------------------------------------------------
    sweep = sorted({int(s) for s in args.depth_sweep.split(",") if s.strip()})
    print("\n" + "=" * 64)
    print(f"Depth extrapolation — OpenMythos accuracy by 'thinking depth' (n_loops)")
    print(f"(trained at n_loops={cfg.max_loop_iters})")
    print("=" * 64)
    header = "  n_loops " + "".join(f"{('L=' + str(L)):>10}" for L in eval_lens)
    print(header)
    for nl in sweep:
        row = f"  {nl:>7} "
        for L in eval_lens:
            eval_gen.manual_seed(args.seed + 1)
            acc = accuracy(mythos, task, L, device, args.eval_samples,
                           args.batch_size, eval_gen, n_loops=nl)
            row += f"{acc:>9.1%} "
        marker = "  ←trained" if nl == cfg.max_loop_iters else ""
        print(row + marker)


if __name__ == "__main__":
    main(build_parser().parse_args())
