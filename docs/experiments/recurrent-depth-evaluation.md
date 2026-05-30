# Does recurrent depth help? An honest small-scale evaluation

**Question:** OpenMythos reuses one transformer block in a loop (the *Recurrent
Block*) instead of stacking many distinct layers. The hypothesis is that this
lets a model with *fewer unique weights* reach *greater effective depth*, and
that at inference it can "think longer" — loop more times than it was trained
with — to solve harder problems (**depth extrapolation**).

This document reports two small-scale experiments that test that hypothesis on a
single consumer GPU (NVIDIA RTX 4090, 24 GB), against a **parameter-matched
vanilla Transformer** built from the same attention/FFN kernels. Results are
reported as measured, including where OpenMythos does *not* win.

> **TL;DR.** At this scale (dim=128, ~0.7 M params, minutes of training) we did
> **not** observe a recurrent-depth advantage. A same-size (or larger) vanilla
> Transformer matched or beat OpenMythos on both an easy language task and a
> depth-bound reasoning task, and the "think longer" (depth-extrapolation)
> effect did not appear. See [Interpretation](#interpretation) for why this is a
> reasonable — not surprising — outcome.

---

## Setup

- **Hardware:** 1× RTX 4090 (24 GB), bf16/fp32, PyTorch 2.x.
- **OpenMythos:** tiny shared config (`dim=128`, MLA attention, 4 routed experts,
  `prelude=1 / recurrent=1 / coda=1` unique blocks).
- **Baseline:** `BaselineTransformer` — a vanilla decoder-only Transformer built
  from the *same* `TransformerBlock` primitive stacked non-recurrently, so any
  difference reflects the recurrent-depth architecture, not the kernels.
- **Fairness:** the baseline is given **at least as many parameters** as
  OpenMythos. Any OpenMythos advantage would therefore come from reusing weights
  for more effective depth, not from being the larger model.

---

## Experiment 1 — Language modelling (TinyStories)

Plain next-token prediction on TinyStories (gpt2 tokenizer), 1000 steps, both
models fed identical batches. Script: `tests/small_benchmark.py`.

```
python tests/small_benchmark.py --device cuda --steps 1000
```

| Metric | OpenMythos | Baseline |
|---|---|---|
| Params | 7.11 M | 6.88 M |
| Final train loss | 3.39 | **3.27** |
| Held-out eval loss | 3.33 | **3.22** |
| Throughput (tok/s) | 124 k | **308 k** (~2.5× faster) |

Depth-extrapolation sweep (trained at `n_loops=4`): eval loss got **worse** when
forced to loop more (8 → 3.87, 16 → 4.21 vs 4 → 3.40).

**Outcome:** the baseline is slightly better and ~2.5× faster; extra loops hurt.
This is expected — next-token prediction on simple stories is not a task that
requires iterative depth, so looping only adds cost.

---

## Experiment 2 — A depth-bound reasoning task (permutation composition)

To give recurrent depth its best chance, we use the **S_k word problem**: given a
sequence of permutations π₁…πₙ of *k* items, output the running composition at
every step. Composition is non-commutative, so step *t* needs the result of step
*t−1* — the answer **cannot be shortcut** and takes a number of sequential
reasoning steps that grows with the chain length. This is the canonical task
class where fixed-depth Transformers are known to struggle and recurrence is
*supposed* to help.

Script: `training/reasoning_compare.py`. Setup: `k=4` (vocab=24), trained on
chains of length 3–8, evaluated on longer unseen lengths. OpenMythos was trained
with a **random loop count (1–8) per step** so it could learn to use variable
depth. Baseline auto-sized to ≥ OpenMythos params.

```
python training/reasoning_compare.py --k 4 --steps 4000 \
    --train-min-len 3 --train-max-len 8 --eval-lens 4,8,12,16,24 \
    --train-loops-min 1 --train-loops-max 8 --depth-sweep 1,2,4,8,16,32
```

Params: **OpenMythos 684 k vs Baseline 752 k (5 layers)** — baseline is larger.

**Exact-match accuracy of the full composition** (random guess for S₄ = 1/24 ≈ 4.2 %):

| Chain length | OpenMythos | Baseline | |
|---|---|---|---|
| 4 (trained) | 95.8 % | **99.9 %** | both solve it |
| 8 (trained) | 8.3 % | **52.7 %** | baseline much better |
| 12 (unseen) | 4.6 % | 5.1 % | both ≈ random |
| 16 (unseen) | 4.6 % | 3.3 % | both ≈ random |
| 24 (unseen) | 4.8 % | 4.3 % | both ≈ random |

**Depth extrapolation** — OpenMythos accuracy by thinking depth (`n_loops`):

| n_loops | L=4 | L=8 | L=12 | L=16 | L=24 |
|---|---|---|---|---|---|
| 1 | 92.8 % | 8.5 % | 4.2 % | 3.9 % | 3.7 % |
| 4 | 95.7 % | 8.6 % | 4.5 % | 3.5 % | 4.9 % |
| 8 (trained) | 95.8 % | 8.3 % | 4.6 % | 4.6 % | 4.8 % |
| 32 | 93.3 % | 8.8 % | 4.2 % | 5.2 % | 4.8 % |

**Outcome:** the (larger) baseline matched or beat OpenMythos at every length;
neither model generalized to unseen lengths; and looping more produced **no
improvement** — the depth-extrapolation effect did not appear.

---

## Interpretation

The recurrent-depth advantage did **not** materialize at this scale. This is a
reasonable outcome rather than a bug:

1. **Length generalization is an open research problem.** Even purpose-built
   looped/universal Transformers usually fail to extrapolate to unseen sequence
   lengths without specialized techniques; a few minutes of training on a tiny
   model is not expected to crack it.
2. **Depth extrapolation needs more than random-loop training.** Producing the
   right answer at more loops than trained is delicate; our attempt (randomizing
   loops 1–8 during training) was not enough to induce it here.
3. **Scale.** dim=128 / ~0.7 M params is far below where emergent
   reasoning/depth benefits are typically reported.
4. **It is a reconstruction.** Per the project README, OpenMythos is "an
   independent, community-driven theoretical reconstruction based solely on
   publicly available research and speculation." The theorized
   depth-extrapolation property may simply not be realized by this
   implementation at this scale.

**What this does *not* claim:** that recurrent-depth Transformers are useless, or
that OpenMythos would not benefit from scale. It claims only what was measured:
no advantage at this small scale, on these tasks, with this training budget.

---

## Reproducing

Both scripts are self-contained and CLI-configurable:

```
# Experiment 1
python tests/small_benchmark.py --device cuda --steps 1000

# Experiment 2 (see --help for all knobs)
python training/reasoning_compare.py --help
python training/reasoning_compare.py --k 4 --steps 4000
```

`reasoning_compare.py` reuses the parameter-matched `BaselineTransformer` and
tiny config from `tests/small_benchmark.py`, implements the S_k
permutation-composition task, trains both models on identical data, and reports
accuracy by chain length plus the depth-extrapolation sweep.

To push the question further (larger `--k`, more `--steps`, bigger config, or a
different task), these harnesses are the starting point — but expect length
generalization and depth extrapolation to remain hard.
