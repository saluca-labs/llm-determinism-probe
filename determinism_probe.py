"""Probe: is the arithmetic deterministic, and does batch composition change the answer?

Tier 1 of the evidence plan for the 'deterministic function, probabilistic-looking output'
piece. No model download required. Two questions:

  Q1  Same input, same shape, repeated -> bitwise identical?
  Q2  Same input, DIFFERENT batch composition (padded with unrelated rows) -> bitwise identical?

Q1 tests determinism. Q2 tests batch-invariance. They are different properties and the
whole argument turns on the difference.
"""
import torch

torch.manual_seed(0)
D = 4096

# A fixed "weight matrix" and a fixed query row.
W = torch.randn(D, D, dtype=torch.float32)
x = torch.randn(1, D, dtype=torch.float32)

# ---- Q1: repeat the identical computation ----
runs = [ (x @ W) for _ in range(5) ]
identical = all(torch.equal(runs[0], r) for r in runs[1:])
print("Q1  same input, same shape, 5 runs, bitwise identical :", identical)

# ---- Q2: same row, different batch composition ----
alone = (x @ W)[0]

diffs = {}
for extra in (1, 7, 31, 127):
    others = torch.randn(extra, D, dtype=torch.float32)   # unrelated 'other users' requests'
    batch = torch.cat([x, others], dim=0)
    batched = (batch @ W)[0]                              # OUR row, out of the batch
    same = torch.equal(alone, batched)
    maxdiff = (alone - batched).abs().max().item()
    ndiff = int((alone != batched).sum().item())
    diffs[extra] = (same, maxdiff, ndiff)
    print(f"Q2  batch of 1+{extra:<4} identical: {str(same):5}  max|delta|: {maxdiff:.3e}  elems differing: {ndiff}/{D}")

# ---- consequence: does a last-bit delta flip an argmax? ----
# Simulate near-tied logits, the regime real next-token distributions live in.
print()
torch.manual_seed(1)
V = 32000
logits_a = torch.randn(V, dtype=torch.float32) * 0.01     # tight distribution -> near ties
# perturb by the magnitude we actually measured above
eps = max(d[1] for d in diffs.values())
logits_b = logits_a + (torch.randn(V) * eps)
flip = torch.argmax(logits_a).item() != torch.argmax(logits_b).item()
top2 = torch.topk(logits_a, 2).values
print(f"perturbation applied            : {eps:.3e}  (the max delta measured above)")
print(f"gap between top-1 and top-2     : {(top2[0]-top2[1]).item():.3e}")
print(f"argmax flipped                  : {flip}")
print(f"fraction of vocab within eps of top-1: {(logits_a > top2[0]-eps).sum().item()}")
