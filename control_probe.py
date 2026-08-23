"""Controls for token_flip_probe.py.

The tier-2 result (bf16: batch of 1 and batch of >1 diverge at token 13) only means
something if two things hold:

  C1  Repeatability WITHIN a configuration. Run the same batch size twice - identical?
      If not, the model is simply nondeterministic run to run and batch composition
      explains nothing.

  C2  Does batch SIZE matter, or only batched-vs-alone? 2, 8 and 32 all diverged at the
      same token and produced the same text as each other, which suggests the latter.

Also re-runs fp32 at the same token budget as bf16, so the precision comparison is fair
rather than an artefact of the shorter budget used in the first fp32 run.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NEW_TOKENS = 120
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL)

PROMPT = "Explain in plain language why a bridge needs expansion joints."
FILLERS = [
    "Describe the water cycle for a school textbook, step by step and clearly.",
    "Summarise the causes of the 1929 crash for a general reader in plain prose.",
    "Write a short paragraph about the migration patterns of arctic terns each year.",
    "Give a concise account of how a heat pump moves warmth from cold outdoor air.",
]


def ids(t):
    return list(tok.apply_chat_template([{"role": "user", "content": t}],
                                        add_generation_prompt=True, tokenize=True)["input_ids"])


ours = ids(PROMPT)
L = len(ours)
fill = []
for f in FILLERS:
    t = ids(f)
    while len(t) < L:
        t = t + t
    fill.append(t[:L])


def run(model, bs):
    rows = [ours] + [fill[i % len(fill)] for i in range(bs - 1)]
    x = torch.tensor(rows, dtype=torch.long)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                         max_new_tokens=NEW_TOKENS, do_sample=False, num_beams=1,
                         pad_token_id=tok.eos_token_id)
    return out[0, x.shape[1]:].tolist()


for dtype in ("bfloat16", "float32"):
    print(f"\n=== {dtype}, {NEW_TOKENS} new tokens, greedy ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, dtype)).eval()

    # C1 - repeatability within a configuration
    for bs in (1, 8):
        a, b = run(model, bs), run(model, bs)
        print(f"C1  batch {bs:<3} run twice -> identical: {a == b}")

    # C2 - batched vs alone, and does size matter
    base = run(model, 1)
    prev = None
    for bs in (2, 4, 8, 32):
        g = run(model, bs)
        first = next((i for i, (p, q) in enumerate(zip(base, g)) if p != q), None)
        same_as_prev = (prev is not None and g == prev)
        print(f"C2  batch {bs:<3} vs batch 1 -> first divergence: "
              f"{'none' if first is None else '#%d' % first}"
              f"   identical to previous batch size: {same_as_prev if prev is not None else '-'}")
        prev = g
    del model
