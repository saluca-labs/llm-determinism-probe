"""Tier 2: does batch-dependent arithmetic actually flip a generated token?

Tier 1 (determinism_probe.py) showed one matmul row changes in the low bits when unrelated
rows share the batch. This asks the only question that matters downstream: does that reach
the output?

Design note - why there is no padding here. Padding plus a mask would introduce a second
variable and invite the objection that we changed the computation rather than the numerics.
Instead every filler sequence is trimmed to EXACTLY the same token length as ours, so the
batch is a clean rectangle with no PAD token anywhere. Under a causal mask our row cannot
attend to any other row, so our row's result is mathematically identical whether computed
alone or in company. Any difference observed is therefore pure floating-point reduction
order, not a change of computation.

Everything is greedy (argmax). There is no sampling anywhere in this file.

Phase A - one forward pass, batch 1 vs batch K, compare the final-position logits.
Phase B - full greedy generation, batch 1 vs batch K, find the first divergent token.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
import os
DTYPE = os.environ.get("PROBE_DTYPE", "float32")
NEW = int(os.environ.get("PROBE_TOKENS", "80"))
BATCH_SIZES = [2, 8, 32]
NEW_TOKENS = 80  # overridden by PROBE_TOKENS below

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, DTYPE)).eval()

PROMPT = os.environ.get("PROBE_PROMPT", "Explain in plain language why a bridge needs expansion joints.")
FILLERS = [
    "Describe the water cycle for a school textbook, step by step and clearly.",
    "Summarise the causes of the 1929 crash for a general reader in plain prose.",
    "Write a short paragraph about the migration patterns of arctic terns each year.",
    "Give a concise account of how a heat pump moves warmth from cold outdoor air.",
]


def ids(text):
    enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True)
    return list(enc["input_ids"])


ours = ids(PROMPT)
L = len(ours)
fillers = []
for f in FILLERS:
    t = ids(f)
    while len(t) < L:
        t = t + t
    fillers.append(t[:L])


def batch_for(bs):
    rows = [ours] + [fillers[i % len(fillers)] for i in range(bs - 1)]
    return torch.tensor(rows, dtype=torch.long)


NEW_TOKENS = NEW
report = {"model": MODEL, "dtype": DTYPE, "device": "cpu", "decoding": "greedy",
          "prompt_tokens": L, "new_tokens": NEW_TOKENS, "phase_a": {}, "phase_b": {}}

print(f"model  : {MODEL}   fp32, cpu, greedy")
print(f"prompt : {L} tokens, no padding anywhere (fillers trimmed to the same length)\n")

# ---------------- Phase A: logits at the final prompt position ----------------
base_logits = model(input_ids=batch_for(1)).logits[0, -1, :].float()
b2 = torch.topk(base_logits, 2)
base_gap = float(b2.values[0] - b2.values[1])
print("PHASE A - final-position logits, batch of 1 vs batch of K")
print(f"  baseline top1-top2 gap: {base_gap:.3e}   top1 token: {int(b2.indices[0])}")

for bs in BATCH_SIZES:
    lg = model(input_ids=batch_for(bs)).logits[0, -1, :].float()
    delta = float((base_logits - lg).abs().max())
    ndiff = int((base_logits != lg).sum())
    flip = int(torch.argmax(base_logits)) != int(torch.argmax(lg))
    within = int((base_logits > b2.values[0] - delta).sum())
    print(f"  batch {bs:<3} max|delta|: {delta:.3e}   logits differing: {ndiff}/{lg.numel()}"
          f"   argmax flipped: {flip}   vocab within delta of top1: {within}")
    report["phase_a"][bs] = {"max_abs_delta": delta, "logits_differing": ndiff,
                             "vocab_size": int(lg.numel()), "argmax_flipped": flip,
                             "vocab_within_delta_of_top1": within,
                             "baseline_top1_top2_gap": base_gap}

# ---------------- Phase B: full greedy generation ----------------
print(f"\nPHASE B - greedy generation of {NEW_TOKENS} tokens")


def gen(bs):
    x = batch_for(bs)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                         max_new_tokens=NEW_TOKENS, do_sample=False,
                         num_beams=1, pad_token_id=tok.eos_token_id)
    return out[0, x.shape[1]:].tolist()


base_gen = gen(1)
print("  baseline generated, comparing...")
report["phase_b"]["baseline_tokens"] = base_gen
texts = {1: tok.decode(base_gen, skip_special_tokens=True)}

for bs in BATCH_SIZES:
    g = gen(bs)
    first = next((i for i, (a, b) in enumerate(zip(base_gen, g)) if a != b), None)
    same_text = tok.decode(g, skip_special_tokens=True) == texts[1]
    texts[bs] = tok.decode(g, skip_special_tokens=True)
    print(f"  batch {bs:<3} first divergent token: "
          f"{('none in %d' % NEW_TOKENS) if first is None else '#%d' % first}"
          f"   identical text: {same_text}")
    report["phase_b"][bs] = {"first_divergent_token": first, "identical_text": same_text,
                             "tokens": g}

print("\n--- baseline (batch of 1) ---")
print(texts[1].replace("\n", " ")[:400])
for bs in BATCH_SIZES:
    if report["phase_b"][bs]["first_divergent_token"] is not None:
        print(f"\n--- batch of {bs}, diverges at token "
              f"#{report['phase_b'][bs]['first_divergent_token']} ---")
        print(texts[bs].replace("\n", " ")[:400])

report["phase_b"]["texts"] = {str(k): v for k, v in texts.items()}
with open(os.environ.get("PROBE_OUT","token_flip_result.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print("\nwrote token_flip_result.json")
