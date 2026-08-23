# llm-determinism-probe

A minimal, CPU-only reproduction of a claim that is easy to state and easy to get wrong:

> **A language model is a deterministic function. The system serving it is not reproducible.
> The gap between those two facts is engineering, not indeterminacy.**

Everything here runs on a laptop. No GPU, no API key, no cluster. `pip install torch
transformers`, one ~1 GB model download, and a few minutes.

**This is a reproduction, not a discovery.** The phenomenon was documented and then *fixed* by
Thinking Machines Lab in *Defeating Nondeterminism in LLM Inference*, which identified the
cause as kernels that are not batch-invariant and shipped batch-invariant replacements. What
this repository adds is a version small enough that anyone can run it and see the effect
themselves, plus two findings we did not find stated elsewhere: that the operative variable is
**alone versus not alone** rather than batch size, and that at float32 the perturbation exists
but never reaches the output.

## The claim, split in two

These are different properties and the whole argument depends on not confusing them:

| Property | Question | Result |
|---|---|---|
| **Determinism** | Same configuration twice, same output? | **Yes.** Bitwise identical, always. |
| **Batch invariance** | Same request, different company in the batch, same output? | **No.** Diverges at token 13 at bf16. |

## Results (2026-08-23, CPU, Qwen2.5-0.5B-Instruct, greedy decoding throughout)

There is no sampling anywhere in this repository. Temperature is not set to zero; it is not in
the loop at all. Every result below is greedy `argmax` decoding.

**Tier 1 - raw matmul, no model.** One 1x4096 row against a 4096x4096 matrix.

```
Q1  same input, same shape, 5 runs, bitwise identical : True
Q2  batch of 1+1    identical: False  max|delta|: 3.052e-04  elems differing: 3941/4096
Q2  batch of 1+7    identical: False  max|delta|: 1.984e-04  elems differing: 3852/4096
Q2  batch of 1+31   identical: False  max|delta|: 1.984e-04  elems differing: 3852/4096
Q2  batch of 1+127  identical: False  max|delta|: 1.984e-04  elems differing: 3852/4096
```

**Tier 2 - a real model, 120 generated tokens.**

| | float32 | bfloat16 |
|---|---|---|
| Same config run twice (batch 1, batch 8) | identical | identical |
| Logit delta from batch composition, final prompt position | 2.193e-05, changing 146,797 / 151,936 logits | 0.0 (rounds away at prefill) |
| Top-1 / top-2 gap at that position | 5.308e-01 | 3.750e-01 |
| **First divergent generated token, batch >1 vs batch 1** | **none in 120** | **#13** |
| Batches of 2, 4, 8, 32 identical to each other | yes | yes |

The divergence is not cosmetic:

```
alone      ... longevity of bridges over various terrains.
           Here's an explanation in plain language:

in company ... longevity of bridges.
           Here's why they need to be expanded:
```

Same question. Same weights. No sampling. A different answer, because of who else was in the
batch. The second reply also drifts from what was asked.

## Why there is no padding anywhere

This is the part that makes the result mean something.

The usual way to batch sequences of different lengths is to pad and mask. That introduces a
second variable and invites the obvious objection: that the computation changed, not the
arithmetic. So there is no padding here. Every filler sequence is trimmed to **exactly** the
token length of the sequence under test, making each batch a clean rectangle containing no
padding token at all.

Under a causal mask, the row under test cannot attend to any other row. Its result is
therefore *mathematically* identical whether computed alone or in company. Every difference
measured here is floating-point reduction order and nothing else.

## Running it

```bash
pip install -r requirements.txt

python determinism_probe.py      # tier 1, seconds, no model download
python token_flip_probe.py       # tier 2, downloads ~1 GB on first run
python control_probe.py          # the controls, both precisions
```

`token_flip_probe.py` reads four environment variables: `PROBE_DTYPE` (`float32` | `bfloat16`),
`PROBE_TOKENS`, `PROBE_PROMPT`, `PROBE_OUT`.

```bash
PROBE_DTYPE=bfloat16 PROBE_TOKENS=120 python token_flip_probe.py
```

Captured output and machine-readable results from our run are in [`results/`](results/).

**Expect your numbers to differ from ours.** Different CPUs, BLAS builds and library versions
change the exact magnitudes and the exact divergence point. What should reproduce is the
*shape*: repeatable within a configuration, not repeatable across batch composition, and worse
at lower precision.

## Honest limits

- One 0.5B model, one prompt, CPU, static rectangular batching. Production serving adds
  continuous batching, chunked prefill, KV-cache reuse, tensor parallelism and expert routing.
  None of that is simulated here.
- float32 showed no output divergence within a 120-token budget. It may diverge over a longer
  one. Not tested.
- At bfloat16 the prefill logits are identical across batch sizes while generation still
  diverges, so the effect enters during incremental decode, a different kernel path. Not
  separately instrumented.
- Only batch composition is varied. Hardware differences, kernel autotuning and quantisation
  are independently known to move outputs and are not tested here.

## Why it matters

Determinism is not predictability. Knowing the function is deterministic tells you nothing
about what it will say; the system is sensitive enough that a difference in the last bits of a
reduction becomes a different sentence thirteen tokens later.

What determinism buys is **reconstruction**. A deterministic function plus a complete record of
its inputs is an answer you can re-run, examine and contest. That is what auditing a model
decision actually requires, and it is why the missing half is almost never the model. It is the
record: the prompt, the system prompt version, the retrieved documents, the tool outputs, the
build, the precision, the batch conditions. Almost nobody keeps them.

"The model decided" is nearly always "the context differed, and nobody kept the context."

## Related

- Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* - 1,000 identical requests
  to a 235B model returning 80 distinct completions; batch-invariant kernels achieving
  1,000/1,000 bit-identical output at roughly 61.5% throughput cost.
- LMSYS / SGLang, *Towards Deterministic Inference in SGLang and Reproducible RL Training*
  (22 September 2025) - the integration that reduced that cost to roughly a third.
- [`saluca-labs/velamen`](https://github.com/saluca-labs/velamen) - steganography by arithmetic
  coding over *frozen* token distributions. Encode and decode round-trip with no live inference
  at all, which only works if the distribution is a well-defined function of the prompt. A
  shipping artifact that would fail silently if the claim in this repository were false. The
  same package carries HCTP, a hash-chain context transfer protocol, which is the record half.

## Citing this

The accompanying note is *Deterministic Function, Non-Reproducible Deployment: A Minimal
Reproduction of Batch-Dependent Output Variation in a Language Model, and What It Implies for
Auditability* (Ruvalcaba and the Saluca Agentic AI Research Team, Saluca LLC, 2026), deposited to
Zenodo under `10.5281/zenodo.22072388`. **That DOI is reserved and the deposit is pending
publication at the time of writing**, so it may not resolve yet. See `CITATION.cff`.

## Licence

Apache License 2.0. Copyright 2026 Saluca LLC.
