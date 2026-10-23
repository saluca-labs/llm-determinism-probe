# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this is

A CPU-only reproduction that a language model is a deterministic function while its serving system
is not batch-invariant. It is three standalone research scripts and their recorded outputs. It is
not a library and there is nothing to import.

## Read in this order

1. `README.md`: the claim, the "How it works" diagram, the results and which command wrote each.
2. `determinism_probe.py`: tier 1, the smallest form of the effect (one matmul).
3. `token_flip_probe.py`: tier 2, the same effect through a real model.
4. `control_probe.py`: the controls that make tier 2 mean something.
5. `results/`: the recorded 2026-08-23 outputs.

## What you can and cannot see

- You can see all of the code. There is no closed part and no other repo it depends on.
- The companion note is on Zenodo (concept DOI 10.5281/zenodo.22072387), not in this repo.
- The model, `Qwen/Qwen2.5-0.5B-Instruct`, is downloaded from Hugging Face on the first tier 2 run
  (about 1 GB). Tier 1 needs no download.

## The only numbers to trust

- Tier 1 deltas: `results/RESULT-2026-08-23.txt`.
- Tier 2 float32 logit delta (2.193e-05) and gap (5.308e-01): `results/token_flip_result.json`.
- Tier 2 bfloat16 divergence at token 13: `results/FLIP-RESULT-bf16-2026-08-23.txt` and
  `results/token_flip_result_bf16.json`.
- Repeatability and batch-size controls: `results/CONTROL-RESULT-2026-08-23.txt`.
- Advisory floors for torch and transformers: `MIN_SAFE` in `tools/check_floors.py`.

A re-run on other hardware or library versions gives other magnitudes and another divergence point
(the 2026-10-22 re-run diverged at token 60). Do not overwrite the recorded files with those.

## Quickstart

```bash
python determinism_probe.py      # tier 1, seconds, no model download
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  -r requirements.txt
```

## Test

The same checks CI runs (`.github/workflows/check.yml`):

```bash
python tools/check_floors.py
python -m py_compile determinism_probe.py token_flip_probe.py control_probe.py
python determinism_probe.py
```

Passing looks like `OK: requirement floors are at or above the known-safe versions`, no compile
error, and tier 1 printing `Q1 ... bitwise identical : True` with every `Q2` line
`identical: False`. Tier 2 (`token_flip_probe.py`, `control_probe.py`) downloads about 1 GB and
takes minutes on CPU; run it when a change touches those scripts.

## Conventions

- Dependencies are torch and transformers only. Do not add others.
- Keep the requirement floors at or above `MIN_SAFE` in `tools/check_floors.py`; raise both
  together when new advisories land.
- No sampling, no padding: greedy decoding and length-matched filler rows are the point of the
  design.
- No em-dashes in docs. If the README changes, regenerate `llms-full.txt` in the same change.
- If a script's steps change, update the Mermaid diagram in the README in the same change.
- LF line endings (`.gitattributes`).

## Do not

- Edit files in `results/`. They are the recorded 2026-08-23 runs the companion note reports; add
  new runs as new dated files instead.
- Commit `token_flip_result.json` written to the repo root by a local run.
- Commit secrets, hostnames, internal paths or personal emails.

## Done means

The three Test commands pass and the README still matches what the code on `main` does.
