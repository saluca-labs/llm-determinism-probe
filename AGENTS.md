# AGENTS.md

Instructions for AI coding agents working in this repository.

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
- LF line endings (`.gitattributes`).

## Do not

- Edit files in `results/`. They are the recorded 2026-08-23 runs the companion note reports; add
  new runs as new dated files instead.
- Commit `token_flip_result.json` written to the repo root by a local run.
- Commit secrets, hostnames, internal paths or personal emails.

## Done means

The three Test commands pass and the README still matches what the code on `main` does.
