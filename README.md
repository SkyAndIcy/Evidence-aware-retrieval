# EARA: Evidence-Aware Retrieval and Abstention for Multi-Hop QA

**EARA** is a modular framework that separates *information acquisition* from
*answer release* in retrieval-augmented agents:

- **CRP — Controllable Retrieval Process.** Decomposes the question into a
  directed list of evidence-seeking sub-questions with explicit dependencies,
  schedules an unresolved sub-question against the accumulated evidence, and
  rewrites each query using only retrieved support (never injected facts).
- **CAC — Calibrated Abstention Controller.** At every answer proposal, a
  reviewer scores whether the displayed evidence fully supports the candidate —
  every required multi-hop link — and the controller chooses **answer**,
  **retrieve-more**, or **abstain** (Eq. 2 of the paper).

The two components expose separate interfaces, so each can be attached to a
retrieval-capable host agent without replacing its answer generator, and their
effects are measured with the factorial gains Δ_C, Δ_A, and interaction
(Eq. 3–4).

> **Status.** This repository accompanies an anonymous submission under
> double-blind review. It is the *implementation* of the evaluation protocol:
> the controller, retrieval stack, grids, audit tooling, and offline
> validation. It contains **no fabricated results** — result cells appear only
> when runs actually execute, and `python -m experiments.eara capabilities`
> prints an honest inventory of what is and is not implemented.

## Repository layout

```
experiments/
  eara/                    the framework package
    agents.py              CRP planning/rewrite/scheduling + CAC gate + host adapters
                           (ReAct, Self-Ask, IRCoT, CRAG-local, Search-o1, tagged
                           search-R1/ZeroSearch adapters, direct/CoT)
    retrieval.py           BM25s index build, exact dense (BGE-M3) index,
                           weighted reciprocal-rank fusion (11 mixture settings)
    data.py                DPR-style 2018-Wikipedia corpus import,
                           five dataset normalizers, frozen dev/test splits
    runner.py              grid construction, resumable runs, run fingerprints
    metrics.py             EM/F1, coverage, selective accuracy, yield,
                           paired cluster bootstrap, transfer effects (Eq. 3-4)
    audit.py               frozen-tuple export, model labeling, blind human
                           audit sampling, kappa/macro-F1, threshold calibration
    gateway.py             concurrency caps, 429 cooldown (no key rotation),
                           request ledger, secret redaction
    protocol.py            protocol guards + capability inventory
  configs/                 paper.yaml + the experiment grids
                           (main, decomposition, planner_crossed, rewriting,
                           rewriting_control, retrieval, transfer)
  validate_framework.py    end-to-end offline validation (synthetic corpus +
                           real BM25s + mock transport)
  test_controller_contract.py   17 unit tests for the controller contract
  serve_qwen.sh            local Qwen3-8B / Qwen3-4B serving helpers
  credentials.example.yaml gateway credential template (never commit real keys)
  fixtures/                tiny synthetic corpus used by the validator
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/requirements.txt

# honest inventory of implemented vs. unimplemented capabilities
python -m experiments.eara capabilities

# end-to-end offline validation: synthetic corpus, real BM25s, mock LLM
PYTHONPATH=. python experiments/validate_framework.py

# controller contract tests
PYTHONPATH=. python experiments/test_controller_contract.py
```

Real runs additionally require (all supplied externally, none included here):
the December-2018 English Wikipedia passage corpus, the five benchmark source
files, model endpoints, and human audit annotations.

```bash
python -m experiments.eara doctor          # readiness report, zero API calls

# data pipeline
python -m experiments.eara download-corpus --output psgs_w100.tsv.gz --max-gib 5
python -m experiments.eara import-corpus --input psgs_w100.tsv.gz
python -m experiments.eara index-bm25
python -m experiments.eara normalize-dataset --name hotpotqa --input ... --source-url ...
python -m experiments.eara freeze-splits

# experiments (add --confirm-api-use to authorize billable requests)
python -m experiments.eara make-grid --family main --output main.json
python -m experiments.eara run --grid main.json --condition-id eara --mode pilot --limit 50
```

## Protocol guarantees enforced in code

- Main comparison: shared 2018 Wikipedia corpus, BM25s, top-5 retrieval,
  10 logical retrieval calls per question (cached queries still count).
- One candidate per proposal; one format-only retry, then abstention.
- Paper mode refuses to run without a development-selected, closed-loop
  validated threshold artifact and frozen model revisions.
- Unimplemented configurations fail loudly instead of silently degrading
  (e.g., five-sample self-consistency, TIR, unresolved checkpoints).
- Hard-limit terminations, abstention reasons, and infrastructure failures
  are reported separately, never silently dropped.
- Every request is logged to a redacted ledger; secrets never enter records.

## License

MIT (anonymous submission; see LICENSE).
