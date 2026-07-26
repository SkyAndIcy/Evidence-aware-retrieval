# EARA: Evidence-Aware Retrieval Agents

---

## Overview

**EARA** (Evidence-Aware Retrieval Agent) is a framework that augments
retrieval-augmented LLM agents with an *evidence-sufficiency gate*. It targets
two systemic failures of current retrieval agents:

1. **Black-box retrieval** — agents issue queries but cannot assess whether the
   retrieved evidence is *sufficient* to answer.
2. **No abstention** — even when evidence is insufficient or contradictory,
   agents are forced to answer, producing overconfident wrong answers.

EARA interleaves ReAct-style step-by-step retrieval with a judge that scores
whether the gathered evidence sufficiently supports a proposed answer. Low
sufficiency triggers one more retrieval round; persistently low sufficiency
triggers honest abstention. This converts the open-loop "retrieve-then-answer"
pipeline into a closed-loop "monitor-then-decide" controller.

### Two components

- **CRP (Controllable Retrieval Process)** — decomposes the question into
  sub-queries, rewrites each query conditioned on the current evidence state,
  retrieves via BM25, and maintains an explicit evidence state
  $E_t = (c_t, \delta_t, \kappa_t)$ (coverage, conflict, credibility).
- **CAC (Calibrated Abstention Controller)** — reads the evidence state at
  every step and decides among **answer**, **retrieve-more**, or **abstain**
  via a conjunction of evidence-sufficiency and self-consistency thresholds.

The three-signal CAC is the general framework; the experiments instantiate it
with a single same-model self-judge sufficiency score (threshold $\tau_{cov}$).

---

## Repository layout

```
run.py                       CLI entry point: --model --benchmark --method --n --workers
core/                        core abstractions (interface + implementation split)
  base.py                    abstract BaseLLM / BaseRetriever contracts
  exceptions.py              SafetyViolation, RateLimitExhausted
  cache.py                   in-process response cache + ${ENV} resolution + call hashing
  tools.py                   OpenAI function-calling tool schemas + normalization
  llm.py                     concrete LLM (OpenAI-compatible gateway, AppId-pool round-robin)
  retriever.py               concrete Retriever (BM25) + Passage dataclass
data/
  datasets.py                loaders for HotpotQA, MuSiQue, MuSiQue-2hop, ...
methods/                     agent implementations (depend only on core interfaces)
  baselines.py               Direct / RAG / Self-Ask / CRAG / ReAct / IRCoT / Con
  eara.py                    EARA (three-signal CRP + CAC, general framework)
  eara_v4.py                 EARA-v4 + eara_ircot (single-signal self-judge instantiation, main config)
prompts/                     centralized prompt constants (decoupled from control flow)
  baselines_prompts.py       ReAct / RAG / Direct / Self-Ask / CRAG / Con prompts
  eara_prompts.py            decompose / rewrite / monitor / answer / self-consistency prompts
  eara_v4_prompts.py         evidence-sufficiency (coverage) judge prompt
evaluation/
  metrics.py                 Coverage@Acc, AUROC, abstention rate, cost summary
analysis/                    extended analyses
  case_agent.py              case-study agent (optional)
  falcon.py                  falcon baseline (optional)
  noise.py                   retrieval-noise utilities (optional)
utils/                       shared utilities
tools/
  scrub_appids.py            verify no AppIds leak in config before repackaging
tests/
  test_imports.py            package import smoke tests (post-refactor integrity)
  test_metrics.py            unit tests for Coverage@Acc / accuracy / normalization
config.yaml                  template config (AppIds / gateway URL are placeholders)
pyproject.toml               project metadata, dependencies, pytest config
requirements.txt             pip requirements (mirror of pyproject dependencies)
LICENSE                      MIT
```

The methods layer depends only on the abstract `BaseLLM` / `BaseRetriever`
interfaces in `core/base.py`, so agents are decoupled from any specific API
client or retrieval backend. Prompt strings live in `prompts/`, versioned
separately from agent control flow. Run `python -m pytest tests/` to verify
import integrity and metric correctness.
tools/
  scrub_appids.py           verify no AppIds leak before repackaging
config.yaml                 template config (AppIds / gateway URL are placeholders)
requirements.txt            python dependencies
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the LLM gateway

The framework talks to an OpenAI-compatible LLM gateway. Set the gateway URL
and your AppId(s) as environment variables (the template `config.yaml`
references them as `${LLM_GATEWAY_URL}` and `${LLM_APPID}`):

```bash
export LLM_GATEWAY_URL="https://your-gateway/v1/openai/native"
export LLM_APPID="your-app-id"        # or a space-separated list for pooling
```

Alternatively, edit `config.yaml` and replace the `${...}` placeholders with
your values directly (do not commit real AppIds).

### 3. Datasets

HotpotQA and MuSiQue are loaded via the HuggingFace `datasets` cache (the
loader falls back to the latest cached version in offline mode). MuSiQue-2hop
is the 2-hop subset of MuSiQue. We use a 500-question test set per benchmark;
the BM25 index is built from the distractor + supporting passages of each
benchmark.

---

---

## Reproducing the paper's results

The main table compares `eara_ircot_tau03` (self-judge sufficiency gate,
`tau_cov=0.3`, `T=10` retrieval steps) against five baselines on **four LLM
backbones** (Qwen-Plus, Gemini-2.5-Flash, GPT-4o-mini, Kimi-k2.5) × **three
benchmarks** (HotpotQA, MuSiQue, MuSiQue-2hop), 500 questions each.

### Running a configuration

```bash
python -m run --config config.yaml --model <model> \
  --benchmark <benchmark> --method eara_ircot_tau03 --n <n> --workers <workers>
```

Results are written to `results/{model}_{benchmark}_{method}.jsonl` with a
matching `_summary.json` (Coverage@Acc, accuracy, abstention rate, AUROC,
average retrievals, token cost). Sweep the model/benchmark/method flags to
cover the full main table.

### Ablation

`eara_ircot_noab` isolates the retrieve-more loop (gate triggers additional
retrieval but the agent never abstains — it commits a best guess at budget
exhaustion). Compare it against `react` and `eara_ircot_tau03` to decompose
the gain into retrieve-more and abstention contributions.

### Calibration

The `eara_ircot_tau0*` sweep produces the Coverage@Acc / AUROC / Accuracy
curves as a function of $\tau_{cov}$ (Figure in the paper). Coverage@Acc is
flat across $\tau \in [0.1, 0.5]$, so $\tau_{cov}=0.3$ is a stable operating
point.

---

## Key metrics

| Metric | Definition |
|---|---|
| **Coverage@Acc** | Accuracy on the answered subset (abstentions excluded). Primary metric. |
| **Accuracy** | Overall accuracy (abstentions counted as incorrect). |
| **Abstention rate** | Fraction of questions withheld. |
| **AUROC** | How well the abstention decision separates incorrect from correct answers. |
| **Avg retrievals** | Mean retrieval calls per question. |

See `src/evaluation/metrics.py` for implementations.

---

## Notes

- `tools/scrub_appids.py --check` verifies no hardcoded AppIds remain in
  `config.yaml` before repackaging.
- Temperature: 0.6 for the main agent loop where the API supports it; 0 for
  deterministic evaluation prompts. Reasoning models that force a specific
  temperature are handled in `src/core/llm.py`.
- All models are accessed as black-box APIs; no model weights are used or
  modified.

---

## Citation

This is an anonymous submission. The citation will be added upon acceptance.
