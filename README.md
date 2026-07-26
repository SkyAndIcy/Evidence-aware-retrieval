# EARA: Evidence-Aware Retrieval Agents

EARA is a training-free framework that makes retrieval-augmented LLM agents
**evidence-aware**: they can tell whether the retrieved evidence is enough to
answer, retrieve more when it is not, and **abstain** honestly when it still
isn't — instead of hallucinating a confident wrong answer.

---

## Why EARA?

Current retrieval agents (ReAct, RAG, Self-Ask, …) suffer two failures:

1. **Black-box retrieval.** The agent fires a query and consumes whatever
   comes back. It has no notion of whether the evidence is *sufficient*.
2. **No abstention.** Even when evidence is missing or contradictory, the
   agent is forced to answer — producing overconfident, wrong answers that a
   user cannot distinguish from correct ones.

EARA adds an **evidence-sufficiency gate** on top of ReAct-style retrieval.
A judge scores whether the gathered evidence supports the proposed answer;
low scores trigger one more retrieval round, and persistently low scores
trigger abstention. This turns the open-loop *retrieve-then-answer* pipeline
into a closed-loop *monitor-then-decide* controller.

---

## How it works

EARA has two co-dependent components:

| Component | Role |
|---|---|
| **CRP** — Controllable Retrieval Process | Decomposes the question into sub-queries, rewrites each query against the current evidence state, retrieves via BM25, and maintains an explicit evidence state $E_t = (c_t, \delta_t, \kappa_t)$ — coverage, conflict, credibility. |
| **CAC** — Calibrated Abstention Controller | Reads $E_t$ at every step and chooses one of: **answer**, **retrieve-more**, or **abstain**, via a conjunction of evidence-sufficiency and self-consistency thresholds. |

The three-signal CAC is the general framework. The experiments instantiate it
with a single **same-model self-judge** sufficiency score (one threshold
$\tau_{cov}$), which already captures the dominant gain.

---

## Repository layout

```
run.py                       CLI entry point
core/                        abstract interfaces + concrete gateway/retriever
  base.py  exceptions.py  cache.py  tools.py  llm.py  retriever.py
data/                        dataset loaders (HotpotQA, MuSiQue, ...)
methods/                     agent implementations (depend only on core interfaces)
  baselines.py  eara.py  eara_v4.py
prompts/                     prompt constants, decoupled from control flow
  baselines_prompts.py  eara_prompts.py  eara_v4_prompts.py
evaluation/                  Coverage@Acc, AUROC, abstention rate, cost
analysis/                    extended analyses (case-study, falcon, noise)
utils/
tools/                       scrub_appids.py — leak check before repackaging
tests/                       import smoke tests + metric unit tests
config.yaml                  template config (placeholders for gateway/AppId)
pyproject.toml  requirements.txt  LICENSE
```

The methods layer depends only on the abstract `BaseLLM` / `BaseRetriever`
interfaces in `core/base.py`, so agents stay decoupled from any specific API
client or retrieval backend.

---

## Setup

**1. Install.**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure the gateway.** The framework talks to any OpenAI-compatible
LLM gateway. Set its URL and your AppId(s) as environment variables (the
template `config.yaml` references them as `${LLM_GATEWAY_URL}` and
`${LLM_APPID}`):

```bash
export LLM_GATEWAY_URL="https://your-gateway/v1/openai/native"
export LLM_APPID="your-app-id"          # space-separated list enables pooling
```

Alternatively, edit `config.yaml` and replace the `${...}` placeholders
directly (never commit real AppIds).

**3. Datasets.** HotpotQA and MuSiQue load via the HuggingFace `datasets`
cache (the loader falls back to the latest cached version offline).
MuSiQue-2hop is the 2-hop subset of MuSiQue. The BM25 index is built from
the distractor + supporting passages of each benchmark.

---

## Reproducing the results

The main table compares `eara_ircot_tau03` (self-judge sufficiency gate,
`tau_cov=0.3`, `T=10` retrieval steps) against five baselines on **four LLM
backbones** × **three benchmarks**.

**Run one configuration:**

```bash
python -m run --config config.yaml --model <model> \
  --benchmark <benchmark> --method eara_ircot_tau03 --n <n> --workers <workers>
```

Results go to `results/{model}_{benchmark}_{method}.jsonl` plus a matching
`_summary.json` (Coverage@Acc, accuracy, abstention rate, AUROC, avg
retrievals, token cost). Sweep the model/benchmark/method flags to cover the
full table.

**Ablation.** `eara_ircot_noab` isolates the retrieve-more loop: the gate
still triggers extra retrieval, but the agent never abstains (it commits a
best guess at budget exhaustion). Comparing `react` → `eara_ircot_noab` →
`eara_ircot_tau03` decomposes the gain into retrieve-more and abstention
contributions.

**Calibration.** The `eara_ircot_tau0*` sweep produces Coverage@Acc / AUROC
/ Accuracy curves as a function of $\tau_{cov}$. Coverage@Acc is flat across
$\tau \in [0.1, 0.5]$, so $\tau_{cov}=0.3$ is a stable operating point.

---

## Key metrics

| Metric | Definition |
|---|---|
| **Coverage@Acc** | Accuracy on the answered subset (abstentions excluded). Primary metric. |
| **Accuracy** | Overall accuracy (abstentions counted as incorrect). |
| **Abstention rate** | Fraction of questions withheld. |
| **AUROC** | How well the abstention decision separates wrong from right answers. |
| **Avg retrievals** | Mean retrieval calls per question. |

Implementation in `evaluation/metrics.py`.

---

## Notes

- `tools/scrub_appids.py --check` verifies no AppIds remain in `config.yaml`
  before repackaging.
- Temperature: 0.6 for the main agent loop where the API supports it; 0 for
  deterministic evaluation prompts. Reasoning models that force a specific
  temperature are handled in `core/llm.py`.
- All models are accessed as black-box APIs; no model weights are used or
  modified.
- Run `python -m pytest tests/` to verify import integrity and metric
  correctness after any change.

---

## Citation

This is an anonymous submission. The citation will be added upon acceptance.
