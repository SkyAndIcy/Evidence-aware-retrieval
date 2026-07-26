"""CASE: Compute-Aware Scheduling for agEnts.

Training-free controller that adaptively allocates inference-time compute
across a retrieval agent's trajectory.

Two components:
  Online Progress Estimator:
    - d_t in [0,1]: high = agent progressing well (cheap step suffices),
      low = agent stuck/far-from-done (expensive step warranted).
    - integrates 4 trajectory signals:
        r_cl  : sub-question closure rate
        dpc   : smoothed confidence trend (delta p_t)
        r_ev  : marginal evidence gain
        u_t   : residual uncertainty
    - d_t = sigmoid(alpha1*r_cl + alpha2*dpc + alpha3*r_ev - alpha4*u_t)

  Tier-Selection Policy:
    - 4 tiers: SHALLOW / DEEP / VERIFY / COMMIT
    - decision tree on (d_t, p_t, budget_remaining)

Reuses the EARA harness (LLM, Retriever, datasets, metrics) — no new infra.

Relationship to EARA (anti-salami-slicing):
  EARA decides {answer, retrieve-more, abstain} from EVIDENCE SUFFICIENCY
  signals (coverage c_t, conflict delta_t, credibility kappa_t).
  CASE decides {shallow, deep, verify, commit} from PROGRESS TREND signals
  (closure rate, confidence trend, evidence gain, residual uncertainty).
  Different decision space, different signals, different metrics
  (CASE: cost-accuracy Pareto; EARA: AUROC / selective acc).
"""
from __future__ import annotations
import json
import re
import math
from dataclasses import dataclass, field
from typing import Optional
from core.llm import LLM
from core.retriever import Retriever, Passage
from data.datasets import Example
from methods.eara import decompose, answer as eara_answer, SC_STUB  # reuse prompts


# ---- Tier definitions --------------------------------------------------------

SHALLOW = "shallow"   # 1 retrieval + 1 reasoning pass  (cost 2)
DEEP    = "deep"      # S sampled reasoning + 1 retrieval (cost S+1)
VERIFY  = "verify"    # V extra self-consistency samples   (cost V)
COMMIT  = "commit"    # stop, return best answer           (cost 0)

TIER_COST = {SHALLOW: 2, VERIFY: None, DEEP: None, COMMIT: 0}  # VERIFY/DEEP set at runtime


@dataclass
class TrajectoryState:
    """Online state tracked across the agent trajectory for CASE."""
    sub_questions: list[str] = field(default_factory=list)
    closed_subqs: set = field(default_factory=set)            # indices closed
    evidence: list[Passage] = field(default_factory=list)     # accumulated snippets
    confidences: list[float] = field(default_factory=list)    # p_1..p_t history
    evidence_gains: list[float] = field(default_factory=list) # g_1..g_t
    spend: int = 0                                            # cumulative LLM calls
    best_answer: str = ""
    best_confidence: float = 0.0


# ---- Online Progress Estimator (Section 3.3) --------------------------------

def closure_rate(state: TrajectoryState, t: int) -> float:
    """r_cl: closed sub-questions per step spent."""
    k = max(1, len(state.sub_questions))
    o_t = len(state.closed_subqs)
    denom = min(max(1, t), k)
    return o_t / denom


def confidence_trend(state: TrajectoryState, window: int = 3) -> float:
    """Smoothed delta p_t over a window. Returns mean of recent deltas, in [-1,1]."""
    ps = state.confidences
    if len(ps) < 2:
        return 0.0
    recent = ps[-(window + 1):]
    deltas = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
    return sum(deltas) / max(1, len(deltas))


def marginal_evidence_gain(state: TrajectoryState) -> float:
    """r_ev: g_t / max(1, g_{1:t-1}). Decays as retriever saturates."""
    gs = state.evidence_gains
    if not gs:
        return 1.0
    cur = gs[-1]
    prior = sum(gs[:-1])
    return cur / max(1.0, prior)


def residual_uncertainty(state: TrajectoryState) -> float:
    """u_t = (1 - o_t/k) + (1 - p_t), in [0,2]; normalize to [0,1] via /2."""
    k = max(1, len(state.sub_questions))
    o_t = len(state.closed_subqs)
    p_t = state.confidences[-1] if state.confidences else 0.0
    return ((1 - o_t / k) + (1 - p_t)) / 2.0


def progress_signal(state: TrajectoryState, t: int,
                    alpha: tuple = (1.0, 1.0, 1.0, 1.0)) -> float:
    """d_t in [0,1]. High = progressing well (cheap tier ok); low = stuck."""
    r_cl = closure_rate(state, t)
    dpc = confidence_trend(state)
    r_ev = marginal_evidence_gain(state)
    u_t = residual_uncertainty(state)
    a1, a2, a3, a4 = alpha
    z = a1 * r_cl + a2 * dpc + a3 * r_ev - a4 * u_t
    return 1.0 / (1.0 + math.exp(-z))


# ---- Confidence (self-consistency, reused from EARA) ------------------------

def confidence(llm: LLM, q: str, state: TrajectoryState, k: int = 5) -> float:
    """p_t via self-consistency. Reuses EARA's SC_STUB prompt."""
    ev = "\n\n".join(repr(s) for s in state.evidence[-6:]) or "(none)"
    answers = []
    for _ in range(k):
        r = llm.chat([{"role": "user", "content": SC_STUB.format(q=q, evidence=ev)}],
                     temperature=0.7)
        a = re.sub(r"[^\w\s]", "", r["content"].lower()).strip().split("\n")[0]
        answers.append(a)
    if not answers:
        return 0.0
    from collections import Counter
    most = Counter(answers).most_common(1)[0][1]
    return most / len(answers)


# ---- Tier execution ---------------------------------------------------------

REASON_PROMPT = """Given the question and evidence so far, reason about the next sub-question to close and what to search for. Output a search query (one line).

Question: {q}
Open sub-questions: {open}
Evidence so far:
{ev}

Next search query:"""


def _open_subqs(state: TrajectoryState) -> list[int]:
    return [i for i in range(len(state.sub_questions)) if i not in state.closed_subqs]


def execute_shallow(llm: LLM, retriever: Retriever, q: str,
                    state: TrajectoryState) -> int:
    """1 reasoning pass + 1 retrieval. Cost = 2."""
    open_idx = _open_subqs(state)
    sq = state.sub_questions[open_idx[0]] if open_idx else q
    ev = "\n".join(repr(s) for s in state.evidence[-3:]) or "(none)"
    r1 = llm.chat([{"role": "user",
                    "content": REASON_PROMPT.format(q=q, open=sq, ev=ev)}], temperature=0.4)
    g = r1["content"].strip().split("\n")[0][:120]
    passages = retriever.search(g, k=3)
    state.evidence.extend(passages)
    gain = len(passages)
    state.evidence_gains.append(float(gain))
    # close sub-question if evidence mentions it
    if open_idx:
        sq_text = state.sub_questions[open_idx[0]].lower()
        if any(sq_text and any(w in p.text.lower() for w in sq_text.split()[:3]) for p in passages):
            state.closed_subqs.add(open_idx[0])
    return 2


def execute_deep(llm: LLM, retriever: Retriever, q: str,
                 state: TrajectoryState, S: int = 4) -> int:
    """S sampled reasoning passes, aggregated, then 1 retrieval. Cost = S+1."""
    open_idx = _open_subqs(state)
    sq = state.sub_questions[open_idx[0]] if open_idx else q
    ev = "\n".join(repr(s) for s in state.evidence[-3:]) or "(none)"
    queries = []
    for _ in range(S):
        r = llm.chat([{"role": "user",
                       "content": REASON_PROMPT.format(q=q, open=sq, ev=ev)}], temperature=0.8)
        queries.append(r["content"].strip().split("\n")[0][:120])
    # aggregate: pick most common query (simple vote)
    from collections import Counter
    g = Counter(queries).most_common(1)[0][0]
    passages = retriever.search(g, k=3)
    state.evidence.extend(passages)
    state.evidence_gains.append(float(len(passages)))
    if open_idx:
        sq_text = state.sub_questions[open_idx[0]].lower()
        if any(any(w in p.text.lower() for w in sq_text.split()[:3]) for p in passages):
            state.closed_subqs.add(open_idx[0])
    return S + 1


def execute_verify(llm: LLM, q: str, state: TrajectoryState,
                   V: int = 4, accept_thresh: float = 0.6) -> tuple[int, bool]:
    """V extra self-consistency samples on candidate answer. Returns (cost, accepted)."""
    p = confidence(llm, q, state, k=V)
    state.confidences.append(p)
    if p >= accept_thresh:
        state.best_answer = eara_answer(llm, q, _wrap_state(state))
        state.best_confidence = p
        return V, True
    return V, False


def _wrap_state(state: TrajectoryState):
    """Adapt TrajectoryState to EARA's EvidenceState shape for answer()."""
    class _S: pass
    s = _S()
    s.snippets = state.evidence
    return s


# ---- Tier-Selection Policy (Section 3.4, Eq. 2) -----------------------------

def select_tier(d_t: float, p_t: float, budget_left: int,
                theta: tuple, S: int) -> str:
    """Decision tree (Eq. 2). theta = (theta1, theta2, theta_p, theta_v)."""
    theta1, theta2, theta_p, theta_v = theta
    if d_t >= theta1 and p_t >= theta_p:
        return COMMIT
    if theta2 <= d_t < theta1 and p_t >= theta_v:
        return VERIFY
    if d_t < theta2 and budget_left >= S + 1:
        return DEEP
    return SHALLOW


# ---- CASE main loop (Algorithm 1) -------------------------------------------

def run_case(llm: LLM, retriever: Retriever, ex: Example,
             budget: int = 24, theta: tuple = (0.7, 0.4, 0.7, 0.5),
             alpha: tuple = (1.0, 1.0, 1.0, 1.0),
             S: int = 4, V: int = 4, sc_k: int = 3) -> dict:
    """Full CASE inference. Returns {answer, abstained, spend, trace}."""
    state = TrajectoryState()
    state.sub_questions = decompose(llm, ex.question) or [ex.question]
    k = len(state.sub_questions)
    trace = []
    t = 0

    # initial confidence
    p_t = confidence(llm, ex.question, state, k=sc_k)
    state.confidences.append(p_t)

    while state.spend < budget:
        t += 1
        d_t = progress_signal(state, t, alpha)
        p_t = state.confidences[-1]
        tier = select_tier(d_t, p_t, budget - state.spend, theta, S)
        trace.append({"t": t, "d_t": round(d_t, 3), "p_t": round(p_t, 3),
                      "tier": tier, "spend_before": state.spend})

        if tier == COMMIT:
            ans = state.best_answer or eara_answer(llm, ex.question, _wrap_state(state))
            return {"answer": ans, "abstained": False, "spend": state.spend,
                    "trace": trace}
        elif tier == SHALLOW:
            state.spend += execute_shallow(llm, retriever, ex.question, state)
        elif tier == DEEP:
            state.spend += execute_deep(llm, retriever, ex.question, state, S=S)
        elif tier == VERIFY:
            cost, accepted = execute_verify(llm, ex.question, state, V=V)
            state.spend += cost
            if accepted:
                return {"answer": state.best_answer, "abstained": False,
                        "spend": state.spend, "trace": trace}
        # re-estimate confidence after action
        state.confidences.append(confidence(llm, ex.question, state, k=sc_k))

    # budget exhausted: if final confidence is low, ABSTAIN instead of
    # returning a low-confidence best_answer. This gives CASE a real
    # risk-coverage curve (the original always-hard-answered, so abstain=0).
    final_p = state.confidences[-1] if state.confidences else 0.0
    abstain_thresh = 0.5   # p below this at budget exhaustion -> abstain
    if final_p < abstain_thresh and not state.best_answer:
        return {"answer": "I cannot answer this question with sufficient evidence.",
                "abstained": True, "spend": state.spend, "trace": trace}
    ans = state.best_answer or eara_answer(llm, ex.question, _wrap_state(state))
    return {"answer": ans, "abstained": False, "spend": state.spend, "trace": trace}


# ---- Calibration (Section 3.4) ---------------------------------------------

def calibrate(llm: LLM, retriever: Retriever, dev_examples: list[Example],
              budget: int = 24) -> tuple:
    """Grid-search theta and alpha on dev set to maximize accuracy under budget.

    Returns (theta, alpha). Coarse grid for the 8-day sprint.
    Training-free: no weights change.
    """
    from .metrics import is_correct
    # Collect per-example (d_t trajectory + correctness) with default params.
    # Then sweep theta on a small grid; alpha kept at default (can be added).
    best_theta, best_acc = (0.7, 0.4, 0.7, 0.5), -1.0
    # Pre-run dev with default theta to collect signals (save API by reusing).
    # For the sprint: direct sweep is acceptable on ~50 dev examples.
    for theta1 in [0.6, 0.7, 0.8]:
        for theta2 in [0.3, 0.4, 0.5]:
            for theta_p in [0.6, 0.7, 0.8]:
                for theta_v in [0.4, 0.5, 0.6]:
                    theta = (theta1, theta2, theta_p, theta_v)
                    correct = 0
                    for ex in dev_examples[:50]:
                        r = run_case(llm, retriever, ex, budget=budget, theta=theta)
                        if is_correct(r["answer"], ex.answer):
                            correct += 1
                    acc = correct / min(50, len(dev_examples))
                    if acc > best_acc:
                        best_acc, best_theta = acc, theta
    return best_theta, (1.0, 1.0, 1.0, 1.0)
