"""EARA: Evidence-Aware Retrieval Agent.

Two components:
  CRP (Controllable Retrieval Process):
    - Decompose question into sub-questions with dependencies.
    - Retrieval-evidence loop: pick open sub-Q, rewrite query, retrieve,
      extract evidence, update evidence state.
    - Evidence monitoring: coverage c_t, conflict delta_t, credibility kappa_t.

  CAC (Calibrated Abstention Controller):
    - decision(t) in {ANSWER, RETRIEVE-MORE, ABSTAIN} based on (c_t, delta_t, p_t)
      and calibrated thresholds (tau_c, tau_delta, tau_p).
    - p_t: self-consistency confidence from k samples.

Training-free: no weight updates. Thresholds calibrated once on a dev set.
"""

from __future__ import annotations
from prompts.eara_prompts import DECOMPOSE_PROMPT, REWRITE_PROMPT, MONITOR_PROMPT, ANSWER_PROMPT, SC_STUB
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from core.llm import LLM
from core.retriever import Retriever, Passage
from data.datasets import Example


@dataclass
class EvidenceState:
    sub_questions: list[str] = field(default_factory=list)
    answered_subqs: dict[int, str] = field(default_factory=dict)   # idx -> evidence text
    coverage: float = 0.0
    conflict: float = 0.0
    credibility: float = 0.0
    confidence: float = 0.0
    snippets: list[Passage] = field(default_factory=list)


# ---- Prompts -----------------------------------------------------------------







# ---- CRP ---------------------------------------------------------------------

def decompose(llm: LLM, q: str) -> list[str]:
    resp = llm.chat([{"role": "user",
                      "content": DECOMPOSE_PROMPT.format(q=q)}], temperature=0.2)
    subs = []
    for line in resp["content"].splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            subs.append(m.group(1).strip())
    return subs[:4] if subs else [q]


def rewrite_query(llm: LLM, q: str, sq: str, state: EvidenceState) -> str:
    ev = "\n".join(repr(s) for s in state.snippets[-3:]) or "(none yet)"
    resp = llm.chat([{"role": "user",
                      "content": REWRITE_PROMPT.format(q=q, sq=sq, evidence=ev)}],
                    temperature=0.3)
    return resp["content"].strip().split("\n")[0][:120]


def subq_addressed(llm: LLM, sq: str, passages: list) -> bool:
    """Judge whether the retrieved passages actually address the sub-question.
    Prevents premature sub-question closure from empty/irrelevant retrievals."""
    ev = "\n\n".join(repr(p) for p in passages[:3]) or "(none)"
    resp = llm.chat([{"role": "user",
                      "content": f"Sub-question: {sq}\n\nEvidence:\n{ev}\n\n"
                                 "Does the evidence directly address and provide "
                                 "information to answer the sub-question? "
                                 "Answer with YES or NO only."}],
                    temperature=0.0)
    return resp["content"].strip().upper().startswith("Y")


def monitor(llm: LLM, q: str, state: EvidenceState) -> tuple[float, float, float]:
    subqs = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state.sub_questions))
    ev = "\n\n".join(repr(s) for s in state.snippets[-6:]) or "(none)"
    resp = llm.chat([{"role": "user",
                      "content": MONITOR_PROMPT.format(q=q, subqs=subqs, evidence=ev)}],
                    temperature=0.0)
    nums = re.findall(r"[01](?:\.\d+)?", resp["content"])
    c = float(nums[0]) if len(nums) > 0 else 0.0
    d = float(nums[1]) if len(nums) > 1 else 0.0
    k = float(nums[2]) if len(nums) > 2 else 0.5
    return max(0.0, min(1.0, c)), max(0.0, min(1.0, d)), max(0.0, min(1.0, k))


def confidence(llm: LLM, q: str, state: EvidenceState, k: int = 5) -> float:
    """Self-consistency: sample k short answers, return agreement fraction."""
    ev = "\n\n".join(repr(s) for s in state.snippets[-6:]) or "(none)"
    answers = []
    for _ in range(k):
        r = llm.chat([{"role": "user", "content": SC_STUB.format(q=q, evidence=ev)}],
                     temperature=0.7)
        # normalize: lowercase, strip punctuation
        a = re.sub(r"[^\w\s]", "", r["content"].lower()).strip().split("\n")[0]
        answers.append(a)
    if not answers:
        return 0.0
    from collections import Counter
    most = Counter(answers).most_common(1)[0][1]
    return most / len(answers)


def answer(llm: LLM, q: str, state: EvidenceState) -> str:
    ev = "\n\n".join(repr(s) for s in state.snippets)
    resp = llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(q=q, evidence=ev)}],
                    temperature=0.0)
    return resp["content"].strip()


# ---- EARA main loop ----------------------------------------------------------

def run_eara(llm: LLM, retriever: Retriever, ex: Example,
             tau_c: float = 0.8, tau_delta: float = 0.2, tau_p: float = 0.6,
             max_steps: int = 40, sc_k: int = 5) -> dict:
    """Full EARA inference. Returns {answer, abstained, n_retrievals, state, trace}."""
    state = EvidenceState()
    state.sub_questions = decompose(llm, ex.question)
    n_retrievals = 0
    trace = []

    for t in range(max_steps):
        # pick first open sub-question (those not yet answered)
        open_idx = next((i for i in range(len(state.sub_questions))
                         if i not in state.answered_subqs), None)
        if open_idx is None:
            break  # all sub-questions addressed
        sq = state.sub_questions[open_idx]

        g = rewrite_query(llm, ex.question, sq, state)
        passages = retriever.search(g, k=3)
        n_retrievals += 1
        state.snippets.extend(passages)

        # Only mark the sub-question answered if retrieval actually returned
        # evidence that addresses it (LLM-judged). This prevents premature
        # closure from empty/irrelevant retrievals inflating coverage.
        if passages and subq_addressed(llm, sq, passages):
            state.answered_subqs[open_idx] = passages[0].text[:200]

        # evidence monitoring
        state.coverage, state.conflict, state.credibility = monitor(llm, ex.question, state)
        state.confidence = confidence(llm, ex.question, state, k=sc_k)
        trace.append({"step": t, "coverage": state.coverage,
                      "conflict": state.conflict, "confidence": state.confidence})

        # CAC decision
        if (state.coverage >= tau_c and state.conflict <= tau_delta
                and state.confidence >= tau_p):
            return {"answer": answer(llm, ex.question, state),
                    "abstained": False, "n_retrievals": n_retrievals,
                    "state": state, "trace": trace}

    # budget exhausted: abstain
    return {"answer": "I cannot answer this question with sufficient evidence.",
            "abstained": True, "n_retrievals": n_retrievals,
            "state": state, "trace": trace}


# ---- Threshold calibration ---------------------------------------------------

def calibrate_thresholds(llm: LLM, retriever: Retriever, dev_examples: list[Example],
                         sc_k: int = 5) -> tuple[float, float, float]:
    """Grid-search (tau_c, tau_delta, tau_p) on a dev set to maximize AUROC
    of the abstention decision.

    For the sprint: run EARA on dev (collect per-example coverage/conflict/confidence
    + correctness), then sweep thresholds. This is threshold calibration, NOT
    training -- no model weights change.
    """
    # Collect per-example signals (run EARA once, record final signals + correctness).
    records = []
    for ex in dev_examples:
        r = run_eara(llm, retriever, ex, tau_c=0.0, tau_delta=1.0, tau_p=0.0,
                     max_steps=10, sc_k=sc_k)
        correct = _is_correct(r["answer"], ex.answer)
        records.append({
            "coverage": r["state"].coverage,
            "conflict": r["state"].conflict,
            "confidence": r["state"].confidence,
            "correct": correct,
        })

    # Grid search (coarse)
    best, best_score = (0.8, 0.2, 0.6), -1.0
    from .metrics import abstention_auroc
    for tc in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for td in [0.1, 0.2, 0.3, 0.4]:
            for tp in [0.4, 0.5, 0.6, 0.7, 0.8]:
                preds, labels = [], []
                for rec in records:
                    will_answer = (rec["coverage"] >= tc and
                                   rec["conflict"] <= td and
                                   rec["confidence"] >= tp)
                    preds.append(0 if will_answer else 1)  # 1=abstain
                    labels.append(0 if rec["correct"] else 1)  # 1=should abstain
                score = abstention_auroc(preds, labels)
                if score > best_score:
                    best_score, best = score, (tc, td, tp)
    return best


def _is_correct(pred: str, gold: str) -> bool:
    """Exact-match-ish normalization for QA scoring."""
    def norm(s):
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return norm(pred).find(norm(gold)) >= 0 if gold else False
