"""Evaluation metrics for EARA.

Metrics:
  - Accuracy (abstain counts as 0)
  - Coverage@Acc (accuracy on answered subset)
  - AUROC (abstention decision quality)
  - True Abstain Rate (abstain on unanswerable)
  - Retrieval Cost (n calls / tokens)
"""
from __future__ import annotations
import re
import numpy as np


def normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower()).strip()


def is_correct(pred: str, gold: str) -> bool:
    """Token-overlap EM proxy (HotpotQA-style F1 can be added)."""
    if not gold:
        return False
    np_ = normalize(pred)
    ng = normalize(gold)
    if not ng:
        return False
    return ng in np_ or np_.find(ng) >= 0


def accuracy(preds: list[str], abstained: list[bool], golds: list[str]) -> float:
    correct = sum(is_correct(p, g) and not a
                  for p, a, g in zip(preds, abstained, golds))
    return correct / len(golds) if golds else 0.0


def coverage_at_acc(preds: list[str], abstained: list[bool], golds: list[str]) -> float:
    """Accuracy among answered (non-abstained) questions."""
    answered = [(p, g) for p, a, g in zip(preds, abstained, golds) if not a]
    if not answered:
        return 0.0
    return sum(is_correct(p, g) for p, g in answered) / len(answered)


def abstention_auroc(abstain_scores: list[float], should_abstain: list[int]) -> float:
    """AUROC where higher score = more likely to abstain.

    abstain_scores: continuous signal (e.g., 1 - coverage, or the CAC decision).
    should_abstain: 1 if the question is unanswerable/wrong, else 0.
    """
    if len(set(should_abstain)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(should_abstain, abstain_scores))


def true_abstain_rate(abstained: list[bool], unanswerable: list[bool]) -> float:
    """Fraction of unanswerable questions that were correctly abstained."""
    un = [a for a, u in zip(abstained, unanswerable) if u]
    return sum(un) / len(un) if un else 0.0


# ---- FALCON-specific metrics -------------------------------------------------

def evidence_precision(results: list[dict]) -> float:
    """Fraction of questions where the agent's final evidence (passages used)
    included at least one gold passage. Higher = less noise admitted.

    Reads optional 'evidence_titles' field from each result record; falls back
    to checking that the gold answer string appears in used passages. If no
    evidence info recorded, returns -1 (not applicable).
    """
    hits = 0
    n_with_info = 0
    for r in results:
        ev = r.get("evidence_titles") or r.get("used_passages") or []
        gold_titles = r.get("gold_titles") or []
        if not ev:
            continue
        n_with_info += 1
        ev_lower = set(t.lower() for t in ev)
        gold_lower = set(t.lower() for t in gold_titles)
        if gold_lower and (ev_lower & gold_lower):
            hits += 1
    if n_with_info == 0:
        return -1.0
    return hits / n_with_info


def noise_robustness(clean_acc: float, noisy_acc: float) -> float:
    """Accuracy retained under noise, relative to clean setting.

    Returns noisy_acc / clean_acc (1.0 = no degradation, 0.0 = total collapse).
    clean_acc of 0 is treated as a tiny epsilon to avoid division by zero.
    """
    if clean_acc <= 0:
        return 0.0 if noisy_acc <= 0 else 1.0
    return noisy_acc / clean_acc


def conflict_resolution_rate(results: list[dict]) -> float:
    """Fraction of questions with an injected conflict where the gold passage
    was retained (not dropped). Reads 'had_conflict' + 'retained_gold' flags.
    Returns -1 if no conflict info recorded.
    """
    conflict_q = [r for r in results if r.get("had_conflict")]
    if not conflict_q:
        return -1.0
    resolved = sum(1 for r in conflict_q if r.get("retained_gold"))
    return resolved / len(conflict_q)


def evaluate_falcon(results: list[dict], clean_acc: float | None = None) -> dict:
    """FALCON-specific evaluation: accuracy + evidence precision + noise metrics.

    If clean_acc is given (accuracy on the clean setting), reports noise
    robustness relative to it.
    """
    base = evaluate(results)
    base["evidence_precision"] = evidence_precision(results)
    base["conflict_resolution_rate"] = conflict_resolution_rate(results)
    if clean_acc is not None:
        base["noise_robustness"] = noise_robustness(clean_acc, base["accuracy"])
    return base


def cost_summary(llm) -> dict:
    return llm.cost_summary()


def evaluate(results: list[dict]) -> dict:
    """results: list of {pred, abstained, gold, unanswerable, n_retrievals}."""
    preds = [r.get("pred", r.get("answer", "")) for r in results]
    abstained = [r["abstained"] for r in results]
    golds = [r["gold"] for r in results]
    unanswerable = [r.get("unanswerable", False) for r in results]

    # abstain signal for AUROC: 1 if abstained
    abstain_signal = [1.0 if a else 0.0 for a in abstained]
    should_abstain = [1 if (u or not is_correct(p, g)) else 0
                      for u, p, g in zip(unanswerable, preds, golds)]

    total_retrievals = sum(r.get("n_retrievals", 0) for r in results)

    return {
        "accuracy": accuracy(preds, abstained, golds),
        "coverage_at_acc": coverage_at_acc(preds, abstained, golds),
        "auroc": abstention_auroc(abstain_signal, should_abstain),
        "true_abstain_rate": true_abstain_rate(abstained, unanswerable),
        "answered_rate": 1 - (sum(abstained) / len(abstained) if abstained else 0),
        "avg_retrievals": total_retrievals / len(results) if results else 0,
    }
