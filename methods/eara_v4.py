"""EARA-v4: LLM-driven retrieval + evidence-coverage gate + abstention.

Why v4 (after v1-v3, v5 all failed to beat ReAct):
  - v1/v3: rigid decompose->rewrite->retrieve pipeline. Retrieval quality is
    WORSE than ReAct's LLM-freely-decided queries, so they lose despite better
    answer-gating. (v3: gpt-4.1 64% vs react 72%; qwen-plus 49% vs 48%.)
  - v5: kept ReAct retrieval + self-consistency confidence gate. But the
    confidence signal is noisy (answer surface forms vary), so it rejected
    correct answers (5 correct turned wrong, 0 added). 62% < react 72%.

  Root cause: the SIGNAL used to decide "answer vs retrieve-more" was unreliable.
  self-consistency on short answers is dominated by surface-form variance, not by
  actual evidence sufficiency.

v4 fix — use a DIFFERENT, more reliable signal: EVIDENCE COVERAGE.
  - Retrieval: LLM-driven (copied from ReAct verbatim — keeps the retrieval
    quality that made ReAct strong).
  - When LLM emits FINAL ANSWER, v4 asks the LLM: "does the gathered evidence
    contain the facts needed to support this answer?" (YES/NO + a 0-1 score).
    - YES (coverage >= tau_cov): commit.
    - NO  (coverage <  tau_cov): reject, ask LLM to retrieve more (up to max_steps).
    - If still NO at budget exhaustion: ABSTAIN (this is the risk-coverage source).

  This signal is about EVIDENCE SUFFICIENCY (grounded in retrieved passages),
  not about answer variance (which self-consistency measured). It should fire
  when the LLM is guessing without evidence — exactly the case ReAct gets wrong.

Baselines now include: Direct (no retrieval), RAG, Self-Ask (same decompose
family as EARA but no control), ReAct. So v4 is compared against 4 baselines.
"""

from __future__ import annotations
from prompts.eara_v4_prompts import COVERAGE_PROMPT
import json
from core.llm import LLM, ALL_TOOLS
from core.retriever import Retriever
from data.datasets import Example
from methods.baselines import REACT_SYSTEM, IRCoT_SYSTEM




def evidence_sufficiency(llm: LLM, q: str, ans: str, evidence: str) -> float:
    """LLM-judge: is the evidence sufficient to support `ans`? Returns 0..1."""
    resp = llm.chat([{"role": "user",
                      "content": COVERAGE_PROMPT.format(q=q, ans=ans, evidence=evidence)}],
                    temperature=0.0)
    import re
    nums = re.findall(r"[01](?:\.\d+)?", resp["content"])
    if not nums:
        return 0.0
    return max(0.0, min(1.0, float(nums[0])))


def _self_consistency(llm, question: str, evidence: str, k: int = 5) -> float:
    """Sample k short answers (temp 0.7), return agreement fraction p_t."""
    import re
    from collections import Counter
    prompt = ("Answer using the evidence. Give only the short answer, no explanation.\n"
              "Evidence:\n{ev}\n\nQuestion: {q}\nAnswer:").format(ev=evidence, q=question)
    answers = []
    for _ in range(k):
        r = llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
        a = re.sub(r"[^\w\s]", "", (r.get("content") or "").lower()).strip().split("\n")[0]
        answers.append(a)
    if not answers:
        return 0.0
    most = Counter(answers).most_common(1)[0][1]
    return most / len(answers)


def run_eara_v4(llm: LLM, retriever: Retriever, ex: Example,
                tau_cov: float = 0.5, max_steps: int = 10,
                system_prompt: str = REACT_SYSTEM,
                max_rejects: int = 99, judge_llm: LLM = None,
                tau_p=None, sc_k: int = 5) -> dict:
    """ReAct retrieval loop + evidence-sufficiency gate before committing.

    When the LLM emits FINAL ANSWER, judge evidence sufficiency:
      - sufficiency >= tau_cov -> commit
      - sufficiency <  tau_cov -> reject, retrieve more (up to max_steps)
      - at budget exhaustion with insufficient evidence -> ABSTAIN

    max_rejects caps how many times the gate can reject+re-retrieve (default
    unlimited). Setting max_rejects=1 limits re-retrieval noise on strong models.

    judge_llm: if provided, the evidence-sufficiency judge uses this (stronger)
    model instead of the agent LLM (self-judge). None = self-judge (same llm).

    system_prompt defaults to ReAct; pass IRCoT_SYSTEM for the ircot variant.
    """
    judge = judge_llm or llm  # use judge_llm if provided, else self-judge
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {ex.question}"}]
    n_retrievals = 0
    n_rejects = 0
    trace = []
    last_answer = None
    last_sufficiency = 0.0

    for step in range(max_steps):
        resp = llm.chat(messages, tools=ALL_TOOLS)
        tool_calls = resp.get("tool_calls") or []
        if tool_calls:
            assistant_tc = []
            for idx, tc in enumerate(tool_calls):
                call_id = tc.get("id") or f"call_{step}_{idx}"
                assistant_tc.append({"id": call_id, "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"])}})
            messages.append({"role": "assistant",
                             "content": resp.get("content") or "",
                             "tool_calls": assistant_tc})
            for idx, tc in enumerate(tool_calls):
                call_id = assistant_tc[idx]["id"]
                if tc["name"] == "wiki_search":
                    n_retrievals += 1
                    passages = retriever.search(tc["arguments"]["query"], k=3)
                    obs = "\n\n".join(repr(p) for p in passages) or "No results."
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": obs})
                elif tc["name"] == "wiki_lookup":
                    p = retriever.lookup(tc["arguments"]["title"])
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": repr(p) if p else "Not found."})
                else:
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": "Unknown tool."})
        else:
            content = resp.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            if "FINAL ANSWER" in content.upper():
                ans = content.upper().split("FINAL ANSWER")[-1].strip(":\n ")
                # gather evidence text from tool results
                evidence_parts = [m.get("content", "") for m in messages
                                  if m.get("role") == "tool"]
                evidence = "\n\n".join(evidence_parts[-6:]) or "(none)"
                suf = evidence_sufficiency(judge, ex.question, ans, evidence)
                trace.append({"step": step, "sufficiency": suf,
                              "n_retrievals": n_retrievals})
                last_answer, last_sufficiency = ans, suf
                # === Evidence-sufficiency gate (the only thing v4 adds) ===
                if suf >= tau_cov:
                    return {"answer": ans, "abstained": False,
                            "n_retrievals": n_retrievals, "state": None,
                            "trace": trace}
                # insufficient: if we've already hit max_rejects, commit anyway
                # (avoid re-retrieval noise on strong models that already know ans)
                if n_rejects >= max_rejects:
                    return {"answer": ans, "abstained": False,
                            "n_retrievals": n_retrievals, "state": None,
                            "trace": trace}
                n_rejects += 1
                # insufficient: reject, ask for one more retrieval
                messages.append({"role": "user",
                    "content": "The gathered evidence does not yet sufficiently "
                               "support that answer. Search once more to find "
                               "the specific supporting fact, then give FINAL ANSWER."})
                # fall through to next loop step
    # budget exhausted. 3-signal CAC (Method §3.3): by default (tau_p=None) use
    # the sufficiency gate alone — abstain if suf < tau_cov. When tau_p is set,
    # use the full conjunction: abstain only if suf < tau_cov AND p_t < tau_p.
    if last_answer is not None:
        if tau_p is not None:
            ev_parts = [m.get("content", "") for m in messages
                        if isinstance(m, dict) and m.get("role") == "tool"]
            evidence = "\n\n".join(ev_parts[-6:]) or "(none)"
            p_t = _self_consistency(llm, ex.question, evidence, k=sc_k)
            abstain = (last_sufficiency < tau_cov) and (p_t < tau_p)
        else:
            p_t = None
            abstain = last_sufficiency < tau_cov
        if abstain:
            return {"answer": None, "abstained": True,
                    "n_retrievals": n_retrievals, "state": None, "trace": trace,
                    "p_t": p_t, "sufficiency": last_sufficiency}
        return {"answer": last_answer, "abstained": False,
                "n_retrievals": n_retrievals, "state": None, "trace": trace,
                "p_t": p_t, "sufficiency": last_sufficiency}
    # never reached a sufficient answer
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    return {"answer": last, "abstained": False,
            "n_retrievals": n_retrievals, "state": None, "trace": trace}


def run_eara_ircot(llm: LLM, retriever: Retriever, ex: Example,
                    tau_cov: float = 0.5, max_steps: int = 8,
                    max_rejects: int = 99, judge_llm: LLM = None,
                    tau_p=None, sc_k: int = 5) -> dict:
    """IRCoT-style retrieval (CoT interleaved) + evidence-sufficiency gate.

    Same as run_eara_v4 but with IRCoT_SYSTEM prompt (bias toward step-by-step
    CoT + targeted retrieval). Combines ircot's retrieval quality with eara4's
    evidence-sufficiency gate + abstention.
    """
    return run_eara_v4(llm, retriever, ex, tau_cov=tau_cov,
                       max_steps=max_steps, system_prompt=IRCoT_SYSTEM,
                       max_rejects=max_rejects, judge_llm=judge_llm,
                       tau_p=tau_p, sc_k=sc_k)
