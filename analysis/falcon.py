"""FALCON: Falsifiability-based Credibility for retrieval ageNts.

Instead of estimating passage credibility directly (a hard self-referential
judgment), FALCON tests whether each retrieved passage can be *falsified*:
  1. Claim extraction: decompose a passage into atomic checkable claims.
  2. Counter-argument generation: for each claim, generate K targeted attacks.
  3. Counter-evidence retrieval: retrieve passages supporting each attack.
  4. Survival-based adjudication: a claim's falsification pressure F(c) is the
     max over attacks of (relevance * support); survival weight w(c) = 1 - F(c).
  5. Evidence aggregation: passage weight = mean claim survival weight; drop
     passages below theta_drop, re-rank the rest, feed to reasoning.

Training-free: every step is an LLM call or a retrieval. No weights change.
Thresholds (theta, theta_drop) calibrated once on a dev set.

FALCON sits between retrieval and reasoning in a ReAct-style loop: at each
retrieval step it cleans the retrieved passages before the agent reasons over
them. It is evidence-level and local, in contrast to trajectory-level conflict
monitors or abstention controllers.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from core.llm import LLM
from core.retriever import Retriever, Passage
from data.datasets import Example


@dataclass
class Claim:
    text: str
    weight: float = 1.0          # survival weight w(c) in [0,1]
    pressure: float = 0.0        # falsification pressure F(c) in [0,1]


@dataclass
class FalconState:
    """Per-step state of the FALCON evidence filter."""
    tested_passages: list[tuple[Passage, float, list[Claim]]] = field(default_factory=list)
    # (passage, passage_weight W(p), claims with survival weights)
    n_claims_tested: int = 0
    n_claims_dropped: int = 0
    n_counter_retrievals: int = 0


# ---- Prompts -----------------------------------------------------------------

CLAIM_PROMPT = """Extract the atomic factual claims made in the passage below.
Each claim should be a single checkable proposition that could in principle be
verified or refuted against other sources (e.g., "X happened in year Y",
"A causes B"). Omit hedged or subjective statements. Output one claim per line,
numbered.

Passage: {passage}

Claims:"""

COUNTERARG_PROMPT = """You are trying to FALSIFY the following claim. Generate
{K} strong, specific counter-arguments that, if true, would contradict or
undermine the claim. Each counter-argument should be a concrete competing fact,
alternative cause, or direct negation — not a vague doubt. Output one per line,
numbered.

Claim: {claim}

Counter-arguments:"""

SUPPORT_PROMPT = """You are judging whether the counter-argument below is
supported by the provided evidence passages. Output a single number between 0
and 1: 1 means the evidence clearly supports the counter-argument, 0 means it
does not support it at all (or contradicts it). Output ONLY the number.

Counter-argument: {counterarg}

Evidence:
{evidence}

Support (0 to 1):"""

RELEVANCE_PROMPT = """You are judging whether the counter-argument, if true,
would actually contradict or undermine the original claim. Output a single
number between 0 and 1: 1 means a direct contradiction, 0 means irrelevant to
the claim. Output ONLY the number.

Original claim: {claim}
Counter-argument: {counterarg}

Relevance (0 to 1):"""

ANSWER_PROMPT = """Answer the question based ONLY on the evidence below. The
evidence has been filtered: passages that could be falsified have been removed.
Give a short final answer.

Question: {q}

Evidence:
{evidence}

Answer (short):"""

FALCON_SYSTEM = """You are a multi-hop question answering agent. Use the
wiki_search tool to find information. Retrieved evidence is automatically
filtered for credibility before you see it. When you have enough evidence,
answer with: FINAL ANSWER: <answer>."""


# ---- FALCON pipeline (per retrieval step) ------------------------------------

def extract_claims(llm: LLM, passage: Passage) -> list[str]:
    """Step 1: decompose a passage into atomic claims."""
    resp = llm.chat([{"role": "user",
                      "content": CLAIM_PROMPT.format(passage=passage.text[:1500])}],
                    temperature=0.0)
    claims = []
    for line in resp["content"].splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            c = m.group(1).strip()
            if len(c) > 5:
                claims.append(c)
    return claims[:6]  # cap claims per passage for cost


def gen_counterargs(llm: LLM, claim: str, K: int) -> list[str]:
    """Step 2: generate K targeted counter-arguments against a claim."""
    resp = llm.chat([{"role": "user",
                      "content": COUNTERARG_PROMPT.format(claim=claim, K=K)}],
                    temperature=0.8)
    args = []
    for line in resp["content"].splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m:
            args.append(m.group(1).strip())
    return args[:K]


def judge_support(llm: LLM, counterarg: str, evidence: list[Passage]) -> float:
    """Step 3b: judge how well the counter-evidence supports a counter-argument."""
    ev = "\n\n".join(repr(p) for p in evidence) or "(none)"
    resp = llm.chat([{"role": "user",
                      "content": SUPPORT_PROMPT.format(counterarg=counterarg,
                                                       evidence=ev)}],
                    temperature=0.0)
    nums = re.findall(r"[01](?:\.\d+)?", resp["content"])
    return max(0.0, min(1.0, float(nums[0]))) if nums else 0.0


def judge_relevance(llm: LLM, claim: str, counterarg: str) -> float:
    """Step 4a: judge whether a counter-argument (if true) actually contradicts the claim."""
    resp = llm.chat([{"role": "user",
                      "content": RELEVANCE_PROMPT.format(claim=claim,
                                                         counterarg=counterarg)}],
                    temperature=0.0)
    nums = re.findall(r"[01](?:\.\d+)?", resp["content"])
    return max(0.0, min(1.0, float(nums[0]))) if nums else 0.0


def falsify_claim(llm: LLM, retriever: Retriever, claim: str, K: int = 2,
                  state: Optional[FalconState] = None) -> tuple[float, int]:
    """Steps 2-4 for one claim: generate attacks, retrieve counter-evidence,
    compute falsification pressure F(c) = max_a relevance(a,c) * support(a).

    Returns (pressure F(c), n_counter_retrievals_used).
    """
    counterargs = gen_counterargs(llm, claim, K)
    pressure = 0.0
    n_retr = 0
    for arg in counterargs:
        # Step 3a: retrieve counter-evidence for this counter-argument
        counter_ev = retriever.search(arg, k=2)
        n_retr += 1
        if state is not None:
            state.n_counter_retrievals += 1
        s = judge_support(llm, arg, counter_ev)
        r = judge_relevance(llm, claim, arg)
        pressure = max(pressure, r * s)
    return pressure, n_retr


def falcon_filter(llm: LLM, retriever: Retriever, passages: list[Passage],
                  theta_drop: float = 0.3, K: int = 1, max_passages: int = 2,
                  max_claims: int = 2) -> tuple[list[Passage], FalconState]:
    """Run the FALCON pipeline on a set of retrieved passages.

    Cost-bounded: only test the top `max_passages` passages, each with at most
    `max_claims` claims, each claim attacked by K counter-arguments. This keeps
    per-step cost to ~ max_passages * max_claims * K * 3 LLM calls.
    Returns (surviving_passages_sorted_by_weight, state).
    """
    state = FalconState()
    # Only test top-N passages (the rest are passed through unfiltered, since
    # BM25 already ranked them; testing all is the cost bottleneck).
    to_test = passages[:max_passages]
    pass_through = passages[max_passages:]
    tested = []
    for p in to_test:
        claims_text = extract_claims(llm, p)[:max_claims]
        claims = [Claim(text=c) for c in claims_text]
        for c in claims:
            c.pressure, _ = falsify_claim(llm, retriever, c.text, K=K, state=state)
            c.weight = 1.0 - c.pressure
            state.n_claims_tested += 1
            if c.weight < 0.5:
                state.n_claims_dropped += 1
        # passage weight = mean claim survival weight (0.5 if no claims extracted)
        W = (sum(c.weight for c in claims) / len(claims)) if claims else 0.5
        tested.append((p, W, claims))
    # drop low-weight tested passages
    tested = [(p, W, cs) for (p, W, cs) in tested if W >= theta_drop]
    # pass-through passages keep default weight 1.0 (untested)
    for p in pass_through:
        tested.append((p, 1.0, []))
    tested.sort(key=lambda x: x[1], reverse=True)
    state.tested_passages = tested
    return [p for (p, W, cs) in tested], state


# ---- FALCON agent loop -------------------------------------------------------

def run_falcon(llm: LLM, retriever: Retriever, ex: Example,
               theta_drop: float = 0.3, K: int = 1, max_steps: int = 4,
               noise_cfg: Optional[dict] = None,
               corpus_passages: Optional[list[Passage]] = None) -> dict:
    """FALCON inference: a ReAct loop where each retrieval step is filtered
    by falsifiability testing before the agent reasons.

    If noise_cfg is given, the agent's retrieved passages are corrupted by a
    NoisyRetriever (random/conflict/adversarial distractors). Crucially, the
    counter-evidence retrieval inside falcon_filter uses the CLEAN retriever —
    falsification must test claims against real evidence, not noise.

    Returns {answer, abstained, n_retrievals, state, trace}.
    """
    # Wrap the agent-facing retriever with noise; keep a clean retriever for
    # counter-evidence lookup inside falcon_filter.
    if noise_cfg and noise_cfg.get("mode", "none") != "none":
        from .noise import NoisyRetriever
        agent_retriever = NoisyRetriever(
            base=retriever, mode=noise_cfg.get("mode", "random"),
            k=noise_cfg.get("k", 2), corpus_passages=corpus_passages,
            gold_titles=getattr(ex, "gold_titles", []),
            gold_answer=getattr(ex, "answer", ""), llm=llm,
            seed=noise_cfg.get("seed", 42))
    else:
        agent_retriever = retriever

    messages = [{"role": "system", "content": FALCON_SYSTEM},
                {"role": "user", "content": f"Question: {ex.question}"}]
    n_retrievals = 0
    n_counter_retrievals = 0
    trace = []
    final_state = FalconState()

    for step in range(max_steps):
        resp = llm.chat(messages, tools=_wiki_tool(), temperature=0.3)
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
                    # Agent retrieves via (possibly noisy) retriever.
                    raw = agent_retriever.search(tc["arguments"]["query"], k=3)
                    # === FALCON filter: falsifiability-test the retrieved passages.
                    # counter-evidence uses the CLEAN retriever, not the noisy one. ===
                    surviving, fstate = falcon_filter(llm, retriever, raw,
                                                     theta_drop=theta_drop, K=K,
                                                     max_passages=2, max_claims=2)
                    n_counter_retrievals += fstate.n_counter_retrievals
                    final_state = fstate
                    obs = "\n\n".join(repr(p) for p in surviving) or "No surviving evidence."
                    trace.append({"step": step, "n_raw": len(raw),
                                  "n_surviving": len(surviving),
                                  "n_counter_retr": fstate.n_counter_retrievals})
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": obs})
                else:
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": "Unknown tool."})
        else:
            content = resp.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            if "FINAL ANSWER" in content.upper():
                ans = content.upper().split("FINAL ANSWER")[-1].strip(":\n ")
                return {"answer": ans, "abstained": False,
                        "n_retrievals": n_retrievals + n_counter_retrievals,
                        "state": final_state, "trace": trace}
    # exhausted steps: answer from last evidence
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    return {"answer": last, "abstained": not bool(last),
            "n_retrievals": n_retrievals + n_counter_retrievals,
            "state": final_state, "trace": trace}


def _wiki_tool() -> list[dict]:
    """Tool spec for wiki_search (FALCON's agent loop only uses search)."""
    return [{"type": "function", "function": {
        "name": "wiki_search",
        "description": "Search Wikipedia for information.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}]


# ---- Calibration -------------------------------------------------------------

def calibrate_falcon(llm: LLM, retriever: Retriever, dev_examples: list[Example],
                     K: int = 2) -> tuple[float, float]:
    """Grid-search (theta_drop) on a dev set to maximize accuracy.

    K is fixed (counter-argument count). Only theta_drop is swept for cost.
    Returns (best_theta_drop, best_K).
    """
    best_theta, best_acc = 0.3, -1.0
    for theta in [0.2, 0.3, 0.4, 0.5]:
        correct = 0
        for ex in dev_examples[:30]:  # small dev subset for calibration cost
            r = run_falcon(llm, retriever, ex, theta_drop=theta, K=K, max_steps=6)
            if _is_correct(r["answer"], ex.answer):
                correct += 1
        acc = correct / min(30, len(dev_examples))
        if acc > best_acc:
            best_acc, best_theta = acc, theta
    return best_theta, K


def _is_correct(pred: str, gold: str) -> bool:
    def norm(s):
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return norm(pred).find(norm(gold)) >= 0 if gold else False
