"""Baselines: ReAct, plain RAG, Direct (no retrieval), Self-Ask.

All share the same LLM and retriever as EARA for fair comparison.
- ReAct: thought-action-observation loop with wiki_search.
- RAG: retrieve top-k once, then answer.
- Direct: no retrieval, LLM parametric knowledge only.
- Self-Ask: decompose into sub-questions, retrieve per sub-question (same family as EARA).
"""

from __future__ import annotations
from prompts.baselines_prompts import REACT_SYSTEM, RAG_SYSTEM, DIRECT_SYSTEM, SELFASK_DECOMPOSE, SELFASK_ANSWER, SELFASK_FINAL, CRAG_EVAL, CRAG_ANSWER, CON_NOTE, CON_ANSWER
import json
import re
from typing import Optional
from core.llm import LLM, ALL_TOOLS
from core.retriever import Retriever
from data.datasets import Example




def run_react(llm: LLM, retriever: Retriever, ex: Example,
              max_steps: int = 8) -> dict:
    """Standard ReAct loop. Returns {answer, abstained, n_retrievals, trace}."""
    messages = [{"role": "system", "content": REACT_SYSTEM},
                {"role": "user", "content": f"Question: {ex.question}"}]
    n_retrievals = 0
    trace = []
    for step in range(max_steps):
        resp = llm.chat(messages, tools=ALL_TOOLS)
        trace.append(resp)
        tool_calls = resp.get("tool_calls") or []
        if tool_calls:
            # Build ONE assistant message carrying ALL tool_calls (OpenAI format),
            # then append one tool-result message per call with matching tool_call_id.
            assistant_tc = []
            for idx, tc in enumerate(tool_calls):
                call_id = tc.get("id") or f"call_{step}_{idx}"
                assistant_tc.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"])},
                })
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
            # no tool call -> check for final answer
            content = resp.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            if "FINAL ANSWER" in content.upper():
                ans = content.upper().split("FINAL ANSWER")[-1].strip(":\n ")
                return {"answer": ans, "abstained": False,
                        "n_retrievals": n_retrievals, "trace": trace}
    # exhausted steps: take last content as best guess
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    return {"answer": last, "abstained": False,
            "n_retrievals": n_retrievals, "trace": trace}




def run_rag(llm: LLM, retriever: Retriever, ex: Example, top_k: int = 5) -> dict:
    """Plain RAG: retrieve top-k once, then answer. Never abstains (always guesses)."""
    passages = retriever.search(ex.question, k=top_k)
    context = "\n\n".join(repr(p) for p in passages)
    messages = [{"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {ex.question}\n\nAnswer:"}]
    resp = llm.chat(messages)
    content = resp["content"]
    abstained = False  # baselines never abstain; only EARA does
    return {"answer": content, "abstained": abstained,
            "n_retrievals": 1, "passages": passages}


# ---- Direct: no retrieval, pure parametric knowledge --------------------------



def run_direct(llm: LLM, retriever: Retriever, ex: Example, top_k: int = 5) -> dict:
    """Direct: no retrieval, LLM parametric knowledge only. Never abstains (always guesses).

    `retriever` is accepted (unused) so run.py can dispatch all methods uniformly.
    """
    messages = [{"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": f"Question: {ex.question}\n\nAnswer:"}]
    resp = llm.chat(messages)
    content = resp["content"]
    abstained = False  # baselines never abstain; only EARA does
    return {"answer": content, "abstained": abstained,
            "n_retrievals": 0, "passages": []}


# ---- Self-Ask: decompose + per-subquestion retrieval (same family as EARA) ---





def run_selfask(llm: LLM, retriever: Retriever, ex: Example,
                top_k: int = 3, max_subqs: int = 4) -> dict:
    """Self-Ask: decompose into sub-questions, retrieve for each, compose.

    Same decomposition family as EARA but WITHOUT coverage/conflict/abstention
    control — pure decompose-retrieve-compose. A fair same-family baseline.
    """
    resp = llm.chat([{"role": "user",
                      "content": SELFASK_DECOMPOSE.format(q=ex.question)}],
                    temperature=0.2)
    subqs = []
    for line in resp["content"].splitlines():
        m = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
        if m and len(m.group(1).strip()) > 3:
            subqs.append(m.group(1).strip())
    if not subqs:
        subqs = [ex.question]
    subqs = subqs[:max_subqs]

    chain = []
    n_retr = 0
    for sq in subqs:
        passages = retriever.search(sq, k=top_k)
        n_retr += 1
        ctx = "\n\n".join(repr(p) for p in passages) or "(none)"
        r = llm.chat([{"role": "user",
                       "content": SELFASK_ANSWER.format(sq=sq, context=ctx)}],
                     temperature=0.0)
        chain.append(f"{sq}\n  -> {r['content'].strip()}")
    final = llm.chat([{"role": "user",
                       "content": SELFASK_FINAL.format(q=ex.question,
                                                        chain='\n'.join(chain))}],
                     temperature=0.0)
    return {"answer": final["content"].strip(), "abstained": False,
            "n_retrievals": n_retr, "passages": []}



# ---- IRCoT: Interleaving Retrieval with CoT (Trivedi et al. 2023) -----------
# Generate a CoT step; if it needs external knowledge, retrieve and continue.
IRCoT_SYSTEM = """Answer the multi-hop question step by step. After each reasoning
step, if you need a fact you don't know, call wiki_search to find it. Continue
until you can give the final answer: FINAL ANSWER: <answer>."""


def run_ircot(llm: LLM, retriever: Retriever, ex: Example,
              max_steps: int = 8) -> dict:
    """IRCoT: CoT and retrieval interleaved. Same tool protocol as ReAct but
    the system prompt biases toward step-by-step reasoning + targeted retrieval."""
    messages = [{"role": "system", "content": IRCoT_SYSTEM},
                {"role": "user", "content": f"Question: {ex.question}"}]
    n_retr = 0
    for step in range(max_steps):
        resp = llm.chat(messages, tools=ALL_TOOLS)
        tool_calls = resp.get("tool_calls") or []
        if tool_calls:
            assistant_tc = []
            for idx, tc in enumerate(tool_calls):
                cid = tc.get("id") or f"c_{step}_{idx}"
                assistant_tc.append({"id": cid, "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"])}})
            messages.append({"role": "assistant", "content": resp.get("content") or "",
                             "tool_calls": assistant_tc})
            for idx, tc in enumerate(tool_calls):
                cid = assistant_tc[idx]["id"]
                if tc["name"] == "wiki_search":
                    n_retr += 1
                    ps = retriever.search(tc["arguments"]["query"], k=3)
                    messages.append({"role": "tool", "tool_call_id": cid,
                        "content": "\n\n".join(repr(p) for p in ps) or "No results."})
                else:
                    messages.append({"role": "tool", "tool_call_id": cid,
                                     "content": "Unknown tool."})
        else:
            content = resp.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            if "FINAL ANSWER" in content.upper():
                ans = content.upper().split("FINAL ANSWER")[-1].strip(":\n ")
                return {"answer": ans, "abstained": False, "n_retrievals": n_retr}
    last = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    return {"answer": last, "abstained": False, "n_retrievals": n_retr}


# ---- CRAG: Corrective Retrieval-Augmented Generation (Yan et al. 2024) -------



def run_crag(llm: LLM, retriever: Retriever, ex: Example, top_k: int = 5) -> dict:
    """CRAG: retrieve -> evaluate retrieval quality -> refine -> answer.

    2024 ICLR. A retrieval evaluator judges CORRECT/AMBIGUOUS/INCORRECT:
      - CORRECT: filter passages (drop low-relevance) then answer
      - AMBIGUOUS/INCORRECT: fall back to parametric knowledge (no web search here)
    """
    passages = retriever.search(ex.question, k=top_k)
    context = "\n\n".join(repr(p) for p in passages)
    # retrieval evaluator
    resp = llm.chat([{"role": "user",
                      "content": CRAG_EVAL.format(q=ex.question, context=context)}],
                    temperature=0.0)
    quality = (resp["content"] or "").strip().upper()
    if quality.startswith("CORRECT"):
        # filter: keep only passages mentioning question entities (simple heuristic)
        kept = [p for p in passages if any(w.lower() in p.text.lower()
                                           for w in ex.question.split() if len(w) > 4)]
        ctx = "\n\n".join(repr(p) for p in (kept or passages))
    else:
        # AMBIGUOUS/INCORRECT: fall back to parametric knowledge, no context
        ctx = "(no reliable retrieved context; use own knowledge)"
    r = llm.chat([{"role": "user",
                   "content": CRAG_ANSWER.format(q=ex.question, context=ctx)}],
                 temperature=0.0)
    abstained = False
    return {"answer": r["content"].strip(), "abstained": abstained,
            "n_retrievals": 1, "passages": passages}


# ---- Chain-of-Note (Li et al. 2023) -----------------------------------------



def run_con(llm: LLM, retriever: Retriever, ex: Example, top_k: int = 5) -> dict:
    """Chain-of-Note: retrieve -> note per doc -> answer from notes (2023).

    Each retrieved passage gets a relevance note; the answer is derived from
    notes rather than raw passages, filtering noise from irrelevant docs.
    """
    passages = retriever.search(ex.question, k=top_k)
    notes = []
    for p in passages:
        r = llm.chat([{"role": "user",
                       "content": CON_NOTE.format(q=ex.question, doc=repr(p)[:1200])}],
                     temperature=0.0)
        notes.append(r["content"].strip())
    notes_text = "\n".join(f"[{i+1}] {n}" for i, n in enumerate(notes))
    r = llm.chat([{"role": "user",
                   "content": CON_ANSWER.format(q=ex.question, notes=notes_text)}],
                 temperature=0.0)
    content = r["content"]
    abstained = "INSUFFICIENT EVIDENCE" in content.upper()
    return {"answer": content, "abstained": abstained,
            "n_retrievals": 1, "passages": passages}
