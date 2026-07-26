"""Prompt constants for src.methods.eara.

Extracted into a dedicated module so prompts are versioned and editable
centrally, decoupled from agent control flow.
"""
DECOMPOSE_PROMPT = """Decompose the following multi-hop question into 2-4 atomic sub-questions that must each be answered to resolve the original. Output one sub-question per line, numbered.

Question: {q}

Sub-questions:"""
REWRITE_PROMPT = """You are looking for evidence to answer this sub-question:
  Sub-question: {sq}

Original question: {q}
Evidence gathered so far:
{evidence}

Write a concise Wikipedia search query (just the query, no preamble) to find the missing evidence:"""
MONITOR_PROMPT = """Given the evidence below, judge each of these for the ORIGINAL question "{q}":

Sub-questions:
{subqs}

Evidence:
{evidence}

Answer these three questions, one per line:
1. COVERAGE (0 to 1): fraction of sub-questions with sufficient direct evidence.
2. CONFLICT (0 to 1): degree of contradiction among evidence snippets (0=no conflict).
3. CREDIBILITY (0 to 1): aggregate credibility of the sources.

Output exactly three numbers, one per line."""
ANSWER_PROMPT = """Answer the question based ONLY on the evidence. Give a short final answer.

Question: {q}
Evidence:
{evidence}

Answer (short):"""
SC_STUB = """Propose a short answer to the question based on the evidence.
Question: {q}
Evidence: {evidence}
Answer:"""
