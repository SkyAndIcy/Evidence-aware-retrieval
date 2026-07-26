"""Prompt constants for src.methods.eara_v4.

Extracted into a dedicated module so prompts are versioned and editable
centrally, decoupled from agent control flow.
"""
COVERAGE_PROMPT = """You are judging whether the gathered evidence is SUFFICIENT to support the
proposed answer to the question. Evidence sufficiency means the passages contain
the specific facts needed to derive the answer (not just topically related).

Question: {q}
Proposed answer: {ans}
Evidence:
{evidence}

Is the evidence sufficient to support the proposed answer? Answer with a single
number 0 to 1: 1 = clearly sufficient, 0.5 = partially, 0 = insufficient (answer
is a guess not grounded in evidence). Output ONLY the number.

Sufficiency (0 to 1):"""
