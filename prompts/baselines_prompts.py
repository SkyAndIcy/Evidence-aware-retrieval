"""Prompt constants for src.methods.baselines.

Extracted into a dedicated module so prompts are versioned and editable
centrally, decoupled from agent control flow.
"""
REACT_SYSTEM = """You are a multi-hop question answering agent. Use the wiki_search tool to find information, then answer the question.

Follow this loop:
1. THINK: reason about what you need to find next.
2. ACT: call wiki_search(query=<your query>).
3. OBSERVE: read the returned passages.
4. Repeat until you have enough evidence, then answer with: FINAL ANSWER: <answer>.

If you cannot find sufficient evidence after several searches, give your best guess."""
RAG_SYSTEM = """Answer the multi-hop question using the provided context passages.
If the context is insufficient, use your own knowledge to give your best-guess
short answer. Always give a concrete short answer; never refuse or say you
cannot answer."""
DIRECT_SYSTEM = """Answer the question directly from your own knowledge.
Always give your best-guess short answer; never refuse or say you do not know."""
SELFASK_DECOMPOSE = """Decompose the question into a chain of follow-up sub-questions, where each
sub-question's answer is needed to answer the next, ending with the original question.
Output one sub-question per line, numbered. End with the original question.

Question: {q}

Sub-questions:"""
SELFASK_ANSWER = """Answer the sub-question using the retrieved passages. If the passages do not
contain the answer, use your own knowledge. Output only the short answer.

Sub-question: {sq}
Passages:
{context}

Short answer:"""""
SELFASK_FINAL = """Answer the original question using the chain of sub-answers below.
Output only the short final answer.

Original question: {q}
Sub-answers:
{chain}

Final answer:"""
CRAG_EVAL = """Judge the quality of retrieved passages for answering the question.
Output one word:
- CORRECT: passages clearly contain the answer
- AMBIGUOUS: partially relevant but not sufficient
- INCORRECT: passages are irrelevant

Question: {q}
Passages:
{context}

Quality:"""
CRAG_ANSWER = """Answer the question using the passages. If they are insufficient,
use your own knowledge. Output only the short answer.

Question: {q}
Passages:
{context}

Answer:"""
CON_NOTE = """Read the retrieved document and write a short note (1-2 sentences) assessing whether
and how it helps answer the question. If the document is irrelevant, write: "NOTE: irrelevant".

Question: {q}
Document: {doc}

Note:"""
CON_ANSWER = """Answer the question using the notes below. If all notes say irrelevant, answer:
INSUFFICIENT EVIDENCE. Otherwise give the short answer.

Question: {q}
Notes:
{notes}

Answer:"""
