"""Core abstractions: base interfaces, LLM gateway, retriever, and helpers.

Layered structure:
  base.py         abstract BaseLLM / BaseRetriever contracts
  exceptions.py   SafetyViolation, RateLimitExhausted
  cache.py        in-process response cache + ${ENV} resolution + call hashing
  tools.py        OpenAI function-calling tool schemas + response normalization
  llm.py          concrete LLM (OpenAI-compatible gateway, AppId-pool round-robin)
  retriever.py    concrete Retriever (BM25) + Passage dataclass
"""
from core.base import BaseLLM, BaseRetriever
from core.exceptions import SafetyViolation, RateLimitExhausted
from core.cache import _resolve_env, _hash_call, cache_get, cache_set
from core.tools import ALL_TOOLS, WIKI_SEARCH_TOOL, WIKI_LOOKUP_TOOL, normalize_tool_calls
from core.llm import LLM
from core.retriever import Retriever, Passage, build_bm25_from_dataset

__all__ = [
    "BaseLLM", "BaseRetriever",
    "SafetyViolation", "RateLimitExhausted",
    "_resolve_env", "_hash_call", "cache_get", "cache_set",
    "ALL_TOOLS", "WIKI_SEARCH_TOOL", "WIKI_LOOKUP_TOOL", "normalize_tool_calls",
    "LLM", "Retriever", "Passage", "build_bm25_from_dataset",
]
