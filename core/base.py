"""Abstract base interfaces for core components.

Defines the contracts that concrete implementations (LLM gateway wrappers,
retrievers) must satisfy. This separation keeps the methods layer decoupled
from any specific API client or retrieval backend, so EARA and the baselines
depend only on these interfaces rather than concrete classes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract LLM interface: a stateful chat client with tool-calling.

    Implementations are expected to handle retries, rate-limit rotation, and
    caching internally; callers see only a normalized ``chat`` response.
    """

    model_name: str

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, use_cache: bool = True) -> dict:
        """Return ``{"content": str, "tool_calls": list[dict]}``."""
        ...

    @abstractmethod
    def cost_summary(self) -> dict:
        """Return cumulative token / call accounting."""
        ...


class BaseRetriever(ABC):
    """Abstract retriever interface over a document corpus."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[Any]:
        """Return the top-k passages for ``query``."""
        ...

    @abstractmethod
    def lookup(self, title: str) -> Any | None:
        """Return the passage whose title matches ``title``, or None."""
        ...
