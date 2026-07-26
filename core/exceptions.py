"""Exception types raised by the LLM gateway layer."""
from __future__ import annotations


class SafetyViolation(Exception):
    """Raised when the gateway returns 450/451 (content policy)."""


class RateLimitExhausted(Exception):
    """Raised when ALL AppIds in the pool are rate-limited."""
