"""In-process response cache for deterministic (temperature 0) LLM calls.

Caching is keyed by a hash of (model, messages, tools); non-zero-temperature
calls bypass the cache. The cache is module-level and survives across calls
within one run, avoiding redundant API spend on repeated identical prompts.
"""
from __future__ import annotations
import os
import json
import hashlib

_CACHE: dict[str, dict] = {}


def _resolve_env(val: str) -> str:
    """Resolve ${ENV_VAR} references. Passes plain strings through unchanged."""
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        env = val[2:-1]
        v = os.environ.get(env)
        if v is None:
            raise RuntimeError(f"Environment variable {env} not set.")
        return v
    return val


def _hash_call(model: str, messages: list[dict], tools: list[dict] | None) -> str:
    blob = json.dumps({"model": model, "messages": messages, "tools": tools},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_get(k: str):
    return _CACHE.get(k)


def cache_set(k: str, v: dict):
    _CACHE[k] = v
