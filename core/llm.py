"""Unified LLM API wrapper for an OpenAI-compatible gateway.

The gateway is reached via a configurable base_url and authenticated with an
AppId used as the Bearer token in the Authorization header. All models
(qwen-plus-latest, gpt-4o-mini, gemini-2.5-flash, kimi-k2.5, ...) are reached
through the same /v1/openai/native/chat/completions endpoint.

AppId pool: config api_key may be a single AppId OR a list of AppIds. When a
list is given, calls round-robin across AppIds and on a 429 (rate limit) the
offending AppId is temporarily benched and the next one is tried immediately
--- this multiplies effective RPM by the number of AppIds, which matters for
large batch runs.

Tool-calling (function calling) is supported via the standard OpenAI tools
schema, so ReAct and EARA share the same tool protocol.

Safety: the gateway returns 450/451 for content-policy violations and 429 for
rate limits. We retry 429/5xx with backoff (and AppId rotation) and surface
450/451 as a clear error so the caller can drop the offending example.

Set the gateway URL and AppId(s) via environment variables or config.yaml
(referenced as ${LLM_GATEWAY_URL} and ${LLM_APPID} in the template config).
"""
from __future__ import annotations
import os
import json
import time
import hashlib
import threading
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type




from core.base import BaseLLM
from core.exceptions import SafetyViolation, RateLimitExhausted
from core.cache import _resolve_env, _hash_call, cache_get, cache_set
from core.tools import ALL_TOOLS, normalize_tool_calls


class LLM(BaseLLM):
    """OpenAI-compatible gateway wrapper with AppId-pool round-robin + retry + cache."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg["model_name"]
        self.temperature = cfg.get("temperature", 0.6)
        self.max_tokens = cfg.get("max_tokens", 2048)
        self.base_url = cfg["base_url"]
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self.n_cache_hits = 0

        # Build AppId pool: accept str, list, or ${ENV} (single).
        raw = cfg["api_key"]
        if isinstance(raw, list):
            self.appids = [_resolve_env(x) for x in raw]
        else:
            self.appids = [_resolve_env(raw)]
        if not self.appids:
            raise ValueError("No AppId configured (api_key empty).")

        # One OpenAI client per AppId (api_key is set per-client).
        from openai import OpenAI
        self._clients = [OpenAI(base_url=self.base_url, api_key=a) for a in self.appids]

        # Round-robin index + per-AppId bench-until timestamps (for 429 cooldown).
        self._idx = 0
        self._bench_until: dict[int, float] = {}  # client_idx -> bench-until unix ts
        self._lock = threading.Lock()
        self._per_appid_calls = {i: 0 for i in range(len(self._clients))}

    def _next_client(self) -> tuple[int, Any]:
        """Pick the next non-benched client (round-robin). Raise if all benched."""
        with self._lock:
            now = time.time()
            n = len(self._clients)
            for _ in range(n):
                i = self._idx % n
                self._idx += 1
                if self._bench_until.get(i, 0) <= now:
                    self._per_appid_calls[i] += 1
                    return i, self._clients[i]
            # all benched: wait for the soonest one
            soonest = min(self._bench_until.get(i, 0) for i in range(n))
            wait = max(0.5, soonest - now)
        raise RateLimitExhausted(
            f"All {n} AppIds rate-limited; soonest available in {wait:.1f}s")

    def _bench(self, idx: int, seconds: float = 60.0):
        with self._lock:
            self._bench_until[idx] = time.time() + seconds

    @retry(stop=stop_after_attempt(8),
           wait=wait_exponential(min=1, max=30),
           retry=retry_if_exception_type((TimeoutError, ConnectionError, RateLimitExhausted)))
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, use_cache: bool = True) -> dict:
        """Return {content, tool_calls}. Normalized.

        Caching: deterministic calls (temperature 0) are memoized by a hash of
        (model, messages, tools). Non-zero temperature bypasses the cache.
        """
        # Some models force a specific temperature (e.g. kimi-k3 requires 1.0).
        # If the model's configured temperature is 1.0, always use it and ignore
        # any caller-supplied temperature, which would otherwise 400.
        if self.temperature == 1.0:
            temp = 1.0
        else:
            temp = self.temperature if temperature is None else temperature
        cache_key = None
        if use_cache and temp == 0.0:
            cache_key = _hash_call(self.model_name, messages, tools)
            cached = cache_get(cache_key)
            if cached is not None:
                self.n_cache_hits += 1
                return cached

        self.n_calls += 1
        last_err = None
        for _attempt in range(len(self._clients) + 2):  # try rotation a few times
            try:
                idx, client = self._next_client()
            except RateLimitExhausted as e:
                # all benched — sleep a bit and let tenacity retry
                time.sleep(2)
                last_err = e
                continue
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                msg = str(e)
                if "450" in msg or "451" in msg or "security" in msg.lower():
                    raise SafetyViolation(msg) from e
                if "429" in msg:
                    self._bench(idx, seconds=60.0)  # bench this AppId 60s
                    last_err = e
                    continue  # try next AppId immediately
                if any(c in msg for c in ("500", "502", "503", "504", "timeout", "timed out")):
                    last_err = TimeoutError(msg)
                    continue  # retry same/next
                raise  # unknown -> bubble up

            choice = resp.choices[0].message
            if resp.usage:
                self.total_prompt_tokens += resp.usage.prompt_tokens or 0
                self.total_completion_tokens += resp.usage.completion_tokens or 0
            out = {"content": choice.content or "", "tool_calls": normalize_tool_calls(choice)}
            if cache_key is not None:
                cache_set(cache_key, out)
            return out

        if last_err:
            raise last_err
        raise RuntimeError("chat() exhausted retries unexpectedly")

    def cost_summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "per_appid_calls": self._per_appid_calls,
            "n_appids": len(self._clients),
        }




