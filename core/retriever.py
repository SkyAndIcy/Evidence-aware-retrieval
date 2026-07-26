"""Retrieval backends for EARA experiments.

Three modes (selected in config):
  - wikipedia_api: MediaWiki API via the `wikipedia` package (no setup, rate-limited).
  - bm25_local:    Build BM25 over the context paragraphs that ship with
                   HotpotQA / 2WikiMultihopQA. This is the most reliable,
                   fully-offline, cost-free mode and the recommended default
                   for the 8-day sprint.
  - webdetective:  Delegate retrieval to the WebDetective sandbox (requires
                   their environment + SERPER_API_KEY). Enable once sandbox is up.

All modes expose the same interface: search(query, k) -> list[Passage].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Passage:
    title: str
    text: str

    def __repr__(self):
        return f"[{self.title}] {self.text[:200]}..."


from core.base import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, mode: str, config: Optional[dict] = None):
        self.mode = mode
        self.config = config or {}
        self._bm25 = None
        self._corpus_titles = []
        if mode == "bm25_local":
            self._init_bm25()
        elif mode == "wikipedia_api":
            pass  # lazy, uses `wikipedia` package
        elif mode == "webdetective":
            # Placeholder: integrate with WebDetective's retrieval module.
            # See WebDetective/utils/wiki_search.py for the exact call.
            raise NotImplementedError("Enable after cloning WebDetective repo")
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def search(self, query: str, k: int = 3) -> list[Passage]:
        if self.mode == "bm25_local":
            return self._bm25_search(query, k)
        elif self.mode == "wikipedia_api":
            return self._wiki_api_search(query, k)
        raise RuntimeError("unsupported mode")

    def lookup(self, title: str) -> Optional[Passage]:
        if self.mode == "bm25_local":
            text = self._title2text.get(title)
            return Passage(title=title, text=text) if text else None
        elif self.mode == "wikipedia_api":
            return self._wiki_api_lookup(title)
        return None

    # ---- BM25 local ---------------------------------------------------------

    def _init_bm25(self):
        """Build a BM25 index over paragraphs collected from the dataset loader.

        Call build_bm25_from_dataset(dataset) before using search().
        """
        from rank_bm25 import BM25Okapi
        # corpus set externally via build_bm25_from_dataset
        self._bm25 = None
        self._title2text = {}

    def build_from_passages(self, passages: list[Passage]):
        """Index a flat list of passages (title + text)."""
        from rank_bm25 import BM25Okapi
        self._corpus_titles = [p.title for p in passages]
        tokenized = [p.text.lower().split() for p in passages]
        self._bm25 = BM25Okapi(tokenized)
        # dedup by title (last wins)
        self._title2text = {p.title: p.text for p in passages}

    def _bm25_search(self, query: str, k: int) -> list[Passage]:
        if self._bm25 is None:
            raise RuntimeError("BM25 not built. Call build_from_passages() first.")
        scores = self._bm25.get_scores(query.lower().split())
        import numpy as np
        top = np.argsort(scores)[::-1][:k]
        return [Passage(self._corpus_titles[i], self._title2text[self._corpus_titles[i]])
                for i in top if scores[i] > 0]

    # ---- Wikipedia API ------------------------------------------------------

    def _wiki_api_search(self, query: str, k: int) -> list[Passage]:
        import wikipedia
        results = []
        try:
            titles = wikipedia.search(query, results=k)
        except Exception:
            return []
        for t in titles[:k]:
            try:
                summary = wikipedia.summary(t, sentences=3)
                results.append(Passage(t, summary))
            except Exception:
                continue
        return results

    def _wiki_api_lookup(self, title: str) -> Optional[Passage]:
        import wikipedia
        try:
            return Passage(title, wikipedia.summary(title, sentences=5))
        except Exception:
            return None


def build_bm25_from_dataset(context_passages: list[Passage]) -> Retriever:
    """Convenience: create a bm25_local retriever pre-indexed on dataset context."""
    r = Retriever("bm25_local")
    r.build_from_passages(context_passages)
    return r
