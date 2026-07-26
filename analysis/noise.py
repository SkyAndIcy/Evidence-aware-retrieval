"""Noise injection for FALCON experiments.

FALCON's claim is robustness to noisy/conflicting/adversarial retrieved
passages. To stress this, we inject controlled noise into the retriever's
output before FALCON's falsifiability filter sees it (and before baselines
see it, for fairness).

Three noise modes:
  - random:       inject k passages sampled at random from the corpus (distractors).
  - conflict:     inject 1 passage that asserts the opposite of a gold passage.
  - adversarial:  inject 1 fluently-written false passage (LLM-generated to look credible).

The injector wraps a Retriever: NoiseyRetriever.search() calls the base
retriever, then mixes in noise passages. Gold titles (from the example) are
used to avoid injecting the actual gold passage as a "distractor" and to
construct conflicting/adversarial passages against the gold.
"""
from __future__ import annotations
import random
from typing import Optional
from core.retriever import Retriever, Passage
from core.llm import LLM


class NoisyRetriever:
    """Wraps a base Retriever, injecting noise into each search() result."""

    def __init__(self, base: Retriever, mode: str = "random", k: int = 2,
                 corpus_passages: Optional[list[Passage]] = None,
                 gold_titles: Optional[list[str]] = None,
                 gold_answer: str = "",
                 llm: Optional[LLM] = None,
                 seed: int = 42):
        self.base = base
        self.mode = mode
        self.k = k
        self.corpus = corpus_passages or []
        self.gold_titles = set(t.lower() for t in (gold_titles or []))
        self.gold_answer = gold_answer
        self.llm = llm
        self.rng = random.Random(seed)
        # expose base retriever's mode for logging
        self.mode_base = getattr(base, "mode", "unknown")

    def search(self, query: str, k: int = 3) -> list[Passage]:
        real = self.base.search(query, k=k)
        if self.mode == "none" or self.k <= 0:
            return real
        noise = self._make_noise(real)
        return real + noise

    def lookup(self, title: str) -> Optional[Passage]:
        return self.base.lookup(title)

    def _make_noise(self, real: list[Passage]) -> list[Passage]:
        """Generate noise passages to append to `real`."""
        if self.mode == "random":
            return self._random_distractors(self.k)
        if self.mode == "conflict":
            return self._conflicting_passage(real)
        if self.mode == "adversarial":
            return self._adversarial_passage(real)
        return []

    def _random_distractors(self, k: int) -> list[Passage]:
        """Sample k passages from the corpus that are NOT gold-titled."""
        pool = [p for p in self.corpus if p.title.lower() not in self.gold_titles]
        if not pool:
            return []
        n = min(k, len(pool))
        return self.rng.sample(pool, n)

    def _conflicting_passage(self, real: list[Passage]) -> list[Passage]:
        """Inject 1 passage asserting the opposite of the gold answer.

        If we know the gold answer, craft a passage that states a different
        (wrong) answer to the same question. This creates a direct conflict.
        """
        if not self.gold_answer:
            return self._random_distractors(1)
        wrong = self._wrong_answer(self.gold_answer)
        return [Passage(title="Conflict",
                        text=f"According to recent records, the answer is {wrong}. "
                             f"This contradicts other sources on the matter.")]

    def _wrong_answer(self, gold: str) -> str:
        """A plausible-looking wrong answer. Deterministic per gold string."""
        # Simple deterministic perturbation: reverse words or append a counter.
        words = gold.split()
        if len(words) > 1:
            return " ".join(reversed(words))
        return f"not {gold}"

    def _adversarial_passage(self, real: list[Passage]) -> list[Passage]:
        """Inject 1 LLM-generated fluent-but-false passage (looks credible).

        The passage is generated to be on-topic (mentions the query entities)
        but states a false claim, written fluently to test whether the agent
        is fooled by surface credibility.
        """
        if self.llm is None:
            return self._conflicting_passage(real)
        topic = real[0].title if real else "the topic"
        try:
            resp = self.llm.chat([{"role": "user",
                "content": f"Write a short (2-3 sentence) passage about {topic} "
                           f"that sounds credible and well-written but contains "
                           f"a FALSE factual claim. Do not mark it as false. "
                           f"Output only the passage."}], temperature=0.9)
            return [Passage(title="Adversarial", text=resp["content"].strip())]
        except Exception:
            return self._conflicting_passage(real)


def make_noisy(base: Retriever, cfg_noise: dict, ex, llm: Optional[LLM] = None,
               corpus_passages: Optional[list[Passage]] = None) -> NoisyRetriever:
    """Build a NoisyRetriever from a config noise block + example."""
    return NoisyRetriever(
        base=base,
        mode=cfg_noise.get("mode", "random"),
        k=cfg_noise.get("k", 2),
        corpus_passages=corpus_passages,
        gold_titles=getattr(ex, "gold_titles", []),
        gold_answer=getattr(ex, "answer", ""),
        llm=llm,
        seed=cfg_noise.get("seed", 42),
    )
