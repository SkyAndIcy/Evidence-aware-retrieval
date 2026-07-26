"""Dataset loaders for HotpotQA and 2WikiMultihopQA.

Both ship with supporting Wikipedia context paragraphs, which we use to build
a local BM25 index (offline, cost-free retrieval). This avoids external API
dependencies during the sprint while keeping the multi-hop retrieval setting
faithful to the benchmarks.

For WebDetective, load its own JSONL files directly (see load_webdetective()).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from core.retriever import Passage


class Example:
    __slots__ = ("qid", "question", "answer", "gold_titles", "context_passages",
                 "unanswerable")

    def __init__(self, qid, question, answer, gold_titles=None,
                 context_passages=None, unanswerable=False):
        self.qid = qid
        self.question = question
        self.answer = answer
        self.gold_titles = gold_titles or []
        self.context_passages = context_passages or []
        self.unanswerable = unanswerable  # for abstention eval


def load_hotpotqa(split: str = "validation", n: Optional[int] = None,
                  cache_dir: Optional[str] = None) -> list[Example]:
    """Load HotpotQA from HuggingFace datasets.

    HotpotQA context is a list of (title, [sentences]) tuples per question.
    We flatten these into Passage objects for the BM25 index.
    """
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, cache_dir=cache_dir)
    examples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        passages = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            text = " ".join(sents)
            passages.append(Passage(title, text))
        examples.append(Example(
            qid=row["id"],
            question=row["question"],
            answer=row["answer"],
            gold_titles=list(row["supporting_facts"]["title"]),
            context_passages=passages,
        ))
    return examples


def load_2wikimultihop(split: str = "dev", n: Optional[int] = None,
                       cache_dir: Optional[str] = None) -> list[Example]:
    """Load 2WikiMultihopQA. HF: 'voidful/2WikiMultihopQA' or local JSON.

    Falls back to a local JSON path if HF unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("voidful/2WikiMultihopQA", split=split, cache_dir=cache_dir)
        examples = []
        for i, row in enumerate(ds):
            if n is not None and i >= n:
                break
            passages = []
            # 2Wiki stores context as list of {title, sentences}
            for c in row.get("context", []):
                title = c["title"]
                text = " ".join(c["sentences"])
                passages.append(Passage(title, text))
            examples.append(Example(
                qid=str(row.get("_id", i)),
                question=row["question"],
                answer=row.get("answer", ""),
                gold_titles=[c["title"] for c in row.get("context", [])],
                context_passages=passages,
            ))
        return examples
    except Exception as e:
        print(f"[2Wiki] HF load failed ({e}); provide local JSON via cache_dir")
        return []


def load_webdetective(jsonl_path: str, n: Optional[int] = None) -> list[Example]:
    """Load WebDetective hint-free questions.

    WebDetective JSONL has fields like {question, answer, ...}.
    The unanswerable flag is set by WebDetective's evaluation harness
    (questions whose supporting evidence was removed). We mark them here
    if the file includes an 'unanswerable' or 'abstain' field.
    """
    examples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            row = json.loads(line)
            examples.append(Example(
                qid=row.get("id", str(i)),
                question=row["question"],
                answer=row.get("answer", ""),
                gold_titles=row.get("gold_titles", []),
                context_passages=[],  # WebDetective provides retrieval via sandbox
                unanswerable=row.get("unanswerable", False),
            ))
    return examples


def all_context_passages(examples: list[Example]) -> list[Passage]:
    """Flatten all context passages across examples for a global BM25 index.

    NOTE: For a fair retrieval setting, index only per-example context
    (open-book per question). For a harder global setting, index all.
    The default per-example setting matches HotpotQA's standard eval.
    """
    seen = {}
    for ex in examples:
        for p in ex.context_passages:
            if p.title not in seen:
                seen[p.title] = p
    return list(seen.values())


# ---- CASE benchmarks: MuSiQue, Bamboogle ------------------------------------

def load_musique(split: str = "validation", n: Optional[int] = None,
                 cache_dir: Optional[str] = None) -> list[Example]:
    """Load MuSiQue (multi-hop QA with composed sub-questions).

    HF: 'dgslibisey/MuSiQue'. Each example has supporting paragraphs.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"[MuSiQue] HF load failed ({e}); check dataset name/availability")
        return []
    examples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        passages = []
        # MuSiQue stores supporting paragraphs in 'paragraphs' list
        for para in row.get("paragraphs", []):
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if text:
                passages.append(Passage(title, text))
        examples.append(Example(
            qid=row.get("id", str(i)),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            gold_titles=[p.get("title", "") for p in row.get("paragraphs", [])
                         if p.get("is_supporting", False)],
            context_passages=passages,
        ))
    return examples


def load_musique_hard(split: str = "validation", n: Optional[int] = None,
                      min_hops: int = 4, cache_dir: Optional[str] = None) -> list[Example]:
    """Load MuSiQue HARD subset: only questions with >= min_hops decomposition steps.

    MuSiQue has 2/3/4-hop questions. The 4-hop subset is substantially harder
    (more retrieval steps needed, more evidence to gather). Used to test whether
    EARA's evidence-sufficiency gate helps when retrieval is genuinely difficult
    (on easy 2-hop hotpotqa, strong models answer from parametric knowledge and
    the gate only interferes).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"[MuSiQue-hard] HF load failed ({e})")
        return []
    examples = []
    for i, row in enumerate(ds):
        decomp = row.get("question_decomposition", [])
        if len(decomp) < min_hops:
            continue
        passages = []
        for para in row.get("paragraphs", []):
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if text:
                passages.append(Passage(title, text))
        examples.append(Example(
            qid=row.get("id", str(i)),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            gold_titles=[p.get("title", "") for p in row.get("paragraphs", [])
                         if p.get("is_supporting", False)],
            context_passages=passages,
        ))
        if n is not None and len(examples) >= n:
            break
    return examples


def load_musique3(split: str = "validation", n: Optional[int] = None,
                  cache_dir: Optional[str] = None) -> list[Example]:
    """Load MuSiQue 3-hop subset (medium difficulty).

    Difficulty gradient: hotpotqa(2hop) < musique3(3hop) < musique_hard(4hop).
    3-hop is the sweet spot: hard enough that retrieval matters (unlike 2-hop
    where strong models use parametric knowledge), but not so hard that all
    methods collapse (like 4-hop where acc~0.1).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"[MuSiQue3] HF load failed ({e})")
        return []
    examples = []
    for i, row in enumerate(ds):
        decomp = row.get("question_decomposition", [])
        if len(decomp) != 3:  # exactly 3 hops
            continue
        passages = []
        for para in row.get("paragraphs", []):
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if text:
                passages.append(Passage(title, text))
        examples.append(Example(
            qid=row.get("id", str(i)),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            gold_titles=[p.get("title", "") for p in row.get("paragraphs", [])
                         if p.get("is_supporting", False)],
            context_passages=passages,
        ))
        if n is not None and len(examples) >= n:
            break
    return examples


def load_musique2(split: str = "validation", n: Optional[int] = None,
                  cache_dir: Optional[str] = None) -> list[Example]:
    """Load MuSiQue 2-hop subset.

    2-hop is the only MuSiQue difficulty that works under API-only + BM25
    retrieval (3/4-hop collapses to ~0.05 for all methods). Slightly harder
    than hotpotqa's 2-hop (more distractor paragraphs), giving a useful 3rd
    dataset with discrimination.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"[MuSiQue2] HF load failed ({e})")
        return []
    examples = []
    for i, row in enumerate(ds):
        decomp = row.get("question_decomposition", [])
        if len(decomp) != 2:  # exactly 2 hops
            continue
        passages = []
        for para in row.get("paragraphs", []):
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if text:
                passages.append(Passage(title, text))
        examples.append(Example(
            qid=row.get("id", str(i)),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            gold_titles=[p.get("title", "") for p in row.get("paragraphs", [])
                         if p.get("is_supporting", False)],
            context_passages=passages,
        ))
        if n is not None and len(examples) >= n:
            break
    return examples


def load_bamboogle(n: Optional[int] = None,
                   cache_dir: Optional[str] = None) -> list[Example]:
    """Load Bamboogle (multi-hop QA over Wikipedia, 2-hop questions).

    HF: 'course/bamboogle' or local. Has question + answer + supporting context.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("course/bamboogle", split="test", cache_dir=cache_dir)
    except Exception as e:
        print(f"[Bamboogle] HF load failed ({e}); trying alternative source")
        try:
            ds = load_dataset("tasksource/bamboogle", split="test", cache_dir=cache_dir)
        except Exception as e2:
            print(f"[Bamboogle] alternative also failed ({e2})")
            return []
    examples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        passages = []
        # Bamboogle may store context in various fields
        ctx = row.get("context") or row.get("supporting_context") or []
        if isinstance(ctx, list):
            for c in ctx:
                if isinstance(c, dict):
                    passages.append(Passage(c.get("title", str(i)),
                                           c.get("text", c.get("paragraph_text", ""))))
                elif isinstance(c, str):
                    passages.append(Passage(str(i), c))
        elif isinstance(ctx, str) and ctx:
            passages.append(Passage("ctx", ctx))
        examples.append(Example(
            qid=str(row.get("id", i)),
            question=row.get("question", ""),
            answer=str(row.get("answer", "")),
            gold_titles=[],
            context_passages=passages,
        ))
    return examples


# ---- FALCON benchmarks: TriviaQA, StrategyQA --------------------------------

def load_triviaqa(split: str = "validation", n: Optional[int] = None,
                  cache_dir: Optional[str] = None) -> list[Example]:
    """Load TriviaQA (factoid QA with Wikipedia evidence documents).

    HF: 'mandarjoshi/trivia_qa', config 'rc.nocontext' (we attach the golden
    entity's Wikipedia passages as context for BM25). For the FALCON noisy-
    retrieval setting, the agent retrieves from the corpus; noise is injected
    by the noise_injector (see src/noise.py).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                          split=split, cache_dir=cache_dir)
    except Exception as e:
        print(f"[TriviaQA] HF load failed ({e})")
        return []
    examples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        q = row.get("question", "")
        # TriviaQA answer aliases
        ans = row.get("answer", {})
        gold = ans.get("value", "") if isinstance(ans, dict) else str(ans)
        examples.append(Example(
            qid=row.get("question_id", str(i)),
            question=q,
            answer=gold,
            gold_titles=[],
            context_passages=[],  # no per-question context; agent retrieves from Wikipedia
        ))
    return examples


def load_strategyqa(split: str = "validation", n: Optional[int] = None,
                    cache_dir: Optional[str] = None) -> list[Example]:
    """Load StrategyQA (implicit multi-step yes/no reasoning).

    HF: 'voidful/StrategyQA' or 'tasksource/strategy-qa'. Answers are yes/no.
    Facts are decomposed; the agent must assemble and judge implicit evidence.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/strategy-qa", split="train", cache_dir=cache_dir)
    except Exception as e:
        print(f"[StrategyQA] tasksource load failed ({e}); trying voidful")
        try:
            ds = load_dataset("voidful/StrategyQA", split="train", cache_dir=cache_dir)
        except Exception as e2:
            print(f"[StrategyQA] voidful also failed ({e2})")
            return []
    examples = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        ans_raw = row.get("answer", row.get("label", False))
        if isinstance(ans_raw, bool):
            gold = "yes" if ans_raw else "no"
        elif isinstance(ans_raw, str):
            gold = ans_raw.lower()
        elif isinstance(ans_raw, int):
            gold = "yes" if ans_raw == 1 else "no"
        else:
            gold = "yes"
        # decomposed facts may be available as supporting evidence
        facts = row.get("facts", row.get("decomposition", []))
        passages = []
        if isinstance(facts, list):
            for j, f in enumerate(facts):
                if isinstance(f, str) and f:
                    passages.append(Passage(f"fact_{j}", f))
        elif isinstance(facts, str) and facts:
            passages.append(Passage("facts", facts))
        examples.append(Example(
            qid=str(row.get("id", i)),
            question=row.get("question", ""),
            answer=gold,
            gold_titles=[],
            context_passages=passages,
        ))
    return examples
