"""Smoke tests: verify the package imports cleanly after refactoring."""
import sys
import pathlib

# ensure repo root is importable when running from tests/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_core_imports():
    from core import (
        BaseLLM, BaseRetriever, LLM, Retriever, Passage,
        SafetyViolation, RateLimitExhausted, ALL_TOOLS,
        cache_get, cache_set, _hash_call,
    )
    assert LLM is not None and Retriever is not None


def test_methods_imports():
    from methods import (
        run_react, run_rag, run_direct, run_selfask,
        run_ircot, run_crag, run_eara, run_eara_v4, run_eara_ircot,
    )
    assert callable(run_react) and callable(run_eara_ircot)


def test_prompts_imports():
    from prompts import baselines_prompts, eara_prompts, eara_v4_prompts
    assert hasattr(baselines_prompts, "REACT_SYSTEM")
    assert hasattr(eara_prompts, "DECOMPOSE_PROMPT")
    assert hasattr(eara_v4_prompts, "COVERAGE_PROMPT")


def test_evaluation_imports():
    from evaluation import evaluate, coverage_at_acc, accuracy, abstention_auroc
    assert callable(evaluate) and callable(coverage_at_acc)


def test_data_imports():
    from data import Example, all_context_passages
    assert Example is not None


def test_run_entrypoint_imports():
    from run import main, run_method, load_config
    assert callable(main)


def test_llm_inherits_base():
    from core import LLM, BaseLLM
    assert issubclass(LLM, BaseLLM)


def test_retriever_inherits_base():
    from core import Retriever, BaseRetriever
    assert issubclass(Retriever, BaseRetriever)
