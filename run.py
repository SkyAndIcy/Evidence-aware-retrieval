"""Main experiment runner.

Usage:
  python -m src.run --model deepseek-v4-flash --benchmark hotpotqa --method eara
  python -m src.run --model deepseek-v4-flash --benchmark hotpotqa --method react
  python -m src.run --model deepseek-v4-flash --benchmark hotpotqa --method rag

Results are saved to results/{model}_{benchmark}_{method}.jsonl with a summary
in results/{model}_{benchmark}_{method}_summary.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import yaml  # pip install pyyaml (in requirements.txt)

# Force HF to use local cache — avoids hangs when spawn'd worker processes
# try to re-verify datasets against huggingface.co (which can time out and
# stall the whole run). Datasets are downloaded once on first use.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# allow `python -m src.run` and `python run.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import LLM
from core.retriever import Retriever, build_bm25_from_dataset
from data import datasets as DS
from methods.baselines import (run_react, run_rag, run_direct, run_selfask,
                           run_ircot, run_crag, run_con)
from methods.eara import run_eara, calibrate_thresholds
from methods.eara_v4 import run_eara_v4, run_eara_ircot
from analysis.case_agent import run_case, calibrate as calibrate_case
from analysis.falcon import run_falcon, calibrate_falcon
from evaluation.metrics import evaluate, is_correct


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_retriever(cfg: dict, examples, benchmark: str = "") -> Retriever:
    mode = cfg["retrieval"]["mode"]
    if mode == "bm25_local":
        # Index per-example context for the standard open-book setting.
        all_passages = []
        for ex in examples:
            all_passages.extend(ex.context_passages)
        # For benchmarks without shipped context (TriviaQA, StrategyQA), build a
        # shared Wikipedia index from HotpotQA's context so retrieval still works.
        if not all_passages and benchmark in ("triviaqa", "strategyqa"):
            print(f"[retriever] {benchmark} has no context; building shared Wikipedia index from HotpotQA...")
            hp = DS.load_hotpotqa(split="validation", n=2000)
            for ex in hp:
                all_passages.extend(ex.context_passages)
        if not all_passages:
            # Avoid BM25 ZeroDivisionError on empty corpus.
            all_passages = [Passage("empty", "no content")]
        return build_bm25_from_dataset(all_passages)
    return Retriever(mode, cfg["retrieval"])


def run_method(method: str, llm, retriever, ex, cfg):
    if method == "react":
        # Cap ReAct retrieval budget at T=4 steps, matching EARA's operating regime.
        r = run_react(llm, retriever, ex, max_steps=4)
    elif method == "rag":
        r = run_rag(llm, retriever, ex, top_k=cfg["retrieval"]["top_k"])
    elif method == "direct":
        r = run_direct(llm, retriever, ex)
    elif method == "selfask":
        r = run_selfask(llm, retriever, ex, top_k=cfg["retrieval"]["top_k"])
    elif method == "ircot":
        r = run_ircot(llm, retriever, ex, max_steps=cfg["experiment"]["max_steps"])
    elif method == "crag":
        r = run_crag(llm, retriever, ex, top_k=cfg["retrieval"]["top_k"])
    elif method == "con":
        r = run_con(llm, retriever, ex, top_k=cfg["retrieval"]["top_k"])
    elif method == "eara":
        r = run_eara(llm, retriever, ex,
                     tau_c=cfg["eara"]["tau_c"],
                     tau_delta=cfg["eara"]["tau_delta"],
                     tau_p=cfg["eara"]["tau_p"],
                     max_steps=cfg["experiment"]["max_steps"],
                     sc_k=cfg["experiment"]["selfconsistency_k"])
    elif method == "eara4":
        # EARA-v4: ReAct retrieval + evidence-sufficiency gate + abstention.
        r = run_eara_v4(llm, retriever, ex,
                        tau_cov=0.5, max_steps=10)
    elif method == "eara_ircot":
        # EARA-ircot (best): IRCoT retrieval + evidence-sufficiency gate.
        r = run_eara_ircot(llm, retriever, ex,
                           tau_cov=0.5, max_steps=8)
    elif method == "eara_ircot_t3":
        # looser gate tau=0.3
        r = run_eara_ircot(llm, retriever, ex, tau_cov=0.3, max_steps=8)
    elif method == "eara_ircot_t2":
        # looser gate tau=0.2
        r = run_eara_ircot(llm, retriever, ex, tau_cov=0.2, max_steps=8)
    elif method == "eara_ircot_t2s10":
        # tau=0.2, max_steps=10, with a separate stronger judge LLM
        judge_llm = getattr(llm, "_judge_llm", None)
        if judge_llm is None:
            judge_llm = LLM(cfg["models"]["gpt-4.1-mini"])
            llm._judge_llm = judge_llm
        r = run_eara_ircot(llm, retriever, ex, tau_cov=0.2, max_steps=10, judge_llm=judge_llm)
    elif method in ("eara_ircot_j3","eara_ircot_j4","eara_ircot_j5","eara_ircot_j6"):
        # stronger-judge variant; judge model configurable via cfg["eara"]["judge_model"]
        tau = {"eara_ircot_j3":0.3,"eara_ircot_j4":0.4,"eara_ircot_j5":0.5,"eara_ircot_j6":0.6}[method]
        judge_model = cfg.get("eara", {}).get("judge_model", "gpt-4.1-mini")
        judge_llm = getattr(llm, "_judge_llm", None)
        if judge_llm is None or getattr(llm, "_judge_model", "") != judge_model:
            judge_llm = LLM(cfg["models"][judge_model])
            llm._judge_llm = judge_llm
            llm._judge_model = judge_model
        r = run_eara_ircot(llm, retriever, ex, tau_cov=tau, max_steps=10, judge_llm=judge_llm)
    elif method in ("eara_ircot_tau001","eara_ircot_tau005","eara_ircot_tau01","eara_ircot_tau02","eara_ircot_tau03","eara_ircot_tau04","eara_ircot_tau05","eara_ircot_tau06","eara_ircot_tau07"):
        # self-judge abstention sweep: no judge_llm, max_steps=10, tau=0.01..0.7
        tau = {"eara_ircot_tau001":0.01,"eara_ircot_tau005":0.05,"eara_ircot_tau01":0.1,"eara_ircot_tau02":0.2,"eara_ircot_tau03":0.3,"eara_ircot_tau04":0.4,"eara_ircot_tau05":0.5,"eara_ircot_tau06":0.6,"eara_ircot_tau07":0.7}[method]
        r = run_eara_ircot(llm, retriever, ex, tau_cov=tau, max_steps=10)
    elif method in ("eara_ircot_tp06","eara_ircot_tp08","eara_ircot_tp09","eara_ircot_tp1"):
        # tau_p sweep (tau_cov=0.6 fixed, vary self-consistency bar)
        tp = {"eara_ircot_tp06":0.6,"eara_ircot_tp08":0.8,"eara_ircot_tp09":0.9,"eara_ircot_tp1":1.0}[method]
        r = run_eara_ircot(llm, retriever, ex, tau_cov=0.6, max_steps=10, tau_p=tp, sc_k=5)
    elif method == "eara_ircot_noab":
        r = run_eara_ircot(llm, retriever, ex, tau_cov=0.3, max_steps=10, tau_p=-1.0)
    elif method == "case":
        case_cfg = cfg.get("case", {})
        r = run_case(llm, retriever, ex,
                     budget=case_cfg.get("budget", 24),
                     theta=tuple(case_cfg.get("theta", [0.7, 0.4, 0.7, 0.5])),
                     alpha=tuple(case_cfg.get("alpha", [1.0, 1.0, 1.0, 1.0])),
                     S=case_cfg.get("S", 4),
                     V=case_cfg.get("V", 4),
                     sc_k=cfg["experiment"]["selfconsistency_k"])
        # normalize to common record shape
        r = {"answer": r["answer"], "abstained": r["abstained"],
             "n_retrievals": r.get("spend", 0), "trace": r.get("trace")}
    elif method == "falcon":
        falcon_cfg = cfg.get("falcon", {})
        # Build corpus passage list once for the noise injector (cached on llm).
        corpus = getattr(llm, "_falcon_corpus", None)
        if corpus is None:
            corpus = []
            # corpus is built from all examples' context in main; attach to llm
            corpus = getattr(run_method, "_corpus", [])
        r = run_falcon(llm, retriever, ex,
                       theta_drop=falcon_cfg.get("theta_drop", 0.3),
                       K=falcon_cfg.get("K", 2),
                       max_steps=cfg["experiment"]["max_steps"],
                       noise_cfg=falcon_cfg.get("noise"),
                       corpus_passages=corpus)
    else:
        raise ValueError(method)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True,
                    choices=["hotpotqa", "2wikimultihop", "webdetective",
                             "musique", "musique_hard", "musique3", "musique2", "bamboogle", "triviaqa", "strategyqa"])
    ap.add_argument("--method", required=True,
                    choices=["react", "rag", "direct", "selfask", "ircot", "crag", "con",
                             "eara", "eara4", "eara_ircot", "eara_ircot_t3", "eara_ircot_t2", "eara_ircot_t2s10", "eara_ircot_j3","eara_ircot_j4","eara_ircot_j5","eara_ircot_j6",
                             "eara_ircot_tau001","eara_ircot_tau005","eara_ircot_tau01","eara_ircot_tau02","eara_ircot_tau03","eara_ircot_tau04","eara_ircot_tau05","eara_ircot_tau06","eara_ircot_tau07","eara_ircot_noab","eara_ircot_noab",
                             "eara_ircot_tp06","eara_ircot_tp08","eara_ircot_tp09","eara_ircot_tp1",
                             "case", "falcon"])
    ap.add_argument("--n", type=int, default=None, help="override n_test")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrate EARA thresholds on dev set first")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel question workers (each uses one AppId from the model's pool)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["models"][args.model]
    llm = LLM(model_cfg)

    n = args.n or cfg["experiment"]["n_test"]

    # Load benchmark
    if args.benchmark == "hotpotqa":
        examples = DS.load_hotpotqa(split="validation", n=n)
    elif args.benchmark == "2wikimultihop":
        examples = DS.load_2wikimultihop(split="dev", n=n)
    elif args.benchmark == "webdetective":
        examples = DS.load_webdetective(
            cfg["retrieval"].get("webdetective_jsonl", "mask_wiki_200.jsonl"), n=n)
    elif args.benchmark == "musique":
        examples = DS.load_musique(split="validation", n=n)
    elif args.benchmark == "musique_hard":
        examples = DS.load_musique_hard(split="validation", n=n, min_hops=4)
    elif args.benchmark == "musique3":
        examples = DS.load_musique3(split="validation", n=n)
    elif args.benchmark == "musique2":
        examples = DS.load_musique2(split="validation", n=n)
    elif args.benchmark == "bamboogle":
        examples = DS.load_bamboogle(n=n)
    elif args.benchmark == "triviaqa":
        examples = DS.load_triviaqa(split="validation", n=n)
    elif args.benchmark == "strategyqa":
        examples = DS.load_strategyqa(split="validation", n=n)

    print(f"[{args.benchmark}] loaded {len(examples)} examples")

    # For contextless benchmarks (TriviaQA, StrategyQA), attach a shared
    # Wikipedia corpus (from HotpotQA) so BM25 retrieval and FALCON's noise
    # injector both have passages to work with. wikipedia_api is unreachable.
    has_ctx = any(ex.context_passages for ex in examples)
    if not has_ctx and args.benchmark in ("triviaqa", "strategyqa"):
        print(f"[retriever] {args.benchmark} has no context; loading shared Wikipedia corpus from HotpotQA...")
        hp = DS.load_hotpotqa(split="validation", n=2000)
        shared = DS.all_context_passages(hp)
        for ex in examples:
            ex.context_passages = shared
        print(f"[retriever] attached {len(shared)} shared Wikipedia passages")

    retriever = build_retriever(cfg, examples, benchmark=args.benchmark)
    print(f"[retriever] mode={retriever.mode}")
    # Cache the full corpus passage list for FALCON's noise injector.
    run_method._corpus = DS.all_context_passages(examples)

    # Optional calibration for EARA
    if args.method == "eara" and args.calibrate:
        dev = examples[:cfg["experiment"]["n_dev"]]
        tau_c, tau_d, tau_p = calibrate_thresholds(llm, retriever, dev,
                                                   sc_k=cfg["experiment"]["selfconsistency_k"])
        cfg["eara"]["tau_c"], cfg["eara"]["tau_delta"], cfg["eara"]["tau_p"] = tau_c, tau_d, tau_p
        print(f"[calibrate] tau_c={tau_c} tau_delta={tau_d} tau_p={tau_p}")

    # Optional calibration for CASE
    if args.method == "case" and args.calibrate:
        dev = examples[:min(50, cfg["experiment"]["n_dev"])]
        theta, alpha = calibrate_case(llm, retriever, dev,
                                      budget=cfg.get("case", {}).get("budget", 24))
        cfg.setdefault("case", {})["theta"] = list(theta)
        cfg["case"]["alpha"] = list(alpha)
        print(f"[calibrate] theta={theta} alpha={alpha}")

    # Optional calibration for FALCON
    if args.method == "falcon" and args.calibrate:
        dev = examples[:min(30, cfg["experiment"]["n_dev"])]
        theta_drop, K = calibrate_falcon(llm, retriever, dev,
                                         K=cfg.get("falcon", {}).get("K", 2))
        cfg.setdefault("falcon", {})["theta_drop"] = theta_drop
        cfg["falcon"]["K"] = K
        print(f"[calibrate] theta_drop={theta_drop} K={K}")

    # Run
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.benchmark}_{args.method}.jsonl"

    if args.workers > 1:
        results = _run_parallel(args, cfg, examples, out_path)
    else:
        results = _run_serial(args, cfg, llm, retriever, examples, out_path)

    # Summary
    summary = evaluate(results)
    summary["model"] = args.model
    summary["benchmark"] = args.benchmark
    summary["method"] = args.method
    summary["llm_cost"] = llm.cost_summary() if args.workers == 1 else {"note": "parallel run, per-worker cost not aggregated"}
    summary_path = out_dir / f"{args.model}_{args.benchmark}_{args.method}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _run_serial(args, cfg, llm, retriever, examples, out_path):
    """Original serial loop (workers=1)."""
    results = []
    with open(out_path, "w") as fout:
        for i, ex in enumerate(examples):
            try:
                r = run_method(args.method, llm, retriever, ex, cfg)
            except Exception as e:
                print(f"[{i}] error: {e}")
                r = {"answer": "", "abstained": True, "n_retrievals": 0, "error": str(e)}
            rec = {
                "qid": ex.qid, "question": ex.question, "gold": ex.answer,
                "pred": r.get("answer", ""), "abstained": r.get("abstained", False),
                "unanswerable": ex.unanswerable,
                "n_retrievals": r.get("n_retrievals", 0),
                "correct": is_correct(r.get("answer", ""), ex.answer),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(examples)}] acc-so-far="
                      f"{sum(x['correct'] and not x['abstained'] for x in results)/len(results):.3f}")
    return results


# ---- Parallel execution (workers > 1) ---------------------------------------
# Each worker process builds its own LLM (one AppId from the model's pool) and
# its own retriever (BM25 index rebuilt from the examples' context — fast).

_WORKER_CFG = None       # set by pool initializer
_WORKER_LLM = None
_WORKER_RETRIEVER = None
_WORKER_METHOD = None


def _worker_init(cfg, model_key, method, context_passages, retriever_mode, retrieval_top_k):
    """Initialize per-worker LLM (one AppId) + retriever."""
    global _WORKER_CFG, _WORKER_LLM, _WORKER_RETRIEVER, _WORKER_METHOD
    import os
    from core.llm import LLM as _LLM
    from core.retriever import build_bm25_from_dataset, Retriever as _Retriever
    _WORKER_CFG = cfg
    _WORKER_METHOD = method
    model_cfg = dict(cfg["models"][model_key])
    # Each worker binds ONE AppId from the model's pool, selected by PID so workers
    # spread across AppIds without explicit rank passing.
    appids = model_cfg["api_key"]
    if isinstance(appids, list) and len(appids) > 1:
        idx = os.getpid() % len(appids)
        model_cfg["api_key"] = [appids[idx]]
    _WORKER_LLM = _LLM(model_cfg)
    if retriever_mode == "bm25_local":
        _WORKER_RETRIEVER = build_bm25_from_dataset(context_passages)
    else:
        _WORKER_RETRIEVER = _Retriever(retriever_mode, {"top_k": retrieval_top_k})


def _worker_run(ex):
    """Process one example in a worker process."""
    try:
        r = run_method(_WORKER_METHOD, _WORKER_LLM, _WORKER_RETRIEVER, ex, _WORKER_CFG)
    except Exception as e:
        r = {"answer": "", "abstained": True, "n_retrievals": 0, "error": str(e)}
    return {
        "qid": ex.qid, "question": ex.question, "gold": ex.answer,
        "pred": r.get("answer", ""), "abstained": r.get("abstained", False),
        "unanswerable": ex.unanswerable,
        "n_retrievals": r.get("n_retrievals", 0),
        "correct": is_correct(r.get("answer", ""), ex.answer),
    }


def _run_parallel(args, cfg, examples, out_path):
    """Run examples across args.workers processes."""
    from multiprocessing import get_context
    # Flatten context passages for BM25 index (rebuilt per worker).
    context_passages = []
    for ex in examples:
        context_passages.extend(ex.context_passages)
    retriever_mode = cfg["retrieval"]["mode"]
    retrieval_top_k = cfg["retrieval"]["top_k"]

    results = []
    ctx = get_context("spawn")
    pool_args = (cfg, args.model, args.method, context_passages,
                 retriever_mode, retrieval_top_k)
    with open(out_path, "w") as fout:
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=pool_args) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker_run, examples,
                                                        chunksize=1)):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                results.append(rec)
                if (i + 1) % 10 == 0:
                    done = len(results)
                    acc = sum(x['correct'] and not x['abstained'] for x in results) / done
                    print(f"[{done}/{len(examples)}] acc-so-far={acc:.3f}")
    return results


if __name__ == "__main__":
    main()
