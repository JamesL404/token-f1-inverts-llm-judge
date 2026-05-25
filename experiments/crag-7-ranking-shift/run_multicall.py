"""Run the multi-call ReAct-lite ablation on LME oracle 14B.

Tests whether the architecture-zero finding survives in a 2-call regime
(planner decides ANSWER-or-SEARCH; if SEARCH, re-retrieve with refined query).

Compares against cell E (flat × tight-per-cat single-call): if multi-call
is no better above bootstrap noise, the architecture-zero claim
generalizes from single-call to multi-call.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import f1_single
from src.longmemeval import iter_questions, load_questions
from src.recipe.generator import get_generator
from src.recipe.multicall import MultiCallReAct
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-7-ranking-shift"


def main() -> None:
    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
    rc = MultiCallReAct(benchmark="longmemeval",
                        router=RuleBasedRouter("longmemeval"),
                        generator=generator, top_k=5)

    questions = load_questions()
    preds = []
    n_search = 0
    f1_per_q = []
    f1_by_label = defaultdict(list)
    label_counter = Counter()
    total_calls = 0; total_p = 0; total_c = 0; t_wall = 0.0
    for i, q in enumerate(iter_questions(questions)):
        result = rc.answer(q.question, q)
        f1 = f1_single(result.prediction, str(q.answer))
        f1_per_q.append(f1)
        f1_by_label[q.label].append(f1)
        label_counter[q.label] += 1
        total_calls += result.n_llm_calls
        total_p += result.total_prompt_chars
        total_c += result.total_completion_chars
        t_wall += result.wall_time_s
        if result.used_search: n_search += 1
        preds.append({
            "sample_id": getattr(q, "sample_id", str(i)),
            "question_type": q.question_type,
            "gold_label": q.label,
            "question": q.question,
            "gold_answer": str(q.answer),
            "prediction": result.prediction,
            "routed_category": result.routed_category,
            "n_retrieved_turns": result.n_retrieved_turns,
            "n_llm_calls": result.n_llm_calls,
            "used_search": result.used_search,
            "refined_query": result.refined_query,
            "f1": f1,
            "wall_time_s": result.wall_time_s,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/500  (search-used = {n_search})", flush=True)

    summary = {
        "system": "crag-7M-multicall-react-tight-per-cat",
        "benchmark": "longmemeval", "split": "oracle",
        "generator_model": "Qwen/Qwen2.5-14B-Instruct",
        "n_questions": len(preds),
        "n_search_used": n_search,
        "f1": sum(f1_per_q) / len(f1_per_q) if f1_per_q else 0.0,
        "by_label": {k: {"n": label_counter[k], "f1": sum(v) / len(v) if v else 0.0}
                     for k, v in f1_by_label.items()},
        "n_llm_calls": total_calls,
        "calls_per_question": total_calls / len(preds) if preds else 0.0,
        "total_prompt_chars": total_p,
        "total_completion_chars": total_c,
        "total_wall_time_s": t_wall,
    }
    out_path = OUT / "results_7M_lme_multicall_tight.json"
    out_path.write_text(json.dumps({"summary": summary, "predictions": preds}, indent=2))
    print(f"\n7M multicall+tight-per-cat: F1={summary['f1']:.4f}  search_used={n_search}/{len(preds)}  calls/Q={summary['calls_per_question']:.2f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
