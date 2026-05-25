"""Faithful A-MEM reproduction with evolution (W3 from expert reviewer).

Uses the upstream agentic_memory package from github.com/agiresearch/A-MEM
(installed via `pip install .` from the cloned repo). This is the actual
A-MEM library: LLM-driven note construction at write time, ChromaDB
embeddings, and a memory-evolution rule that consolidates / contradicts /
supersedes notes as the store grows.

Pipeline:
  1. Per question, instantiate a fresh AgenticMemorySystem (gpt-4o-mini
     for the LLM layer, all-MiniLM-L6-v2 for the embedder).
  2. Set evo_threshold so memory evolution triggers within the haystack
     (default upstream is 100; LongMemEval oracle haystacks are ~22 turns,
     so we use evo_threshold=10 to ensure at least one evolution pass).
  3. add_note() each haystack turn with its date.
  4. search_agentic(question, k=5) -> top-5 evolved memory notes.
  5. Hand to Qwen2.5-14B-Instruct under the same FCS prompt template as cell J.
  6. Score with token-F1.

Compare paired against:
  cell J flat-BM25+FCS:   F1 = 0.447 (n=500)
  Mem0 paired (n=200):    F1 = 0.483 (Δ=+0.021, p=0.249 n.s.)
"""
from __future__ import annotations
import argparse, json, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.longmemeval import iter_questions, load_questions
from src.recipe.generator import get_generator
from src.recipe.prompts import load_template
from src.recipe.router import RuleBasedRouter
from src.eval import f1_single


def load_env(env_path: Path) -> None:
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--evo_threshold", type=int, default=10,
                        help="trigger consolidate_memories every N notes")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.smoke: args.limit = 10

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY missing"); sys.exit(1)

    from agentic_memory.memory_system import AgenticMemorySystem

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
    router = RuleBasedRouter("longmemeval")

    # Stratified sample (same seed as Mem0 / HippoRAG runs for direct comparability)
    all_qs = list(iter_questions(load_questions()))
    from collections import defaultdict
    import random as _random
    by_label = defaultdict(list)
    for q in all_qs: by_label[q.label].append(q)
    per_label = max(1, args.limit // len(by_label))
    rng = _random.Random(20260506)
    questions = []
    for label, qs in by_label.items():
        rng.shuffle(qs); questions.extend(qs[:per_label])
    rng.shuffle(questions); questions = questions[: args.limit]

    print(f"Running A-MEM (with evolution, evo_threshold={args.evo_threshold}) on {len(questions)} LME oracle, top_k={args.top_k}")
    print(f"Memory layer: agentic_memory (gpt-4o-mini, all-MiniLM-L6-v2 embedder, ChromaDB)")
    print(f"Generator: Qwen2.5-14B-Instruct under FCS prompt (matches cell J)\n")

    predictions = []
    f1_sum = 0.0
    t0 = time.time()

    for q_idx, q in enumerate(questions):
        try:
            mem = AgenticMemorySystem(
                model_name="all-MiniLM-L6-v2",
                llm_backend="openai",
                llm_model="gpt-4o-mini",
                evo_threshold=args.evo_threshold,
            )
        except Exception as e:
            print(f"  [{q_idx}] AgenticMemorySystem init failed: {str(e)[:120]}"); continue

        # Ingest haystack — each turn becomes a note
        ingested = 0
        for session in q.sessions:
            for turn in session:
                role = "user" if turn.role.lower() == "user" else "assistant"
                date = turn.session_date or ""
                content = f"[{date}] {role}: {turn.text}".strip()
                try:
                    mem.add_note(content=content, time=date)
                    ingested += 1
                except Exception as e:
                    print(f"  [{q_idx}] add_note failed (skipped): {str(e)[:80]}")
                    continue

        # Query
        try:
            results = mem.search_agentic(q.question, k=args.top_k)
            mem_texts = [r.get("content","") for r in (results or [])]
        except Exception as e:
            print(f"  [{q_idx}] search_agentic failed: {str(e)[:120]}")
            mem_texts = []

        retrieved_block = "\n".join(f"- {p}" for p in mem_texts) if mem_texts else "(no memories retrieved)"

        # FCS prompt (matches cell J)
        route = router.route(q.question)
        try: template = load_template("longmemeval-fcs", route.category)
        except Exception: template = load_template("longmemeval", "single-session")
        kwargs = {"question": q.question, "retrieved_turns": retrieved_block,
                  "retrieved_turns_with_timestamps": retrieved_block}
        try: prompt = template.format(**kwargs)
        except KeyError:
            for k, v in kwargs.items(): template = template.replace("{"+k+"}", str(v))
            prompt = template

        call = generator.generate(prompt, max_tokens=96)
        pred = call.completion.strip()
        f1 = f1_single(pred, q.answer)
        f1_sum += f1

        predictions.append({
            "question_id": q.question_id, "question_type": q.question_type,
            "gold_label": q.label, "question": q.question, "gold_answer": q.answer,
            "prediction": pred, "n_memories_retrieved": len(mem_texts),
            "n_notes_ingested": ingested, "completion_chars": len(pred), "f1": f1,
        })

        if (q_idx + 1) % 10 == 0:
            print(f"  [{q_idx+1}/{len(questions)}] {time.time()-t0:.0f}s running F1={f1_sum/(q_idx+1):.4f}")

    n = len(predictions)
    aggregate_f1 = sum(p["f1"] for p in predictions) / max(n, 1)
    avg_chars = sum(p["completion_chars"] for p in predictions) / max(n, 1)
    by_label_summary = {}
    from collections import defaultdict as dd
    per_label = dd(list)
    for p in predictions: per_label[p["gold_label"]].append(p["f1"])
    for k, v in per_label.items():
        by_label_summary[k] = {"n": len(v), "f1": sum(v)/len(v)}

    summary = {
        "system": "amem-faithful-with-evolution",
        "memory_layer": "agentic_memory (upstream A-MEM repo, gpt-4o-mini + all-MiniLM-L6-v2)",
        "generator": "Qwen/Qwen2.5-14B-Instruct, FCS prompt (matches cell J)",
        "evo_threshold": args.evo_threshold,
        "n_questions": n, "top_k": args.top_k,
        "f1": aggregate_f1, "avg_completion_chars": avg_chars,
        "total_wall_time_s": time.time() - t0,
        "by_label": by_label_summary,
    }
    print(f"\n=== A-MEM faithful (with evolution) summary ===")
    print(f"  n={n}  F1={aggregate_f1:.4f}  avg_chars={avg_chars:.1f}")
    print(f"  vs cell J flat-BM25+FCS: F1=0.447 (n=500)")
    print(f"  vs Mem0 paired (n=200):  F1=0.483")
    print(f"  by_label: {by_label_summary}")

    out_path = Path(args.out) if args.out else (
        ROOT / "experiments/crag-9-faithful-repro" / f"results_amem_lme_oracle_n{n}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "predictions": predictions}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
