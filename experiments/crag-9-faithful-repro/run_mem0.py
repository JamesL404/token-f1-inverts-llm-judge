"""Faithful Mem0 reproduction (Item #8 from final to-do).

Per-question pipeline (matched to our flat-RAG cells):
  1. Build a fresh Mem0 store per question (unique user_id)
  2. Ingest the LongMemEval oracle haystack turn-by-turn as a conversation
     -- this triggers Mem0's LLM-driven memory extraction (gpt-4o-mini)
  3. Query Mem0 with the question text -> top-K extracted memories
  4. Pass retrieved memories to Qwen2.5-14B-Instruct with our FCS prompt
     (same prompt template as cell J for fairness)
  5. Score against gold with token-F1

Compare against:
  cell J (flat BM25 + tight-fcs): F1 = 0.447
  cell E (flat BM25 + tight-per-cat): F1 = 0.365

If Mem0 beats J above bootstrap noise: faithful reproduction supports
their architectural claim under our matched protocol.
If Mem0 ties or loses to J: faithful reproduction does NOT clear our bar.
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
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="total questions across labels")
    parser.add_argument("--top_k", type=int, default=5, help="memories returned per query")
    parser.add_argument("--stratified", action="store_true", help="sample uniformly across labels")
    parser.add_argument("--chunk", type=int, default=10, help="ingest in chunks of N turns to avoid embedding length limits")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY missing"); sys.exit(1)

    # Mem0 maintains a shared telemetry vector store, so we use ONE Memory instance
    # and isolate questions via user_id namespacing.
    os.environ["MEM0_TELEMETRY"] = "False"
    from mem0 import Memory
    config = {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini",
                                                 "temperature": 0.0, "max_tokens": 1000}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    }
    shared_memory = Memory.from_config(config)

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
    router = RuleBasedRouter("longmemeval")

    all_qs = list(iter_questions(load_questions()))
    if args.stratified:
        import random as _random
        from collections import defaultdict
        by_label = defaultdict(list)
        for q in all_qs:
            by_label[q.label].append(q)
        per_label = max(1, args.limit // len(by_label))
        rng = _random.Random(42)
        questions = []
        for label, qs in by_label.items():
            rng.shuffle(qs)
            questions.extend(qs[:per_label])
        rng.shuffle(questions)
        questions = questions[: args.limit]
    else:
        questions = all_qs[: args.limit]
    print(f"Running Mem0 reproduction on {len(questions)} LME oracle questions, top_k={args.top_k}")
    print(f"Memory layer: gpt-4o-mini extraction + text-embedding-3-small")
    print(f"Generator: Qwen2.5-14B-Instruct (FCS prompt, same as cell J)\n")

    predictions = []
    n_correct_token_f1 = 0
    f1_sum = 0.0
    t0 = time.time()

    for q_idx, q in enumerate(questions):
        m = shared_memory
        user_id = f"lme_oracle_q{q_idx}_{uuid.uuid4().hex[:8]}"

        # Ingest haystack: each session's turns as a conversation
        ingest_messages = []
        for session in q.sessions:           # session is list[LMETurn]
            for turn in session:
                role = "user" if turn.role.lower() == "user" else "assistant"
                ingest_messages.append({"role": role, "content": turn.text})

        # Ingest in chunks to keep each embedding call under the 8192-token limit
        ingest_failed = False
        for chunk_start in range(0, len(ingest_messages), args.chunk):
            batch = ingest_messages[chunk_start:chunk_start + args.chunk]
            try:
                m.add(batch, user_id=user_id)
            except Exception as e:
                print(f"  [{q_idx}] mem0.add chunk {chunk_start} failed: {str(e)[:120]}")
                ingest_failed = True
                break
        if ingest_failed:
            # Fall through with whatever was ingested before failure; still attempt query.
            pass

        # Query Mem0 for top-K memories
        try:
            sr = m.search(q.question, filters={"user_id": user_id}, limit=args.top_k)
            mems = sr.get("results", []) if isinstance(sr, dict) else sr
            mem_texts = [r.get("memory","") if isinstance(r, dict) else str(r) for r in mems]
        except Exception as e:
            print(f"  [{q_idx}] mem0.search failed: {e}")
            mem_texts = []

        # Format prompt with retrieved memories. Use our FCS template per category.
        route = router.route(q.question)
        category = route.category
        try:
            template = load_template("longmemeval-fcs", category)
        except Exception:
            template = load_template("longmemeval", "single-session")

        # Compose retrieved-turns text-block from Mem0 memories
        if mem_texts:
            retrieved_block = "\n".join(f"- {t}" for t in mem_texts)
        else:
            retrieved_block = "(no relevant memories retrieved)"
        prompt_kwargs = {
            "question": q.question,
            "retrieved_turns": retrieved_block,
            "retrieved_turns_with_timestamps": retrieved_block,
        }
        try:
            prompt = template.format(**prompt_kwargs)
        except KeyError:
            for k, v in prompt_kwargs.items():
                template = template.replace("{" + k + "}", str(v))
            prompt = template

        call = generator.generate(prompt, max_tokens=96)
        pred = call.completion.strip()

        f1 = f1_single(pred, q.answer)
        f1_sum += f1
        if f1 >= 0.5:
            n_correct_token_f1 += 1

        predictions.append({
            "question_id": q.question_id,
            "question_type": q.question_type,
            "gold_label": q.label,
            "question": q.question,
            "gold_answer": q.answer,
            "prediction": pred,
            "n_memories_retrieved": len(mem_texts),
            "memories": mem_texts,
            "completion_chars": len(pred),
            "f1": f1,
        })

        if (q_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{q_idx+1}/{len(questions)}] {elapsed:.0f}s  running F1={f1_sum/(q_idx+1):.4f}")

    n = len(predictions)
    aggregate_f1 = sum(p["f1"] for p in predictions) / max(n, 1)
    avg_chars = sum(p["completion_chars"] for p in predictions) / max(n, 1)
    avg_mems = sum(p["n_memories_retrieved"] for p in predictions) / max(n, 1)

    by_label = {}
    for p in predictions:
        label = p["gold_label"]
        by_label.setdefault(label, []).append(p["f1"])
    by_label_summary = {k: {"n": len(v), "f1": sum(v)/len(v)} for k, v in by_label.items()}

    summary = {
        "system": "mem0-faithful-repro",
        "memory_layer": "mem0ai==2.0.1, gpt-4o-mini extraction, text-embedding-3-small",
        "generator": "Qwen/Qwen2.5-14B-Instruct, FCS prompt (matches cell J)",
        "n_questions": n,
        "top_k": args.top_k,
        "f1": aggregate_f1,
        "avg_completion_chars": avg_chars,
        "avg_memories_retrieved": avg_mems,
        "total_wall_time_s": time.time() - t0,
        "by_label": by_label_summary,
    }
    print(f"\n=== Mem0 reproduction summary ===")
    print(f"  n={n}  F1={aggregate_f1:.4f}  avg_chars={avg_chars:.1f}  avg_mems_retrieved={avg_mems:.1f}")
    print(f"  vs cell J (flat × tight-fcs): F1=0.447 (n=500)")
    print(f"  vs cell E (flat × tight-per-cat): F1=0.365 (n=500)")
    print(f"  by_label: {by_label_summary}")

    out_path = Path(args.out) if args.out else (
        ROOT / "experiments/crag-9-faithful-repro" / f"results_mem0_lme_oracle_n{n}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "predictions": predictions}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
