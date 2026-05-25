"""Faithful HippoRAG reproduction (Recommendation 3 / Weakness 3 from expert reviewer).

Per-question pipeline (matched against cell J flat-BM25+FCS):
  1. Build a fresh HippoRAG store per question (unique save_dir).
  2. Index the haystack passages: each LongMemEval turn is a passage.
     This triggers HippoRAG's full pipeline (entity extraction via
     gpt-4o-mini, OpenIE triple extraction, bipartite passage-entity
     graph construction, dense embedding for each passage and entity).
  3. Query HippoRAG with the question text -> top-5 passages via
     Personalized PageRank from query-entity seeds.
  4. Pass retrieved passages to Qwen2.5-14B-Instruct under the same
     FCS prompt template as cell J.
  5. Score against gold with token-F1.

Compare against:
  cell J flat-BM25+FCS:   F1 = 0.447 (n=500)
  Mem0 paired (J subset): Δ = +0.021, p=0.249 n.s.

If full HippoRAG beats J above paired-bootstrap noise: faithful
reproduction supports their published gain at the matched protocol.
If not: the architecture-null finding strengthens to cover full HippoRAG
(not just our HippoRAG-lite proxy).
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
    parser.add_argument("--limit", type=int, default=200, help="number of questions (stratified)")
    parser.add_argument("--top_k", type=int, default=5, help="passages returned per query")
    parser.add_argument("--stratified", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", help="run on n=10 only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.limit = 10

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY missing"); sys.exit(1)

    from hipporag import HippoRAG
    import tempfile

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
    router = RuleBasedRouter("longmemeval")

    all_qs = list(iter_questions(load_questions()))
    if args.stratified:
        from collections import defaultdict
        import random as _random
        by_label = defaultdict(list)
        for q in all_qs:
            by_label[q.label].append(q)
        per_label = max(1, args.limit // len(by_label))
        rng = _random.Random(20260506)
        questions = []
        for label, qs in by_label.items():
            rng.shuffle(qs)
            questions.extend(qs[:per_label])
        rng.shuffle(questions)
        questions = questions[: args.limit]
    else:
        questions = all_qs[: args.limit]

    print(f"Running HippoRAG reproduction on {len(questions)} LME oracle questions, top_k={args.top_k}")
    print(f"Memory layer: hipporag (gpt-4o-mini for OpenIE + entity extraction; OpenAI embeddings)")
    print(f"Generator: Qwen2.5-14B-Instruct under FCS prompt (same as cell J)\n")

    base_save = Path(tempfile.gettempdir()) / "hipporag_repro"
    base_save.mkdir(parents=True, exist_ok=True)

    predictions = []
    f1_sum = 0.0
    t0 = time.time()

    for q_idx, q in enumerate(questions):
        # Fresh HippoRAG per question (each question has its own oracle haystack)
        save_dir = base_save / f"q{q_idx}_{uuid.uuid4().hex[:8]}"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            hr = HippoRAG(
                save_dir=str(save_dir),
                llm_model_name="gpt-4o-mini",
                embedding_model_name="text-embedding-3-small",
            )
        except Exception as e:
            print(f"  [{q_idx}] HippoRAG init failed: {str(e)[:120]}"); continue

        # Build passages — each turn is a passage
        passages = []
        for session in q.sessions:
            for turn in session:
                role = "user" if turn.role.lower() == "user" else "assistant"
                date = turn.session_date or ""
                passages.append(f"[{date}] {role}: {turn.text}".strip())

        try:
            hr.index(passages)
        except Exception as e:
            print(f"  [{q_idx}] hr.index failed: {str(e)[:120]}"); continue

        try:
            results = hr.retrieve(queries=[q.question], num_to_retrieve=args.top_k)
            if isinstance(results, tuple):
                results = results[0]
            sol = results[0]
            retrieved_passages = sol.docs if hasattr(sol, "docs") else []
            mem_texts = retrieved_passages[: args.top_k]
        except Exception as e:
            print(f"  [{q_idx}] hr.retrieve failed: {str(e)[:120]}")
            mem_texts = []

        retrieved_block = "\n".join(f"- {p}" for p in mem_texts) if mem_texts else "(no passages retrieved)"

        # Format with FCS template
        route = router.route(q.question)
        try:
            template = load_template("longmemeval-fcs", route.category)
        except Exception:
            template = load_template("longmemeval", "single-session")
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

        predictions.append({
            "question_id": q.question_id,
            "question_type": q.question_type,
            "gold_label": q.label,
            "question": q.question,
            "gold_answer": q.answer,
            "prediction": pred,
            "n_passages_retrieved": len(mem_texts),
            "completion_chars": len(pred),
            "f1": f1,
        })

        if (q_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{q_idx+1}/{len(questions)}] {elapsed:.0f}s  running F1={f1_sum/(q_idx+1):.4f}")

    n = len(predictions)
    aggregate_f1 = sum(p["f1"] for p in predictions) / max(n, 1)
    avg_chars = sum(p["completion_chars"] for p in predictions) / max(n, 1)

    from collections import defaultdict
    by_label = defaultdict(list)
    for p in predictions:
        by_label[p["gold_label"]].append(p["f1"])
    by_label_summary = {k: {"n": len(v), "f1": sum(v)/len(v)} for k, v in by_label.items()}

    summary = {
        "system": "hipporag-faithful-repro",
        "memory_layer": "hipporag (gpt-4o-mini OpenIE + text-embedding-3-small)",
        "generator": "Qwen/Qwen2.5-14B-Instruct, FCS prompt (matches cell J)",
        "n_questions": n,
        "top_k": args.top_k,
        "f1": aggregate_f1,
        "avg_completion_chars": avg_chars,
        "total_wall_time_s": time.time() - t0,
        "by_label": by_label_summary,
    }
    print(f"\n=== HippoRAG reproduction summary ===")
    print(f"  n={n}  F1={aggregate_f1:.4f}  avg_chars={avg_chars:.1f}")
    print(f"  vs cell J flat-BM25+FCS: F1=0.447 (n=500)")
    print(f"  vs Mem0 paired (n=200):  F1=0.483")
    print(f"  by_label: {by_label_summary}")

    out_path = Path(args.out) if args.out else (
        ROOT / "experiments/crag-9-faithful-repro" / f"results_hipporag_lme_oracle_n{n}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "predictions": predictions}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
