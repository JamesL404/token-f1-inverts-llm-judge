"""Native-prompt sanity cells for the three faithful reproductions.

Holds the generator (Qwen2.5-14B) and retrieval/memory layer fixed at the
faithful-repro versions, but swaps the answer prompt for a native loose
prompt with a 256-token budget instead of cell J's FCS prompt + 96-token
cap. Run on the same n=200 stratified subset to enable paired comparison.

Output: results_native_prompt_summary.json with per-system native-prompt
F1, delta vs cell-J-FCS version, and delta vs cell J flat-BM25+FCS reference.
"""
from __future__ import annotations
import argparse, json, os, sys, time, uuid, tempfile
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.longmemeval import iter_questions, load_questions
from src.recipe.generator import get_generator
from src.recipe.router import RuleBasedRouter
from src.eval import f1_single


NATIVE_PROMPT = """You are answering a question about a long conversation. Use the retrieved excerpts/memories to answer the question accurately. You may answer in a complete sentence if helpful.

Memories / excerpts:
{retrieved_block}

Question: {question}

Answer:"""


def load_env(env_path: Path) -> None:
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


def stratified_sample(qs, limit, seed):
    import random
    by_label = defaultdict(list)
    for q in qs: by_label[q.label].append(q)
    per = max(1, limit // len(by_label))
    rng = random.Random(seed)
    out = []
    for lab, lst in by_label.items():
        rng.shuffle(lst); out.extend(lst[:per])
    rng.shuffle(out); return out[:limit]


def run_mem0_native(questions, generator):
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig
    config = {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    }
    preds = []
    t0 = time.time()
    for k, q in enumerate(questions):
        try:
            mem = Memory.from_config(config_dict=config)
            for session in q.sessions:
                msgs = []
                for turn in session:
                    role = "user" if turn.role.lower() == "user" else "assistant"
                    date = turn.session_date or ""
                    msgs.append({"role": role, "content": f"[{date}] {turn.text}"})
                if msgs:
                    mem.add(msgs, user_id=f"q{k}")
            results = mem.search(query=q.question, user_id=f"q{k}", limit=5)
            mems = [r.get("memory", "") for r in (results.get("results", []) if isinstance(results, dict) else results)]
        except Exception as e:
            print(f"  Mem0 [{k}] error: {str(e)[:80]}"); mems = []
        retrieved = "\n".join(f"- {m}" for m in mems) if mems else "(no memories retrieved)"
        prompt = NATIVE_PROMPT.format(retrieved_block=retrieved, question=q.question)
        call = generator.generate(prompt, max_tokens=256)
        pred = call.completion.strip()
        f1 = f1_single(pred, q.answer)
        preds.append({"question_id": q.question_id, "label": q.label, "f1": f1, "chars": len(pred)})
        if (k + 1) % 20 == 0:
            print(f"  Mem0 native [{k+1}/{len(questions)}] {time.time()-t0:.0f}s F1={sum(p['f1'] for p in preds)/len(preds):.4f}")
    return preds


def run_hipporag_native(questions, generator):
    from hipporag import HippoRAG
    base = Path(tempfile.gettempdir()) / "hipporag_native"
    base.mkdir(parents=True, exist_ok=True)
    preds = []
    t0 = time.time()
    for k, q in enumerate(questions):
        save_dir = base / f"q{k}_{uuid.uuid4().hex[:8]}"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            hr = HippoRAG(save_dir=str(save_dir), llm_model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-small")
        except Exception as e:
            print(f"  HippoRAG [{k}] init failed: {str(e)[:80]}"); preds.append({"question_id": q.question_id, "label": q.label, "f1": 0.0, "chars": 0}); continue
        passages = []
        for session in q.sessions:
            for turn in session:
                role = "user" if turn.role.lower() == "user" else "assistant"
                date = turn.session_date or ""
                passages.append(f"[{date}] {role}: {turn.text}".strip())
        try: hr.index(passages)
        except Exception as e:
            print(f"  HippoRAG [{k}] index failed: {str(e)[:80]}"); preds.append({"question_id": q.question_id, "label": q.label, "f1": 0.0, "chars": 0}); continue
        try:
            r = hr.retrieve(queries=[q.question], num_to_retrieve=5)
            if isinstance(r, tuple): r = r[0]
            sol = r[0]
            mems = (sol.docs if hasattr(sol, "docs") else [])[:5]
        except Exception as e:
            print(f"  HippoRAG [{k}] retrieve failed: {str(e)[:80]}"); mems = []
        retrieved = "\n".join(f"- {m}" for m in mems) if mems else "(no passages retrieved)"
        prompt = NATIVE_PROMPT.format(retrieved_block=retrieved, question=q.question)
        call = generator.generate(prompt, max_tokens=256)
        pred = call.completion.strip()
        f1 = f1_single(pred, q.answer)
        preds.append({"question_id": q.question_id, "label": q.label, "f1": f1, "chars": len(pred)})
        if (k + 1) % 20 == 0:
            print(f"  HippoRAG native [{k+1}/{len(questions)}] {time.time()-t0:.0f}s F1={sum(p['f1'] for p in preds)/len(preds):.4f}")
    return preds


def run_amem_native(questions, generator):
    from agentic_memory.memory_system import AgenticMemorySystem
    preds = []
    t0 = time.time()
    for k, q in enumerate(questions):
        try:
            mem = AgenticMemorySystem(model_name="all-MiniLM-L6-v2", llm_backend="openai", llm_model="gpt-4o-mini", evo_threshold=10)
        except Exception as e:
            print(f"  A-MEM [{k}] init failed: {str(e)[:80]}"); preds.append({"question_id": q.question_id, "label": q.label, "f1": 0.0, "chars": 0}); continue
        for session in q.sessions:
            for turn in session:
                role = "user" if turn.role.lower() == "user" else "assistant"
                date = turn.session_date or ""
                content = f"[{date}] {role}: {turn.text}".strip()
                try: mem.add_note(content=content, time=date)
                except Exception: pass
        try:
            results = mem.search_agentic(q.question, k=5)
            mems = [r.get("content", "") for r in (results or [])]
        except Exception as e:
            print(f"  A-MEM [{k}] search failed: {str(e)[:80]}"); mems = []
        retrieved = "\n".join(f"- {m}" for m in mems) if mems else "(no notes retrieved)"
        prompt = NATIVE_PROMPT.format(retrieved_block=retrieved, question=q.question)
        call = generator.generate(prompt, max_tokens=256)
        pred = call.completion.strip()
        f1 = f1_single(pred, q.answer)
        preds.append({"question_id": q.question_id, "label": q.label, "f1": f1, "chars": len(pred)})
        if (k + 1) % 20 == 0:
            print(f"  A-MEM native [{k+1}/{len(questions)}] {time.time()-t0:.0f}s F1={sum(p['f1'] for p in preds)/len(preds):.4f}")
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--systems", nargs="+", default=["mem0", "hipporag", "amem"])
    args = parser.parse_args()

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY missing"); sys.exit(1)

    qs = list(iter_questions(load_questions()))
    questions = stratified_sample(qs, args.limit, args.seed)
    print(f"Native-prompt sanity on n={len(questions)} (same stratified seed=20260506 as faithful-repro)")
    print(f"Generator: Qwen2.5-14B + loose prompt + 256-token budget (vs cell-J's FCS + 96-token)\n")

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    runners = {"mem0": run_mem0_native, "hipporag": run_hipporag_native, "amem": run_amem_native}
    out = {"n_questions": len(questions), "seed": args.seed, "by_system": {}}
    for system in args.systems:
        if system not in runners:
            print(f"WARN: unknown system {system}, skipping"); continue
        print(f"\n=== {system.upper()} ===")
        preds = runners[system](questions, generator)
        n = len(preds)
        agg = sum(p["f1"] for p in preds) / max(n, 1)
        avg_chars = sum(p["chars"] for p in preds) / max(n, 1)
        by_lab = defaultdict(list)
        for p in preds: by_lab[p["label"]].append(p["f1"])
        by_label_summary = {k: {"n": len(v), "f1": sum(v)/len(v)} for k, v in by_lab.items()}
        out["by_system"][system] = {
            "n": n, "f1": agg, "avg_chars": avg_chars,
            "by_label": by_label_summary, "predictions": preds,
        }
        print(f"  {system} native: F1={agg:.4f} avg_chars={avg_chars:.1f} (n={n})")

    out_path = ROOT / "experiments/crag-9-faithful-repro/results_native_prompt_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    print("\n=== Comparison vs faithful-repro (FCS prompt) version ===")
    refs = {"mem0": 0.483, "hipporag": 0.469, "amem": 0.394}
    for system in args.systems:
        if system in out["by_system"]:
            native = out["by_system"][system]["f1"]
            print(f"  {system:>10}: native (loose, 256-tok) F1={native:.4f}  vs  FCS (96-tok) F1={refs[system]:.4f}  ∆={native-refs[system]:+.4f}")


if __name__ == "__main__":
    main()
