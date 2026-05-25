"""LLM-as-judge cell on a 200-Q LME oracle subset.

Compares two cells via GPT-4o-mini as judge:
  - flat × tight-per-cat (cell E from crag-6/6a, F1 = 0.365 token)
  - flat × tight-fcs     (cell J from crag-7, F1 = 0.447 token)

Loads OpenAI key from the project root .env. Asks the judge whether
the prediction is semantically correct given the gold answer. Returns
binary correct/incorrect per question. Reports judge accuracy alongside
token-F1 for each cell.

The point: if FCS-only's +0.082 token-F1 advantage is mostly format-driven,
a fairer LLM judge should narrow the gap (or invert it). If the gap
SURVIVES under LLM-judge, FCS-only is genuinely better — strengthening
the 'task hints are harmful' claim.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a gold (reference) answer, and a system's predicted answer. Your job is to decide whether the prediction conveys the same factual answer as the gold, regardless of length or phrasing.

Question: {question}

Gold answer: {gold}

Predicted answer: {pred}

Reply with exactly one token: CORRECT or INCORRECT.
- CORRECT: the prediction conveys the same factual answer as the gold (semantically equivalent; minor format differences are fine).
- INCORRECT: the prediction is factually wrong, contradictory, missing, or refuses when the gold has an answer.

Reply (CORRECT or INCORRECT only):"""


def judge_one(client, model: str, question: str, gold: str, pred: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return "CORRECT", verdict
        if "INCORRECT" in verdict:
            return "INCORRECT", verdict
        return "AMBIGUOUS", verdict
    except Exception as e:
        return "ERROR", str(e)[:200]


def judge_cell(cell_path: Path, *, indices: list[int], model: str, client) -> dict:
    with cell_path.open() as f:
        d = json.load(f)
    preds = d["predictions"]
    out = []
    correct = 0
    incorrect = 0
    other = 0
    for i, idx in enumerate(indices):
        if idx >= len(preds):
            continue
        p = preds[idx]
        verdict, raw = judge_one(client, model, p["question"], p.get("gold_answer", ""), p.get("prediction", ""))
        out.append({
            "idx": idx,
            "question": p["question"][:200],
            "gold": str(p.get("gold_answer", ""))[:200],
            "pred": str(p.get("prediction", ""))[:200],
            "verdict": verdict,
            "f1_token": float(p.get("f1", 0.0)) if "f1" in p else None,
        })
        if verdict == "CORRECT":
            correct += 1
        elif verdict == "INCORRECT":
            incorrect += 1
        else:
            other += 1
        if (i + 1) % 25 == 0:
            print(f"  judged {i+1}/{len(indices)}  CORRECT={correct} INCORRECT={incorrect} other={other}", flush=True)
        time.sleep(0.05)  # gentle rate-limit
    return {
        "n_judged": len(out),
        "correct": correct,
        "incorrect": incorrect,
        "other": other,
        "judge_accuracy": correct / max(1, correct + incorrect),
        "verdicts": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--cell-e", default=str(ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"))
    parser.add_argument("--cell-j", default=str(ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"))
    parser.add_argument("--output", default=str(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"))
    args = parser.parse_args()

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not in environment", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    # Load both cells; ensure same questions are paired by index
    with open(args.cell_e) as f:
        e = json.load(f)
    with open(args.cell_j) as f:
        j = json.load(f)
    n_e, n_j = len(e["predictions"]), len(j["predictions"])
    n = min(n_e, n_j)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(n), min(args.n, n)))

    print(f"Cell E: {args.cell_e} (n={n_e})")
    print(f"Cell J: {args.cell_j} (n={n_j})")
    print(f"Judging {len(indices)} matched questions with {args.model}")
    print()

    print("=== Judging Cell E (flat + tight-per-cat) ===")
    e_result = judge_cell(Path(args.cell_e), indices=indices, model=args.model, client=client)
    print(f"  E judge accuracy: {e_result['judge_accuracy']:.4f}  ({e_result['correct']}/{e_result['correct']+e_result['incorrect']})")
    print()

    print("=== Judging Cell J (flat + tight-fcs) ===")
    j_result = judge_cell(Path(args.cell_j), indices=indices, model=args.model, client=client)
    print(f"  J judge accuracy: {j_result['judge_accuracy']:.4f}  ({j_result['correct']}/{j_result['correct']+j_result['incorrect']})")
    print()

    delta = j_result["judge_accuracy"] - e_result["judge_accuracy"]
    print(f"DELTA (J - E) judge_accuracy: {delta:+.4f}")
    print(f"DELTA (J - E) token_F1:       {0.4469 - 0.3652:+.4f}  (LME aggregate from prior runs)")

    out = {
        "model": args.model,
        "n_judged": len(indices),
        "indices_seed": args.seed,
        "cell_e_path": str(args.cell_e),
        "cell_j_path": str(args.cell_j),
        "cell_e": e_result,
        "cell_j": j_result,
        "delta_judge_accuracy": delta,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
