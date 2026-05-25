"""Full LLM-judge matrix: judge every cell on a 200-Q paired LME / LoCoMo subset.

For LongMemEval (n=500/cell, paired indices, seed=42):
  9 main matrix cells + cell J (FCS-only) = 10 cells
For LoCoMo (n=1986/cell):
  9 main matrix cells = 9 cells
Total: 19 cells × 200 judgments = 3,800 judge calls. ~$1.50 on gpt-4o-mini.

Reuses the judge logic from llm_judge.py.
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

from experiments.__init__ import *  # noqa


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


JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a gold (reference) answer, and a system's predicted answer. Your job is to decide whether the prediction conveys the same factual answer as the gold, regardless of length or phrasing.

Question: {question}

Gold answer: {gold}

Predicted answer: {pred}

Reply with exactly one token: CORRECT or INCORRECT.
- CORRECT: the prediction conveys the same factual answer as the gold (semantically equivalent; minor format differences are fine).
- INCORRECT: the prediction is factually wrong, contradictory, missing, or refuses when the gold has an answer.

Reply (CORRECT or INCORRECT only):"""


def judge_one(client, model: str, question: str, gold: str, pred: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8, temperature=0.0,
        )
        v = response.choices[0].message.content.strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v:
            return "CORRECT"
        if "INCORRECT" in v:
            return "INCORRECT"
        return "AMBIGUOUS"
    except Exception:
        return "ERROR"


def judge_cell(cell_path: Path, indices: list[int], model: str, client) -> dict:
    with cell_path.open() as f:
        d = json.load(f)
    preds = d["predictions"]
    correct = 0; incorrect = 0; other = 0
    verdicts = []
    for i, idx in enumerate(indices):
        if idx >= len(preds):
            continue
        p = preds[idx]
        v = judge_one(client, model, p["question"], p.get("gold_answer", ""), p.get("prediction", ""))
        verdicts.append(v)
        if v == "CORRECT": correct += 1
        elif v == "INCORRECT": incorrect += 1
        else: other += 1
        time.sleep(0.03)
    return {
        "n_judged": len(verdicts),
        "correct": correct,
        "incorrect": incorrect,
        "other": other,
        "judge_accuracy": correct / max(1, correct + incorrect),
        "verdicts": verdicts,
    }


CELLS_LME = {
    "E_flat_tight_per_cat": "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
    "F_linked_tight_per_cat": "experiments/crag-6-linking-baseline/results_6b_lme_linked.json",
    "G_session_bank_tight_per_cat": "experiments/crag-7-ranking-shift/results_7G_lme_session_bank.json",
    "A_flat_tight_generic": "experiments/crag-7-ranking-shift/results_7A_lme_flat_tight_generic.json",
    "C_linked_tight_generic": "experiments/crag-7-ranking-shift/results_7C_lme_linked_tight_generic.json",
    "H_session_bank_tight_generic": "experiments/crag-7-ranking-shift/results_7H_lme_session_bank.json",
    "B_flat_loose": "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json",
    "D_linked_loose": "experiments/crag-7-ranking-shift/results_7D_lme_linked_loose.json",
    "I_session_bank_loose": "experiments/crag-7-ranking-shift/results_7I_lme_session_bank.json",
    "J_flat_tight_fcs": "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json",
}

CELLS_LOCOMO = {
    "LE_flat_tight_per_cat": "experiments/crag-1-headline/results_locomo_promptfix.json",
    "LF_linked_tight_per_cat": "experiments/crag-7-ranking-shift/results_7LF_locomo.json",
    "LG_session_bank_tight_per_cat": "experiments/crag-7-ranking-shift/results_7LG_locomo.json",
    "LA_flat_tight_generic": "experiments/crag-7-ranking-shift/results_7LA_locomo.json",
    "LC_linked_tight_generic": "experiments/crag-7-ranking-shift/results_7LC_locomo.json",
    "LH_session_bank_tight_generic": "experiments/crag-7-ranking-shift/results_7LH_locomo.json",
    "LB_flat_loose": "experiments/crag-7-ranking-shift/results_7LB_locomo.json",
    "LD_linked_loose": "experiments/crag-7-ranking-shift/results_7LD_locomo.json",
    "LI_session_bank_loose": "experiments/crag-7-ranking-shift/results_7LI_locomo.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--scope", choices=["lme", "locomo", "both"], default="both")
    ap.add_argument("--output", default=str(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json"))
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not in environment", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    rng = random.Random(args.seed)
    output = {"model": args.model, "n": args.n, "seed": args.seed, "cells": {}}

    cells = []
    if args.scope in ("lme", "both"):
        # LME has 500 questions per cell; pair indices once
        with open(ROOT / CELLS_LME["E_flat_tight_per_cat"]) as f:
            n_lme = len(json.load(f)["predictions"])
        lme_indices = sorted(rng.sample(range(n_lme), min(args.n, n_lme)))
        for label, path in CELLS_LME.items():
            cells.append(("lme", label, ROOT / path, lme_indices))
    if args.scope in ("locomo", "both"):
        # Re-seed for LoCoMo so the index choice is independent
        rng2 = random.Random(args.seed + 1)
        with open(ROOT / CELLS_LOCOMO["LE_flat_tight_per_cat"]) as f:
            n_locomo = len(json.load(f)["predictions"])
        locomo_indices = sorted(rng2.sample(range(n_locomo), min(args.n, n_locomo)))
        for label, path in CELLS_LOCOMO.items():
            cells.append(("locomo", label, ROOT / path, locomo_indices))

    print(f"Running {len(cells)} cells × n={args.n} judgments each = {len(cells)*args.n} total judge calls")
    print(f"Model: {args.model}")
    print()

    for benchmark, label, path, indices in cells:
        if not path.exists():
            print(f"  [skip] {label}: missing {path}")
            continue
        result = judge_cell(path, indices, args.model, client)
        output["cells"][f"{benchmark}.{label}"] = result
        print(f"  [{benchmark}] {label:36s}  {result['correct']:>3d}/{result['correct']+result['incorrect']:>3d} = {result['judge_accuracy']:.4f}  (other={result['other']})", flush=True)
        Path(args.output).write_text(json.dumps(output, indent=2))

    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
