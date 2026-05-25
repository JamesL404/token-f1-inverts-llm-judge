"""Cross-judge replication: judge cells E vs J on the same 200-Q subset
with a different judge model. Compares verdict agreement with gpt-4o-mini.
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


def judge_anthropic(client, model: str, q: str, gold: str, pred: str) -> str:
    prompt = JUDGE_PROMPT.format(question=q[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        msg = client.messages.create(
            model=model, max_tokens=12,
            messages=[{"role": "user", "content": prompt}],
        )
        v = msg.content[0].text.strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception:
        return "ERROR"


def judge_google(client_key: str, model: str, q: str, gold: str, pred: str) -> str:
    """Google Gemini via google.genai."""
    import google.genai as genai
    client = genai.Client(api_key=client_key)
    prompt = JUDGE_PROMPT.format(question=q[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        v = (resp.text or "").strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception:
        return "ERROR"


def judge_one(provider, client, model, q, gold, pred, google_key=None):
    if provider == "anthropic":
        return judge_anthropic(client, model, q, gold, pred)
    if provider == "google":
        return judge_google(google_key, model, q, gold, pred)
    raise ValueError(f"unknown provider {provider}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", choices=["anthropic", "google"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell-e", default=str(ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"))
    ap.add_argument("--cell-j", default=str(ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if args.provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("ERROR: ANTHROPIC_API_KEY not in environment"); sys.exit(1)
        from anthropic import Anthropic
        client = Anthropic()
        google_key = None
    else:
        if "GOOGLE_API_KEY" not in os.environ:
            print("ERROR: GOOGLE_API_KEY not in environment"); sys.exit(1)
        client = None
        google_key = os.environ["GOOGLE_API_KEY"]

    rng = random.Random(args.seed)
    with open(args.cell_e) as f:
        e = json.load(f)
    with open(args.cell_j) as f:
        j = json.load(f)
    n_min = min(len(e["predictions"]), len(j["predictions"]))
    indices = sorted(rng.sample(range(n_min), min(args.n, n_min)))

    print(f"Judging {len(indices)} paired questions with {args.provider}:{args.model}")
    out = {"provider": args.provider, "model": args.model, "n": len(indices),
           "indices_seed": args.seed, "cells": {}}
    for label, payload in [("E_flat_tight_per_cat", e), ("J_flat_tight_fcs", j)]:
        verdicts = []
        correct = 0; incorrect = 0; other = 0
        preds = payload["predictions"]
        for i, idx in enumerate(indices):
            p = preds[idx]
            v = judge_one(args.provider, client, args.model,
                          p["question"], p.get("gold_answer", ""), p.get("prediction", ""),
                          google_key=google_key)
            verdicts.append(v)
            if v == "CORRECT": correct += 1
            elif v == "INCORRECT": incorrect += 1
            else: other += 1
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{len(indices)}  CORRECT={correct} INCORRECT={incorrect} other={other}", flush=True)
            time.sleep(0.05)
        out["cells"][label] = {
            "n_judged": len(verdicts), "correct": correct, "incorrect": incorrect,
            "other": other, "judge_accuracy": correct / max(1, correct + incorrect),
            "verdicts": verdicts,
        }
        print(f"  {label}: judge_acc = {out['cells'][label]['judge_accuracy']:.4f}")

    e_acc = out["cells"]["E_flat_tight_per_cat"]["judge_accuracy"]
    j_acc = out["cells"]["J_flat_tight_fcs"]["judge_accuracy"]
    out["delta_judge_accuracy"] = j_acc - e_acc
    print(f"\n[{args.model}] J − E judge_acc delta: {j_acc - e_acc:+.4f}")

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
