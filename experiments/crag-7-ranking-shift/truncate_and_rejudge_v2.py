"""Truncate-and-rejudge v2: multiple truncation budgets to map the curve.

The v1 result (cutting loose to tight's exact char budget) gave acc 0.105 vs tight 0.53,
indicating the answer often surfaces beyond the tight char budget in loose outputs.
v2 maps the accuracy-vs-truncation-budget curve at three settings:
  - first-sentence: take the first complete sentence of the loose prediction
  - 2x-tight: truncate loose to 2 * per-question tight length
  - 4x-tight: truncate loose to 4 * per-question tight length

If first-sentence still beats tight, mechanism is "tight underspecifies the answer location."
If 4x-tight matches loose-full (acc ~0.60), there is no marginal gain past that length.
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/crag-7-ranking-shift"))
from truncate_and_rejudge import load_env, judge_one


def first_sentence(text: str) -> str:
    text = text.strip()
    m = re.search(r"[.!?](\s|$)", text)
    if m:
        return text[: m.end()].strip()
    return text[:120]


def main():
    load_env(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    judge_path = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"
    judge_data = json.load(open(judge_path))
    indices = sorted({v["idx"] for v in judge_data["cell_e"]["verdicts"]})

    cell_b_path = ROOT / "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json"
    cell_e_path = ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"
    b_preds = json.load(open(cell_b_path))["predictions"]
    e_preds = json.load(open(cell_e_path))["predictions"]

    variants = {}
    for idx in indices:
        b_pred = b_preds[idx]["prediction"]
        e_pred = e_preds[idx]["prediction"]
        e_len = len(e_pred)
        variants.setdefault("first_sentence", []).append((idx, first_sentence(b_pred), b_preds[idx]))
        variants.setdefault("2x_tight",       []).append((idx, b_pred[: max(2 * e_len, 60)], b_preds[idx]))
        variants.setdefault("4x_tight",       []).append((idx, b_pred[: max(4 * e_len, 120)], b_preds[idx]))

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"

    out = {"model": model, "n": len(indices), "variants": {}}
    for name, items in variants.items():
        avg_len = sum(len(t[1]) for t in items) / len(items)
        print(f"\n=== variant: {name}  avg_len={avg_len:.0f} chars ===")
        correct = 0; per_idx_v = {}
        t0 = time.time()
        for k, (idx, trunc_pred, src) in enumerate(items):
            v = judge_one(client, model, src["question"], src["gold_answer"], trunc_pred)
            per_idx_v[idx] = v
            if v == "CORRECT": correct += 1
            if (k + 1) % 50 == 0:
                print(f"  [{k+1}/{len(items)}] {time.time()-t0:.0f}s acc_so_far={correct/(k+1):.3f}")
        acc = correct / len(items)
        print(f"  -> judge acc = {acc:.3f}  ({correct}/{len(items)})")
        out["variants"][name] = {"avg_len_chars": avg_len, "judge_accuracy": acc, "verdicts": per_idx_v}

    # Compare to tight (E) and loose-full (B)
    tight_acc = 0.53
    loose_full = 0.60
    print("\n=== Summary ===")
    print(f"  tight-per-cat (cell E):  {tight_acc:.3f}  (~47 chars)")
    print(f"  truncated-tight-len:     0.105  (~47 chars; from v1)")
    for name, v in out["variants"].items():
        print(f"  {name:<20} {v['judge_accuracy']:.3f}  ({v['avg_len_chars']:.0f} chars)")
    print(f"  loose-full (cell B):     {loose_full:.3f}  (~311 chars)")

    # McNemar paired vs tight, for each variant
    tight_v = {v["idx"]: v["verdict"] for v in judge_data["cell_e"]["verdicts"]}
    for name, v in out["variants"].items():
        tv = v["verdicts"]
        a = sum(1 for i in tv if tv[i] == "CORRECT" and tight_v.get(i) == "INCORRECT")
        b = sum(1 for i in tv if tv[i] == "INCORRECT" and tight_v.get(i) == "CORRECT")
        disc = a + b
        chi2 = (abs(a - b) - 1) ** 2 / disc if disc else 0.0
        v["mcnemar_vs_tight"] = {"variant_only": a, "tight_only": b, "chi2": chi2, "sig": chi2 > 3.841}
        print(f"  McNemar {name} vs tight: {a}/{b} discordant, χ²={chi2:.2f} {'sig' if chi2>3.841 else 'n.s.'}")

    out_path = ROOT / "experiments/crag-7-ranking-shift/results_truncate_rejudge_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
