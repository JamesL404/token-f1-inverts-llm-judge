"""50-example blind manual audit (Item #3 from final to-do).

Sample 50 paired B-vs-E questions from the LongMemEval n=200 paired judged subset.
For each question, randomize the order of the two predictions (call them OUT_X, OUT_Y),
hide which prompt produced which, label each as correct / partial / incorrect.
Then unblind and compare against gpt-4o-mini and Claude-Sonnet-4.5 judge verdicts.

This is a sanity check, not an independent annotation study.
The author labeling is single-rater and not adjudicated.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JUDGE_PATH = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"
B_PATH = ROOT / "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json"
E_PATH = ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"

CLAUDE_PATH = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_claude.json"


def main():
    judge_data = json.load(open(JUDGE_PATH))
    indices = sorted({v["idx"] for v in judge_data["cell_e"]["verdicts"]})

    # Sample 50 of the 200 judge indices
    rng = random.Random(20260505)
    sample_idx = sorted(rng.sample(indices, 50))

    b_preds = json.load(open(B_PATH))["predictions"]
    e_preds = json.load(open(E_PATH))["predictions"]

    items = []
    for idx in sample_idx:
        # Randomize order: 50/50 chance B is shown first
        b_first = rng.random() < 0.5
        if b_first:
            out_X, out_Y = b_preds[idx]["prediction"], e_preds[idx]["prediction"]
            X_is_B = True
        else:
            out_X, out_Y = e_preds[idx]["prediction"], b_preds[idx]["prediction"]
            X_is_B = False
        items.append({
            "idx": idx,
            "question": b_preds[idx]["question"],
            "gold": b_preds[idx]["gold_answer"],
            "out_X": out_X,
            "out_Y": out_Y,
            "X_is_B": X_is_B,            # ground-truth blind key (for unblinding)
            "B_truncated": (b_preds[idx]["prediction"][:300] + "…") if len(b_preds[idx]["prediction"]) > 300 else b_preds[idx]["prediction"],
            "E_full": e_preds[idx]["prediction"],
        })

    out_path = ROOT / "experiments/crag-7-ranking-shift/blind_audit_items.json"
    out_path.write_text(json.dumps(items, indent=2))
    print(f"wrote {out_path} ({len(items)} items)")
    print(f"X_is_B distribution: {sum(1 for it in items if it['X_is_B'])}/50 X-is-loose")


if __name__ == "__main__":
    main()
