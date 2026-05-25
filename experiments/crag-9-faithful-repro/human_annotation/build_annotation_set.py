"""Build the n=100 human-annotation package (Recommendation 1 from expert reviewer).

Procedure:
  1. Sample 100 paired indices from the LongMemEval oracle 200-question
     LLM-judge subset (seed=20260506).
  2. For each paired index, take cell B (flat × loose) and cell E (flat ×
     tight-per-cat) predictions.
  3. Randomize per-question whether B is shown as "Output X" or "Output Y".
  4. Write annotator-facing CSV with columns:
       qid, question, gold, output_X, output_Y, rater_label_X, rater_label_Y, notes
     The labeler fills in rater_label_X / rater_label_Y / notes; everything
     else is read-only context.
  5. Save the unblind key separately so the author can score later without
     contaminating the labeler's blinding.
"""
from __future__ import annotations
import csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments/crag-9-faithful-repro/human_annotation"

JUDGE_PATH = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"
B_PATH = ROOT / "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json"
E_PATH = ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"


def main(n_items: int = 100, seed: int = 20260506):
    judge_data = json.load(open(JUDGE_PATH))
    indices_pool = sorted({v["idx"] for v in judge_data["cell_e"]["verdicts"]})
    assert len(indices_pool) == 200

    rng = random.Random(seed)
    sample_idx = sorted(rng.sample(indices_pool, n_items))

    b_preds = json.load(open(B_PATH))["predictions"]
    e_preds = json.load(open(E_PATH))["predictions"]

    rows_for_annotator = []   # blinded
    unblind_key = []           # author-only

    for i, idx in enumerate(sample_idx, start=1):
        b_text = b_preds[idx]["prediction"].strip()
        e_text = e_preds[idx]["prediction"].strip()
        question = b_preds[idx]["question"]
        gold = b_preds[idx]["gold_answer"]
        # Randomize order
        x_is_b = rng.random() < 0.5
        if x_is_b:
            out_X, out_Y = b_text, e_text
        else:
            out_X, out_Y = e_text, b_text
        rows_for_annotator.append({
            "item_id": i,
            "question": question,
            "gold_answer": gold,
            "output_X": out_X,
            "output_Y": out_Y,
            "rater_label_X": "",      # FILL: correct / partial / incorrect
            "rater_label_Y": "",      # FILL: correct / partial / incorrect
            "rater_preference": "",   # FILL: X / Y / tie
            "notes": "",
        })
        unblind_key.append({
            "item_id": i,
            "qid": b_preds[idx]["question_id"],
            "lme_idx": idx,
            "X_is_B": x_is_b,
            "X_cell": "B (loose)" if x_is_b else "E (tight-per-cat)",
            "Y_cell": "E (tight-per-cat)" if x_is_b else "B (loose)",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "annotation_items.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_annotator[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_annotator)
    print(f"wrote {csv_path} ({len(rows_for_annotator)} items)")

    json_path = OUT_DIR / "annotation_items.json"
    json_path.write_text(json.dumps(rows_for_annotator, indent=2, ensure_ascii=False))
    print(f"wrote {json_path}")

    key_path = OUT_DIR / "UNBLIND_KEY_DO_NOT_SHOW_RATERS.json"
    key_path.write_text(json.dumps({
        "seed": seed,
        "n_items": n_items,
        "key": unblind_key,
        "note": (
            "DO NOT share with raters. Used after rater submission to map "
            "X/Y back to cell B (loose) / cell E (tight-per-cat)."
        ),
    }, indent=2))
    print(f"wrote {key_path}")

    n_x_is_b = sum(1 for k in unblind_key if k["X_is_B"])
    print(f"\nbalance check: X is cell B (loose) on {n_x_is_b}/{n_items} items")


if __name__ == "__main__":
    main()
