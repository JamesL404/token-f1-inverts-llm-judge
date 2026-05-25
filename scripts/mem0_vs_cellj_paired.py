"""Paired Mem0-vs-cellJ comparison on the same 200 indices Mem0 ran (Item #8 follow-up).

Uses existing cell J predictions from results_7J_lme_flat_tight_fcs.json,
filters to the 200 question_ids Mem0 sampled, and computes:
  - paired aggregate F1 (Mem0 and J on same questions)
  - per-label F1 on the same matched subset
  - paired-bootstrap p-value on Δ_aggregate
  - per-question wins / losses / ties
"""
from __future__ import annotations
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mem0 = json.load(open(ROOT / "experiments/crag-9-faithful-repro/results_mem0_lme_oracle_n200.json"))
cellj = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"))

mem0_preds = {p["question_id"]: p for p in mem0["predictions"]}
cellj_preds = {p["question_id"]: p for p in cellj["predictions"]}

paired = []
for qid in mem0_preds:
    if qid in cellj_preds:
        m = mem0_preds[qid]
        j = cellj_preds[qid]
        paired.append({"qid": qid, "label": m["gold_label"],
                       "mem0_f1": m["f1"], "cellj_f1": j["f1"],
                       "delta": m["f1"] - j["f1"]})

n = len(paired)
print(f"paired matched on qid: {n}/200")

mem0_agg = sum(p["mem0_f1"] for p in paired) / n
j_agg    = sum(p["cellj_f1"] for p in paired) / n
print(f"\n=== aggregate F1 on same {n} indices ===")
print(f"  Mem0 + Qwen + FCS:        {mem0_agg:.4f}")
print(f"  Cell J flat-BM25 + FCS:   {j_agg:.4f}")
print(f"  Δ (Mem0 − J):             {mem0_agg - j_agg:+.4f}")

# Per-label
from collections import defaultdict
by_label = defaultdict(list)
for p in paired:
    by_label[p["label"]].append(p)
print("\n=== per-label paired ===")
for label, items in by_label.items():
    m = sum(x["mem0_f1"] for x in items) / len(items)
    j = sum(x["cellj_f1"] for x in items) / len(items)
    print(f"  {label:>20}  n={len(items):>3}  Mem0={m:.4f}  J={j:.4f}  Δ={m-j:+.4f}")

# paired bootstrap p-value: prob Δ > 0 under resampling
B = 10000
rng = random.Random(20260506)
deltas = [p["delta"] for p in paired]
boot_means = []
for _ in range(B):
    sample = [deltas[rng.randint(0, n-1)] for _ in range(n)]
    boot_means.append(sum(sample) / n)
p_le_zero = sum(1 for d in boot_means if d <= 0) / B
ci_low = sorted(boot_means)[int(0.05 * B)]
ci_high = sorted(boot_means)[int(0.95 * B)]
print(f"\n=== paired bootstrap (B={B}) on Δ_aggregate ===")
print(f"  observed Δ: {mem0_agg - j_agg:+.4f}")
print(f"  90% CI on Δ: [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"  one-sided p (H0: Δ ≤ 0): {p_le_zero:.4f}  (sig at α=0.05 if p < 0.05)")

# per-question pairwise
mem0_wins = sum(1 for p in paired if p["delta"] > 0.05)
j_wins = sum(1 for p in paired if p["delta"] < -0.05)
ties = n - mem0_wins - j_wins
print(f"\n=== per-question pairwise (|Δ| > 0.05 threshold) ===")
print(f"  Mem0 wins: {mem0_wins}, Cell J wins: {j_wins}, ties: {ties}")

# Save
out = {
    "n_paired": n,
    "mem0_aggregate_f1": mem0_agg,
    "cellj_aggregate_f1": j_agg,
    "delta_mem0_minus_j": mem0_agg - j_agg,
    "paired_bootstrap": {
        "B": B,
        "ci_90_low": ci_low,
        "ci_90_high": ci_high,
        "one_sided_p_le_zero": p_le_zero,
    },
    "per_label": {label: {
        "n": len(items),
        "mem0_f1": sum(x["mem0_f1"] for x in items) / len(items),
        "cellj_f1": sum(x["cellj_f1"] for x in items) / len(items),
    } for label, items in by_label.items()},
    "pairwise": {"mem0_wins": mem0_wins, "j_wins": j_wins, "ties": ties},
}
out_path = ROOT / "experiments/crag-9-faithful-repro/results_mem0_vs_cellj_paired.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nwrote {out_path}")
