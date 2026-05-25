"""Paired comparison vs cellJ for HippoRAG and A-MEM faithful reproductions."""
from __future__ import annotations
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cellj = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"))
cellj_preds = {p["question_id"]: p for p in cellj["predictions"]}

for system, path in [
    ("HippoRAG", "experiments/crag-9-faithful-repro/results_hipporag_lme_oracle_n200.json"),
    ("A-MEM",    "experiments/crag-9-faithful-repro/results_amem_lme_oracle_n200.json"),
]:
    d = json.load(open(ROOT / path))
    sys_preds = {p["question_id"]: p for p in d["predictions"]}
    paired = []
    for qid in sys_preds:
        if qid in cellj_preds:
            s = sys_preds[qid]; j = cellj_preds[qid]
            paired.append({"qid": qid, "label": s["gold_label"],
                           "sys_f1": s["f1"], "j_f1": j["f1"], "delta": s["f1"]-j["f1"]})
    n = len(paired)
    sys_agg = sum(p["sys_f1"] for p in paired)/n
    j_agg = sum(p["j_f1"] for p in paired)/n
    print(f"\n=== {system} vs cell J (paired n={n}) ===")
    print(f"  {system}: F1={sys_agg:.4f}")
    print(f"  Cell J:   F1={j_agg:.4f}")
    print(f"  Δ:        {sys_agg-j_agg:+.4f}")

    B = 10000
    rng = random.Random(20260506)
    deltas = [p["delta"] for p in paired]
    boot = [sum(deltas[rng.randint(0,n-1)] for _ in range(n))/n for _ in range(B)]
    boot.sort()
    p_le_zero = sum(1 for d in boot if d <= 0)/B
    p_ge_zero = sum(1 for d in boot if d >= 0)/B
    print(f"  90% CI: [{boot[int(0.05*B)]:+.4f}, {boot[int(0.95*B)]:+.4f}]")
    print(f"  one-sided p (Δ≤0): {p_le_zero:.4f}  (Δ≥0): {p_ge_zero:.4f}")

    sys_wins = sum(1 for p in paired if p["delta"]>0.05)
    j_wins = sum(1 for p in paired if p["delta"]<-0.05)
    ties = n-sys_wins-j_wins
    print(f"  per-question (|Δ|>0.05): {system} wins {sys_wins}, J wins {j_wins}, ties {ties}")

    from collections import defaultdict
    by_label = defaultdict(list)
    for p in paired: by_label[p["label"]].append(p)
    for lab, items in by_label.items():
        m = sum(x["sys_f1"] for x in items)/len(items)
        j = sum(x["j_f1"] for x in items)/len(items)
        print(f"    {lab:>20}  n={len(items):>3}  {system}={m:.4f}  J={j:.4f}  Δ={m-j:+.4f}")

    out = {"system": system, "n_paired": n,
           "sys_f1": sys_agg, "cellj_f1": j_agg, "delta": sys_agg-j_agg,
           "paired_bootstrap": {"B": B,
                "ci_90": [boot[int(0.05*B)], boot[int(0.95*B)]],
                "one_sided_p_le_zero": p_le_zero,
                "one_sided_p_ge_zero": p_ge_zero},
           "pairwise": {"sys_wins": sys_wins, "j_wins": j_wins, "ties": ties},
           "per_label": {lab: {"n": len(items),
                "sys_f1": sum(x["sys_f1"] for x in items)/len(items),
                "cellj_f1": sum(x["j_f1"] for x in items)/len(items)}
                for lab, items in by_label.items()}}
    out_path = ROOT / f"experiments/crag-9-faithful-repro/results_{system.lower().replace('-','')}_vs_cellj_paired.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}")
