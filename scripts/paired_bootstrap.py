"""Paired-bootstrap p-values for cell-vs-cell aggregate-F1 comparisons.

For each (cell_A, cell_B) pair, compute the paired delta distribution by
resampling questions with replacement. Report:
  - delta in observed F1 (mean)
  - 90% CI on the delta
  - one-sided p-value (fraction of bootstrap samples where delta <= 0)
  - Wilcoxon signed-rank statistic on per-question deltas

Both cells must be on the same question set with paired indices.
LongMemEval prediction files have a per-question 'f1' field; LoCoMo
files compute F1 on the fly via src.eval.f1_single.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def per_q_f1s(payload: dict) -> list[float]:
    preds = payload.get("predictions") or []
    if not preds:
        return []
    if "f1" in preds[0]:
        return [float(p["f1"]) for p in preds]
    from src.eval import f1_single
    out = []
    for p in preds:
        gold = str(p.get("gold_answer") or "")
        pred = str(p.get("prediction") or "")
        if p.get("gold_label") == "adversarial":
            out.append(1.0 if "no information" in pred.lower() else 0.0)
        else:
            out.append(f1_single(pred, gold))
    return out


def paired_bootstrap(a_f1s: list[float], b_f1s: list[float], n_bootstrap: int = 10000, seed: int = 42) -> dict:
    n = min(len(a_f1s), len(b_f1s))
    a = a_f1s[:n]
    b = b_f1s[:n]
    deltas = [bi - ai for ai, bi in zip(a, b)]
    observed_delta = sum(deltas) / n

    rng = random.Random(seed)
    boot_deltas = []
    for _ in range(n_bootstrap):
        sample_deltas = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_deltas.append(sum(sample_deltas) / n)
    boot_deltas.sort()
    p5 = boot_deltas[int(0.05 * n_bootstrap)]
    p95 = boot_deltas[int(0.95 * n_bootstrap)]

    # one-sided p-value (testing B > A)
    n_le_zero = sum(1 for d in boot_deltas if d <= 0)
    p_one_sided = n_le_zero / n_bootstrap

    # two-sided
    p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)

    # Wilcoxon signed-rank: count signs of nonzero deltas
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_disc = n_pos + n_neg

    return {
        "n": n,
        "observed_delta": observed_delta,
        "ci_5th": p5,
        "ci_95th": p95,
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n - n_disc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=ROOT / "experiments/crag-7-ranking-shift/paired_bootstrap_summary.json")
    args = ap.parse_args()

    pairs = [
        ("LME E flat × tight-per-cat",   "LME J flat × tight-fcs",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"),
        ("LME E flat × tight-per-cat",   "LME F linked × tight-per-cat",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-6-linking-baseline/results_6b_lme_linked.json"),
        ("LME E flat × tight-per-cat",   "LME G session-bank × tight-per-cat",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7G_lme_session_bank.json"),
        ("LME E flat × tight-per-cat",   "LME B flat × loose",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json"),
        ("LME E flat × tight-per-cat",   "LME A flat × tight-generic",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7A_lme_flat_tight_generic.json"),
        ("LME E flat × tight-per-cat",   "LME K1 amem-faithful × tight-per-cat",
         ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7K_lme_amem_tight.json"),
        ("LME J flat × tight-fcs",       "LME K2 amem-faithful × tight-fcs",
         ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7K_lme_amem_fcs.json"),
        ("LoCoMo LE flat × tight-per-cat",  "LoCoMo LF linked × tight-per-cat",
         ROOT / "experiments/crag-1-headline/results_locomo_promptfix.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7LF_locomo.json"),
        ("LoCoMo LE flat × tight-per-cat",  "LoCoMo LG session-bank × tight-per-cat",
         ROOT / "experiments/crag-1-headline/results_locomo_promptfix.json",
         ROOT / "experiments/crag-7-ranking-shift/results_7LG_locomo.json"),
    ]

    summaries = []
    print(f"{'comparison':75s}  {'n':>4s}  {'Δ':>9s}  {'5th':>9s}  {'95th':>9s}  {'p1':>8s}  {'verdict'}")
    print("-" * 130)
    for label_a, label_b, path_a, path_b in pairs:
        if not path_a.exists() or not path_b.exists():
            print(f"{label_a} → {label_b:50s}  (missing files)")
            continue
        a = per_q_f1s(json.loads(path_a.read_text()))
        b = per_q_f1s(json.loads(path_b.read_text()))
        if not a or not b:
            print(f"{label_a} → {label_b:50s}  (empty predictions)")
            continue
        result = paired_bootstrap(a, b, n_bootstrap=args.n_bootstrap, seed=args.seed)
        verdict = (
            "B > A (real, p<0.05)" if result["p_one_sided"] < 0.05
            else "marginal (p<0.10)" if result["p_one_sided"] < 0.10
            else "B < A (real, p>0.95)" if result["p_one_sided"] > 0.95
            else "n.s. (within noise)"
        )
        summaries.append({"a": label_a, "b": label_b, **result, "verdict": verdict})
        comparison = f"{label_a} → {label_b}"
        print(f"{comparison:75s}  {result['n']:>4d}  {result['observed_delta']:+9.4f}  {result['ci_5th']:+9.4f}  {result['ci_95th']:+9.4f}  {result['p_one_sided']:>8.4f}  {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
