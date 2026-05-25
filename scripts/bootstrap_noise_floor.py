"""Bootstrap noise floor for a single C-RAG result file.

Per `paper/proposal.md` section "Noise-floor run", any ranking-shift
claim must exceed the bootstrap noise floor. For deterministic-greedy
generation (do_sample=False) the model output is fixed, so the natural
noise estimate is bootstrap resampling of the question set.

Reports:
  - mean aggregate F1 across `n_bootstrap` resamples
  - 5th-95th percentile spread (the noise floor)
  - per-category spreads
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def bootstrap_f1(per_q_f1: list[float], n_bootstrap: int, seed: int) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    n = len(per_q_f1)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = [per_q_f1[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(means) / len(means)
    p5 = means[int(0.05 * len(means))]
    p95 = means[int(0.95 * len(means))]
    std = statistics.stdev(means) if len(means) > 1 else 0.0
    return mean, p5, p95, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = load(args.path)
    preds = payload.get("predictions") or []
    if not preds:
        print("no predictions in file")
        return

    summary = payload.get("summary", {})
    print(f"=== Bootstrap noise floor: {args.path.name} ===")
    print(f"n_questions: {len(preds)}, point F1: {summary.get('f1', 'n/a')}")
    print(f"n_bootstrap: {args.n_bootstrap}, seed: {args.seed}")
    print()

    if preds and "f1" not in preds[0]:
        from src.eval import f1_single
        def per_q(p: dict) -> float:
            gold = str(p.get("gold_answer") or "")
            pred = str(p.get("prediction") or "")
            if p.get("gold_label") == "adversarial":
                return 1.0 if "no information" in pred.lower() else 0.0
            return f1_single(pred, gold)
        f1s = [per_q(p) for p in preds]
    else:
        f1s = [float(p.get("f1", 0.0)) for p in preds]
    mean, p5, p95, std = bootstrap_f1(f1s, args.n_bootstrap, args.seed)
    spread = p95 - p5
    print(f"aggregate:  mean={mean:.4f}  5th={p5:.4f}  95th={p95:.4f}  std={std:.4f}  spread={spread:.4f}")
    print(f"  -> noise floor: any delta < {spread:.4f} (90% CI half-width = {spread/2:.4f}) is within noise")
    print()

    by_label: dict[str, list[float]] = defaultdict(list)
    for p in preds:
        by_label[p.get("gold_label", "?")].append(float(p.get("f1", 0.0)))

    print("per-category bootstrap (n_bootstrap, mean, 5th, 95th, spread):")
    for cat, items in sorted(by_label.items()):
        m, p5, p95, _ = bootstrap_f1(items, args.n_bootstrap, args.seed + 1)
        print(f"  {cat:25s}  n={len(items):4d}  mean={m:.4f}  5th={p5:.4f}  95th={p95:.4f}  spread={p95 - p5:.4f}")


if __name__ == "__main__":
    main()
