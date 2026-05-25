"""Compare two C-RAG result JSONs head-to-head.

Designed for the crag-6 (linked-notes vs flat) and crag-7 (uncontrolled
vs controlled) comparisons. Reports delta F1, completion-length, and
flags pre-registered decision-tree outcomes.

Usage:
  python scripts/compare_crag_runs.py <baseline.json> <variant.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def per_label(summary: dict) -> dict[str, dict]:
    return summary.get("by_label") or summary.get("by_type") or {}


def fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    if abs(d) >= 0.05:
        return f"\033[1m{sign}{d:.4f}\033[0m"
    return f"{sign}{d:.4f}"


def compare(baseline: dict, variant: dict, *, baseline_label: str, variant_label: str) -> dict:
    b = baseline.get("summary", baseline)
    v = variant.get("summary", variant)

    if b.get("benchmark") != v.get("benchmark"):
        print(f"WARNING: baseline benchmark={b.get('benchmark')} variant benchmark={v.get('benchmark')}", file=sys.stderr)
    if b.get("n_questions") != v.get("n_questions"):
        print(
            f"WARNING: n differs: baseline {b.get('n_questions')} vs variant {v.get('n_questions')}",
            file=sys.stderr,
        )

    deltas: dict[str, dict] = {}
    print(f"\n=== Comparison: {baseline_label} -> {variant_label} ===")
    print(f"benchmark: {b.get('benchmark')}  n_questions: {b.get('n_questions')}")
    print()
    print(f"{'metric':30s} {baseline_label:>14s} {variant_label:>14s} {'delta':>14s}")
    print("-" * 75)
    for key in ("f1", "containment", "rubric_coverage", "calls_per_question", "total_completion_chars"):
        if key in b and key in v:
            db = b[key]
            dv = v[key]
            try:
                delta = dv - db
                print(f"{key:30s} {db:14.4f} {dv:14.4f} {fmt_delta(delta):>14s}")
                deltas[key] = {"baseline": db, "variant": dv, "delta": delta}
            except (TypeError, ValueError):
                print(f"{key:30s} {db!r:>14} {dv!r:>14} -")
    print()
    print(f"--- per-category F1 ---")
    print(f"{'category':30s} {baseline_label:>14s} {variant_label:>14s} {'delta':>14s} {'n':>6s}")
    print("-" * 80)
    pb = per_label(b)
    pv = per_label(v)
    for cat in sorted(set(pb) | set(pv)):
        bf = pb.get(cat, {}).get("f1")
        vf = pv.get(cat, {}).get("f1")
        n = pb.get(cat, {}).get("n") or pv.get(cat, {}).get("n", 0)
        if bf is None or vf is None:
            continue
        delta = vf - bf
        print(f"{cat:30s} {bf:14.4f} {vf:14.4f} {fmt_delta(delta):>14s} {n:>6}")
        deltas[f"by_label.{cat}.f1"] = {"baseline": bf, "variant": vf, "delta": delta}

    return deltas


def crag6_decision_tree(deltas: dict) -> str:
    """Apply the pre-registered crag-6 decision tree from
    experiments/crag-6-linking-baseline/protocol.md.
    """
    f1d = deltas.get("f1", {}).get("delta", 0.0)
    ms_key_lme = "by_label.multi-session.f1"
    ms_d = deltas.get(ms_key_lme, {}).get("delta", 0.0)

    if f1d >= 0.005:
        return "VERDICT: linking baseline non-trivially helps aggregate (>=+0.5 F1). Promote to ranking-shift table."
    if abs(f1d) < 0.005 and ms_d >= 0.03:
        return "VERDICT: aggregate ties but multi-session lifts >=+3 F1. Useful category-specific contribution."
    if f1d <= -0.005:
        return "VERDICT: linking baseline underperforms flat by >=0.5 F1. Strengthening negative result; escalate to embedding-link v2 or memory-bank baseline."
    return "VERDICT: ambiguous (delta within +-0.005 F1). Need more cells to decide."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--variant-label", default="variant")
    parser.add_argument("--decision-tree", choices=["crag6"], default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    b = load_run(args.baseline)
    v = load_run(args.variant)
    deltas = compare(b, v, baseline_label=args.baseline_label, variant_label=args.variant_label)

    if args.decision_tree == "crag6":
        print()
        print(crag6_decision_tree(deltas))

    if args.output_json:
        args.output_json.write_text(json.dumps(deltas, indent=2))
        print(f"\nwrote deltas to {args.output_json}")


if __name__ == "__main__":
    main()
