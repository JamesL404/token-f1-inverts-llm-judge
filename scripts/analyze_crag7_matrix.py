"""Compute the crag-7 2x3 ranking-shift matrix and decision-tree verdicts.

Reads the 6 result JSONs (4 from crag-7 + 2 reused from crag-6) and
prints the per-cell aggregate F1 + per-category F1 grid, plus the four
key effect deltas relevant to the proposal's success criteria.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def s(payload: dict) -> dict:
    return payload.get("summary", payload)


def per_cat(summary: dict) -> dict[str, float]:
    src = summary.get("by_label") or summary.get("by_type") or {}
    if src and isinstance(next(iter(src.values())), dict):
        return {k: v.get("f1", 0.0) for k, v in src.items()}
    # LoCoMo schema uses f1_by_label : {cat: float}
    fbl = summary.get("f1_by_label") or {}
    if fbl:
        return dict(fbl)
    return {}


def fmt(v: float | None) -> str:
    if v is None:
        return "    -   "
    return f"{v:8.4f}"


def fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    if abs(d) >= 0.01:
        return f"\033[1m{sign}{d:.4f}\033[0m"
    return f"{sign}{d:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("experiments"))
    p.add_argument("--benchmark", choices=["lme", "locomo"], default="lme")
    args = p.parse_args()
    root = args.root

    if args.benchmark == "lme":
        cells = {
            "E_flat_tight_per_cat":   root / "crag-6-linking-baseline" / "results_6a_lme_flat.json",
            "F_linked_tight_per_cat": root / "crag-6-linking-baseline" / "results_6b_lme_linked.json",
            "A_flat_tight_generic":   root / "crag-7-ranking-shift" / "results_7A_lme_flat_tight_generic.json",
            "B_flat_loose":           root / "crag-7-ranking-shift" / "results_7B_lme_flat_loose.json",
            "C_linked_tight_generic": root / "crag-7-ranking-shift" / "results_7C_lme_linked_tight_generic.json",
            "D_linked_loose":         root / "crag-7-ranking-shift" / "results_7D_lme_linked_loose.json",
        }
    else:
        cells = {
            "E_flat_tight_per_cat":   root / "crag-1-headline" / "results_locomo_promptfix.json",
            "F_linked_tight_per_cat": root / "crag-7-ranking-shift" / "results_7LF_locomo.json",
            "A_flat_tight_generic":   root / "crag-7-ranking-shift" / "results_7LA_locomo.json",
            "B_flat_loose":           root / "crag-7-ranking-shift" / "results_7LB_locomo.json",
            "C_linked_tight_generic": root / "crag-7-ranking-shift" / "results_7LC_locomo.json",
            "D_linked_loose":         root / "crag-7-ranking-shift" / "results_7LD_locomo.json",
        }

    summaries: dict[str, dict] = {}
    for name, path in cells.items():
        if not path.exists():
            print(f"WARNING: missing {name} at {path}")
            continue
        summaries[name] = s(load(path))

    bench_label = "LongMemEval oracle 14B (n=500)" if args.benchmark == "lme" else "LoCoMo 14B (n=1986)"
    print(f"\n=== crag-7 2x3 ranking-shift matrix on {bench_label} ===\n")
    print(f"{'':30s} {'tight-per-cat':>14s} {'tight-generic':>14s} {'loose':>14s}")
    print("-" * 76)
    for ret_name, e_key, a_key, b_key in (
        ("flat",         "E_flat_tight_per_cat",   "A_flat_tight_generic",   "B_flat_loose"),
        ("linked-notes", "F_linked_tight_per_cat", "C_linked_tight_generic", "D_linked_loose"),
    ):
        e = summaries.get(e_key, {}).get("f1")
        a = summaries.get(a_key, {}).get("f1")
        b = summaries.get(b_key, {}).get("f1")
        print(f"{ret_name:30s} {fmt(e):>14s} {fmt(a):>14s} {fmt(b):>14s}")

    # Key effects
    print("\n=== Key effects (aggregate F1) ===\n")

    def get(name: str) -> float | None:
        return summaries.get(name, {}).get("f1")

    E = get("E_flat_tight_per_cat")
    F = get("F_linked_tight_per_cat")
    A = get("A_flat_tight_generic")
    B = get("B_flat_loose")
    C = get("C_linked_tight_generic")
    D = get("D_linked_loose")

    if all(x is not None for x in (E, F, A, B, C, D)):
        print(f"  format effect (no-cat, flat):       A - B = {fmt_delta(A - B)}")
        print(f"  format effect (no-cat, linked):     C - D = {fmt_delta(C - D)}")
        print(f"  category-aware effect (tight, flat):  E - A = {fmt_delta(E - A)}")
        print(f"  category-aware effect (tight, linked): F - C = {fmt_delta(F - C)}")
        print(f"  architecture effect (per-cat tight):    F - E = {fmt_delta(F - E)}")
        print(f"  architecture effect (generic tight):    C - A = {fmt_delta(C - A)}")
        print(f"  architecture effect (loose):            D - B = {fmt_delta(D - B)}")

        # Ranking-shift-style metric:
        # the linked-vs-flat gap at tight-per-cat vs at loose
        gap_tight = F - E
        gap_loose = D - B
        if gap_tight != 0:
            shrinkage = (gap_tight - gap_loose) / gap_tight * 100
        else:
            shrinkage = float("inf")
        print()
        print(f"  Architecture gap (linked - flat) under different format settings:")
        print(f"    under tight-per-cat:  {fmt_delta(gap_tight)}")
        print(f"    under loose:          {fmt_delta(gap_loose)}")
        print(f"    relative change:      {shrinkage:+.1f}%")

        # Pairwise ordering check
        print()
        print(f"  Pairwise orderings on aggregate F1:")
        print(f"    tight-per-cat:   {'flat > linked' if E > F else 'linked > flat' if F > E else 'tie'}  ({E:.4f} vs {F:.4f})")
        print(f"    tight-generic:   {'flat > linked' if A > C else 'linked > flat' if C > A else 'tie'}  ({A:.4f} vs {C:.4f})")
        print(f"    loose:           {'flat > linked' if B > D else 'linked > flat' if D > B else 'tie'}  ({B:.4f} vs {D:.4f})")
    else:
        print("  (some cells missing — partial summary above)")

    # Per-category matrices (just temporal and multi-session, the most paper-relevant)
    for cat in ("temporal", "multi-session", "knowledge-update", "single-session"):
        if not all(cat in per_cat(summaries.get(k, {})) for k in cells):
            continue
        print(f"\n--- {cat} F1 ---")
        for ret_name, e_key, a_key, b_key in (
            ("flat",         "E_flat_tight_per_cat",   "A_flat_tight_generic",   "B_flat_loose"),
            ("linked-notes", "F_linked_tight_per_cat", "C_linked_tight_generic", "D_linked_loose"),
        ):
            e = per_cat(summaries.get(e_key, {})).get(cat)
            a = per_cat(summaries.get(a_key, {})).get(cat)
            b = per_cat(summaries.get(b_key, {})).get(cat)
            print(f"  {ret_name:14s}  per-cat {fmt(e):>10s}   generic {fmt(a):>10s}   loose {fmt(b):>10s}")


if __name__ == "__main__":
    main()
