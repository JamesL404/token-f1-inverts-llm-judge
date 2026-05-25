"""Plot a head-to-head per-category F1 comparison between two C-RAG runs.

Usage:
  python scripts/plot_crag_comparison.py <baseline.json> <variant.json> \
      --baseline-label "flat" --variant-label "linked-notes" \
      --title "crag-6: linked-notes vs flat (LME oracle 14B)" \
      --output figures/crag6_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _per_label(summary: dict) -> dict[str, float]:
    src = summary.get("by_label") or summary.get("by_type") or {}
    return {k: v.get("f1", 0.0) for k, v in src.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("variant", type=Path)
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--variant-label", default="variant")
    p.add_argument("--title", default="C-RAG comparison")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    b = load(args.baseline)["summary"]
    v = load(args.variant)["summary"]
    pb = _per_label(b)
    pv = _per_label(v)
    cats = sorted(set(pb) | set(pv))

    cats_with_aggregate = ["aggregate"] + cats
    bf = [b.get("f1", 0.0)] + [pb.get(c, 0.0) for c in cats]
    vf = [v.get("f1", 0.0)] + [pv.get(c, 0.0) for c in cats]
    deltas = [vf[i] - bf[i] for i in range(len(cats_with_aggregate))]

    x = range(len(cats_with_aggregate))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(cats_with_aggregate)), 4.5))
    bars1 = ax.bar([i - width / 2 for i in x], bf, width, label=args.baseline_label, color="#4477aa")
    bars2 = ax.bar([i + width / 2 for i in x], vf, width, label=args.variant_label, color="#ee6677")

    for i, (b1, b2, d) in enumerate(zip(bars1, bars2, deltas)):
        ymax = max(b1.get_height(), b2.get_height())
        sign = "+" if d >= 0 else ""
        color = "#117733" if d > 0.005 else ("#cc3311" if d < -0.005 else "#666")
        ax.text(i, ymax + 0.012, f"{sign}{d:.3f}", ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats_with_aggregate, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, max(max(bf), max(vf)) * 1.18)
    ax.set_title(args.title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
