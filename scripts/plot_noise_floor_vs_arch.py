"""Visualize architecture spread vs bootstrap CI band for each prompt setting.

For each (benchmark, prompt-variant) cell, plot:
  - architecture spread (max - min across 3 flavors): a thin bar
  - bootstrap 90% CI half-width of the highest-F1 cell: a wider bar

Visual punch: architecture spreads are dwarfed by CI widths, making the
null result visceral in one glance.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)["summary"]


def bootstrap_spread(path: Path, n_bootstrap: int = 1000, seed: int = 42) -> float:
    """Return 90% CI half-width."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.eval import f1_single

    with path.open() as f:
        d = json.load(f)
    preds = d["predictions"]
    if "f1" in preds[0]:
        f1s = [float(p["f1"]) for p in preds]
    else:
        f1s = []
        for p in preds:
            gold = str(p.get("gold_answer") or "")
            pred = str(p.get("prediction") or "")
            if p.get("gold_label") == "adversarial":
                f1s.append(1.0 if "no information" in pred.lower() else 0.0)
            else:
                f1s.append(f1_single(pred, gold))
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [f1s[rng.randrange(len(f1s))] for _ in range(len(f1s))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.95 * n_bootstrap)] - means[int(0.05 * n_bootstrap)]) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "experiments")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root

    cells = {
        "LME tight-per-cat": [
            ("flat", root / "crag-6-linking-baseline" / "results_6a_lme_flat.json"),
            ("linked", root / "crag-6-linking-baseline" / "results_6b_lme_linked.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7G_lme_session_bank.json"),
        ],
        "LME tight-generic": [
            ("flat", root / "crag-7-ranking-shift" / "results_7A_lme_flat_tight_generic.json"),
            ("linked", root / "crag-7-ranking-shift" / "results_7C_lme_linked_tight_generic.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7H_lme_session_bank.json"),
        ],
        "LME loose": [
            ("flat", root / "crag-7-ranking-shift" / "results_7B_lme_flat_loose.json"),
            ("linked", root / "crag-7-ranking-shift" / "results_7D_lme_linked_loose.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7I_lme_session_bank.json"),
        ],
        "LoCoMo tight-per-cat": [
            ("flat", root / "crag-1-headline" / "results_locomo_promptfix.json"),
            ("linked", root / "crag-7-ranking-shift" / "results_7LF_locomo.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7LG_locomo.json"),
        ],
        "LoCoMo tight-generic": [
            ("flat", root / "crag-7-ranking-shift" / "results_7LA_locomo.json"),
            ("linked", root / "crag-7-ranking-shift" / "results_7LC_locomo.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7LH_locomo.json"),
        ],
        "LoCoMo loose": [
            ("flat", root / "crag-7-ranking-shift" / "results_7LB_locomo.json"),
            ("linked", root / "crag-7-ranking-shift" / "results_7LD_locomo.json"),
            ("session-bank", root / "crag-7-ranking-shift" / "results_7LI_locomo.json"),
        ],
    }

    spreads = []
    cis = []
    labels = []
    for label, items in cells.items():
        f1_vals = []
        ci_widths = []
        for arch, path in items:
            if not path.exists():
                continue
            f1_vals.append(load_summary(path).get("f1", 0.0))
            ci_widths.append(bootstrap_spread(path))
        if not f1_vals:
            continue
        arch_spread = max(f1_vals) - min(f1_vals)
        max_ci = max(ci_widths)
        spreads.append(arch_spread)
        cis.append(max_ci)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(labels)))
    width = 0.36

    bars_arch = ax.bar([i - width / 2 for i in x], spreads, width,
                       label="architecture spread (max-min across 3 flavors)",
                       color="#cc3311", edgecolor="#222")
    bars_ci = ax.bar([i + width / 2 for i in x], cis, width,
                     label="bootstrap 90% CI half-width (largest cell)",
                     color="#4477aa", edgecolor="#222")

    # Annotate values on each bar
    for i, (s, c) in enumerate(zip(spreads, cis)):
        ax.text(i - width / 2, s + 0.002, f"{s:.3f}", ha="center", fontsize=9, color="#cc3311", fontweight="bold")
        ax.text(i + width / 2, c + 0.002, f"{c:.3f}", ha="center", fontsize=9, color="#4477aa", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("F1 difference")
    ax.set_title("Architecture spread vs bootstrap noise floor — every cell, both benchmarks")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(max(spreads), max(cis)) * 1.18)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")
    print()
    print("Architecture spreads vs noise floors:")
    for lbl, s, c in zip(labels, spreads, cis):
        ratio = c / s if s > 0 else float("inf")
        print(f"  {lbl:25s}  arch_spread={s:.4f}  CI_half={c:.4f}  ratio={ratio:.1f}x")


if __name__ == "__main__":
    main()
