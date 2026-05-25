"""Plot the crag-7 2x3 ranking-shift matrix as a grouped bar chart with
the bootstrap noise floor (90% CI) drawn as error bars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    with path.open() as f:
        d = json.load(f)
    return d.get("summary", d)


def bootstrap_spread(path: Path, n: int = 1000, seed: int = 42) -> float:
    import random
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from src.eval import f1_single

    with path.open() as f:
        d = json.load(f)
    preds = d.get("predictions", [])
    if not preds:
        return 0.0
    if "f1" in preds[0]:
        f1s = [float(p.get("f1", 0.0)) for p in preds]
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
    for _ in range(n):
        sample = [f1s[rng.randrange(len(f1s))] for _ in range(len(f1s))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.95 * n)] - means[int(0.05 * n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "experiments")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", choices=["lme", "locomo"], default="lme")
    args = p.parse_args()
    root = args.root

    if args.benchmark == "lme":
        cells = {
            ("flat", "tight-per-cat"):    root / "crag-6-linking-baseline" / "results_6a_lme_flat.json",
            ("flat", "tight-generic"):    root / "crag-7-ranking-shift" / "results_7A_lme_flat_tight_generic.json",
            ("flat", "loose"):            root / "crag-7-ranking-shift" / "results_7B_lme_flat_loose.json",
            ("linked-notes", "tight-per-cat"):    root / "crag-6-linking-baseline" / "results_6b_lme_linked.json",
            ("linked-notes", "tight-generic"):    root / "crag-7-ranking-shift" / "results_7C_lme_linked_tight_generic.json",
            ("linked-notes", "loose"):            root / "crag-7-ranking-shift" / "results_7D_lme_linked_loose.json",
        }
        title = "crag-7 ranking-shift matrix on LongMemEval oracle 14B (n=500)"
    else:
        cells = {
            ("flat", "tight-per-cat"):    root / "crag-1-headline" / "results_locomo_promptfix.json",
            ("flat", "tight-generic"):    root / "crag-7-ranking-shift" / "results_7LA_locomo.json",
            ("flat", "loose"):            root / "crag-7-ranking-shift" / "results_7LB_locomo.json",
            ("linked-notes", "tight-per-cat"):    root / "crag-7-ranking-shift" / "results_7LF_locomo.json",
            ("linked-notes", "tight-generic"):    root / "crag-7-ranking-shift" / "results_7LC_locomo.json",
            ("linked-notes", "loose"):            root / "crag-7-ranking-shift" / "results_7LD_locomo.json",
        }
        title = "crag-7 ranking-shift matrix on LoCoMo 14B (n=1986)"

    prompt_variants = ["tight-per-cat", "tight-generic", "loose"]
    retrieval_modes = ["flat", "linked-notes"]

    f1_grid: dict[tuple[str, str], float] = {}
    spread_grid: dict[tuple[str, str], float] = {}
    for key, path in cells.items():
        if not path.exists():
            f1_grid[key] = 0.0
            spread_grid[key] = 0.0
            continue
        s = load_summary(path)
        f1_grid[key] = s.get("f1", 0.0)
        spread_grid[key] = bootstrap_spread(path)

    # Filter to cells that exist
    available_variants = [
        v for v in prompt_variants
        if cells[("flat", v)].exists() and cells[("linked-notes", v)].exists()
    ]
    if not available_variants:
        print("no matching cell pairs available; abort")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(available_variants)))
    width = 0.36

    flat_y = [f1_grid[("flat", v)] for v in available_variants]
    flat_err = [spread_grid[("flat", v)] / 2 for v in available_variants]
    linked_y = [f1_grid[("linked-notes", v)] for v in available_variants]
    linked_err = [spread_grid[("linked-notes", v)] / 2 for v in available_variants]

    ax.bar(
        [i - width / 2 for i in x], flat_y, width,
        yerr=flat_err, capsize=4, label="flat BM25 (vanilla RAG)", color="#4477aa", ecolor="#222",
    )
    ax.bar(
        [i + width / 2 for i in x], linked_y, width,
        yerr=linked_err, capsize=4, label="linked-notes (A-MEM-style)", color="#ee6677", ecolor="#222",
    )

    for i, (fy, ly) in enumerate(zip(flat_y, linked_y)):
        ax.text(i - width / 2, fy + 0.005, f"{fy:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, ly + 0.005, f"{ly:.3f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(available_variants)
    ax.set_xlabel("Prompt variant")
    ax.set_ylabel("Aggregate F1")
    ax.set_ylim(0, max(max(flat_y), max(linked_y)) * 1.18)
    ax.set_title(f"{title}  (90% bootstrap CI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
