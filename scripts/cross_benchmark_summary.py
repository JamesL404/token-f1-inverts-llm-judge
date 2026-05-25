"""Produce the cross-benchmark summary table for the paper's headline.

Reads the LongMemEval and LoCoMo crag-7 matrices and reports:
  - per-cell aggregate F1 (and MC10 where available)
  - 3 effects per benchmark: architecture, format-tightness, category-aware
  - bootstrap CI for each cell
  - cross-benchmark architecture verdict
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def get_f1(payload: dict) -> float:
    s = payload.get("summary", payload)
    return s.get("f1", 0.0)


def get_no_adv_f1(payload: dict) -> float | None:
    """LoCoMo-specific: aggregate F1 excluding adversarial. The 'honest' number."""
    s = payload.get("summary", payload)
    fbl = s.get("f1_by_label") or {}
    if "adversarial" not in fbl:
        return None
    preds = payload.get("predictions") or []
    if not preds:
        return None
    from collections import Counter
    counts = Counter(p.get("gold_label") for p in preds)
    total = 0.0
    n = 0
    for cat, f1 in fbl.items():
        if cat == "adversarial":
            continue
        total += f1 * counts[cat]
        n += counts[cat]
    return total / n if n else 0.0


def per_q_f1s(payload: dict) -> list[float]:
    """Get per-question F1, computing on-the-fly for LoCoMo."""
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


def bootstrap(f1s: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    if not f1s:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n):
        sample = [f1s[rng.randrange(len(f1s))] for _ in range(len(f1s))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.05 * n)], means[int(0.95 * n)], means[int(0.95 * n)] - means[int(0.05 * n)]


def lme_cells(root: Path) -> dict:
    return {
        "tight-per-cat × flat":         root / "crag-6-linking-baseline" / "results_6a_lme_flat.json",
        "tight-per-cat × linked":       root / "crag-6-linking-baseline" / "results_6b_lme_linked.json",
        "tight-per-cat × session-bank": root / "crag-7-ranking-shift" / "results_7G_lme_session_bank.json",
        "tight-generic × flat":         root / "crag-7-ranking-shift" / "results_7A_lme_flat_tight_generic.json",
        "tight-generic × linked":       root / "crag-7-ranking-shift" / "results_7C_lme_linked_tight_generic.json",
        "tight-generic × session-bank": root / "crag-7-ranking-shift" / "results_7H_lme_session_bank.json",
        "loose × flat":                 root / "crag-7-ranking-shift" / "results_7B_lme_flat_loose.json",
        "loose × linked":               root / "crag-7-ranking-shift" / "results_7D_lme_linked_loose.json",
        "loose × session-bank":         root / "crag-7-ranking-shift" / "results_7I_lme_session_bank.json",
        "tight-fcs × flat":             root / "crag-7-ranking-shift" / "results_7J_lme_flat_tight_fcs.json",
    }


def locomo_cells(root: Path) -> dict:
    return {
        "tight-per-cat × flat":         root / "crag-1-headline" / "results_locomo_promptfix.json",
        "tight-per-cat × linked":       root / "crag-7-ranking-shift" / "results_7LF_locomo.json",
        "tight-generic × flat":         root / "crag-7-ranking-shift" / "results_7LA_locomo.json",
        "loose × flat":                 root / "crag-7-ranking-shift" / "results_7LB_locomo.json",
        "tight-generic × linked":       root / "crag-7-ranking-shift" / "results_7LC_locomo.json",
        "loose × linked":               root / "crag-7-ranking-shift" / "results_7LD_locomo.json",
    }


def summarize_benchmark(name: str, paths: dict, *, also_no_adv: bool = False) -> dict:
    print(f"\n=== {name} ===\n")
    header = f"{'cell':30s} {'F1':>8s}"
    if also_no_adv:
        header += f" {'F1_no_adv':>10s}"
    header += f" {'5th':>8s} {'95th':>8s} {'spread':>8s}"
    print(header)
    cell_f1 = {}
    cell_no_adv = {}
    cell_spread = {}
    for label, path in paths.items():
        if not path.exists():
            print(f"  {label:30s} (missing)")
            cell_f1[label] = None
            cell_no_adv[label] = None
            cell_spread[label] = None
            continue
        payload = load(path)
        f1 = get_f1(payload)
        f1_na = get_no_adv_f1(payload) if also_no_adv else None
        f1s = per_q_f1s(payload)
        p5, p95, spread = bootstrap(f1s)
        line = f"  {label:30s} {f1:8.4f}"
        if also_no_adv:
            line += f" {f1_na:10.4f}" if f1_na is not None else f" {'':>10s}"
        line += f" {p5:8.4f} {p95:8.4f} {spread:8.4f}"
        print(line)
        cell_f1[label] = f1
        cell_no_adv[label] = f1_na
        cell_spread[label] = spread

    e = cell_f1.get("tight-per-cat × flat")
    f = cell_f1.get("tight-per-cat × linked")
    a = cell_f1.get("tight-generic × flat")
    b = cell_f1.get("loose × flat")
    c = cell_f1.get("tight-generic × linked")
    d = cell_f1.get("loose × linked")

    if all(x is not None for x in (e, f, a, b, c, d)):
        print(f"\n  Effects (aggregate F1):")
        print(f"    architecture (linked - flat) under tight-per-cat:   {f - e:+.4f}")
        print(f"    architecture (linked - flat) under tight-generic:   {c - a:+.4f}")
        print(f"    architecture (linked - flat) under loose:           {d - b:+.4f}")
        print(f"    format-tightness (tight-generic - loose) flat:      {a - b:+.4f}")
        print(f"    format-tightness (tight-generic - loose) linked:    {c - d:+.4f}")
        print(f"    category-aware (tight-per-cat - tight-generic) flat:   {e - a:+.4f}")
        print(f"    category-aware (tight-per-cat - tight-generic) linked: {f - c:+.4f}")
        print(f"    combined prompt engineering (tight-per-cat - loose) flat:   {e - b:+.4f}")
        print(f"    combined prompt engineering (tight-per-cat - loose) linked: {f - d:+.4f}")

    if also_no_adv:
        e_na = cell_no_adv.get("tight-per-cat × flat")
        f_na = cell_no_adv.get("tight-per-cat × linked")
        a_na = cell_no_adv.get("tight-generic × flat")
        b_na = cell_no_adv.get("loose × flat")
        c_na = cell_no_adv.get("tight-generic × linked")
        d_na = cell_no_adv.get("loose × linked")
        if all(x is not None for x in (e_na, f_na, a_na)):
            print(f"\n  No-adversarial Effects (aggregate F1, no adversarial):")
            print(f"    architecture (linked - flat) under tight-per-cat:   {f_na - e_na:+.4f}")
            if c_na is not None:
                print(f"    architecture (linked - flat) under tight-generic:   {c_na - a_na:+.4f}")
            if all(x is not None for x in (b_na, d_na)):
                print(f"    architecture (linked - flat) under loose:           {d_na - b_na:+.4f}")
            if b_na is not None:
                print(f"    format-tightness (tight-generic - loose) flat:      {a_na - b_na:+.4f}")
            if all(x is not None for x in (c_na, d_na)):
                print(f"    format-tightness (tight-generic - loose) linked:    {c_na - d_na:+.4f}")
            print(f"    category-aware (tight-per-cat - tight-generic) flat:   {e_na - a_na:+.4f}")
            if c_na is not None:
                print(f"    category-aware (tight-per-cat - tight-generic) linked: {f_na - c_na:+.4f}")
            if b_na is not None:
                print(f"    combined prompt engineering (tight-per-cat - loose) flat:   {e_na - b_na:+.4f}")

    return {"cell_f1": cell_f1, "cell_no_adv": cell_no_adv, "cell_spread": cell_spread}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("experiments"))
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    print("=" * 76)
    print("Cross-benchmark crag-7 ranking-shift summary")
    print("matched 14B Qwen2.5-Instruct, BM25 retrieval, rule-based router, 1 LLM call/Q")
    print("=" * 76)

    lme = summarize_benchmark("LongMemEval oracle (n=500 / cell)", lme_cells(args.root))
    locomo = summarize_benchmark("LoCoMo (n=1986 / cell)", locomo_cells(args.root), also_no_adv=True)

    print("\n=== Cross-benchmark architecture verdict ===\n")
    for label in ("tight-per-cat × flat", "tight-per-cat × linked", "tight-generic × flat", "tight-generic × linked", "loose × flat", "loose × linked"):
        ll = lme["cell_f1"].get(label)
        lc = locomo["cell_f1"].get(label)
        l_str = f"{ll:.4f}" if ll is not None else "  -   "
        c_str = f"{lc:.4f}" if lc is not None else "  -   "
        print(f"  {label:30s}  LME {l_str}   LoCoMo {c_str}")

    if args.output_json:
        args.output_json.write_text(json.dumps({"lme": lme, "locomo": locomo}, indent=2))
        print(f"\nwrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
