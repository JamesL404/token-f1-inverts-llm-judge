"""19-cell inversion synthesis (Item #2 from final to-do).

For every comparable pair of cells (cell_X, cell_Y) in the full LLM-judge matrix:
  - Δ_F1 = token_F1(X) - token_F1(Y)
  - Δ_judge = judge_acc(X) - judge_acc(Y)
  - Pair is 'agree' if sign(Δ_F1) == sign(Δ_judge) (and both nonzero)
  - Pair is 'disagree' if sign(Δ_F1) != sign(Δ_judge) (and both nonzero)
  - Pair is 'tie' if either Δ is below a noise threshold (we use 0.005 for judge,
    which equals the gpt-4o-mini McNemar-n.s. threshold we already document)

Pair types:
  - prompt-only: same architecture, different prompt variant
  - architecture-only: same prompt, different architecture
  - retrieval-substrate: not in main 9-cell matrix (BM25 only)
  - all comparable pairs: union of prompt-only and architecture-only

Outputs a markdown table to stdout and writes results_inversion_synthesis.json.
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json"

# Token-F1 numbers from the headline matrices in §5.1 (n=500/cell LME, n=1986 LoCoMo no-adv)
LME_F1 = {
    "E": 0.365,  "A": 0.182,  "B": 0.120,
    "F": 0.366,  "C": 0.174,  "D": 0.116,
    "G": 0.367,  "H": 0.180,  "I": 0.119,
    "J": 0.447,
}
LOC_F1 = {
    "LE": 0.301, "LA": 0.150, "LB": 0.073,
    "LF": 0.278, "LC": 0.147, "LD": 0.072,
    "LG": 0.308, "LH": 0.156, "LI": 0.080,
}

# Mapping of judge keys → letter cell ID
JUDGE_KEY_TO_CELL = {
    "lme.E_flat_tight_per_cat": "E",
    "lme.F_linked_tight_per_cat": "F",
    "lme.G_session_bank_tight_per_cat": "G",
    "lme.A_flat_tight_generic": "A",
    "lme.C_linked_tight_generic": "C",
    "lme.H_session_bank_tight_generic": "H",
    "lme.B_flat_loose": "B",
    "lme.D_linked_loose": "D",
    "lme.I_session_bank_loose": "I",
    "lme.J_flat_tight_fcs": "J",
    "locomo.LE_flat_tight_per_cat": "LE",
    "locomo.LF_linked_tight_per_cat": "LF",
    "locomo.LG_session_bank_tight_per_cat": "LG",
    "locomo.LA_flat_tight_generic": "LA",
    "locomo.LC_linked_tight_generic": "LC",
    "locomo.LH_session_bank_tight_generic": "LH",
    "locomo.LB_flat_loose": "LB",
    "locomo.LD_linked_loose": "LD",
    "locomo.LI_session_bank_loose": "LI",
}

# Cell metadata for pair classification
META_LME = {
    "E": ("flat", "tight-per-cat"),
    "F": ("linked-notes", "tight-per-cat"),
    "G": ("session-bank", "tight-per-cat"),
    "A": ("flat", "tight-generic"),
    "C": ("linked-notes", "tight-generic"),
    "H": ("session-bank", "tight-generic"),
    "B": ("flat", "loose"),
    "D": ("linked-notes", "loose"),
    "I": ("session-bank", "loose"),
    "J": ("flat", "tight-fcs"),
}
META_LOC = {
    "LE": ("flat", "tight-per-cat"),
    "LF": ("linked-notes", "tight-per-cat"),
    "LG": ("session-bank", "tight-per-cat"),
    "LA": ("flat", "tight-generic"),
    "LC": ("linked-notes", "tight-generic"),
    "LH": ("session-bank", "tight-generic"),
    "LB": ("flat", "loose"),
    "LD": ("linked-notes", "loose"),
    "LI": ("session-bank", "loose"),
}

JUDGE_TIE_THRESHOLD = 0.005   # gpt-4o-mini McNemar-n.s. threshold from cells E vs J
F1_TIE_THRESHOLD = 0.005      # symmetric


def classify_pair(cell_a, cell_b, meta):
    arch_a, prompt_a = meta[cell_a]
    arch_b, prompt_b = meta[cell_b]
    if arch_a == arch_b and prompt_a != prompt_b:
        return "prompt-only"
    if prompt_a == prompt_b and arch_a != arch_b:
        return "architecture-only"
    return "off-axis"   # both differ → not in our pair-type taxonomy


def main():
    judge_data = json.load(open(JUDGE_PATH))
    judge_acc = {JUDGE_KEY_TO_CELL[k]: v["judge_accuracy"] for k, v in judge_data["cells"].items() if k in JUDGE_KEY_TO_CELL}

    pairs_by_type = {"prompt-only": [], "architecture-only": []}

    for benchmark, f1_dict, meta in [("LongMemEval", LME_F1, META_LME), ("LoCoMo", LOC_F1, META_LOC)]:
        cells = list(meta.keys())
        for a, b in combinations(cells, 2):
            ptype = classify_pair(a, b, meta)
            if ptype == "off-axis":
                continue
            if a not in judge_acc or b not in judge_acc:
                continue
            d_f1 = f1_dict[a] - f1_dict[b]
            d_judge = judge_acc[a] - judge_acc[b]
            f1_tie = abs(d_f1) <= F1_TIE_THRESHOLD
            judge_tie = abs(d_judge) <= JUDGE_TIE_THRESHOLD
            if f1_tie or judge_tie:
                verdict = "tie"
            elif (d_f1 > 0) == (d_judge > 0):
                verdict = "agree"
            else:
                verdict = "disagree"
            pairs_by_type[ptype].append({
                "benchmark": benchmark,
                "cell_a": a,
                "cell_b": b,
                "delta_f1_a_minus_b": round(d_f1, 4),
                "delta_judge_a_minus_b": round(d_judge, 4),
                "verdict": verdict,
            })

    # Add the FCS pair (J vs E) explicitly — same architecture (flat), different prompt
    # Already covered by prompt-only since J shares flat with E/A/B.

    summary = {}
    for ptype, pairs in pairs_by_type.items():
        n = len(pairs)
        agree = sum(1 for p in pairs if p["verdict"] == "agree")
        disagree = sum(1 for p in pairs if p["verdict"] == "disagree")
        tie = sum(1 for p in pairs if p["verdict"] == "tie")
        summary[ptype] = {"n": n, "agree": agree, "disagree": disagree, "tie": tie}

    # All-comparable
    all_pairs = pairs_by_type["prompt-only"] + pairs_by_type["architecture-only"]
    summary["all-comparable"] = {
        "n": len(all_pairs),
        "agree": sum(1 for p in all_pairs if p["verdict"] == "agree"),
        "disagree": sum(1 for p in all_pairs if p["verdict"] == "disagree"),
        "tie": sum(1 for p in all_pairs if p["verdict"] == "tie"),
    }

    print("\n=== Inversion synthesis ===")
    print(f"{'Pair type':<22} {'#pairs':>7} {'Agree':>7} {'Disagree':>10} {'Tie':>5}")
    for ptype, s in summary.items():
        print(f"{ptype:<22} {s['n']:>7} {s['agree']:>7} {s['disagree']:>10} {s['tie']:>5}")

    print("\n=== prompt-only pair details ===")
    for p in pairs_by_type["prompt-only"]:
        print(f"  {p['benchmark']:>12}  {p['cell_a']:>3} vs {p['cell_b']:>3}  "
              f"ΔF1={p['delta_f1_a_minus_b']:+.4f}  Δjudge={p['delta_judge_a_minus_b']:+.4f}  -> {p['verdict']}")

    print("\n=== architecture-only pair details ===")
    for p in pairs_by_type["architecture-only"]:
        print(f"  {p['benchmark']:>12}  {p['cell_a']:>3} vs {p['cell_b']:>3}  "
              f"ΔF1={p['delta_f1_a_minus_b']:+.4f}  Δjudge={p['delta_judge_a_minus_b']:+.4f}  -> {p['verdict']}")

    out = {
        "summary": summary,
        "pairs_by_type": pairs_by_type,
        "thresholds": {"f1_tie": F1_TIE_THRESHOLD, "judge_tie": JUDGE_TIE_THRESHOLD},
        "judge_acc": judge_acc,
        "f1": {**LME_F1, **LOC_F1},
    }
    out_path = ROOT / "experiments/crag-7-ranking-shift/results_inversion_synthesis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
