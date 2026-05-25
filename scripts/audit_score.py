"""Score the author-led blind audit (Item #3 from final to-do).

I labeled out_X / out_Y per item as C / P / I (correct / partial / incorrect)
while blind to which prompt produced which output. This script:
 1. unblinds via the X_is_B key
 2. maps my labels to B-vs-E correctness
 3. compares against gpt-4o-mini and Claude-Sonnet-4.5 judge verdicts

Treat partial as correct OR as half — we report both.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# My labels in order Q1-Q50, format (X_label, Y_label) where each is "C"/"P"/"I"
AUTHOR_LABELS = [
    ("I","I"),  # Q1
    ("P","C"),  # Q2
    ("I","I"),  # Q3
    ("I","I"),  # Q4
    ("C","C"),  # Q5
    ("I","I"),  # Q6
    ("C","C"),  # Q7
    ("I","I"),  # Q8
    ("I","C"),  # Q9
    ("I","I"),  # Q10
    ("I","I"),  # Q11
    ("I","I"),  # Q12
    ("C","C"),  # Q13
    ("C","C"),  # Q14
    ("I","I"),  # Q15
    ("P","C"),  # Q16
    ("C","C"),  # Q17
    ("C","I"),  # Q18
    ("C","C"),  # Q19
    ("I","I"),  # Q20
    ("C","C"),  # Q21
    ("C","C"),  # Q22
    ("I","I"),  # Q23
    ("I","I"),  # Q24
    ("C","C"),  # Q25
    ("C","C"),  # Q26
    ("C","C"),  # Q27
    ("C","C"),  # Q28
    ("C","C"),  # Q29
    ("C","C"),  # Q30
    ("C","C"),  # Q31
    ("C","P"),  # Q32
    ("I","P"),  # Q33
    ("C","C"),  # Q34
    ("C","C"),  # Q35
    ("I","I"),  # Q36
    ("I","I"),  # Q37
    ("I","I"),  # Q38
    ("I","I"),  # Q39
    ("C","C"),  # Q40
    ("C","C"),  # Q41
    ("I","I"),  # Q42
    ("I","I"),  # Q43
    ("I","I"),  # Q44
    ("C","C"),  # Q45
    ("I","I"),  # Q46
    ("I","I"),  # Q47
    ("I","I"),  # Q48
    ("I","I"),  # Q49
    ("C","C"),  # Q50
]


def lab_to_correct(lab: str, partial_as_correct: bool) -> bool:
    """Convert C/P/I to a binary CORRECT bool."""
    if lab == "C": return True
    if lab == "I": return False
    return partial_as_correct  # "P"


def main():
    items = json.load(open(ROOT / "experiments/crag-7-ranking-shift/blind_audit_items.json"))
    # Indices used by the original 200-question paired judge run (same seed=42 as full_matrix)
    e_vs_j = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"))
    judge_indices = sorted({v["idx"] for v in e_vs_j["cell_e"]["verdicts"]})
    assert len(judge_indices) == 200

    fm = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json"))
    e_v_list = fm["cells"]["lme.E_flat_tight_per_cat"]["verdicts"]
    b_v_list = fm["cells"]["lme.B_flat_loose"]["verdicts"]
    gpt_e_verdicts = dict(zip(judge_indices, e_v_list))
    gpt_b_verdicts = dict(zip(judge_indices, b_v_list))

    # Claude run only covered E vs J — we have claude_E but not claude_B.
    # Load whatever's there:
    claude = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_claude.json"))
    claude_e_v = {}
    claude_b_v = {}
    if "cell_e" in claude and isinstance(claude["cell_e"].get("verdicts"), list):
        first = claude["cell_e"]["verdicts"][0] if claude["cell_e"]["verdicts"] else None
        if isinstance(first, dict):
            claude_e_v = {v["idx"]: v["verdict"] for v in claude["cell_e"]["verdicts"]}
        else:
            claude_e_v = dict(zip(judge_indices, claude["cell_e"]["verdicts"]))
    print(f"gpt B/E n={len(gpt_b_verdicts)}/{len(gpt_e_verdicts)}, claude E n={len(claude_e_v)}, claude B n={len(claude_b_v)}")

    assert len(items) == len(AUTHOR_LABELS) == 50

    rows = []
    for it, (lab_X, lab_Y) in zip(items, AUTHOR_LABELS):
        # unblind: if X_is_B, then X_label is for B (loose), Y_label for E (tight)
        if it["X_is_B"]:
            lab_B, lab_E = lab_X, lab_Y
        else:
            lab_B, lab_E = lab_Y, lab_X
        rows.append({
            "idx": it["idx"],
            "lab_B": lab_B,
            "lab_E": lab_E,
            "gpt_B": gpt_b_verdicts.get(it["idx"]),
            "gpt_E": gpt_e_verdicts.get(it["idx"]),
            "claude_B": claude_b_v.get(it["idx"]),
            "claude_E": claude_e_v.get(it["idx"]),
        })

    for partial_as in [True, False]:
        tag = "partial→correct" if partial_as else "partial→incorrect"
        print(f"\n=== Author labels ({tag}) ===")
        n = 0
        author_b_correct = author_e_correct = 0
        author_prefers_b = author_prefers_e = author_tie = 0
        for r in rows:
            n += 1
            b = lab_to_correct(r["lab_B"], partial_as)
            e = lab_to_correct(r["lab_E"], partial_as)
            author_b_correct += b
            author_e_correct += e
            if b and not e: author_prefers_b += 1
            elif e and not b: author_prefers_e += 1
            else: author_tie += 1
        print(f"  author B correct: {author_b_correct}/{n}, E correct: {author_e_correct}/{n}")
        print(f"  author prefers B (loose): {author_prefers_b}, prefers E (tight): {author_prefers_e}, tie: {author_tie}")

        # Agreement with judges per cell
        for which_cell in ("B", "E"):
            for judge_name in ("gpt", "claude"):
                lab_key = f"lab_{which_cell}"
                judge_key = f"{judge_name}_{which_cell}"
                agree = 0; total = 0
                for r in rows:
                    if r[judge_key] in ("CORRECT","INCORRECT"):
                        total += 1
                        au = lab_to_correct(r[lab_key], partial_as)
                        ju = (r[judge_key] == "CORRECT")
                        if au == ju: agree += 1
                if total:
                    print(f"  agreement (author/{judge_name}, cell {which_cell}): {agree}/{total} = {agree/total:.2%}")

    out_path = ROOT / "experiments/crag-7-ranking-shift/results_blind_audit.json"
    out_path.write_text(json.dumps({"rows": rows, "author_labels_per_X_Y": AUTHOR_LABELS}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
