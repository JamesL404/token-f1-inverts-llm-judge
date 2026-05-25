"""Score the n=100 human-annotation submission against B-vs-E + LLM judges.

Maps annotator labels c/p/i (correct/partial/incorrect) for X/Y back to
cell B (loose) vs cell E (tight-per-cat) using UNBLIND_KEY, then:
  - per-cell correctness rate
  - human B-vs-E preference distribution
  - human / gpt-4o-mini agreement per cell
  - human / Claude-Sonnet-4.5 agreement per cell (where available)
  - paired McNemar B vs E under human labels
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HA = ROOT / "experiments/crag-9-faithful-repro/human_annotation"


def lab_norm(s: str, partial_as: str = "correct") -> bool | None:
    """Normalize to bool (True=correct, False=incorrect). Optional partial-as."""
    s = s.strip().lower()
    if not s: return None
    if s in ("c", "correct", "yes"): return True
    if s in ("i", "incorrect", "no"): return False
    if s in ("p", "partial"):
        return True if partial_as == "correct" else False
    return None


def main():
    items = json.load(open(HA / "annotation_items.json"))
    key = json.load(open(HA / "UNBLIND_KEY_DO_NOT_SHOW_RATERS.json"))["key"]
    key_by_id = {k["item_id"]: k for k in key}

    # gpt-4o-mini judge verdicts on the full 200-question subset
    e_vs_j = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"))
    judge_idxs = sorted({v["idx"] for v in e_vs_j["cell_e"]["verdicts"]})
    fm = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json"))
    gpt_b = dict(zip(judge_idxs, fm["cells"]["lme.B_flat_loose"]["verdicts"]))
    gpt_e = dict(zip(judge_idxs, fm["cells"]["lme.E_flat_tight_per_cat"]["verdicts"]))

    # Claude verdicts where available
    try:
        claude = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_claude.json"))
        claude_e_v = {}
        if "cell_e" in claude and isinstance(claude["cell_e"].get("verdicts"), list):
            first = claude["cell_e"]["verdicts"][0] if claude["cell_e"]["verdicts"] else None
            if isinstance(first, dict):
                claude_e_v = {v["idx"]: v["verdict"] for v in claude["cell_e"]["verdicts"]}
            else:
                claude_e_v = dict(zip(judge_idxs, claude["cell_e"]["verdicts"]))
    except Exception:
        claude_e_v = {}

    # Build per-question (B, E) human-derived correctness on the n=100 subset
    rows = []
    for it in items:
        if not it.get("rater_label_X","").strip(): continue
        kid = it["item_id"]
        u = key_by_id[kid]
        idx = u["lme_idx"]
        lab_X = lab_norm(it["rater_label_X"])
        lab_Y = lab_norm(it["rater_label_Y"])
        lab_pref = it.get("rater_preference","").strip().upper()
        if u["X_is_B"]:
            human_B, human_E = lab_X, lab_Y
            human_pref_B_over_E = (lab_pref == "X")
            human_pref_E_over_B = (lab_pref == "Y")
        else:
            human_B, human_E = lab_Y, lab_X
            human_pref_B_over_E = (lab_pref == "Y")
            human_pref_E_over_B = (lab_pref == "X")
        rows.append({
            "item_id": kid, "lme_idx": idx,
            "human_B": human_B, "human_E": human_E,
            "human_pref_B_over_E": human_pref_B_over_E,
            "human_pref_E_over_B": human_pref_E_over_B,
            "human_pref_tie": (lab_pref == "TIE"),
            "gpt_B": gpt_b.get(idx), "gpt_E": gpt_e.get(idx),
            "claude_E": claude_e_v.get(idx),
        })

    n = len(rows)
    print(f"=== Human audit n={n} (partial -> correct) ===")
    nB = sum(1 for r in rows if r["human_B"] is not None)
    nE = sum(1 for r in rows if r["human_E"] is not None)
    print(f"  human B correct: {sum(1 for r in rows if r['human_B'] is True)}/{nB} = {sum(1 for r in rows if r['human_B'] is True)/max(nB,1):.3f}")
    print(f"  human E correct: {sum(1 for r in rows if r['human_E'] is True)}/{nE} = {sum(1 for r in rows if r['human_E'] is True)/max(nE,1):.3f}")
    delta = (sum(1 for r in rows if r['human_B'] is True)/max(nB,1)) - (sum(1 for r in rows if r['human_E'] is True)/max(nE,1))
    print(f"  Δ(B - E) human: {delta:+.4f}")

    print(f"\n=== Pairwise preference (explicit) ===")
    nB_pref = sum(1 for r in rows if r["human_pref_B_over_E"])
    nE_pref = sum(1 for r in rows if r["human_pref_E_over_B"])
    n_tie = sum(1 for r in rows if r["human_pref_tie"])
    print(f"  human prefers B (loose): {nB_pref}")
    print(f"  human prefers E (tight): {nE_pref}")
    print(f"  human ties: {n_tie}")
    print(f"  Δ(B-pref - E-pref): {nB_pref - nE_pref:+d}")
    # binomial sign test on non-tie subset
    discordant = nB_pref + nE_pref
    if discordant:
        # one-sided p (H0: P(B-pref) = 0.5)
        from math import comb
        k = max(nB_pref, nE_pref)
        n_d = discordant
        p_one = sum(comb(n_d, i) for i in range(k, n_d+1)) / 2**n_d
        print(f"  binomial sign test on n={discordant} non-ties: p_one_sided={p_one:.4f}")

    print(f"\n=== Paired McNemar B vs E (human-derived correctness) ===")
    n_only_B = sum(1 for r in rows if r["human_B"] is True and r["human_E"] is False)
    n_only_E = sum(1 for r in rows if r["human_E"] is True and r["human_B"] is False)
    n_both_C = sum(1 for r in rows if r["human_B"] is True and r["human_E"] is True)
    n_both_I = sum(1 for r in rows if r["human_B"] is False and r["human_E"] is False)
    disc = n_only_B + n_only_E
    chi2 = (abs(n_only_B - n_only_E) - 1)**2 / disc if disc else 0
    print(f"  only-B-correct: {n_only_B}, only-E-correct: {n_only_E}, both-C: {n_both_C}, both-I: {n_both_I}")
    print(f"  McNemar χ² (cont.corr.): {chi2:.2f}, sig at α=0.05 if > 3.841")

    print(f"\n=== Agreement: human vs gpt-4o-mini per cell ===")
    for cell in ("B", "E"):
        agree = total = 0
        for r in rows:
            hum = r[f"human_{cell}"]
            j = r[f"gpt_{cell}"]
            if hum is None or j not in ("CORRECT", "INCORRECT"): continue
            total += 1
            if hum == (j == "CORRECT"): agree += 1
        print(f"  cell {cell}: {agree}/{total} = {agree/max(total,1):.2%}")

    if claude_e_v:
        print(f"\n=== Agreement: human vs Claude-Sonnet-4.5 per cell ===")
        for cell in ("E",):
            agree = total = 0
            for r in rows:
                hum = r[f"human_{cell}"]
                j = r.get(f"claude_{cell}")
                if hum is None or j not in ("CORRECT", "INCORRECT"): continue
                total += 1
                if hum == (j == "CORRECT"): agree += 1
            print(f"  cell {cell}: {agree}/{total} = {agree/max(total,1):.2%}")

    # Also: 3-judge consensus rate (human, gpt, claude all agree on E)
    if claude_e_v:
        print(f"\n=== Cell E: 3-judge agreement rate ===")
        n_3 = 0; n_total = 0
        for r in rows:
            hum = r["human_E"]; gpt = r["gpt_E"]; cla = r.get("claude_E")
            if hum is None or gpt not in ("CORRECT","INCORRECT") or cla not in ("CORRECT","INCORRECT"): continue
            n_total += 1
            if hum == (gpt == "CORRECT") == (cla == "CORRECT"): n_3 += 1
        print(f"  all-3-agree: {n_3}/{n_total} = {n_3/max(n_total,1):.2%}")

    # Save scored result
    out = {
        "n_items_scored": n,
        "human_B_correct_rate": sum(1 for r in rows if r["human_B"] is True)/max(nB,1),
        "human_E_correct_rate": sum(1 for r in rows if r["human_E"] is True)/max(nE,1),
        "human_pref_B_over_E_count": nB_pref,
        "human_pref_E_over_B_count": nE_pref,
        "human_pref_tie_count": n_tie,
        "mcnemar_only_B": n_only_B, "mcnemar_only_E": n_only_E,
        "mcnemar_chi2_continuity": chi2,
        "rows": rows,
    }
    out_path = HA / "results_human_audit_n100_scored.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
