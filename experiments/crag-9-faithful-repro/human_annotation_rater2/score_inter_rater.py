"""Inter-rater analysis: author (rater 1) vs outside rater (rater 2).

Loads:
  - Rater 1 labels:  human_annotation/annotation_items.json (author, filled)
  - Rater 2 labels:  human_annotation_rater2/annotation_items_rater2.json (outside)
  - Unblind key:     human_annotation/UNBLIND_KEY_DO_NOT_SHOW_RATERS.json

Computes:
  - Per-output Cohen's κ (X-labels and Y-labels separately, then combined)
  - Cell-level agreement (after un-blinding X/Y → B/E)
  - Pairwise B-vs-E preference (each rater) with Wilson 95% CI on B-share-of-non-ties
  - McNemar χ² between rater 1 vs rater 2 preferences
  - Per-rater agreement with gpt-4o-mini judge (for cross-check)

Output: results_inter_rater.json
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
from math import sqrt
import statistics

ROOT = Path(__file__).resolve().parents[3]
R1_PATH  = ROOT / "experiments/crag-9-faithful-repro/human_annotation/annotation_items.json"
R2_PATH  = ROOT / "experiments/crag-9-faithful-repro/human_annotation_rater2/annotation_items_rater2.json"
KEY_PATH = ROOT / "experiments/crag-9-faithful-repro/human_annotation/UNBLIND_KEY_DO_NOT_SHOW_RATERS.json"
OUT_PATH = ROOT / "experiments/crag-9-faithful-repro/human_annotation_rater2/results_inter_rater.json"


def norm(lbl):
    if lbl is None: return None
    s = str(lbl).strip().lower()
    if s in ("c", "correct"): return "correct"
    if s in ("p", "partial"): return "partial"
    if s in ("i", "incorrect", "wrong"): return "incorrect"
    return s


def norm_pref(p):
    if p is None: return None
    s = str(p).strip().lower()
    if s == "x": return "X"
    if s == "y": return "Y"
    if s == "tie": return "tie"
    return s


def cohens_kappa(a, b, labels):
    """Cohen's κ for two raters on the same items, given a list of labels."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0: return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # marginals
    a_counts = Counter(a); b_counts = Counter(b)
    pe = sum((a_counts[l] / n) * (b_counts[l] / n) for l in labels)
    if pe == 1.0: return None
    return (po - pe) / (1 - pe)


def wilson_ci(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    halfw = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - halfw, center + halfw)


def mcnemar_continuity(b_only, c_only):
    n = b_only + c_only
    if n == 0: return 0.0, 1.0
    chi2 = (abs(b_only - c_only) - 1) ** 2 / n
    chi2 = max(0.0, chi2)
    from math import erfc, sqrt as msqrt
    p = erfc(msqrt(chi2 / 2.0))
    return chi2, p


def main():
    r1 = json.load(open(R1_PATH))
    r2_obj = json.load(open(R2_PATH))
    r2 = r2_obj["items"]
    key = {k["item_id"]: k for k in json.load(open(KEY_PATH))["key"]}

    # align by item_id
    r1_by_id = {it["item_id"]: it for it in r1}
    r2_by_id = {it["item_id"]: it for it in r2}
    ids = sorted(set(r1_by_id) & set(r2_by_id))
    assert len(ids) == 100, f"expected 100 paired items, got {len(ids)}"

    # ---- per-output label agreement (Cohen's κ) ----
    LABELS = ["correct", "partial", "incorrect"]
    r1_x = [norm(r1_by_id[i].get("rater_label_X")) for i in ids]
    r2_x = [norm(r2_by_id[i].get("rater_label_X")) for i in ids]
    r1_y = [norm(r1_by_id[i].get("rater_label_Y")) for i in ids]
    r2_y = [norm(r2_by_id[i].get("rater_label_Y")) for i in ids]

    kappa_x = cohens_kappa(r1_x, r2_x, LABELS)
    kappa_y = cohens_kappa(r1_y, r2_y, LABELS)
    # combined: pool all (X-label, Y-label) per item across raters
    r1_all = r1_x + r1_y
    r2_all = r2_x + r2_y
    kappa_combined = cohens_kappa(r1_all, r2_all, LABELS)

    raw_agree_x = sum(1 for a, b in zip(r1_x, r2_x) if a == b) / len(ids)
    raw_agree_y = sum(1 for a, b in zip(r1_y, r2_y) if a == b) / len(ids)
    raw_agree_combined = sum(1 for a, b in zip(r1_all, r2_all) if a == b) / len(r1_all)

    # ---- translate preferences X/Y → B/E using unblind key ----
    def to_be(item_id, pref):
        if pref is None: return None
        if pref == "tie": return "tie"
        k = key[item_id]
        if k["X_is_B"]:
            return "B" if pref == "X" else "E"
        else:
            return "B" if pref == "Y" else "E"

    r1_pref_be = [to_be(i, norm_pref(r1_by_id[i].get("rater_preference"))) for i in ids]
    r2_pref_be = [to_be(i, norm_pref(r2_by_id[i].get("rater_preference"))) for i in ids]

    def pref_summary(pref_be_list):
        c = Counter(pref_be_list)
        nonties = c.get("B", 0) + c.get("E", 0)
        return {
            "B": c.get("B", 0),
            "E": c.get("E", 0),
            "tie": c.get("tie", 0),
            "nonties": nonties,
            "B_share_of_nonties": c.get("B", 0) / nonties if nonties else None,
            "wilson_95_ci_for_B_share": list(wilson_ci(c.get("B", 0), nonties)) if nonties else None,
        }

    r1_pref_sum = pref_summary(r1_pref_be)
    r2_pref_sum = pref_summary(r2_pref_be)

    # ---- McNemar between raters on preference (B vs not-B; ignore ties on either side) ----
    # Restrict to items where both raters expressed a preference (not tie)
    both_pref = [(p1, p2) for p1, p2 in zip(r1_pref_be, r2_pref_be) if p1 in ("B", "E") and p2 in ("B", "E")]
    r1_B_r2_E = sum(1 for p1, p2 in both_pref if p1 == "B" and p2 == "E")
    r1_E_r2_B = sum(1 for p1, p2 in both_pref if p1 == "E" and p2 == "B")
    chi2_pref, p_pref = mcnemar_continuity(r1_B_r2_E, r1_E_r2_B)

    # ---- Agreement between raters on preferences (including ties) ----
    pref_agreement = sum(1 for p1, p2 in zip(r1_pref_be, r2_pref_be) if p1 == p2) / len(ids)
    kappa_pref = cohens_kappa(r1_pref_be, r2_pref_be, ["B", "E", "tie"])

    # ---- 84-tie decomposition (rater 2 version) ----
    # For items where rater 2 said "tie", what were the per-output labels?
    r2_tie_decomp = Counter()
    for i in ids:
        if norm_pref(r2_by_id[i].get("rater_preference")) == "tie":
            x = norm(r2_by_id[i].get("rater_label_X"))
            y = norm(r2_by_id[i].get("rater_label_Y"))
            pair = tuple(sorted([x or "NA", y or "NA"]))
            if pair == ("correct", "correct"): r2_tie_decomp["both_correct"] += 1
            elif pair == ("incorrect", "incorrect"): r2_tie_decomp["both_incorrect"] += 1
            elif pair == ("partial", "partial"): r2_tie_decomp["both_partial"] += 1
            elif pair == ("correct", "partial"): r2_tie_decomp["correct_partial"] += 1
            elif pair == ("correct", "incorrect"): r2_tie_decomp["correct_incorrect_mixed"] += 1
            elif pair == ("incorrect", "partial"): r2_tie_decomp["partial_incorrect"] += 1
            else: r2_tie_decomp["other"] += 1

    out = {
        "n_paired_items": len(ids),
        "label_agreement": {
            "kappa_X_outputs": kappa_x,
            "kappa_Y_outputs": kappa_y,
            "kappa_combined": kappa_combined,
            "raw_agreement_X": raw_agree_x,
            "raw_agreement_Y": raw_agree_y,
            "raw_agreement_combined": raw_agree_combined,
            "n_label_decisions": len(r1_all),
        },
        "preference_summary": {
            "rater_1_author": r1_pref_sum,
            "rater_2_outside": r2_pref_sum,
        },
        "preference_agreement": {
            "raw_agreement_3way": pref_agreement,
            "kappa_preference": kappa_pref,
        },
        "mcnemar_between_raters_on_BE_preference": {
            "r1_B_r2_E": r1_B_r2_E,
            "r1_E_r2_B": r1_E_r2_B,
            "n_both_expressed_pref": len(both_pref),
            "chi2_continuity": chi2_pref,
            "p_two_sided": p_pref,
        },
        "rater_2_tie_decomposition_n86": dict(r2_tie_decomp),
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
