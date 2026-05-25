"""Held-out FCS calibration ablation (W4 from senior reviewer).

The reviewer concern: FCS prompts in cell J were calibrated on
LongMemEval gold answers; the +0.082 token-F1 gain (J vs E) is
"partially a direct measurement of test-set leakage" since the same
gold answers used to render FCS are scored against.

This experiment tests how much of the FCS gain is leakage:

  1. Split LongMemEval oracle 50/50 (seed=20260506) into calibration
     and test halves (~n=250 each).
  2. Profile gold answers on calibration half only — compute per-category
     SurfaceForm (median/p25/p75 words, dominant form, abstention strings,
     length cap).
  3. Compare to the full-data profile cell J was rendered against.
  4. If per-category profiles are essentially identical (the format-
     alignment hypothesis), this directly disproves the "leakage" reading
     of FCS: FCS captures stable surface-form regularities that any held-
     out split would surface the same way.
  5. If profiles differ in ways that would change rendered prompts,
     regenerate prompts from calibration-only profile and re-run cell J
     on the test half; report token-F1 vs the calibration-on-test
     reference (cell J on the same test indices using full-data FCS).

Output: experiments/crag-10-fcs-heldout/results.json with the profile
comparison + (if needed) the held-out cell J re-run.
"""
from __future__ import annotations
import json, random, statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.longmemeval import iter_questions, load_questions
from src.recipe.fcs import _classify_form, profile_gold_answers, SurfaceForm


def profile_from_pairs(pairs, label):
    return profile_gold_answers(pairs, benchmark=f"longmemeval-{label}")


def fcs_render_constraints(sf: SurfaceForm) -> dict:
    """Extract the constraint values that drive FCS prompt rendering.

    These are the values that, if they changed across calibration vs full,
    would change the rendered FCS prompt text. The current hand-rendered
    prompts (src/recipe/prompts/lme_*_fcs.txt) bake these values in:
      - "at most N words" (where N is roughly p75)
      - dominant answer form (short_span, date, label, etc.)
      - abstention strings (verbatim from gold)
    """
    return {
        "p25_words": sf.p25_words,
        "median_words": sf.median_words,
        "p75_words": sf.p75_words,
        "max_words": sf.max_words,
        "inferred_form": sf.inferred_form,
        "form_counts": sf.form_counts,
        "abstention_strings": sf.abstention_strings,
        "n": sf.n,
    }


def main():
    seed = 20260506
    rng = random.Random(seed)

    qs = list(iter_questions(load_questions()))
    print(f"LongMemEval oracle: {len(qs)} questions")

    # 50/50 split, deterministic
    indices = list(range(len(qs)))
    rng.shuffle(indices)
    cal_idx = sorted(indices[: len(qs) // 2])
    test_idx = sorted(indices[len(qs) // 2 :])
    print(f"calibration n={len(cal_idx)}, test n={len(test_idx)}")

    pairs_cal = [(qs[i].label, qs[i].answer) for i in cal_idx]
    pairs_test = [(qs[i].label, qs[i].answer) for i in test_idx]
    pairs_full = [(q.label, q.answer) for q in qs]

    profile_cal = profile_from_pairs(pairs_cal, "cal")
    profile_test = profile_from_pairs(pairs_test, "test")
    profile_full = profile_from_pairs(pairs_full, "full")

    print("\n=== Per-category SurfaceForm comparison ===")
    print(f"{'category':<20} {'split':<5} {'n':>4} {'p25':>4} {'med':>4} {'p75':>4} {'max':>4} {'form':<14}")
    rows = []
    cats = sorted(set(profile_cal) | set(profile_test) | set(profile_full))
    for cat in cats:
        for label, prof in (("full", profile_full), ("cal", profile_cal), ("test", profile_test)):
            sf = prof.get(cat)
            if sf is None:
                continue
            print(f"  {cat:<20} {label:<5} {sf.n:>4} {sf.p25_words:>4} {sf.median_words:>4} {sf.p75_words:>4} {sf.max_words:>4} {sf.inferred_form:<14}")
        rows.append({
            "category": cat,
            "full": fcs_render_constraints(profile_full[cat]) if cat in profile_full else None,
            "cal": fcs_render_constraints(profile_cal[cat]) if cat in profile_cal else None,
            "test": fcs_render_constraints(profile_test[cat]) if cat in profile_test else None,
        })

    # Diff analysis: does any FCS-rendering-relevant value differ between
    # cal and full (or cal and test) in a way that would change the
    # rendered prompt?
    print("\n=== Render-relevant differences (cal vs full) ===")
    diffs = []
    for cat in cats:
        c_cal = profile_cal.get(cat)
        c_full = profile_full.get(cat)
        if c_cal is None or c_full is None:
            continue
        # The current FCS prompts cap at "at most N words" where N comes
        # from p75 (rounded). Check delta on p75.
        delta_p75 = c_cal.p75_words - c_full.p75_words
        delta_med = c_cal.median_words - c_full.median_words
        same_form = c_cal.inferred_form == c_full.inferred_form
        same_abst = set(c_cal.abstention_strings) == set(c_full.abstention_strings)
        diffs.append({
            "category": cat,
            "delta_p75": delta_p75,
            "delta_median": delta_med,
            "same_dominant_form": same_form,
            "same_abstention_strings": same_abst,
            "would_change_prompt": (
                abs(delta_p75) >= 1 or not same_form or not same_abst
            ),
        })
        print(f"  {cat:<20} Δp75={delta_p75:+d}  Δmed={delta_med:+d}  same_form={same_form}  same_abst={same_abst}  changes_prompt={diffs[-1]['would_change_prompt']}")

    n_changed = sum(1 for d in diffs if d["would_change_prompt"])
    print(f"\nCategories whose FCS prompt would change under held-out calibration: {n_changed}/{len(diffs)}")

    out = {
        "seed": seed,
        "n_calibration": len(cal_idx),
        "n_test": len(test_idx),
        "n_full": len(qs),
        "calibration_indices": cal_idx,
        "test_indices": test_idx,
        "per_category_profiles": rows,
        "render_diffs_cal_vs_full": diffs,
        "n_categories_whose_prompt_would_change": n_changed,
    }

    out_path = ROOT / "experiments/crag-10-fcs-heldout/results_profile_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    if n_changed == 0:
        print("\n=== HELD-OUT FCS CALIBRATION RESULT ===")
        print("Per-category surface-form profiles on the calibration half (n=250)")
        print("are RENDER-EQUIVALENT to the full-data profiles cell J was built from:")
        print("  - same p75 word cap (the 'at most N words' constraint)")
        print("  - same dominant answer form per category")
        print("  - same abstention string set")
        print("Therefore the FCS prompts rendered from the held-out half would be")
        print("identical to the deployed FCS prompts, and the +0.082 token-F1 gain")
        print("(J vs E on full n=500) is NOT a direct measurement of test-set")
        print("leakage in the calibration-on-test sense. The format-alignment effect")
        print("captured by FCS is a stable per-category regularity, not a memorized")
        print("test-gold artifact.")
    else:
        print("\n=== HELD-OUT FCS CALIBRATION RESULT ===")
        print(f"{n_changed} categories' FCS prompts would change under held-out")
        print("calibration. To complete the ablation, regenerate prompts from")
        print("cal-only profile and re-run cell J on test indices.")
        print("(See render_diffs_cal_vs_full in results JSON.)")


if __name__ == "__main__":
    main()
