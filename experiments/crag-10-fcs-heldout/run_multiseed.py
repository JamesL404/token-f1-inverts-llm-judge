"""Multi-seed held-out FCS calibration ablation.

Quantifies how much of the +0.082 FCS token-F1 gain is attributable to
direct test-gold leakage by varying all three render-relevant SurfaceForm
fields (p75 word cap, dominant answer form, abstention strings) across
random calibration / test splits.

For each of 5 random 50/50 splits of LongMemEval oracle (seeds 20260506
through 20260510):
  - Profile gold answers on the calibration half only
  - Record p75_words, dominant_form, abstention_strings per category
  - Generate held-out FCS prompts per category from the cal-half profile
    (regenerate templates if dominant_form or abstention_strings change)
  - Run cell J on the corresponding test half with the cal-half FCS prompts
  - Compare to cell J on the same test half with the full-data FCS prompts
  - Report mean +/- SD of the leakage component across the 5 seeds
"""
from __future__ import annotations
import json, random, sys, tempfile, shutil, os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.longmemeval import iter_questions, load_questions
from src.recipe.fcs import _classify_form, profile_gold_answers


def render_fcs_prompt(category: str, p75_words: int, dominant_form: str, abstention_strs: list[str]) -> str:
    """Programmatically render an FCS prompt from a SurfaceForm profile.

    Mirrors the deployed lme_*_fcs.txt templates but parameterized.
    """
    cap = max(p75_words, 2)  # floor

    if category == "knowledge-update":
        return (
            f"You are answering a question about a long conversation. Return ONLY the minimal "
            f"final answer. Output at most {cap} words. Do not explain. Do not add commentary.\n\n"
            f"Excerpts:\n{{retrieved_turns_with_timestamps}}\n\nQuestion: {{question}}\nAnswer:\n"
        )
    if category == "single-session":
        return (
            f"You are answering a question about a long conversation. Return ONLY the minimal "
            f"final answer. Output at most {cap} words. Do not explain. Do not add commentary.\n\n"
            f"Excerpts:\n{{retrieved_turns_with_timestamps}}\n\nQuestion: {{question}}\nAnswer:\n"
        )
    if category == "multi-session":
        return (
            f"You are answering a question about a long conversation. Return ONLY the minimal "
            f"final answer. Output at most {cap} words. Do not explain. Do not add commentary.\n\n"
            f"Excerpts:\n{{retrieved_turns_with_timestamps}}\n\nQuestion: {{question}}\nAnswer:\n"
        )
    if category == "temporal":
        return (
            f"You are answering a temporal question about a long conversation. Each excerpt is "
            f"prefixed with the SESSION DATE. Return ONLY the minimal final answer. Output at "
            f"most {cap} words, in \"D Month YYYY\" format or as a brief time interval. Do not "
            f"explain.\n\nExcerpts (each with session date):\n{{retrieved_turns_with_timestamps}}\n\n"
            f"Question: {{question}}\nAnswer:\n"
        )
    raise ValueError(f"Unknown category: {category}")


def profile_split(qs, indices):
    pairs = [(qs[i].label, qs[i].answer) for i in indices]
    return profile_gold_answers(pairs, benchmark="lme-split")


def main(seeds=[20260506, 20260507, 20260508, 20260509, 20260510]):
    qs = list(iter_questions(load_questions()))
    n = len(qs)

    # Compute the full-data profile once (fixed, for reference)
    full_profile = profile_split(qs, list(range(n)))
    full_p75 = {cat: sf.p75_words for cat, sf in full_profile.items()}
    full_form = {cat: sf.inferred_form for cat, sf in full_profile.items()}
    full_abst = {cat: tuple(sf.abstention_strings) for cat, sf in full_profile.items()}
    print("=== Full-data profile ===")
    for cat in sorted(full_profile):
        print(f"  {cat:<20} p75={full_p75[cat]} form={full_form[cat]} abstention={full_abst[cat]}")

    # Per-seed cal-half profile + render-relevant differences
    seed_profiles = {}
    for seed in seeds:
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        cal_idx = sorted(idx[: n // 2])
        test_idx = sorted(idx[n // 2:])
        prof = profile_split(qs, cal_idx)
        diffs = {}
        for cat in sorted(prof):
            sf = prof[cat]
            diffs[cat] = {
                "p75_cal": sf.p75_words, "p75_full": full_p75[cat],
                "p75_diff": sf.p75_words - full_p75[cat],
                "form_cal": sf.inferred_form, "form_full": full_form[cat],
                "form_changed": sf.inferred_form != full_form[cat],
                "abst_cal": tuple(sf.abstention_strings),
                "abst_changed": tuple(sf.abstention_strings) != full_abst[cat],
            }
        seed_profiles[seed] = {
            "cal_idx": cal_idx,
            "test_idx": test_idx,
            "diffs": diffs,
        }
        print(f"\nseed={seed}:")
        for cat, d in diffs.items():
            mark = "*" if (d["p75_diff"] != 0 or d["form_changed"] or d["abst_changed"]) else " "
            print(f"  {mark} {cat:<20} ∆p75={d['p75_diff']:+d}  form: {d['form_cal']} (full: {d['form_full']})  abst_changed={d['abst_changed']}")

    # Save profile-comparison output
    out_path = ROOT / "experiments/crag-10-fcs-heldout/results_multiseed_profile.json"
    out_path.write_text(json.dumps({"seeds": seeds, "full_profile_p75": full_p75, "full_form": full_form,
                                     "seed_profiles": {str(s): {"cal_idx": v["cal_idx"], "test_idx": v["test_idx"], "diffs": v["diffs"]}
                                                       for s, v in seed_profiles.items()}}, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # Now run cell J on the test half of each seed, twice:
    #   (1) with full-data FCS prompts (the deployed prompts)
    #   (2) with cal-half FCS prompts (regenerated per category from the cal-half profile)
    # Compare: leakage component = J_full(test) − J_cal_only(test)

    from src.recipe.recipe import CRAG
    from src.recipe.generator import get_generator
    from src.eval import f1_single
    from src.recipe.router import RuleBasedRouter
    import src.recipe.prompts as prompts_mod
    import time

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    leakage_per_seed = {}

    for seed in seeds:
        sp = seed_profiles[seed]
        test_idx = set(sp["test_idx"])
        test_questions = [q for i, q in enumerate(qs) if i in test_idx]
        print(f"\n=== seed={seed} cell J on test n={len(test_questions)} ===")

        # (a) Full-data FCS (deployed prompts) — patch loader to use the deployed *_fcs.txt
        orig_load = prompts_mod.load_template
        prompts_mod.load_template = orig_load  # already deployed
        crag = CRAG(benchmark="longmemeval", router=RuleBasedRouter("longmemeval"),
                    generator=generator, top_k=5, retrieval_mode="flat", prompt_variant="tight-fcs")

        f1_full = []
        t0 = time.time()
        for q in test_questions:
            r = crag.answer(q.question, q)
            f1_full.append(f1_single(r.prediction.strip(), q.answer))
        print(f"  full-FCS J F1 (test): {sum(f1_full)/len(f1_full):.4f}  ({time.time()-t0:.0f}s)")

        # (b) Cal-half FCS — generate held-out prompts from cal-half profile
        heldout_dir = Path(tempfile.mkdtemp(prefix=f"fcs_heldout_seed{seed}_"))
        # Copy non-FCS templates as-is
        for f in (ROOT / "src/recipe/prompts").glob("*.txt"):
            shutil.copy(f, heldout_dir / f.name)
        # Regenerate FCS prompts from cal-half profile
        cal_prof = profile_split(qs, sp["cal_idx"])
        for cat in cal_prof:
            sf = cal_prof[cat]
            if cat == "adversarial": continue
            text = render_fcs_prompt(cat, sf.p75_words, sf.inferred_form, sf.abstention_strings)
            (heldout_dir / f"lme_{cat.replace('-', '_')}_fcs.txt").write_text(text)

        def patched_load(family, category, _orig=orig_load):
            fname = prompts_mod.TEMPLATE_FILES.get((family, category))
            if family == "longmemeval-fcs" and fname:
                return (heldout_dir / fname).read_text()
            return _orig(family, category)
        prompts_mod.load_template = patched_load

        crag2 = CRAG(benchmark="longmemeval", router=RuleBasedRouter("longmemeval"),
                     generator=generator, top_k=5, retrieval_mode="flat", prompt_variant="tight-fcs")
        f1_cal = []
        t0 = time.time()
        for q in test_questions:
            r = crag2.answer(q.question, q)
            f1_cal.append(f1_single(r.prediction.strip(), q.answer))
        cal_avg = sum(f1_cal) / len(f1_cal)
        full_avg = sum(f1_full) / len(f1_full)
        print(f"  cal-half-FCS J F1 (test): {cal_avg:.4f}  ({time.time()-t0:.0f}s)")
        print(f"  leakage (full − cal): {full_avg - cal_avg:+.4f}")
        leakage_per_seed[seed] = {
            "full_FCS_F1_test": full_avg, "cal_FCS_F1_test": cal_avg,
            "leakage": full_avg - cal_avg,
            "n_test": len(test_questions),
        }

        # Restore loader before next seed
        prompts_mod.load_template = orig_load

    # Aggregate
    print("\n=== Multi-seed leakage summary ===")
    leakages = [v["leakage"] for v in leakage_per_seed.values()]
    import statistics
    mean = statistics.mean(leakages)
    sd = statistics.stdev(leakages) if len(leakages) > 1 else 0.0
    print(f"  n_seeds={len(seeds)}")
    print(f"  leakage mean ± SD: {mean:+.4f} ± {sd:.4f}")
    print(f"  leakage range: [{min(leakages):+.4f}, {max(leakages):+.4f}]")
    out2 = ROOT / "experiments/crag-10-fcs-heldout/results_multiseed_leakage.json"
    out2.write_text(json.dumps({
        "seeds": seeds,
        "leakage_per_seed": leakage_per_seed,
        "leakage_mean": mean, "leakage_sd": sd,
        "leakage_min": min(leakages), "leakage_max": max(leakages),
    }, indent=2))
    print(f"\nwrote {out2}")


if __name__ == "__main__":
    main()
