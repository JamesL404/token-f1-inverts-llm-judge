"""Run cell J (flat-BM25 + FCS) on the test-250 indices using held-out FCS prompts.

Only knowledge-update's prompt changes ("at most 3 words" -> "at most 4 words"),
because that is the only category whose calibration-half SurfaceForm differs in a
render-relevant way from the full-data profile (see results_profile_comparison.json).

Compare resulting token-F1 against the original cell J run (full-data FCS) on the
same 250 test indices to measure the calibration-on-test leakage component of the
+0.082 cell-J vs cell-E gain.
"""
from __future__ import annotations
import json, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Patch FCS prompt directory: copy original prompts, override KU with 4-word version.
PROMPTS_DIR = ROOT / "src/recipe/prompts"
HELDOUT_DIR = Path(tempfile.mkdtemp(prefix="fcs_heldout_"))
for f in PROMPTS_DIR.glob("*.txt"):
    shutil.copy(f, HELDOUT_DIR / f.name)
ku_orig = (PROMPTS_DIR / "lme_knowledge_update_fcs.txt").read_text()
ku_heldout = ku_orig.replace("at most 3 words", "at most 4 words")
assert ku_heldout != ku_orig, "expected text 'at most 3 words' to be in KU FCS prompt"
(HELDOUT_DIR / "lme_knowledge_update_fcs.txt").write_text(ku_heldout)
print(f"held-out FCS prompts in {HELDOUT_DIR}")
print("KU prompt diff: 'at most 3 words' -> 'at most 4 words' (cal-half p75=4, full p75=3)")

# Override prompt loader to read from the held-out dir for the run.
import src.recipe.prompts as prompts_module
_orig_load = prompts_module.load_template
def patched_load(family, category):
    fname = prompts_module.TEMPLATE_FILES.get((family, category))
    if family == "longmemeval-fcs" and fname:
        return (HELDOUT_DIR / fname).read_text()
    return _orig_load(family, category)
prompts_module.load_template = patched_load

# Build cell J pipeline (with patched FCS prompts) and run on test 250 only.
from src.longmemeval import iter_questions, load_questions
from src.recipe.recipe import CRAG
from src.recipe.generator import get_generator
from src.eval import f1_single
from src.recipe.router import RuleBasedRouter

def build_crag(*, benchmark, generator, retrieval_mode, prompt_variant="tight", template_category_override=None):
    return CRAG(
        benchmark=benchmark, router=RuleBasedRouter(benchmark), generator=generator,
        top_k=5, retrieval_mode=retrieval_mode, prompt_variant=prompt_variant,
        template_category_override=template_category_override,
    )

# Reuse the test indices saved by the profile-comparison run.
prof = json.load(open(ROOT / "experiments/crag-10-fcs-heldout/results_profile_comparison.json"))
test_idx = set(prof["test_indices"])
print(f"test_idx: n={len(test_idx)}")

generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
crag = build_crag(
    benchmark="longmemeval",
    generator=generator,
    retrieval_mode="flat",
    prompt_variant="tight-fcs",
)

questions = list(iter_questions(load_questions()))
test_questions = [q for i, q in enumerate(questions) if i in test_idx]
print(f"running cell J on test {len(test_questions)} questions with held-out FCS prompts")

predictions = []
import time
t0 = time.time()
for k, q in enumerate(test_questions):
    rec = crag.answer(q.question, q)
    pred = rec.prediction.strip()
    f1 = f1_single(pred, q.answer)
    predictions.append({
        "question_id": q.question_id, "question_type": q.question_type,
        "gold_label": q.label, "question": q.question, "gold_answer": q.answer,
        "prediction": pred, "completion_chars": len(pred), "f1": f1,
    })
    if (k+1) % 50 == 0:
        print(f"  [{k+1}/{len(test_questions)}] {time.time()-t0:.0f}s F1={sum(p['f1'] for p in predictions)/len(predictions):.4f}")

n = len(predictions)
agg = sum(p["f1"] for p in predictions) / n
avg_chars = sum(p["completion_chars"] for p in predictions) / n
from collections import defaultdict
by_label = defaultdict(list)
for p in predictions: by_label[p["gold_label"]].append(p["f1"])
by_label_summary = {k: {"n": len(v), "f1": sum(v)/len(v)} for k, v in by_label.items()}

# Cell J reference on the SAME test indices, original FCS prompts
cellj = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"))
cellj_test = [p for i, p in enumerate(cellj["predictions"]) if i in test_idx]
ref_agg = sum(p["f1"] for p in cellj_test) / len(cellj_test)
ref_chars = sum(p["completion_chars"] for p in cellj_test) / len(cellj_test) if cellj_test and "completion_chars" in cellj_test[0] else None
ref_by_label = defaultdict(list)
for p in cellj_test: ref_by_label[p["gold_label"]].append(p["f1"])
ref_by_label_summary = {k: {"n": len(v), "f1": sum(v)/len(v)} for k, v in ref_by_label.items()}

# Cell E reference (tight-per-cat, baseline being compared against)
celle = json.load(open(ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"))
celle_test = [p for i, p in enumerate(celle["predictions"]) if i in test_idx]
celle_test_agg = sum(p["f1"] for p in celle_test) / len(celle_test)

print("\n=== HELD-OUT FCS RESULT (test n=250) ===")
print(f"  cell J held-out (cal-only FCS) F1:   {agg:.4f}  avg_chars={avg_chars:.1f}")
print(f"  cell J reference (full FCS) F1:      {ref_agg:.4f}  avg_chars={ref_chars}")
print(f"  cell E reference (tight-per-cat) F1: {celle_test_agg:.4f}")
print(f"  Δ(J_heldout - J_ref):    {agg - ref_agg:+.4f}  (residual after held-out)")
print(f"  Δ(J_heldout - E):        {agg - celle_test_agg:+.4f}  (held-out FCS gain)")
print(f"  Δ(J_ref - E):            {ref_agg - celle_test_agg:+.4f}  (full-data FCS gain on test 250)")
print(f"\nLeakage component (J_ref - J_heldout): {ref_agg - agg:+.4f}")
print(f"Held-out-defensible FCS gain (J_heldout - E): {agg - celle_test_agg:+.4f}")
print(f"  vs full-data FCS gain on full n=500 (J - E): +0.082")

out = {
    "n_test": n,
    "test_indices": sorted(test_idx),
    "j_heldout_f1": agg,
    "j_heldout_avg_chars": avg_chars,
    "j_heldout_by_label": by_label_summary,
    "j_reference_f1": ref_agg,
    "j_reference_by_label": ref_by_label_summary,
    "e_reference_f1": celle_test_agg,
    "delta_heldout_minus_full_J": agg - ref_agg,
    "delta_heldout_minus_E": agg - celle_test_agg,
    "delta_full_minus_E_on_test": ref_agg - celle_test_agg,
    "leakage_component": ref_agg - agg,
    "predictions": predictions,
}
out_path = ROOT / "experiments/crag-10-fcs-heldout/results_heldout_cellj.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nwrote {out_path}")
