"""Truncate-and-rejudge ablation.

Goal: distinguish whether loose prompts win under LLM-judge because
(a) they contain more correct information (informational-content mechanism), or
(b) judges share a length bias toward longer answers (length-bias mechanism).

Method:
  - Take the n=200 paired judge indices from results_llm_judge_e_vs_j.json
  - For each judged index i:
      target_len_chars = len(cell_E_pred[i])
      truncated_loose[i] = cell_B_pred[i][:target_len_chars]
  - Re-run gpt-4o-mini judge on truncated_loose vs gold.
  - Compare to:
      original loose verdicts (cell B, judge_acc = 0.60)
      tight-per-cat verdicts (cell E, judge_acc = 0.53)
  - If truncated-loose still beats tight: mechanism is informational.
  - If truncated-loose loses to tight: mechanism is judge length-bias.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a gold (reference) answer, and a system's predicted answer. Your job is to decide whether the prediction conveys the same factual answer as the gold, regardless of length or phrasing.

Question: {question}

Gold answer: {gold}

Predicted answer: {pred}

Reply with exactly one token: CORRECT or INCORRECT.
- CORRECT: the prediction conveys the same factual answer as the gold (semantically equivalent; minor format differences are fine).
- INCORRECT: the prediction is factually wrong, contradictory, missing, or refuses when the gold has an answer.

Reply (CORRECT or INCORRECT only):"""


def judge_one(client, model, question, gold, pred):
    prompt = JUDGE_PROMPT.format(question=question[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return "CORRECT"
        if "INCORRECT" in verdict:
            return "INCORRECT"
        return "AMBIGUOUS"
    except Exception as e:
        return f"ERROR:{str(e)[:80]}"


def main():
    load_env(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    # Load existing E-vs-J judge file just to get the 200 indices
    judge_path = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"
    judge_data = json.load(open(judge_path))
    indices = sorted({v["idx"] for v in judge_data["cell_e"]["verdicts"]})
    print(f"loaded {len(indices)} judge indices, range [{min(indices)},{max(indices)}]")

    # Cell B = flat × loose
    cell_b_path = ROOT / "experiments/crag-7-ranking-shift/results_7B_lme_flat_loose.json"
    # Cell E = flat × tight-per-cat
    cell_e_path = ROOT / "experiments/crag-6-linking-baseline/results_6a_lme_flat.json"
    b_preds = json.load(open(cell_b_path))["predictions"]
    e_preds = json.load(open(cell_e_path))["predictions"]
    assert len(b_preds) == 500 and len(e_preds) == 500

    # Sanity check: questions match by index
    for idx in indices[:5]:
        assert b_preds[idx]["question"] == e_preds[idx]["question"], f"qmismatch idx={idx}"

    # Length stats on the 200 subset
    e_lens = [len(e_preds[i]["prediction"]) for i in indices]
    b_lens = [len(b_preds[i]["prediction"]) for i in indices]
    import statistics as st
    print(f"cell E lens: median={st.median(e_lens):.0f} mean={st.mean(e_lens):.0f}")
    print(f"cell B lens: median={st.median(b_lens):.0f} mean={st.mean(b_lens):.0f}")
    print(f"length ratio B/E: {st.mean(b_lens)/st.mean(e_lens):.2f}x")

    # Build truncated-loose
    truncated = []
    for idx in indices:
        b_pred = b_preds[idx]["prediction"]
        e_pred = e_preds[idx]["prediction"]
        target_len = len(e_pred)
        # Per-question matched truncation: cut B to E's length
        trunc_b = b_pred[:target_len]
        truncated.append({
            "idx": idx,
            "question": b_preds[idx]["question"],
            "gold": b_preds[idx]["gold_answer"],
            "b_pred_full": b_pred,
            "b_pred_trunc": trunc_b,
            "e_pred": e_pred,
            "len_b_full": len(b_pred),
            "len_e": len(e_pred),
            "len_b_trunc": len(trunc_b),
        })

    avg_full = sum(t["len_b_full"] for t in truncated) / len(truncated)
    avg_trunc = sum(t["len_b_trunc"] for t in truncated) / len(truncated)
    print(f"avg loose-full len: {avg_full:.0f} -> trunc to avg {avg_trunc:.0f} chars")

    # Run judge on truncated loose
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"

    out_path = ROOT / "experiments/crag-7-ranking-shift/results_truncate_rejudge.json"
    print(f"\njudging {len(truncated)} truncated-loose preds with {model}...")
    t0 = time.time()
    correct = 0; incorrect = 0; other = 0
    for k, item in enumerate(truncated):
        v = judge_one(client, model, item["question"], item["gold"], item["b_pred_trunc"])
        item["verdict_trunc"] = v
        if v == "CORRECT": correct += 1
        elif v == "INCORRECT": incorrect += 1
        else: other += 1
        if (k + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{k+1}/{len(truncated)}] {elapsed:.0f}s  acc_so_far={correct/(k+1):.3f}")

    n = len(truncated)
    acc_trunc_loose = correct / n
    print(f"\ntruncated-loose judge accuracy: {acc_trunc_loose:.3f}  ({correct}/{n})")
    print(f"  ambiguous/error: {other}")

    # Compare to existing cell B and cell E judge accuracies (re-load from full matrix)
    fm_path = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json"
    fm = json.load(open(fm_path))
    acc_loose_full = fm["cells"]["lme.B_flat_loose"]["judge_accuracy"]
    acc_tight = fm["cells"]["lme.E_flat_tight_per_cat"]["judge_accuracy"]
    print(f"\noriginal loose-full judge accuracy: {acc_loose_full:.3f}")
    print(f"tight-per-cat (cell E) judge accuracy: {acc_tight:.3f}")
    print(f"Δ(trunc-loose − tight): {acc_trunc_loose - acc_tight:+.3f}")
    print(f"Δ(loose-full − tight):  {acc_loose_full - acc_tight:+.3f}")
    print(f"compression effect on judge: {acc_loose_full - acc_trunc_loose:+.3f}")

    # Mechanism interpretation
    if acc_trunc_loose - acc_tight > 0.03:
        verdict = "INFORMATIONAL: truncated loose still beats tight; loose contains more correct info"
    elif acc_trunc_loose - acc_tight < -0.03:
        verdict = "LENGTH-BIAS: truncating kills loose's advantage; judges prefer longer answers"
    else:
        verdict = "MIXED: truncating partially eliminates loose's advantage"
    print(f"\nmechanism: {verdict}")

    # Paired McNemar against tight (need tight verdicts on same 200)
    tight_verdicts = {v["idx"]: v["verdict"] for v in judge_data["cell_e"]["verdicts"]}
    trunc_verdicts = {item["idx"]: item["verdict_trunc"] for item in truncated}
    common_idx = set(tight_verdicts) & set(trunc_verdicts)
    b_yes_t_no = 0  # trunc-loose CORRECT, tight INCORRECT (loose-better)
    b_no_t_yes = 0  # trunc-loose INCORRECT, tight CORRECT (tight-better)
    both_yes = 0; both_no = 0
    for i in common_idx:
        tv = tight_verdicts[i]; trv = trunc_verdicts[i]
        if tv == "CORRECT" and trv == "CORRECT": both_yes += 1
        elif tv == "INCORRECT" and trv == "INCORRECT": both_no += 1
        elif tv == "INCORRECT" and trv == "CORRECT": b_yes_t_no += 1
        elif tv == "CORRECT" and trv == "INCORRECT": b_no_t_yes += 1
    discordant = b_yes_t_no + b_no_t_yes
    if discordant > 0:
        mc_chi2 = (abs(b_yes_t_no - b_no_t_yes) - 1) ** 2 / discordant
    else:
        mc_chi2 = 0.0
    print(f"\nPaired McNemar (truncated-loose vs tight, n={len(common_idx)}):")
    print(f"  trunc-loose CORRECT, tight INCORRECT: {b_yes_t_no}")
    print(f"  trunc-loose INCORRECT, tight CORRECT: {b_no_t_yes}")
    print(f"  both CORRECT: {both_yes}, both INCORRECT: {both_no}")
    print(f"  McNemar χ² (cont.corr.) = {mc_chi2:.3f}  (sig at p<0.05 if χ² > 3.841)")

    summary = {
        "model": model,
        "n_judged": n,
        "indices_seed": 42,
        "cell_b_path": str(cell_b_path),
        "cell_e_path": str(cell_e_path),
        "judge_indices_source": str(judge_path),
        "len_stats": {
            "cell_b_full_mean_chars": avg_full,
            "cell_b_trunc_mean_chars": avg_trunc,
            "cell_e_mean_chars": sum(t["len_e"] for t in truncated)/n,
            "compression_ratio_b_full_over_e": avg_full / (sum(t["len_e"] for t in truncated)/n),
        },
        "judge_acc": {
            "trunc_loose": acc_trunc_loose,
            "loose_full": acc_loose_full,
            "tight_per_cat": acc_tight,
            "delta_trunc_minus_tight": acc_trunc_loose - acc_tight,
            "delta_full_minus_tight": acc_loose_full - acc_tight,
            "compression_effect": acc_loose_full - acc_trunc_loose,
        },
        "mcnemar_paired_trunc_vs_tight": {
            "n": len(common_idx),
            "trunc_only": b_yes_t_no,
            "tight_only": b_no_t_yes,
            "both_correct": both_yes,
            "both_incorrect": both_no,
            "chi2_continuity_corrected": mc_chi2,
            "significant_p005": mc_chi2 > 3.841,
        },
        "mechanism_verdict": verdict,
        "items": truncated,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
