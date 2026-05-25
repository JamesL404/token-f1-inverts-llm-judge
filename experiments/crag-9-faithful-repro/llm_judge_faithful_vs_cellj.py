"""Paired LLM-judge on the three faithful-repro vs cell-J cell pairs.

For each of (Mem0, HippoRAG, A-MEM with evolution) vs cell J on the same
n=200 stratified LongMemEval subset, runs gpt-4o-mini and Claude-Sonnet-4.5
as paired correctness judges and computes paired McNemar.

Predictions are paired by question_id (not positional index), because the
faithful-repro JSONs were drawn from a stratified n=200 subset of LongMemEval
oracle while cell J covers n=500.

Output: experiments/crag-9-faithful-repro/results_judge_<system>_vs_cellj.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a gold (reference) answer, and a system's predicted answer. Your job is to decide whether the prediction conveys the same factual answer as the gold, regardless of length or phrasing.

Question: {question}

Gold answer: {gold}

Predicted answer: {pred}

Reply with exactly one token: CORRECT or INCORRECT.
- CORRECT: the prediction conveys the same factual answer as the gold (semantically equivalent; minor format differences are fine).
- INCORRECT: the prediction is factually wrong, contradictory, missing, or refuses when the gold has an answer.

Reply (CORRECT or INCORRECT only):"""


def load_env(env_path: Path) -> None:
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


def judge_openai(client, model, question, gold, pred):
    prompt = JUDGE_PROMPT.format(question=str(question)[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        r = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=8, temperature=0.0)
        v = r.choices[0].message.content.strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception as e:
        return f"ERROR:{str(e)[:80]}"


def judge_anthropic(client, model, question, gold, pred):
    prompt = JUDGE_PROMPT.format(question=str(question)[:600], gold=str(gold)[:600], pred=str(pred)[:600])
    try:
        r = client.messages.create(model=model, max_tokens=8, temperature=0.0, messages=[{"role":"user","content":prompt}])
        v = "".join(b.text for b in r.content if hasattr(b,"text")).strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception as e:
        return f"ERROR:{str(e)[:80]}"


def mcnemar(b_only_correct, c_only_correct):
    """Paired McNemar with continuity correction. Returns (chi2, two_sided_p)."""
    n = b_only_correct + c_only_correct
    if n == 0: return 0.0, 1.0
    chi2 = (abs(b_only_correct - c_only_correct) - 1) ** 2 / n if n > 0 else 0.0
    chi2 = max(0.0, chi2)
    # one-tailed via binomial-like (large-sample approximation): p = chi2-survival 1df
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2.0))  # two-sided survival of chi2-1
    return chi2, p


def run_pair(sys_path, cellj_path, label, oai, anth, oai_model, anth_model, out_path):
    sys_d = json.load(open(sys_path))
    j_d = json.load(open(cellj_path))
    sys_by_qid = {p["question_id"]: p for p in sys_d["predictions"]}
    j_by_qid   = {p["question_id"]: p for p in j_d["predictions"]}
    qids = sorted(set(sys_by_qid) & set(j_by_qid))
    print(f"\n=== {label}: paired n = {len(qids)} ===")
    rows = []
    sys_oai_correct = 0; j_oai_correct = 0
    sys_ant_correct = 0; j_ant_correct = 0
    b_only_oai = c_only_oai = both_oai = neither_oai = 0
    b_only_ant = c_only_ant = both_ant = neither_ant = 0
    t0 = time.time()
    for k, qid in enumerate(qids):
        s = sys_by_qid[qid]; jr = j_by_qid[qid]
        q = s["question"]; gold = s["gold_answer"]
        s_pred = s["prediction"]; j_pred = jr["prediction"]

        s_oai = judge_openai(oai, oai_model, q, gold, s_pred)
        j_oai = judge_openai(oai, oai_model, q, gold, j_pred)
        s_ant = judge_anthropic(anth, anth_model, q, gold, s_pred)
        j_ant = judge_anthropic(anth, anth_model, q, gold, j_pred)

        sys_oai_correct += (s_oai == "CORRECT")
        j_oai_correct   += (j_oai == "CORRECT")
        sys_ant_correct += (s_ant == "CORRECT")
        j_ant_correct   += (j_ant == "CORRECT")

        sb, jb = (s_oai=="CORRECT"), (j_oai=="CORRECT")
        if sb and not jb: b_only_oai += 1
        elif jb and not sb: c_only_oai += 1
        elif sb and jb: both_oai += 1
        else: neither_oai += 1

        sa, ja = (s_ant=="CORRECT"), (j_ant=="CORRECT")
        if sa and not ja: b_only_ant += 1
        elif ja and not sa: c_only_ant += 1
        elif sa and ja: both_ant += 1
        else: neither_ant += 1

        rows.append({"qid": qid, "label": s.get("gold_label",""),
                     "sys_oai": s_oai, "j_oai": j_oai, "sys_ant": s_ant, "j_ant": j_ant})
        if (k+1) % 25 == 0:
            print(f"  [{k+1}/{len(qids)}] {time.time()-t0:.0f}s  oai sys={sys_oai_correct} j={j_oai_correct} | ant sys={sys_ant_correct} j={j_ant_correct}", flush=True)

    n = len(qids)
    sys_oai_acc = sys_oai_correct/n; j_oai_acc = j_oai_correct/n
    sys_ant_acc = sys_ant_correct/n; j_ant_acc = j_ant_correct/n
    chi2_oai, p_oai = mcnemar(b_only_oai, c_only_oai)
    chi2_ant, p_ant = mcnemar(b_only_ant, c_only_ant)
    out = {
        "system": label, "n_paired": n,
        "openai_judge": {"model": oai_model, "sys_acc": sys_oai_acc, "j_acc": j_oai_acc, "delta": sys_oai_acc-j_oai_acc,
                         "discordant": {"sys_only": b_only_oai, "j_only": c_only_oai}, "concordant": {"both": both_oai, "neither": neither_oai},
                         "mcnemar_chi2": chi2_oai, "mcnemar_p_two_sided": p_oai},
        "anthropic_judge": {"model": anth_model, "sys_acc": sys_ant_acc, "j_acc": j_ant_acc, "delta": sys_ant_acc-j_ant_acc,
                            "discordant": {"sys_only": b_only_ant, "j_only": c_only_ant}, "concordant": {"both": both_ant, "neither": neither_ant},
                            "mcnemar_chi2": chi2_ant, "mcnemar_p_two_sided": p_ant},
        "verdicts": rows,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"  saved {out_path}")
    print(f"  GPT-4o-mini  : sys={sys_oai_acc:.3f} cellJ={j_oai_acc:.3f} Δ={sys_oai_acc-j_oai_acc:+.3f} McN χ²={chi2_oai:.2f} p={p_oai:.3f}")
    print(f"  Claude-S-4.5 : sys={sys_ant_acc:.3f} cellJ={j_ant_acc:.3f} Δ={sys_ant_acc-j_ant_acc:+.3f} McN χ²={chi2_ant:.2f} p={p_ant:.3f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["mem0","hipporag","amem"])
    parser.add_argument("--oai-model", default="gpt-4o-mini")
    parser.add_argument("--anth-model", default="claude-sonnet-4-5")
    args = parser.parse_args()

    load_env(ROOT/".env")
    if "OPENAI_API_KEY" not in os.environ: print("ERROR: OPENAI_API_KEY missing"); sys.exit(1)
    if "ANTHROPIC_API_KEY" not in os.environ: print("ERROR: ANTHROPIC_API_KEY missing"); sys.exit(1)

    from openai import OpenAI
    import anthropic
    oai = OpenAI(); anth = anthropic.Anthropic()

    cellj = ROOT/"experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json"
    paths = {
        "mem0":     ROOT/"experiments/crag-9-faithful-repro/results_mem0_lme_oracle_n200.json",
        "hipporag": ROOT/"experiments/crag-9-faithful-repro/results_hipporag_lme_oracle_n200.json",
        "amem":     ROOT/"experiments/crag-9-faithful-repro/results_amem_lme_oracle_n200.json",
    }
    for s in args.systems:
        if s not in paths: print(f"WARN unknown {s}"); continue
        out = ROOT/f"experiments/crag-9-faithful-repro/results_judge_{s}_vs_cellj.json"
        run_pair(paths[s], cellj, s, oai, anth, args.oai_model, args.anth_model, out)


if __name__ == "__main__":
    main()
