"""LLM-judge cross-backbone replication of the FCS gain collapse.

For each of 4 non-primary backbones (Qwen2.5-3B/7B-Instruct,
Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct), run gpt-4o-mini and
Claude-Sonnet-4.5 as judges on the cell E (flat × tight-per-cat) and
cell J (flat × tight-fcs) predictions, using the same n=200 paired
indices the primary Qwen-14B run used.

Goal: verify that the body claim "FCS gain collapses under LLM-judge"
replicates across backbones, not just on Qwen-14B.

Output: results_llm_judge_cross_backbone.json with per-backbone × per-
judge accuracy on E vs J + paired McNemar.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_env(env_path: Path) -> None:
    if not env_path.exists(): return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a gold (reference) answer, and a system's predicted answer. Your job is to decide whether the prediction conveys the same factual answer as the gold, regardless of length or phrasing.

Question: {question}

Gold answer: {gold}

Predicted answer: {pred}

Reply with exactly one token: CORRECT or INCORRECT.
- CORRECT: the prediction conveys the same factual answer as the gold (semantically equivalent; minor format differences are fine).
- INCORRECT: the prediction is factually wrong, contradictory, missing, or refuses when the gold has an answer.

Reply (CORRECT or INCORRECT only):"""


def judge_openai(client, model, q, gold, pred):
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=8, temperature=0.0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=q[:600], gold=str(gold)[:600], pred=str(pred)[:600])}],
        )
        v = r.choices[0].message.content.strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception as e:
        return f"ERROR:{str(e)[:60]}"


def judge_anthropic(client, model, q, gold, pred):
    try:
        msg = client.messages.create(
            model=model, max_tokens=12,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=q[:600], gold=str(gold)[:600], pred=str(pred)[:600])}],
        )
        v = msg.content[0].text.strip().upper()
        if "CORRECT" in v and "INCORRECT" not in v: return "CORRECT"
        if "INCORRECT" in v: return "INCORRECT"
        return "AMBIGUOUS"
    except Exception as e:
        return f"ERROR:{str(e)[:60]}"


def judge_cell(predictions, indices, judge_fn, label=""):
    out = []
    for k, idx in enumerate(indices):
        if idx >= len(predictions): continue
        p = predictions[idx]
        v = judge_fn(p["question"], p.get("gold_answer", ""), p.get("prediction", ""))
        out.append({"idx": idx, "verdict": v, "f1": float(p.get("f1", 0.0))})
        if (k + 1) % 50 == 0:
            c = sum(1 for x in out if x["verdict"] == "CORRECT")
            i = sum(1 for x in out if x["verdict"] == "INCORRECT")
            print(f"  [{label}] {k+1}/{len(indices)} CORRECT={c} INCORRECT={i}", flush=True)
    return out


def main():
    load_env(ROOT / ".env")

    # Reuse the same n=200 indices as the primary Qwen-14B judge run
    ej = json.load(open(ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json"))
    indices = sorted({v["idx"] for v in ej["cell_e"]["verdicts"]})
    print(f"using n={len(indices)} paired indices from the primary Qwen-14B judge run")

    backbones = [
        ("Qwen2.5-3B", "results_7xb_qwen3b_flat_tight.json", "results_7xb_qwen3b_flat_fcs.json"),
        ("Qwen2.5-7B", "results_7xb_qwen7b_flat_tight.json", "results_7xb_qwen7b_flat_fcs.json"),
        ("Mistral-7B", "results_7xb_mistral7b_flat_tight.json", "results_7xb_mistral7b_flat_fcs.json"),
        ("Llama-3.1-8B", "results_7xb_llama8b_flat_tight.json", "results_7xb_llama8b_flat_fcs.json"),
    ]

    from openai import OpenAI
    openai_client = OpenAI()
    import anthropic
    anthropic_client = anthropic.Anthropic()

    results = {"indices": indices, "n": len(indices), "backbones": {}}

    for name, e_file, j_file in backbones:
        print(f"\n=== {name} ===")
        e_preds = json.load(open(ROOT / "experiments/crag-7-ranking-shift" / e_file))["predictions"]
        j_preds = json.load(open(ROOT / "experiments/crag-7-ranking-shift" / j_file))["predictions"]

        gpt_e = judge_cell(e_preds, indices, lambda q, g, p: judge_openai(openai_client, "gpt-4o-mini", q, g, p), label=f"{name} E gpt")
        gpt_j = judge_cell(j_preds, indices, lambda q, g, p: judge_openai(openai_client, "gpt-4o-mini", q, g, p), label=f"{name} J gpt")
        cla_e = judge_cell(e_preds, indices, lambda q, g, p: judge_anthropic(anthropic_client, "claude-sonnet-4-5", q, g, p), label=f"{name} E claude")
        cla_j = judge_cell(j_preds, indices, lambda q, g, p: judge_anthropic(anthropic_client, "claude-sonnet-4-5", q, g, p), label=f"{name} J claude")

        def acc(verdicts):
            c = sum(1 for v in verdicts if v["verdict"] == "CORRECT")
            i = sum(1 for v in verdicts if v["verdict"] == "INCORRECT")
            return c / max(1, c + i), c, i

        def mcnemar(e_verds, j_verds):
            n01 = sum(1 for x, y in zip(e_verds, j_verds)
                      if x["verdict"] == "CORRECT" and y["verdict"] == "INCORRECT")
            n10 = sum(1 for x, y in zip(e_verds, j_verds)
                      if x["verdict"] == "INCORRECT" and y["verdict"] == "CORRECT")
            disc = n01 + n10
            chi2 = (abs(n01 - n10) - 1) ** 2 / disc if disc > 0 else 0.0
            return n01, n10, chi2

        gpt_e_acc, _, _ = acc(gpt_e)
        gpt_j_acc, _, _ = acc(gpt_j)
        cla_e_acc, _, _ = acc(cla_e)
        cla_j_acc, _, _ = acc(cla_j)
        gpt_n01, gpt_n10, gpt_chi2 = mcnemar(gpt_e, gpt_j)
        cla_n01, cla_n10, cla_chi2 = mcnemar(cla_e, cla_j)

        print(f"  gpt-4o-mini: E={gpt_e_acc:.3f} J={gpt_j_acc:.3f} ΔJE={gpt_j_acc-gpt_e_acc:+.3f}  McNemar χ²={gpt_chi2:.2f} (n01={gpt_n01}, n10={gpt_n10})")
        print(f"  Claude-S4.5: E={cla_e_acc:.3f} J={cla_j_acc:.3f} ΔJE={cla_j_acc-cla_e_acc:+.3f}  McNemar χ²={cla_chi2:.2f} (n01={cla_n01}, n10={cla_n10})")

        results["backbones"][name] = {
            "gpt_4o_mini": {
                "e_accuracy": gpt_e_acc, "j_accuracy": gpt_j_acc,
                "delta_j_minus_e": gpt_j_acc - gpt_e_acc,
                "mcnemar": {"n01_e_only_correct": gpt_n01, "n10_j_only_correct": gpt_n10, "chi2": gpt_chi2},
                "verdicts_e": gpt_e, "verdicts_j": gpt_j,
            },
            "claude_sonnet_4_5": {
                "e_accuracy": cla_e_acc, "j_accuracy": cla_j_acc,
                "delta_j_minus_e": cla_j_acc - cla_e_acc,
                "mcnemar": {"n01_e_only_correct": cla_n01, "n10_j_only_correct": cla_n10, "chi2": cla_chi2},
                "verdicts_e": cla_e, "verdicts_j": cla_j,
            },
        }

    out_path = ROOT / "experiments/crag-7-ranking-shift/results_llm_judge_cross_backbone.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
