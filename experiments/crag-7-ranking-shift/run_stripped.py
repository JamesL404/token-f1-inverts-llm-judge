"""Tight-per-cat-stripped ablation (Item #7).

Runs flat × tight-stripped on LongMemEval oracle with Qwen2.5-14B-Instruct, n=500.
Compare to:
  cell E (flat × tight-per-cat) F1=0.365  avg_chars=51.8
  cell J (flat × tight-fcs)     F1=0.447  avg_chars=14.4

If stripped matches FCS in length+F1: linguistic mechanism (task-framing words)
established. If stripped sits between E and J: surface-form template structure
(short answer-form templates) carries some of the gain.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recipe.experiment import run_longmemeval, write_json
from src.recipe.generator import get_generator
from src.recipe.recipe import CRAG
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-7-ranking-shift"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run on n=20 for sanity check")
    args = parser.parse_args()

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")
    crag = CRAG(
        benchmark="longmemeval",
        router=RuleBasedRouter("longmemeval"),
        generator=generator,
        top_k=5,
        retrieval_mode="flat",
        prompt_variant="tight-stripped",
    )
    limit = 20 if args.smoke else None
    result = run_longmemeval(
        crag, system_name="crag-7Q-flat-tight-stripped",
        split="oracle", progress_every=50, limit=limit,
    )
    suffix = "_smoke" if args.smoke else ""
    out_path = OUT / f"results_7Q_lme_flat_tight_stripped{suffix}.json"
    write_json(out_path, result)
    s = result["summary"]
    print(f"\n7Q flat × tight-stripped: F1={s['f1']:.4f}  "
          f"avg_chars={s['total_completion_chars']/s['n_questions']:.1f}  "
          f"wall={s['total_wall_time_s']:.0f}s")
    print(f"  vs cell E (tight-per-cat): F1=0.365  avg_chars=~52")
    print(f"  vs cell J (tight-fcs):     F1=0.447  avg_chars=~14")


if __name__ == "__main__":
    main()
