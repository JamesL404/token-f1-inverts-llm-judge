"""Run session-bank architecture-family baseline on LongMemEval oracle 14B.

Per paper/proposal.md "Baseline taxonomy" §"session-summary / memory-bank"
this is the must-have second baseline (after linked-notes from crag-6).

Cells:
  G: session-bank + tight-per-category (matched to E flat / F linked)
  H: session-bank + tight-generic
  I: session-bank + loose

If session-bank ALSO produces no F1 signal above noise vs flat (cell E),
then the negative claim is stable across two architecture flavors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recipe.experiment import run_longmemeval, write_json
from src.recipe.generator import get_generator
from src.recipe.recipe import CRAG
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-7-ranking-shift"


CELL_DEFS = {
    "G": dict(prompt_variant="tight", template_category_override=None,
              name="crag-7G-lme-session-bank-tight-per-cat"),
    "H": dict(prompt_variant="tight", template_category_override="generic",
              name="crag-7H-lme-session-bank-tight-generic"),
    "I": dict(prompt_variant="loose", template_category_override=None,
              name="crag-7I-lme-session-bank-loose"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="hf-local", choices=["dummy", "openai", "hf-local"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--cells", nargs="+", default=list(CELL_DEFS.keys()),
                        help="G=session-bank+tight-per-cat, H=+tight-generic, I=+loose")
    args = parser.parse_args()

    generator_kwargs = {"model": args.model} if args.model else {}
    generator = get_generator(args.generator, **generator_kwargs)

    for cell_id in args.cells:
        if cell_id not in CELL_DEFS:
            print(f"unknown cell: {cell_id}")
            continue
        cfg = CELL_DEFS[cell_id]
        crag = CRAG(
            benchmark="longmemeval",
            router=RuleBasedRouter("longmemeval"),
            generator=generator,
            top_k=5,
            retrieval_mode="session-bank",
            prompt_variant=cfg["prompt_variant"],
            template_category_override=cfg["template_category_override"],
        )
        result = run_longmemeval(
            crag,
            system_name=cfg["name"],
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / f"results_7{cell_id}_lme_session_bank.json", result)
        print(f"7{cell_id} ({cfg['name']}) summary:", result["summary"])
    print(f"saved JSON to {OUT}")


if __name__ == "__main__":
    main()
