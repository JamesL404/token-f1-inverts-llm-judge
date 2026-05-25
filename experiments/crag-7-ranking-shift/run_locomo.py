"""Run the crag-7 ranking-shift matrix on LoCoMo (1986 Qs).

Same 2x3 cell structure as the LongMemEval version, but heavier:
1986 Qs/cell × ~0.5-0.7s/Q ~= 18-24 min/cell × 5 new cells ~= 1.5-2 hours
on GPUs 3+4.

Cell labels are L-prefixed to distinguish from the LME crag-7 cells:
  LE: flat        + tight-per-category  (already run as crag-1 prompt-fix on LoCoMo, 0.4065)
  LF: linked      + tight-per-category
  LA: flat        + tight-generic
  LB: flat        + loose
  LC: linked      + tight-generic
  LD: linked      + loose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recipe.experiment import run_locomo, run_locomo_mc10, write_json
from src.recipe.generator import get_generator
from src.recipe.recipe import CRAG
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-7-ranking-shift"


CELL_DEFS = {
    "LF": dict(retrieval_mode="linked-notes", prompt_variant="tight", template_category_override=None,
               name="crag-7LF-locomo-linked-tight-per-cat"),
    "LA": dict(retrieval_mode="flat", prompt_variant="tight", template_category_override="generic",
               name="crag-7LA-locomo-flat-tight-generic"),
    "LB": dict(retrieval_mode="flat", prompt_variant="loose", template_category_override=None,
               name="crag-7LB-locomo-flat-loose"),
    "LC": dict(retrieval_mode="linked-notes", prompt_variant="tight", template_category_override="generic",
               name="crag-7LC-locomo-linked-tight-generic"),
    "LD": dict(retrieval_mode="linked-notes", prompt_variant="loose", template_category_override=None,
               name="crag-7LD-locomo-linked-loose"),
    # Session-bank cells (proposal must-have #2 architecture flavor on LoCoMo)
    "LG": dict(retrieval_mode="session-bank", prompt_variant="tight", template_category_override=None,
               name="crag-7LG-locomo-session-bank-tight-per-cat"),
    "LH": dict(retrieval_mode="session-bank", prompt_variant="tight", template_category_override="generic",
               name="crag-7LH-locomo-session-bank-tight-generic"),
    "LI": dict(retrieval_mode="session-bank", prompt_variant="loose", template_category_override=None,
               name="crag-7LI-locomo-session-bank-loose"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="hf-local", choices=["dummy", "openai", "hf-local"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(CELL_DEFS.keys()),
        help="LF=linked+tight-per-cat, LA=flat+tight-generic, LB=flat+loose, LC=linked+tight-generic, LD=linked+loose",
    )
    args = parser.parse_args()

    generator_kwargs = {"model": args.model} if args.model else {}
    generator = get_generator(args.generator, **generator_kwargs)

    for cell_id in args.cells:
        if cell_id not in CELL_DEFS:
            print(f"unknown cell: {cell_id}")
            continue
        cfg = CELL_DEFS[cell_id]
        crag = CRAG(
            benchmark="locomo",
            router=RuleBasedRouter("locomo"),
            generator=generator,
            top_k=5,
            retrieval_mode=cfg["retrieval_mode"],
            prompt_variant=cfg["prompt_variant"],
            template_category_override=cfg["template_category_override"],
        )
        result = run_locomo(
            crag,
            system_name=cfg["name"],
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        result["mc10_summary"] = run_locomo_mc10(result["predictions"])
        write_json(OUT / f"results_7{cell_id}_locomo.json", result)
        print(f"7{cell_id} ({cfg['name']}) summary:", result["summary"])
        print(f"7{cell_id} MC10 summary:", result["mc10_summary"])
    print(f"saved JSON to {OUT}")


if __name__ == "__main__":
    main()
