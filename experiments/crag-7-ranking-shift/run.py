"""Run the crag-7 ranking-shift mini matrix on LongMemEval oracle 14B.

Per `experiments/crag-7-ranking-shift/protocol.md`, this script runs
the 4 cells not already covered by run_011 (flat + tight-per-category)
and run_015 (linked-notes + tight-per-category):

  cell A: flat + tight-generic
  cell B: flat + loose
  cell C: linked-notes + tight-generic
  cell D: linked-notes + loose

Combined with E = run_011 = 6a and F = run_015 = 6b, this gives a
2 x 3 matrix over (retrieval in {flat, linked}) x (prompt in
{tight-per-cat, tight-generic, loose}).
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


def build_crag(
    *,
    benchmark: str,
    generator,
    retrieval_mode: str,
    prompt_variant: str = "tight",
    template_category_override: str | None = None,
) -> CRAG:
    return CRAG(
        benchmark=benchmark,
        router=RuleBasedRouter(benchmark),
        generator=generator,
        top_k=5,
        retrieval_mode=retrieval_mode,
        prompt_variant=prompt_variant,
        template_category_override=template_category_override,
    )




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="hf-local", choices=["dummy", "openai", "hf-local"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="A=flat+tight-generic, B=flat+loose, C=linked+tight-generic, D=linked+loose, J=flat+tight-fcs",
    )
    args = parser.parse_args()

    generator_kwargs = {"model": args.model} if args.model else {}
    generator = get_generator(args.generator, **generator_kwargs)

    cells_to_run = set(args.cells)

    if "J" in cells_to_run:
        crag = build_crag(
            benchmark="longmemeval",
            generator=generator,
            retrieval_mode="flat",
            prompt_variant="tight-fcs",
        )
        result = run_longmemeval(
            crag,
            system_name="crag-7J-flat-tight-fcs",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_7J_lme_flat_tight_fcs.json", result)
        print("7J flat+tight-fcs:", result["summary"])

    if "A" in cells_to_run:
        crag = build_crag(
            benchmark="longmemeval",
            generator=generator,
            retrieval_mode="flat",
            prompt_variant="tight",
            template_category_override="generic",
        )
        result = run_longmemeval(
            crag,
            system_name="crag-7A-flat-tight-generic",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_7A_lme_flat_tight_generic.json", result)
        print("7A flat+tight-generic:", result["summary"])

    if "B" in cells_to_run:
        crag = build_crag(
            benchmark="longmemeval",
            generator=generator,
            retrieval_mode="flat",
            prompt_variant="loose",
        )
        result = run_longmemeval(
            crag,
            system_name="crag-7B-flat-loose",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_7B_lme_flat_loose.json", result)
        print("7B flat+loose:", result["summary"])

    if "C" in cells_to_run:
        crag = build_crag(
            benchmark="longmemeval",
            generator=generator,
            retrieval_mode="linked-notes",
            prompt_variant="tight",
            template_category_override="generic",
        )
        result = run_longmemeval(
            crag,
            system_name="crag-7C-linked-tight-generic",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_7C_lme_linked_tight_generic.json", result)
        print("7C linked+tight-generic:", result["summary"])

    if "D" in cells_to_run:
        crag = build_crag(
            benchmark="longmemeval",
            generator=generator,
            retrieval_mode="linked-notes",
            prompt_variant="loose",
        )
        result = run_longmemeval(
            crag,
            system_name="crag-7D-linked-loose",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_7D_lme_linked_loose.json", result)
        print("7D linked+loose:", result["summary"])

    print(f"saved JSON to {OUT}")


if __name__ == "__main__":
    main()
