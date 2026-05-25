"""Run the locked crag-6 linking-baseline experiment.

Per `experiments/crag-6-linking-baseline/protocol.md`, this script runs
flat BM25 vs linked-notes retrieval against the same matched 14B
generator on LongMemEval oracle and (optionally) LoCoMo. Both
retrieval modes share router, generator, top_k, call budget.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recipe.experiment import run_locomo, run_longmemeval, write_json
from src.recipe.generator import get_generator
from src.recipe.recipe import CRAG
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-6-linking-baseline"


def build_crag(
    *,
    benchmark: str,
    generator_name: str,
    model: str | None,
    retrieval_mode: str,
    top_k: int = 5,
) -> CRAG:
    generator_kwargs = {"model": model} if model else {}
    generator = get_generator(generator_name, **generator_kwargs)
    return CRAG(
        benchmark=benchmark,
        router=RuleBasedRouter(benchmark),
        generator=generator,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
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
        default=["6a", "6b"],
        help="which cells to run: 6a=LME flat, 6b=LME linked, 6e=LoCoMo flat, 6f=LoCoMo linked",
    )
    args = parser.parse_args()

    cells = set(args.cells)

    if "6a" in cells:
        lme_flat = run_longmemeval(
            build_crag(
                benchmark="longmemeval",
                generator_name=args.generator,
                model=args.model,
                retrieval_mode="flat",
            ),
            system_name="crag-6a-lme-flat",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_6a_lme_flat.json", lme_flat)
        print("6a LME flat:", lme_flat["summary"])

    if "6b" in cells:
        lme_linked = run_longmemeval(
            build_crag(
                benchmark="longmemeval",
                generator_name=args.generator,
                model=args.model,
                retrieval_mode="linked-notes",
            ),
            system_name="crag-6b-lme-linked",
            split="oracle",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_6b_lme_linked.json", lme_linked)
        print("6b LME linked-notes:", lme_linked["summary"])

    if "6e" in cells:
        locomo_flat = run_locomo(
            build_crag(
                benchmark="locomo",
                generator_name=args.generator,
                model=args.model,
                retrieval_mode="flat",
            ),
            system_name="crag-6e-locomo-flat",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_6e_locomo_flat.json", locomo_flat)
        print("6e LoCoMo flat:", locomo_flat["summary"])

    if "6f" in cells:
        locomo_linked = run_locomo(
            build_crag(
                benchmark="locomo",
                generator_name=args.generator,
                model=args.model,
                retrieval_mode="linked-notes",
            ),
            system_name="crag-6f-locomo-linked",
            limit=args.limit,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        write_json(OUT / "results_6f_locomo_linked.json", locomo_linked)
        print("6f LoCoMo linked-notes:", locomo_linked["summary"])

    print(f"saved JSON to {OUT}")


if __name__ == "__main__":
    main()
