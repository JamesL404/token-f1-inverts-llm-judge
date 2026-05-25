"""Run the faithful-ish A-MEM cell on LongMemEval oracle 14B.

Tests whether a heavier architecture (LLM-driven note construction at
index time + content linking at query time) beats the FCS-only flat
result (0.4469) on the same backbone.
"""
from __future__ import annotations

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


def main() -> None:
    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    # Cell K1: amem-faithful + tight-per-cat
    crag = CRAG(
        benchmark="longmemeval",
        router=RuleBasedRouter("longmemeval"),
        generator=generator,
        top_k=5,
        retrieval_mode="amem-faithful",
        prompt_variant="tight",
    )
    result = run_longmemeval(
        crag, system_name="crag-7K-amem-tight-per-cat", split="oracle", progress_every=50,
    )
    write_json(OUT / "results_7K_lme_amem_tight.json", result)
    print("7K amem-faithful + tight-per-cat:", result["summary"])

    # Cell K2: amem-faithful + tight-fcs
    crag = CRAG(
        benchmark="longmemeval",
        router=RuleBasedRouter("longmemeval"),
        generator=generator,
        top_k=5,
        retrieval_mode="amem-faithful",
        prompt_variant="tight-fcs",
    )
    result = run_longmemeval(
        crag, system_name="crag-7K-amem-tight-fcs", split="oracle", progress_every=50,
    )
    write_json(OUT / "results_7K_lme_amem_fcs.json", result)
    print("7K amem-faithful + tight-fcs:", result["summary"])


if __name__ == "__main__":
    main()
