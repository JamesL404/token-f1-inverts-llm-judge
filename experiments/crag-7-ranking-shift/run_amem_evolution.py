"""Run faithful A-MEM with memory evolution on LME oracle 14B.

Heavier than amem-faithful: per-session note extraction + GLOBAL evolution
pass that merges/updates/deduplicates notes across sessions in the haystack.

Cells:
  K3: amem-evolution × tight-per-cat
  K4: amem-evolution × tight-fcs
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

    crag = CRAG(
        benchmark="longmemeval", router=RuleBasedRouter("longmemeval"),
        generator=generator, top_k=5,
        retrieval_mode="amem-evolution", prompt_variant="tight",
    )
    result = run_longmemeval(crag, system_name="crag-7K3-amem-evol-tight",
                              split="oracle", progress_every=50)
    write_json(OUT / "results_7K3_lme_amem_evol_tight.json", result)
    print("7K3 amem-evolution + tight-per-cat:", result["summary"])

    crag = CRAG(
        benchmark="longmemeval", router=RuleBasedRouter("longmemeval"),
        generator=generator, top_k=5,
        retrieval_mode="amem-evolution", prompt_variant="tight-fcs",
    )
    result = run_longmemeval(crag, system_name="crag-7K4-amem-evol-fcs",
                              split="oracle", progress_every=50)
    write_json(OUT / "results_7K4_lme_amem_evol_fcs.json", result)
    print("7K4 amem-evolution + tight-fcs:", result["summary"])


if __name__ == "__main__":
    main()
