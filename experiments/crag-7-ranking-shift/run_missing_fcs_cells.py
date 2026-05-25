"""Fill in the missing tight-fcs × architecture cells (reviewer comment #6).

The original crag-7 run only had tight-fcs × flat (cell J). To make the
prompt × architecture interaction matrix as complete as possible, we add:
  - linked-notes × tight-fcs (cell L)
  - session-bank × tight-fcs (cell M)
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


def build_crag(*, retrieval_mode, prompt_variant, generator):
    return CRAG(
        benchmark="longmemeval",
        router=RuleBasedRouter("longmemeval"),
        generator=generator,
        top_k=5,
        retrieval_mode=retrieval_mode,
        prompt_variant=prompt_variant,
    )


def main():
    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    # cell L: linked-notes × tight-fcs
    crag_L = build_crag(retrieval_mode="linked-notes", prompt_variant="tight-fcs", generator=generator)
    result_L = run_longmemeval(crag_L, system_name="crag-7L-linked-tight-fcs", split="oracle", limit=None, seed=None, progress_every=50)
    write_json(OUT / "results_7L_lme_linked_tight_fcs.json", result_L)
    print("7L linked-notes+tight-fcs:", result_L["summary"])

    # cell M: session-bank × tight-fcs
    crag_M = build_crag(retrieval_mode="session-bank", prompt_variant="tight-fcs", generator=generator)
    result_M = run_longmemeval(crag_M, system_name="crag-7M-session-bank-tight-fcs", split="oracle", limit=None, seed=None, progress_every=50)
    write_json(OUT / "results_7M_lme_session_bank_tight_fcs.json", result_M)
    print("7M session-bank+tight-fcs:", result_M["summary"])


if __name__ == "__main__":
    main()
