"""LoCoMo tight-fcs cells: flat / linked-notes / session-bank × tight-fcs.

Symmetric extension to the LongMemEval cells L (linked × FCS) and O
(session-bank × FCS) — completes the prompt × architecture interaction
matrix on LoCoMo.

Cell labels:
  LJ = flat × tight-fcs
  LL = linked-notes × tight-fcs
  LO = session-bank × tight-fcs
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recipe.experiment import run_locomo, write_json
from src.recipe.generator import get_generator
from src.recipe.recipe import CRAG
from src.recipe.router import RuleBasedRouter

OUT = ROOT / "experiments" / "crag-7-ranking-shift"


def build_crag(*, retrieval_mode, prompt_variant, generator):
    return CRAG(
        benchmark="locomo",
        router=RuleBasedRouter("locomo"),
        generator=generator,
        top_k=5,
        retrieval_mode=retrieval_mode,
        prompt_variant=prompt_variant,
    )


def main():
    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    crag_LJ = build_crag(retrieval_mode="flat", prompt_variant="tight-fcs", generator=generator)
    res = run_locomo(crag_LJ, system_name="crag-7LJ-locomo-flat-tight-fcs", limit=None, seed=None, progress_every=200)
    write_json(OUT / "results_7LJ_locomo_flat_tight_fcs.json", res)
    print("7LJ flat+tight-fcs:", res["summary"])

    crag_LL = build_crag(retrieval_mode="linked-notes", prompt_variant="tight-fcs", generator=generator)
    res = run_locomo(crag_LL, system_name="crag-7LL-locomo-linked-tight-fcs", limit=None, seed=None, progress_every=200)
    write_json(OUT / "results_7LL_locomo_linked_tight_fcs.json", res)
    print("7LL linked-notes+tight-fcs:", res["summary"])

    crag_LO = build_crag(retrieval_mode="session-bank", prompt_variant="tight-fcs", generator=generator)
    res = run_locomo(crag_LO, system_name="crag-7LO-locomo-session-bank-tight-fcs", limit=None, seed=None, progress_every=200)
    write_json(OUT / "results_7LO_locomo_session_bank_tight_fcs.json", res)
    print("7LO session-bank+tight-fcs:", res["summary"])


if __name__ == "__main__":
    main()
