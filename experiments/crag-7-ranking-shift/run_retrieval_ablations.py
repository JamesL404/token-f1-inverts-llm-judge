"""Retrieval-substrate ablations: dense, BM25+rerank, HippoRAG.

Each runs (flat-style retrieval + tight-per-cat) and (flat-style + tight-fcs)
on LongMemEval oracle 14B. Compares against:
  - cell E (flat BM25 + tight-per-cat) F1=0.3652
  - cell J (flat BM25 + tight-fcs)     F1=0.4469

Cell labels:
  N1 = dense (E5)         + tight-per-cat
  N2 = dense (E5)         + tight-fcs
  P1 = BM25+rerank        + tight-per-cat
  P2 = BM25+rerank        + tight-fcs
  R1 = hipporag-lite      + tight-per-cat
  R2 = hipporag-lite      + tight-fcs
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


CELLS = {
    "N1": dict(retrieval_mode="dense",        prompt_variant="tight",     label="dense-tight"),
    "N2": dict(retrieval_mode="dense",        prompt_variant="tight-fcs", label="dense-fcs"),
    "P1": dict(retrieval_mode="bm25-rerank",  prompt_variant="tight",     label="rerank-tight"),
    "P2": dict(retrieval_mode="bm25-rerank",  prompt_variant="tight-fcs", label="rerank-fcs"),
    "R1": dict(retrieval_mode="hipporag",     prompt_variant="tight",     label="hipporag-tight"),
    "R2": dict(retrieval_mode="hipporag",     prompt_variant="tight-fcs", label="hipporag-fcs"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", default=list(CELLS.keys()))
    args = parser.parse_args()

    generator = get_generator("hf-local", model="Qwen/Qwen2.5-14B-Instruct")

    for cell_id in args.cells:
        if cell_id not in CELLS:
            print(f"unknown cell: {cell_id}")
            continue
        cfg = CELLS[cell_id]
        crag = CRAG(
            benchmark="longmemeval",
            router=RuleBasedRouter("longmemeval"),
            generator=generator, top_k=5,
            retrieval_mode=cfg["retrieval_mode"],
            prompt_variant=cfg["prompt_variant"],
        )
        result = run_longmemeval(
            crag, system_name=f"crag-7{cell_id}-{cfg['label']}",
            split="oracle", progress_every=100,
        )
        out_path = OUT / f"results_7{cell_id}_lme_{cfg['label']}.json"
        write_json(out_path, result)
        print(f"7{cell_id} ({cfg['label']}): F1={result['summary']['f1']:.4f}  wall={result['summary']['total_wall_time_s']:.0f}s")


if __name__ == "__main__":
    main()
