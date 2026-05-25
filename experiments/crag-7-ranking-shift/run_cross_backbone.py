"""Cross-backbone replication of the architecture-zero finding on LongMemEval.

For each backbone, run the two most informative cells:
  - flat retrieval × tight-per-cat (existing-prompt strong baseline)
  - flat retrieval × tight-fcs    (FCS-only, the new headline)

The architecture-zero claim becomes cross-backbone evidence iff the FCS
gain over hand-tuned tight-per-cat replicates on >= 1 other backbone
in similar magnitude.
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
    "tight": dict(prompt_variant="tight", template_category_override=None, label="tight-per-cat"),
    "fcs":   dict(prompt_variant="tight-fcs", template_category_override=None, label="tight-fcs"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="Short label, e.g. qwen7b, llama8b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--cells", nargs="+", default=["tight", "fcs"])
    args = parser.parse_args()

    generator = get_generator("hf-local", model=args.model)

    for cell_id in args.cells:
        if cell_id not in CELLS:
            print(f"unknown cell: {cell_id}")
            continue
        cfg = CELLS[cell_id]
        crag = CRAG(
            benchmark="longmemeval",
            router=RuleBasedRouter("longmemeval"),
            generator=generator,
            top_k=5,
            retrieval_mode="flat",
            prompt_variant=cfg["prompt_variant"],
            template_category_override=cfg["template_category_override"],
        )
        result = run_longmemeval(
            crag,
            system_name=f"crag-7xb-{args.label}-flat-{cfg['label']}",
            split="oracle",
            limit=args.limit,
            progress_every=args.progress_every,
        )
        write_json(OUT / f"results_7xb_{args.label}_flat_{cell_id}.json", result)
        print(f"[{args.label}] flat+{cfg['label']}:", result["summary"])


if __name__ == "__main__":
    main()
