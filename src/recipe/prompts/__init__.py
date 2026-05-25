"""Per-category prompt templates for C-RAG.

The templates are stored as plain text files in this directory and loaded
on demand. They are deliberately minimal (~5 lines each) so the C-RAG
contribution is attributable to the *idea of category-aware prompting*,
not to fancy prompting tricks.

Templates expect the following formatting placeholders:
  {speaker_a}, {speaker_b}    speaker names (LoCoMo only)
  {retrieved_turns}           concatenated retrieved-turn text
  {retrieved_turns_with_timestamps}  same but with session date prefixes
  {question}                  the question text
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

# Map (benchmark, category) → template filename
TEMPLATE_FILES: dict[tuple[str, str], str] = {
    # Loose / format-uncontrolled per-benchmark prompts. Used for the
    # crag-7 ranking-shift "format-uncontrolled" arm. They allow the
    # model to explain and write full sentences. They do not key on
    # category — same prompt regardless of category — to ensure the
    # uncontrolled setting reflects what a benchmark-default RAG
    # pipeline (no per-category prompt engineering) would look like.
    ("locomo", "loose"):         "locomo_loose.txt",
    ("longmemeval", "loose"):    "lme_loose.txt",
    ("beam", "loose"):           "beam_loose.txt",
    # LoCoMo
    ("locomo", "generic"):       "locomo_generic.txt",
    ("locomo", "single-hop"):    "locomo_single_hop.txt",
    ("locomo", "multi-hop"):     "locomo_multi_hop.txt",
    ("locomo", "temporal"):      "locomo_temporal.txt",
    ("locomo", "open-domain"):   "locomo_open_domain.txt",
    ("locomo", "adversarial"):   "locomo_adversarial.txt",
    # LongMemEval
    ("longmemeval", "generic"):          "lme_generic.txt",
    ("longmemeval", "single-session"):   "lme_single_session.txt",
    ("longmemeval", "multi-session"):    "lme_multi_session.txt",
    ("longmemeval", "temporal"):         "lme_temporal.txt",
    ("longmemeval", "knowledge-update"): "lme_knowledge_update.txt",
    # LoCoMo FCS-only (surface-form constraint, NO task hints)
    ("locomo-fcs", "single-hop"):    "locomo_single_hop_fcs.txt",
    ("locomo-fcs", "multi-hop"):     "locomo_multi_hop_fcs.txt",
    ("locomo-fcs", "temporal"):      "locomo_temporal_fcs.txt",
    ("locomo-fcs", "open-domain"):   "locomo_open_domain_fcs.txt",
    ("locomo-fcs", "adversarial"):   "locomo_adversarial_fcs.txt",
    # LongMemEval FCS-only (surface-form constraint, NO task hints)
    ("longmemeval-fcs", "single-session"):   "lme_single_session_fcs.txt",
    ("longmemeval-fcs", "multi-session"):    "lme_multi_session_fcs.txt",
    ("longmemeval-fcs", "temporal"):         "lme_temporal_fcs.txt",
    ("longmemeval-fcs", "knowledge-update"): "lme_knowledge_update_fcs.txt",
    # LongMemEval tight-stripped (task-framing verbs removed, surface-form kept) — §5.5b ablation
    ("longmemeval-stripped", "single-session"):   "lme_single_session_stripped.txt",
    ("longmemeval-stripped", "multi-session"):    "lme_multi_session_stripped.txt",
    ("longmemeval-stripped", "temporal"):         "lme_temporal_stripped.txt",
    ("longmemeval-stripped", "knowledge-update"): "lme_knowledge_update_stripped.txt",
    # BEAM
    ("beam", "generic"):                     "beam_generic.txt",
    ("beam", "abstention"):                  "beam_abstention.txt",
    ("beam", "contradiction-resolution"):    "beam_contradiction_resolution.txt",
    ("beam", "event-ordering"):              "beam_event_ordering.txt",
    ("beam", "information-extraction"):      "beam_information_extraction.txt",
    ("beam", "knowledge-update"):            "beam_knowledge_update.txt",
    ("beam", "multi-session"):               "beam_multi_session.txt",
    ("beam", "temporal"):                    "beam_temporal.txt",
}


def load_template(benchmark: str, category: str) -> str:
    """Load a prompt template by (benchmark, category).

    Falls back to the single-hop template if the requested template doesn't exist.
    """
    key = (benchmark, category)
    if key not in TEMPLATE_FILES:
        # Fallback: single-hop / single-session
        fallback_cat = (
            "single-hop"
            if benchmark == "locomo"
            else ("single-session" if benchmark == "longmemeval" else "information-extraction")
        )
        key = (benchmark, fallback_cat)
    path = _PROMPTS_DIR / TEMPLATE_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text()


def list_templates() -> dict[tuple[str, str], str]:
    """Return all known (benchmark, category) → filename mappings."""
    return dict(TEMPLATE_FILES)
