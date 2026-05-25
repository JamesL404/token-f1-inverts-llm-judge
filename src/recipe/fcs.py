"""Format Control Specification (FCS).

Per `paper/proposal.md` §"Format Control Specification (FCS)" and
§"Ranking Shift Protocol", every method × benchmark cell of the headline
experiment runs in TWO settings:

  - format-uncontrolled : prompt allows the model to explain, write full
                          sentences, mirror gold's verbosity. This is the
                          "natural" setting that benchmark-published
                          systems implicitly use.
  - format-controlled   : prompt constrains SURFACE FORM ONLY (length,
                          form, normalization, abstention behaviour) to
                          match gold-answer surface statistics. It must
                          NOT contain task-solving hints.

The controlled setting must be DERIVED from the gold answers, not
hand-tuned for score. This module is the deriver.

Protocol invariants the controlled prompt must respect:
  1. constrains output surface form only (length, form, normalization)
  2. does NOT leak task-specific reasoning hints
  3. uses the same evidence budget (same retrieval, same top_k)
  4. uses the same router and same backbone

If a controlled prompt makes ALL methods score worse on a benchmark,
that's a signal the FCS itself is malformed (proposal risk R2). The
sanity check below detects this before we trust ranking-shift claims.

This is currently a skeleton: the SurfaceForm profiler runs on existing
benchmarks today; the prompt-rendering and sanity-check functions are
placeholders that will be filled in alongside the crag-7 ranking-shift
experiment.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal


AnswerForm = Literal[
    "short_span",       # 1-3 word factual answer
    "numeric_span",     # bare number, possibly with unit
    "date",             # date or date range
    "label",            # category label / preference
    "list",             # comma-separated list
    "abstention",       # fixed abstention string
    "free_text",        # multi-clause sentence (verbose gold)
    "unknown",
]


@dataclass
class SurfaceForm:
    """Per-(benchmark, category) gold-answer surface-form profile."""

    benchmark: str
    category: str
    n: int
    median_words: int
    p25_words: int
    p75_words: int
    max_words: int
    inferred_form: AnswerForm
    form_counts: dict[AnswerForm, int] = field(default_factory=dict)
    abstention_strings: list[str] = field(default_factory=list)
    sample_golds: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "category": self.category,
            "n": self.n,
            "median_words": self.median_words,
            "p25_words": self.p25_words,
            "p75_words": self.p75_words,
            "max_words": self.max_words,
            "inferred_form": self.inferred_form,
            "form_counts": dict(self.form_counts),
            "abstention_strings": self.abstention_strings[:5],
            "sample_golds": self.sample_golds[:5],
        }


_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|"
    r"\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^\s*[\$£€]?[-+]?\d+(\.\d+)?[\s%]*[a-zA-Z]*\s*$")
_ABSTENTION_PATTERNS = [
    "no information available",
    "not specified",
    "based on the provided",
    "there is no information",
]


def _classify_form(gold: str) -> AnswerForm:
    """Best-effort surface-form classification from a single gold string.

    Conservative: returns the LEAST verbose form consistent with the answer.
    This biases the controlled prompt toward shorter, more constrained
    output; if the gold turns out to be longer, the F1 score still works
    because gold is gold.
    """
    g = (gold or "").strip()
    if not g:
        return "unknown"
    g_lower = g.lower()
    for pat in _ABSTENTION_PATTERNS:
        if pat in g_lower:
            return "abstention"
    words = g.split()
    if _NUMERIC_RE.match(g):
        return "numeric_span"
    if _DATE_RE.search(g) and len(words) <= 8:
        return "date"
    if "," in g and len(words) <= 12:
        return "list"
    if len(words) <= 3:
        return "short_span"
    if len(words) <= 8:
        return "label"
    return "free_text"


def profile_gold_answers(
    pairs: Iterable[tuple[str, str]],
    *,
    benchmark: str,
) -> dict[str, SurfaceForm]:
    """Profile gold-answer surface form per category.

    `pairs` yields (category, gold_answer) tuples for the benchmark of
    interest. The function aggregates by category and returns one
    SurfaceForm per category with the inferred dominant form.
    """
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cat, gold in pairs:
        if gold is None:
            continue
        by_cat[cat].append(str(gold))

    out: dict[str, SurfaceForm] = {}
    for cat, golds in by_cat.items():
        if not golds:
            continue
        word_lens = [len(g.split()) for g in golds]
        forms = Counter(_classify_form(g) for g in golds)
        dominant = forms.most_common(1)[0][0] if forms else "unknown"
        abstain = [g for g in golds if _classify_form(g) == "abstention"][:3]
        sample = golds[:5]
        if len(word_lens) >= 4:
            qs = statistics.quantiles(word_lens, n=4)
            p25, p75 = int(qs[0]), int(qs[2])
        else:
            p25, p75 = min(word_lens), max(word_lens)
        sf = SurfaceForm(
            benchmark=benchmark,
            category=cat,
            n=len(golds),
            median_words=int(statistics.median(word_lens)),
            p25_words=p25,
            p75_words=p75,
            max_words=max(word_lens),
            inferred_form=dominant,
            form_counts={k: int(v) for k, v in forms.items()},
            abstention_strings=abstain,
            sample_golds=sample,
        )
        out[cat] = sf
    return out


def render_uncontrolled_prefix(_sf: SurfaceForm) -> str:
    """Loose prompt prefix that allows verbose, explanatory answers.

    Skeleton: does not yet differentiate per category.
    """
    return (
        "You are answering a question about a long conversation. "
        "Use the retrieved excerpts to answer. You may explain your "
        "reasoning if useful, and you may answer in a complete sentence."
    )


def render_controlled_prefix(sf: SurfaceForm) -> str:
    """Surface-form-only constraint prefix derived from `sf`.

    MUST NOT include task-solving hints. Only:
      - target length window
      - target answer form
      - abstention string (if present in golds)
    """
    bits: list[str] = ["You are answering a question about a long conversation."]
    bits.append("Return ONLY the minimal final answer.")

    form = sf.inferred_form
    if form == "numeric_span":
        bits.append("Output exactly one number, optionally followed by a unit.")
    elif form == "date":
        bits.append("Output exactly one date or date range, no extra words.")
    elif form == "label":
        bits.append("Output a 1 to 3 word category label.")
    elif form == "list":
        bits.append("Output items separated by commas, no surrounding sentence.")
    elif form == "short_span":
        bits.append("Output at most 3 words.")
    elif form == "free_text":
        cap = max(8, sf.p75_words)
        bits.append(f"Output at most {cap} words.")
    elif form == "abstention":
        if sf.abstention_strings:
            bits.append(f'If the answer is not supported by the excerpts, reply exactly: "{sf.abstention_strings[0]}".')

    if sf.abstention_strings and form != "abstention":
        bits.append(
            'If the answer is not supported by the excerpts, reply exactly: '
            f'"{sf.abstention_strings[0]}".'
        )

    bits.append("Do not explain. Do not add commentary.")
    return " ".join(bits)


def fcs_sanity_check_passes(
    uncontrolled_scores: dict[str, float],
    controlled_scores: dict[str, float],
    *,
    min_methods_above_floor: int = 2,
) -> bool:
    """Proposal §"protocol 的最低 sanity check" item 4.

    If the controlled setting causes every method to collapse on a
    benchmark, the controlled prompt is bad-prompt, not good-control.
    Require at least `min_methods_above_floor` methods to retain >=80%
    of their uncontrolled score.

    Placeholder logic — refine when real ranking-shift data lands.
    """
    if not uncontrolled_scores or not controlled_scores:
        return False
    above = 0
    for method, u in uncontrolled_scores.items():
        c = controlled_scores.get(method)
        if c is None or u <= 0:
            continue
        if c >= 0.8 * u:
            above += 1
    return above >= min_methods_above_floor


def profile_locomo() -> dict[str, SurfaceForm]:
    from src.locomo import iter_qas, load_dialogues
    pairs = []
    for d, qa in iter_qas(load_dialogues()):
        if qa.label == "adversarial":
            pairs.append(("adversarial", "no information available"))
        else:
            pairs.append((qa.label, qa.answer))
    return profile_gold_answers(pairs, benchmark="locomo")


def profile_longmemeval() -> dict[str, SurfaceForm]:
    from src.longmemeval import iter_questions, load_questions
    pairs = []
    for q in iter_questions(load_questions()):
        pairs.append((q.label, q.answer))
    return profile_gold_answers(pairs, benchmark="longmemeval")


def profile_beam(split: str = "100K") -> dict[str, SurfaceForm]:
    from src.beam import iter_qas, load_conversations
    pairs = []
    for conv, q in iter_qas(load_conversations(split)):
        pairs.append((q.label, q.answer))
    return profile_gold_answers(pairs, benchmark=f"beam-{split}")


if __name__ == "__main__":
    import json
    print("=== LoCoMo gold-answer surface-form profile ===")
    print(json.dumps({k: sf.as_dict() for k, sf in profile_locomo().items()}, indent=2)[:3000])
    print("\n=== LongMemEval gold-answer surface-form profile ===")
    print(json.dumps({k: sf.as_dict() for k, sf in profile_longmemeval().items()}, indent=2)[:3000])
    print("\n=== BEAM 100K gold-answer surface-form profile ===")
    print(json.dumps({k: sf.as_dict() for k, sf in profile_beam('100K').items()}, indent=2)[:3000])
