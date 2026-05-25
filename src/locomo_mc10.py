"""LoCoMo-MC10 loader with corrected category labels.

LoCoMo-MC10 (Percena/locomo-mc10 on HuggingFace) is the same 1986 questions
as LoCoMo in 10-option multiple-choice format. We use it as a metric
robustness check for our F1-based findings.

IMPORTANT — category label inversion:
  MC10 reports a `question_type` field, but the labels do not match the
  underlying data. The data is unambiguous (cat 1 = multi-answer / multi-hop;
  cat 2 = date answers / temporal; etc.). MC10's `question_type` field
  labels them with a different ordering. We discard MC10's `question_type`
  and re-derive the correct label by joining with `data/locomo/data/locomo10.json`
  via `(sample_id, question_text)`.

This loader exposes:
  - load_mc10(): list[MC10Question] with the *correct* label attached
  - score_mc(prediction, choices, correct_index): pick choice by token-F1
    similarity, return (chosen_index, correct_bool)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.eval import f1_single
from src.locomo import CATEGORY_TO_LABEL, load_dialogues

# Project root resolved from this file location (src/...py → repo root)
ROOT = Path(__file__).resolve().parents[1]

MC10_PATH = ROOT / "data/locomo_mc10.json"


@dataclass
class MC10Question:
    question_id: str
    sample_id: str
    question: str
    choices: list[str]
    correct_choice_index: int
    answer: str
    category: int
    label: str
    n_sessions: int


def _build_locomo_index() -> dict[tuple[str, str], int]:
    """Map (sample_id, question_text) → category integer using locomo10.json.

    This is the source of truth for category labels.
    """
    dialogues = load_dialogues()
    idx: dict[tuple[str, str], int] = {}
    for d in dialogues:
        for q in d.qas:
            idx[(d.sample_id, q.question)] = q.category
    return idx


def load_mc10(path: Path | str = MC10_PATH) -> list[MC10Question]:
    """Load LoCoMo-MC10 with the corrected category labels."""
    items: list[MC10Question] = []
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    cat_index = _build_locomo_index()
    n_unmatched = 0
    for r in raw:
        # MC10 question_id format: f"{sample_id}_q{idx}"
        qid = r["question_id"]
        sample_id = qid.rsplit("_q", 1)[0]
        cat = cat_index.get((sample_id, r["question"]))
        if cat is None:
            n_unmatched += 1
            continue
        items.append(MC10Question(
            question_id=qid,
            sample_id=sample_id,
            question=r["question"],
            choices=list(r.get("choices", [])),
            correct_choice_index=int(r.get("correct_choice_index", -1)),
            answer=str(r.get("answer", "")),
            category=cat,
            label=CATEGORY_TO_LABEL.get(cat, f"cat-{cat}"),
            n_sessions=int(r.get("num_sessions", 0)),
        ))
    if n_unmatched:
        print(f"warning: {n_unmatched} MC10 questions did not match LoCoMo (category unknown)")
    return items


def score_mc(prediction: str, choices: list[str], correct_choice_index: int) -> tuple[int, bool]:
    """Pick the choice with highest token F1 to the prediction.

    Returns (chosen_index, correct). Ties are broken by lowest index (deterministic).
    """
    if not choices:
        return -1, False
    scores = [f1_single(prediction, c) for c in choices]
    best = -1.0
    best_i = 0
    for i, s in enumerate(scores):
        if s > best:
            best = s
            best_i = i
    return best_i, (best_i == correct_choice_index)


def random_baseline_accuracy(items: list[MC10Question], n_options: int = 10) -> float:
    """Theoretical random baseline = 1/n_options for n-way MC."""
    return 1.0 / n_options


if __name__ == "__main__":
    items = load_mc10()
    print(f"loaded {len(items)} MC10 questions with corrected labels")
    import collections
    by_label = collections.Counter(q.label for q in items)
    print("by label:", dict(by_label))
    n_choices = collections.Counter(len(q.choices) for q in items)
    print("choice counts:", dict(n_choices))
