"""LongMemEval dataset loader.

Loads data/LongMemEval/data/longmemeval_oracle.json
(500 questions, 6 question types, 1-6 sessions per question, ~22 turns/question avg).

Per-turn `has_answer` flag is the gold evidence — much cleaner than LoCoMo's
dia_id-based scheme. We expose a unified interface so the same eval/retriever
code we wrote for LoCoMo works here too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Project root resolved from this file location (src/...py → repo root)
ROOT = Path(__file__).resolve().parents[1]

ORACLE_PATH = ROOT / "data/LongMemEval/data/longmemeval_oracle.json"
S_CLEANED_PATH = ROOT / "data/LongMemEval/data/longmemeval_s_cleaned.json"

# LongMemEval's six question types map cleanly to our internal label space.
# Keep both the raw type and the simplified label.
TYPE_TO_LABEL: dict[str, str] = {
    "temporal-reasoning":          "temporal",          # 133
    "multi-session":               "multi-session",     # 133
    "knowledge-update":            "knowledge-update",  #  78
    "single-session-preference":   "single-session",    #  30
    "single-session-assistant":    "single-session",    #  56
    "single-session-user":         "single-session",    #  70
}


@dataclass
class LMETurn:
    """One turn in a LongMemEval haystack session."""
    session_id: str          # opaque LongMemEval session id
    session_index: int       # 0-based index within question's haystack
    session_date: str        # e.g. '2023/04/10 (Mon) 17:50'
    turn_index: int          # 0-based within session
    role: str                # 'user' | 'assistant'
    text: str
    has_answer: bool = False
    dia_id: str = ""         # synthesized: f"{session_index}:{turn_index}" — matches retriever interface


@dataclass
class LMEQuestion:
    """One LongMemEval question with its haystack."""
    question_id: str
    question_type: str       # raw e.g. 'temporal-reasoning'
    label: str               # simplified e.g. 'temporal'
    question: str
    answer: str
    question_date: str       # when the question was asked, e.g. '2023/04/10 (Mon) 23:07'
    answer_session_ids: list[str] = field(default_factory=list)
    sessions: list[list[LMETurn]] = field(default_factory=list)  # haystack sessions in original order

    @property
    def turns(self) -> list[LMETurn]:
        """Flat list of all turns across all sessions, in haystack order."""
        return [t for s in self.sessions for t in s]

    @property
    def gold_turns(self) -> list[LMETurn]:
        """Turns flagged has_answer=True."""
        return [t for t in self.turns if t.has_answer]

    @property
    def num_turns(self) -> int:
        return sum(len(s) for s in self.sessions)


def load_questions(path: Path | str = ORACLE_PATH) -> list[LMEQuestion]:
    raw = json.loads(Path(path).read_text())
    out: list[LMEQuestion] = []
    for q in raw:
        sessions: list[list[LMETurn]] = []
        haystack_dates = q.get("haystack_dates", []) or []
        haystack_sids = q.get("haystack_session_ids", []) or []
        for s_idx, session in enumerate(q.get("haystack_sessions", [])):
            date = haystack_dates[s_idx] if s_idx < len(haystack_dates) else ""
            sid = haystack_sids[s_idx] if s_idx < len(haystack_sids) else f"_s{s_idx}"
            turns: list[LMETurn] = []
            for t_idx, t in enumerate(session):
                turns.append(LMETurn(
                    session_id=sid,
                    session_index=s_idx,
                    session_date=date,
                    turn_index=t_idx,
                    role=t.get("role", ""),
                    text=t.get("content", ""),
                    has_answer=bool(t.get("has_answer", False)),
                    dia_id=f"{s_idx}:{t_idx}",
                ))
            sessions.append(turns)

        raw_type = q.get("question_type", "")
        out.append(LMEQuestion(
            question_id=q["question_id"],
            question_type=raw_type,
            label=TYPE_TO_LABEL.get(raw_type, raw_type),
            question=q["question"],
            answer=str(q.get("answer", "")),
            question_date=q.get("question_date", ""),
            answer_session_ids=q.get("answer_session_ids", []) or [],
            sessions=sessions,
        ))
    return out


def iter_questions(
    questions: list[LMEQuestion] | None = None,
    *,
    labels: list[str] | None = None,
    question_types: list[str] | None = None,
) -> Iterator[LMEQuestion]:
    questions = questions if questions is not None else load_questions()
    label_set = set(labels) if labels else None
    type_set = set(question_types) if question_types else None
    for q in questions:
        if label_set and q.label not in label_set:
            continue
        if type_set and q.question_type not in type_set:
            continue
        yield q


def stats(questions: list[LMEQuestion] | None = None) -> dict:
    questions = questions if questions is not None else load_questions()
    out = {
        "n_questions": len(questions),
        "by_type": {},
        "by_label": {},
        "turns": {
            "min": min((q.num_turns for q in questions), default=0),
            "max": max((q.num_turns for q in questions), default=0),
            "avg": sum(q.num_turns for q in questions) / max(1, len(questions)),
        },
        "sessions_per_q": {
            "min": min((len(q.sessions) for q in questions), default=0),
            "max": max((len(q.sessions) for q in questions), default=0),
            "avg": sum(len(q.sessions) for q in questions) / max(1, len(questions)),
        },
        "gold_turns_per_q_avg": (
            sum(len(q.gold_turns) for q in questions) / max(1, len(questions))
        ),
    }
    for q in questions:
        out["by_type"][q.question_type] = out["by_type"].get(q.question_type, 0) + 1
        out["by_label"][q.label] = out["by_label"].get(q.label, 0) + 1
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(stats())
