"""BEAM benchmark loader.

Loads the public BEAM dataset exports downloaded under
`data/beam/hf/data/{100K,500K,1M}-00000-of-00001.parquet`.

This loader intentionally supports only the BEAM question types that have
explicit gold answers in the public release and therefore fit our current
no-LLM evaluation discipline.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Project root resolved from this file location (src/...py → repo root)
ROOT = Path(__file__).resolve().parents[1]


BEAM_PATHS: dict[str, Path] = {
    "100K": ROOT / "data/beam/hf/data/100K-00000-of-00001.parquet",
    "500K": ROOT / "data/beam/hf/data/500K-00000-of-00001.parquet",
    "1M": ROOT / "data/beam/hf/data/1M-00000-of-00001.parquet",
}

# These are the BEAM types that have explicit gold answers in the public
# release and are therefore currently usable in our pipeline.
USABLE_QUESTION_TYPES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "knowledge_update",
    "multi_session_reasoning",
    "temporal_reasoning",
)

# The four types most directly aligned with the current paper's main story.
IDEA2_CORE_TYPES: tuple[str, ...] = (
    "abstention",
    "knowledge_update",
    "multi_session_reasoning",
    "temporal_reasoning",
)

TYPE_TO_LABEL: dict[str, str] = {
    "abstention": "abstention",
    "contradiction_resolution": "contradiction-resolution",
    "event_ordering": "event-ordering",
    "information_extraction": "information-extraction",
    "knowledge_update": "knowledge-update",
    "multi_session_reasoning": "multi-session",
    "temporal_reasoning": "temporal",
}


@dataclass
class BeamTurn:
    """One turn within a BEAM conversation session."""

    session_id: int
    session_date: str
    turn_index: int
    role: str
    text: str
    chat_id: int
    index: str = ""
    question_type: str = ""
    dia_id: str = ""


@dataclass
class BeamQuestion:
    """One usable BEAM probing question."""

    conversation_id: str
    question: str
    answer: str
    question_type: str
    label: str
    difficulty: str = ""
    source_chat_ids: list[int] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamConversation:
    """One BEAM conversation with its turns and usable probing questions."""

    conversation_id: str
    split: str
    category: str = ""
    title: str = ""
    sessions: list[list[BeamTurn]] = field(default_factory=list)
    questions: list[BeamQuestion] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    conversation_seed: dict[str, Any] = field(default_factory=dict)

    @property
    def turns(self) -> list[BeamTurn]:
        return [t for s in self.sessions for t in s]

    @property
    def num_turns(self) -> int:
        return sum(len(s) for s in self.sessions)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    def turn_by_chat_id(self, chat_id: int) -> BeamTurn | None:
        for t in self.turns:
            if t.chat_id == chat_id:
                return t
        return None


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    except Exception:
        pass

    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Loading BEAM parquet requires either pandas+pyarrow/fastparquet or duckdb."
        ) from exc

    rel = str(path).replace("'", "''")
    reln = duckdb.sql(f"SELECT * FROM read_parquet('{rel}')")
    columns = [d[0] for d in reln.description]
    rows = reln.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def _parse_probing_questions(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        return {str(k): list(v) for k, v in raw.items() if isinstance(v, list)}
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {str(k): list(v) for k, v in obj.items() if isinstance(v, list)}


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    to_list = getattr(obj, "tolist", None)
    if callable(to_list):
        converted = to_list()
        if isinstance(converted, list):
            return converted
    try:
        return list(obj)
    except Exception:
        return []


def _normalize_source_chat_ids(raw: Any) -> list[int]:
    out: list[int] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, int):
            out.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)

    visit(raw)
    deduped = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _extract_answer(item: dict[str, Any]) -> str:
    for key in ("answer", "ideal_answer", "ideal_response"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _question_metadata(item: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key, value in item.items():
        if key in {"question", "answer", "ideal_answer", "ideal_response", "difficulty", "rubric", "source_chat_ids"}:
            continue
        keep[key] = value
    return keep


def _build_conversation(row: dict[str, Any], split: str, usable_types: set[str]) -> BeamConversation:
    conv_seed = row.get("conversation_seed") if isinstance(row.get("conversation_seed"), dict) else {}
    user_profile = row.get("user_profile") if isinstance(row.get("user_profile"), dict) else {}
    sessions: list[list[BeamTurn]] = []
    for s_idx, session in enumerate(_as_list(row.get("chat")), start=1):
        session = _as_list(session)
        if not session:
            continue
        turns: list[BeamTurn] = []
        session_date = ""
        for t_idx, item in enumerate(session, start=1):
            if not isinstance(item, dict):
                continue
            session_date = str(item.get("time_anchor") or session_date or "")
            chat_id = item.get("id")
            try:
                chat_id = int(chat_id)
            except (TypeError, ValueError):
                chat_id = -1
            turns.append(
                BeamTurn(
                    session_id=s_idx,
                    session_date=session_date,
                    turn_index=t_idx,
                    role=str(item.get("role", "")),
                    text=str(item.get("content", "")),
                    chat_id=chat_id,
                    index=str(item.get("index", "")),
                    question_type=str(item.get("question_type", "")),
                    dia_id=f"{s_idx}:{t_idx}",
                )
            )
        sessions.append(turns)

    probing = _parse_probing_questions(row.get("probing_questions"))
    questions: list[BeamQuestion] = []
    conversation_id = str(row.get("conversation_id", ""))
    for q_type, items in probing.items():
        if q_type not in usable_types:
            continue
        label = TYPE_TO_LABEL.get(q_type, q_type)
        for item in items:
            if not isinstance(item, dict):
                continue
            answer = _extract_answer(item)
            if not answer:
                continue
            questions.append(
                BeamQuestion(
                    conversation_id=conversation_id,
                    question=str(item.get("question", "")),
                    answer=answer,
                    question_type=q_type,
                    label=label,
                    difficulty=str(item.get("difficulty", "")),
                    source_chat_ids=_normalize_source_chat_ids(item.get("source_chat_ids")),
                    rubric=[str(x) for x in (item.get("rubric") or [])],
                    metadata=_question_metadata(item),
                )
            )

    return BeamConversation(
        conversation_id=conversation_id,
        split=split,
        category=str(conv_seed.get("category", "")),
        title=str(conv_seed.get("title", "")),
        sessions=sessions,
        questions=questions,
        user_profile=user_profile,
        conversation_seed=conv_seed,
    )


def load_conversations(
    split: str = "100K",
    *,
    question_types: list[str] | None = None,
    core_only: bool = False,
) -> list[BeamConversation]:
    if split not in BEAM_PATHS:
        raise ValueError(f"unknown BEAM split: {split}")
    if core_only and question_types is not None:
        raise ValueError("pass either core_only=True or question_types=..., not both")

    usable_types = set(IDEA2_CORE_TYPES if core_only else USABLE_QUESTION_TYPES)
    if question_types is not None:
        usable_types &= set(question_types)

    rows = _load_parquet_rows(BEAM_PATHS[split])
    return [_build_conversation(row, split, usable_types) for row in rows]


def iter_qas(
    conversations: list[BeamConversation] | None = None,
    *,
    labels: list[str] | None = None,
    question_types: list[str] | None = None,
) -> Iterator[tuple[BeamConversation, BeamQuestion]]:
    conversations = conversations if conversations is not None else load_conversations()
    label_set = set(labels) if labels else None
    type_set = set(question_types) if question_types else None
    for conv in conversations:
        for q in conv.questions:
            if label_set is not None and q.label not in label_set:
                continue
            if type_set is not None and q.question_type not in type_set:
                continue
            yield conv, q


def stats(conversations: list[BeamConversation] | None = None) -> dict[str, Any]:
    conversations = conversations if conversations is not None else load_conversations()
    out: dict[str, Any] = {
        "num_conversations": len(conversations),
        "total_qas": sum(len(c.questions) for c in conversations),
        "by_type": {},
        "by_label": {},
        "turns": {
            "min": min((c.num_turns for c in conversations), default=0),
            "max": max((c.num_turns for c in conversations), default=0),
            "avg": (sum(c.num_turns for c in conversations) / max(1, len(conversations))),
        },
        "sessions": {
            "min": min((c.num_sessions for c in conversations), default=0),
            "max": max((c.num_sessions for c in conversations), default=0),
            "avg": (sum(c.num_sessions for c in conversations) / max(1, len(conversations))),
        },
    }
    for conv in conversations:
        for q in conv.questions:
            out["by_type"][q.question_type] = out["by_type"].get(q.question_type, 0) + 1
            out["by_label"][q.label] = out["by_label"].get(q.label, 0) + 1
    return out


if __name__ == "__main__":
    import pprint

    for split in ("100K", "500K", "1M"):
        print(f"=== {split} ===")
        pprint.pprint(stats(load_conversations(split)))
