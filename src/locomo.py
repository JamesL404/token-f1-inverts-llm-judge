"""LoCoMo dataset loader.

Loads data/locomo/data/locomo10.json (10 dialogues, 1986 QAs)
and exposes a clean iterator over QAExample plus per-dialogue conversation access.

Category integer → reasoning type mapping was inferred by cross-referencing:
  (a) the snap-research/locomo eval code at task_eval/evaluation.py, where
      cat 1 uses multi-answer split F1 (multi-hop), cat 5 is adversarial;
  (b) the LoCoMo paper appendix B.1 percentage breakdown:
      single-hop 36%, multi-hop 14.6%, temporal 20.6%, open-domain 3.9%, adversarial 24.9%.

Category mapping (verified against the LoCoMo paper):
  1 = multi-hop
  2 = temporal
  3 = open-domain
  4 = single-hop
  5 = adversarial
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Project root resolved from this file location (src/...py → repo root)
ROOT = Path(__file__).resolve().parents[1]

LOCOMO_PATH = ROOT / "data/locomo/data/locomo10.json"

CATEGORY_TO_LABEL: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
LABEL_TO_CATEGORY: dict[str, int] = {v: k for k, v in CATEGORY_TO_LABEL.items()}


@dataclass
class Turn:
    """A single conversational turn within a session."""
    session_id: int          # 1-indexed
    session_date_time: str   # e.g. "2:21 pm on 8 May, 2023"
    turn_index: int          # 1-indexed within session
    speaker: str
    text: str
    dia_id: str              # e.g. "D1:3"  (matches LoCoMo evidence format)
    img_url: str | None = None
    blip_caption: str | None = None


@dataclass
class QAExample:
    """One QA pair tied to a dialogue."""
    sample_id: str           # dialogue id, e.g. "conv-26"
    question: str
    answer: str | int | float | list  # gold; may be empty for adversarial
    evidence: list[str]      # list of dia_ids like ["D2:8"]; may be empty
    category: int            # raw integer (1..5)
    label: str               # human label, see CATEGORY_TO_LABEL


@dataclass
class Dialogue:
    """One full LoCoMo dialogue with all sessions and QAs."""
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[Turn] = field(default_factory=list)
    qas: list[QAExample] = field(default_factory=list)
    session_summaries: dict[int, str] = field(default_factory=dict)
    event_summary: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)

    @property
    def num_sessions(self) -> int:
        return max((t.session_id for t in self.turns), default=0)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def session_turns(self, session_id: int) -> list[Turn]:
        return [t for t in self.turns if t.session_id == session_id]

    def turn_by_dia_id(self, dia_id: str) -> Turn | None:
        for t in self.turns:
            if t.dia_id == dia_id:
                return t
        return None


def _parse_session_id(key: str) -> int | None:
    # Keys like "session_3" → 3, "session_3_date_time" → None
    if key.startswith("session_") and not key.endswith("date_time"):
        try:
            return int(key.split("_", 1)[1])
        except ValueError:
            return None
    return None


def load_dialogues(path: Path | str = LOCOMO_PATH) -> list[Dialogue]:
    raw = json.loads(Path(path).read_text())
    dialogues: list[Dialogue] = []
    for sample in raw:
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a", "")
        speaker_b = conv.get("speaker_b", "")

        turns: list[Turn] = []
        session_ids = sorted({sid for k in conv if (sid := _parse_session_id(k)) is not None})
        for sid in session_ids:
            session_key = f"session_{sid}"
            dt_key = f"session_{sid}_date_time"
            session = conv.get(session_key)
            session_dt = conv.get(dt_key, "")
            if not isinstance(session, list):
                continue
            for i, t in enumerate(session, start=1):
                turns.append(
                    Turn(
                        session_id=sid,
                        session_date_time=session_dt,
                        turn_index=i,
                        speaker=t.get("speaker", ""),
                        text=t.get("text", ""),
                        dia_id=t.get("dia_id", f"D{sid}:{i}"),
                        img_url=t.get("img_url"),
                        blip_caption=t.get("blip_caption"),
                    )
                )

        qas: list[QAExample] = []
        for q in sample.get("qa", []):
            cat = q.get("category")
            qas.append(
                QAExample(
                    sample_id=sample["sample_id"],
                    question=q["question"],
                    answer=q.get("answer", ""),
                    evidence=q.get("evidence", []) or [],
                    category=cat,
                    label=CATEGORY_TO_LABEL.get(cat, f"cat-{cat}"),
                )
            )

        # session_summary in LoCoMo is dict like {"1": "...", "2": "..."}
        sess_summary: dict[int, str] = {}
        for k, v in (sample.get("session_summary") or {}).items():
            try:
                sess_summary[int(k)] = v
            except (TypeError, ValueError):
                pass

        dialogues.append(
            Dialogue(
                sample_id=sample["sample_id"],
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                turns=turns,
                qas=qas,
                session_summaries=sess_summary,
                event_summary=sample.get("event_summary") or {},
                observation=sample.get("observation") or {},
            )
        )
    return dialogues


def iter_qas(
    dialogues: list[Dialogue] | None = None,
    *,
    labels: list[str] | None = None,
    skip_empty_answers: bool = False,
) -> Iterator[tuple[Dialogue, QAExample]]:
    """Yield (dialogue, qa) pairs, optionally filtered by reasoning label."""
    dialogues = dialogues if dialogues is not None else load_dialogues()
    label_set = set(labels) if labels else None
    for d in dialogues:
        for q in d.qas:
            if label_set is not None and q.label not in label_set:
                continue
            if skip_empty_answers and (q.answer is None or q.answer == ""):
                continue
            yield d, q


def stats(dialogues: list[Dialogue] | None = None) -> dict:
    dialogues = dialogues if dialogues is not None else load_dialogues()
    out = {
        "num_dialogues": len(dialogues),
        "total_qas": sum(len(d.qas) for d in dialogues),
        "by_label": {},
        "turns": {
            "min": min((d.num_turns for d in dialogues), default=0),
            "max": max((d.num_turns for d in dialogues), default=0),
            "avg": (sum(d.num_turns for d in dialogues) / max(1, len(dialogues))),
        },
        "sessions": {
            "min": min((d.num_sessions for d in dialogues), default=0),
            "max": max((d.num_sessions for d in dialogues), default=0),
            "avg": (sum(d.num_sessions for d in dialogues) / max(1, len(dialogues))),
        },
    }
    for d in dialogues:
        for q in d.qas:
            out["by_label"][q.label] = out["by_label"].get(q.label, 0) + 1
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(stats())
