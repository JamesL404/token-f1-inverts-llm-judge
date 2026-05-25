#!/usr/bin/env python3
"""Audit a local export of the BEAM benchmark.

This script is designed for the research workflow in this repo:

1. Take a local BEAM export (`.json`, `.jsonl`, or `.parquet`).
2. Normalize the top-level structure enough to inspect it consistently.
3. Parse `probing_questions` into a Python object when possible.
4. Summarize whether the benchmark appears compatible with our current
   long-horizon evaluation pipeline:
   - question text present?
   - gold answer present?
   - chat/history present?
   - question type present?
5. Emit a compact JSON report for a later paper/protocol decision.

It is intentionally an audit tool, not yet a production loader.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Common HF export pattern: {"train": [...]} or {"data": [...]}
        for key in ("train", "test", "validation", "data", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    except Exception:
        pass

    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability is env-specific
        raise RuntimeError(
            "Parquet input requires either pandas+pyarrow/fastparquet or duckdb."
        ) from exc

    rel = str(path).replace("'", "''")
    rows = duckdb.sql(f"SELECT * FROM read_parquet('{rel}')").df().to_dict(orient="records")
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".parquet":
        return _load_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def parse_probing_questions(raw: Any) -> tuple[Any, str]:
    """Return (parsed_obj, parse_mode)."""
    if isinstance(raw, (list, dict)):
        return raw, "native"
    if raw is None:
        return None, "missing"
    if not isinstance(raw, str):
        return raw, "opaque"

    text = raw.strip()
    if not text:
        return None, "empty"

    # Try JSON first, then Python literal syntax.
    try:
        return json.loads(text), "json-string"
    except Exception:
        pass
    try:
        return ast.literal_eval(text), "python-literal"
    except Exception:
        return raw, "unparsed-string"


def iter_question_records(parsed: Any) -> list[dict[str, Any]]:
    """Flatten a parsed probing_questions object into question-like dicts.

    We do not assume a single BEAM schema. Instead we search recursively for
    dicts that look like QA records.
    """
    found: list[dict[str, Any]] = []

    def visit(obj: Any, parent_key: str | None = None) -> None:
        if isinstance(obj, dict):
            lowered = {str(k).lower() for k in obj.keys()}
            if "question" in lowered or "ideal_answer" in lowered or "ideal_response" in lowered:
                enriched = dict(obj)
                if parent_key is not None and "__container_type" not in enriched:
                    enriched["__container_type"] = parent_key
                found.append(enriched)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item, parent_key=parent_key)

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            visit(value, parent_key=str(key))
    else:
        visit(parsed)
    return found


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def audit_rows(rows: list[dict[str, Any]], sample_limit: int = 3) -> dict[str, Any]:
    top_level_keys = Counter()
    parse_modes = Counter()
    question_type_counts = Counter()
    question_field_presence = Counter()
    missing_components = Counter()
    sample_rows: list[dict[str, Any]] = []
    question_examples: list[dict[str, Any]] = []

    n_with_chat = 0
    n_with_probing = 0
    n_rows_with_any_question = 0
    total_question_records = 0

    for idx, row in enumerate(rows):
        for key in row.keys():
            top_level_keys[key] += 1

        if row.get("chat") is not None:
            n_with_chat += 1

        parsed, mode = parse_probing_questions(row.get("probing_questions"))
        parse_modes[mode] += 1
        if row.get("probing_questions") is not None:
            n_with_probing += 1

        question_records = iter_question_records(parsed)
        if question_records:
            n_rows_with_any_question += 1
        total_question_records += len(question_records)

        if idx < sample_limit:
            sample_rows.append(
                {
                    "conversation_id": row.get("conversation_id"),
                    "top_level_keys": sorted(row.keys()),
                    "probing_parse_mode": mode,
                    "n_question_records_found": len(question_records),
                }
            )

        for q in question_records:
            q_text = _first_present(q, "question", "query", "prompt")
            q_answer = _first_present(
                q,
                "answer",
                "gold_answer",
                "reference_answer",
                "ideal_answer",
                "ideal_response",
                "label",
            )
            q_type = _first_present(
                q,
                "type",
                "question_type",
                "category",
                "ability",
                "__container_type",
            )

            if q_text not in (None, ""):
                question_field_presence["question"] += 1
            else:
                missing_components["question_text"] += 1

            if q_answer not in (None, ""):
                question_field_presence["answer"] += 1
            else:
                missing_components["gold_answer"] += 1

            if q_type not in (None, ""):
                question_field_presence["type"] += 1
                question_type_counts[str(q_type)] += 1
            else:
                missing_components["question_type"] += 1

            if len(question_examples) < sample_limit:
                question_examples.append(
                    {
                        "question": q_text,
                        "answer": q_answer,
                        "type": q_type,
                        "raw_keys": sorted(q.keys()),
                    }
                )

    n_rows = len(rows)
    benchmark_fit = {
        "chat_history_present_rate": round(n_with_chat / max(1, n_rows), 4),
        "probing_questions_present_rate": round(n_with_probing / max(1, n_rows), 4),
        "rows_with_any_question_records_rate": round(n_rows_with_any_question / max(1, n_rows), 4),
        "total_question_records_found": total_question_records,
    }

    if total_question_records == 0:
        scoreability = "unknown-no-question-records-found"
    else:
        q_has_question = question_field_presence["question"] / total_question_records
        q_has_answer = question_field_presence["answer"] / total_question_records
        q_has_type = question_field_presence["type"] / total_question_records
        if min(q_has_question, q_has_answer) >= 0.95:
            scoreability = "likely-loader-compatible"
        elif min(q_has_question, q_has_answer) >= 0.5:
            scoreability = "partial-loader-compatible"
        else:
            scoreability = "needs-benchmark-specific-scoring"

    return {
        "n_rows": n_rows,
        "top_level_key_counts": dict(sorted(top_level_keys.items())),
        "probing_parse_modes": dict(sorted(parse_modes.items())),
        "benchmark_fit": benchmark_fit,
        "question_field_presence": dict(sorted(question_field_presence.items())),
        "missing_components": dict(sorted(missing_components.items())),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "scoreability_judgment": scoreability,
        "sample_rows": sample_rows,
        "sample_question_records": question_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a local BEAM dataset export.")
    parser.add_argument("input", type=Path, help="Path to a local BEAM export (.json/.jsonl/.parquet)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/beam_schema_audit.json"),
        help="Where to write the JSON report",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="How many sample rows/question records to include in the report",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "audit": audit_rows(rows, sample_limit=args.sample_limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote audit report to {args.output}")


if __name__ == "__main__":
    main()
