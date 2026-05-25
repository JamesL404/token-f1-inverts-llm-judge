"""Reusable runners for C-RAG experiments."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.eval import _normalize_answer, evaluate_predictions, f1_single
from src.eval import adversarial_correct
from src.beam import iter_qas as iter_beam_qas, load_conversations
from src.locomo import iter_qas, load_dialogues
from src.locomo_mc10 import load_mc10, score_mc
from src.longmemeval import S_CLEANED_PATH, load_questions
from src.recipe.recipe import CRAG


def _telemetry_from_result(result) -> dict[str, Any]:
    return {
        "generator_calls": result.n_llm_calls,
        "memory_calls": 0,
        "prompt_chars": result.total_prompt_chars,
        "completion_chars": result.total_completion_chars,
        "wall_time_s": result.wall_time_s,
    }


def run_locomo(
    crag: CRAG,
    *,
    system_name: str,
    limit: int | None = None,
    seed: int | None = None,
    labels: list[str] | None = None,
    progress_every: int | None = None,
) -> dict[str, Any]:
    items = list(iter_qas(load_dialogues(), labels=labels))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)
    if limit is not None:
        items = items[:limit]

    pairs = []
    predictions = []
    total = len(items)
    for idx, (dialogue, qa) in enumerate(items, start=1):
        result = crag.answer(qa.question, dialogue)
        pairs.append((qa, result.prediction, _telemetry_from_result(result)))
        predictions.append(
            {
                "sample_id": dialogue.sample_id,
                "question": qa.question,
                "gold_answer": qa.answer,
                "gold_label": qa.label,
                "gold_category": qa.category,
                "prediction": result.prediction,
                "routed_category": result.routed_category,
                "rule_fired": result.rule_fired,
                "n_retrieved_turns": result.n_retrieved_turns,
                "n_llm_calls": result.n_llm_calls,
                "prompt_chars": result.total_prompt_chars,
                "completion_chars": result.total_completion_chars,
                "wall_time_s": round(result.wall_time_s, 4),
            }
        )
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"[{system_name}] locomo {idx}/{total}")

    run = evaluate_predictions(
        pairs,
        system_name=system_name,
        backbone=crag.generator.model,
    )
    summary = run.summary()
    summary["benchmark"] = "locomo"
    summary["router"] = crag.router.name
    summary["generator_model"] = crag.generator.model
    summary["top_k"] = crag.top_k
    summary["template_category_override"] = crag.template_category_override
    return {"summary": summary, "predictions": predictions}


def run_locomo_mc10(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    mc10_index = {(q.sample_id, q.question): q for q in load_mc10()}
    correct_by_label = defaultdict(int)
    total_by_label = defaultdict(int)
    n = 0
    correct = 0
    missing = 0
    for pred in predictions:
        q = mc10_index.get((pred["sample_id"], pred["question"]))
        if q is None:
            missing += 1
            continue
        _, ok = score_mc(pred["prediction"], q.choices, q.correct_choice_index)
        total_by_label[q.label] += 1
        n += 1
        correct += int(ok)
        if ok:
            correct_by_label[q.label] += 1
    return {
        "n": n,
        "missing_questions": missing,
        "accuracy": round(correct / max(1, n), 4),
        "by_label_accuracy": {
            label: round(correct_by_label[label] / max(1, total_by_label[label]), 4)
            for label in sorted(total_by_label)
        },
        "by_label_n": {label: total_by_label[label] for label in sorted(total_by_label)},
    }


def containment(prediction: str, gold: str) -> float:
    p = _normalize_answer(prediction)
    g = _normalize_answer(gold)
    if not g:
        return 0.0
    return float(g in p)


def rubric_coverage(prediction: str, rubric: list[str]) -> float:
    if not rubric:
        return 0.0
    p = _normalize_answer(prediction)
    hits = 0
    for item in rubric:
        r = _normalize_answer(item)
        if r and r in p:
            hits += 1
    return hits / max(1, len(rubric))


def run_longmemeval(
    crag: CRAG,
    *,
    system_name: str,
    split: str = "oracle",
    limit: int | None = None,
    seed: int | None = None,
    labels: list[str] | None = None,
    question_types: list[str] | None = None,
    progress_every: int | None = None,
) -> dict[str, Any]:
    questions = load_questions() if split == "oracle" else load_questions(S_CLEANED_PATH)
    if labels is not None:
        label_set = set(labels)
        questions = [q for q in questions if q.label in label_set]
    if question_types is not None:
        type_set = set(question_types)
        questions = [q for q in questions if q.question_type in type_set]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(questions)
    if limit is not None:
        questions = questions[:limit]

    by_label = defaultdict(lambda: {"n": 0, "f1_sum": 0.0, "cont_sum": 0.0})
    by_type = defaultdict(lambda: {"n": 0, "f1_sum": 0.0, "cont_sum": 0.0})
    total_f1 = 0.0
    total_cont = 0.0
    total_calls = 0
    total_prompt_chars = 0
    total_completion_chars = 0
    total_wall_time_s = 0.0
    predictions = []

    total = len(questions)
    for idx, q in enumerate(questions, start=1):
        result = crag.answer(q.question, q)
        f1 = f1_single(result.prediction, q.answer)
        cont = containment(result.prediction, q.answer)
        total_f1 += f1
        total_cont += cont
        total_calls += result.n_llm_calls
        total_prompt_chars += result.total_prompt_chars
        total_completion_chars += result.total_completion_chars
        total_wall_time_s += result.wall_time_s
        by_label[q.label]["n"] += 1
        by_label[q.label]["f1_sum"] += f1
        by_label[q.label]["cont_sum"] += cont
        by_type[q.question_type]["n"] += 1
        by_type[q.question_type]["f1_sum"] += f1
        by_type[q.question_type]["cont_sum"] += cont
        predictions.append(
            {
                "question_id": q.question_id,
                "question_type": q.question_type,
                "gold_label": q.label,
                "question": q.question,
                "gold_answer": q.answer,
                "prediction": result.prediction,
                "routed_category": result.routed_category,
                "rule_fired": result.rule_fired,
                "n_retrieved_turns": result.n_retrieved_turns,
                "n_llm_calls": result.n_llm_calls,
                "prompt_chars": result.total_prompt_chars,
                "completion_chars": result.total_completion_chars,
                "wall_time_s": round(result.wall_time_s, 4),
                "f1": round(f1, 4),
                "containment": round(cont, 4),
            }
        )
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"[{system_name}] longmemeval-{split} {idx}/{total}")

    n = len(questions)
    summary = {
        "benchmark": "longmemeval",
        "split": split,
        "system": system_name,
        "router": crag.router.name,
        "generator_model": crag.generator.model,
        "top_k": crag.top_k,
        "template_category_override": crag.template_category_override,
        "n_questions": n,
        "f1": round(total_f1 / max(1, n), 4),
        "containment": round(total_cont / max(1, n), 4),
        "n_llm_calls": total_calls,
        "calls_per_question": round(total_calls / max(1, n), 3),
        "total_wall_time_s": round(total_wall_time_s, 2),
        "total_prompt_chars": total_prompt_chars,
        "total_completion_chars": total_completion_chars,
        "by_label": {
            label: {
                "n": stats["n"],
                "f1": round(stats["f1_sum"] / max(1, stats["n"]), 4),
                "containment": round(stats["cont_sum"] / max(1, stats["n"]), 4),
            }
            for label, stats in sorted(by_label.items())
        },
        "by_type": {
            label: {
                "n": stats["n"],
                "f1": round(stats["f1_sum"] / max(1, stats["n"]), 4),
                "containment": round(stats["cont_sum"] / max(1, stats["n"]), 4),
            }
            for label, stats in sorted(by_type.items())
        },
    }
    return {"summary": summary, "predictions": predictions}


def _score_beam_prediction(question_type: str, prediction: str, gold: str) -> float:
    if question_type == "abstention":
        return adversarial_correct(prediction)
    return f1_single(prediction, gold)


def run_beam(
    crag: CRAG,
    *,
    system_name: str,
    split: str = "100K",
    limit: int | None = None,
    seed: int | None = None,
    labels: list[str] | None = None,
    question_types: list[str] | None = None,
    core_only: bool = False,
    progress_every: int | None = None,
) -> dict[str, Any]:
    conversations = load_conversations(split, question_types=question_types, core_only=core_only)
    items = list(iter_beam_qas(conversations, labels=labels, question_types=question_types))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)
    if limit is not None:
        items = items[:limit]

    by_label = defaultdict(lambda: {"n": 0, "f1_sum": 0.0, "cont_sum": 0.0, "rubric_sum": 0.0})
    by_type = defaultdict(lambda: {"n": 0, "f1_sum": 0.0, "cont_sum": 0.0, "rubric_sum": 0.0})
    total_f1 = 0.0
    total_cont = 0.0
    total_rubric = 0.0
    total_calls = 0
    total_prompt_chars = 0
    total_completion_chars = 0
    total_wall_time_s = 0.0
    predictions = []

    total = len(items)
    for idx, (conv, q) in enumerate(items, start=1):
        result = crag.answer(q.question, conv)
        f1 = _score_beam_prediction(q.question_type, result.prediction, q.answer)
        cont = containment(result.prediction, q.answer)
        rub = rubric_coverage(result.prediction, q.rubric)
        total_f1 += f1
        total_cont += cont
        total_rubric += rub
        total_calls += result.n_llm_calls
        total_prompt_chars += result.total_prompt_chars
        total_completion_chars += result.total_completion_chars
        total_wall_time_s += result.wall_time_s
        by_label[q.label]["n"] += 1
        by_label[q.label]["f1_sum"] += f1
        by_label[q.label]["cont_sum"] += cont
        by_label[q.label]["rubric_sum"] += rub
        by_type[q.question_type]["n"] += 1
        by_type[q.question_type]["f1_sum"] += f1
        by_type[q.question_type]["cont_sum"] += cont
        by_type[q.question_type]["rubric_sum"] += rub
        predictions.append(
            {
                "conversation_id": conv.conversation_id,
                "split": conv.split,
                "question_type": q.question_type,
                "gold_label": q.label,
                "question": q.question,
                "gold_answer": q.answer,
                "prediction": result.prediction,
                "routed_category": result.routed_category,
                "rule_fired": result.rule_fired,
                "n_retrieved_turns": result.n_retrieved_turns,
                "n_llm_calls": result.n_llm_calls,
                "prompt_chars": result.total_prompt_chars,
                "completion_chars": result.total_completion_chars,
                "wall_time_s": round(result.wall_time_s, 4),
                "f1": round(f1, 4),
                "containment": round(cont, 4),
                "rubric_coverage": round(rub, 4),
                "source_chat_ids": q.source_chat_ids,
            }
        )
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(f"[{system_name}] beam-{split} {idx}/{total}")

    n = len(items)
    summary = {
        "benchmark": "beam",
        "split": split,
        "system": system_name,
        "router": crag.router.name,
        "generator_model": crag.generator.model,
        "top_k": crag.top_k,
        "template_category_override": crag.template_category_override,
        "core_only": core_only,
        "n_questions": n,
        "f1": round(total_f1 / max(1, n), 4),
        "containment": round(total_cont / max(1, n), 4),
        "rubric_coverage": round(total_rubric / max(1, n), 4),
        "n_llm_calls": total_calls,
        "calls_per_question": round(total_calls / max(1, n), 3),
        "total_wall_time_s": round(total_wall_time_s, 2),
        "total_prompt_chars": total_prompt_chars,
        "total_completion_chars": total_completion_chars,
        "by_label": {
            label: {
                "n": stats["n"],
                "f1": round(stats["f1_sum"] / max(1, stats["n"]), 4),
                "containment": round(stats["cont_sum"] / max(1, stats["n"]), 4),
                "rubric_coverage": round(stats["rubric_sum"] / max(1, stats["n"]), 4),
            }
            for label, stats in sorted(by_label.items())
        },
        "by_type": {
            label: {
                "n": stats["n"],
                "f1": round(stats["f1_sum"] / max(1, stats["n"]), 4),
                "containment": round(stats["cont_sum"] / max(1, stats["n"]), 4),
                "rubric_coverage": round(stats["rubric_sum"] / max(1, stats["n"]), 4),
            }
            for label, stats in sorted(by_type.items())
        },
    }
    return {"summary": summary, "predictions": predictions}


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
