"""LoCoMo evaluation harness.

Provides:
  - LoCoMo-faithful F1 / EM scoring (ported from snap-research/locomo
    task_eval/evaluation.py so our numbers are comparable to published baselines).
  - A pluggable LLM-judge interface (OpenAI / vLLM-OpenAI / dummy).
  - A `Scorer` aggregator that tracks per-reasoning-type metrics, LLM-call
    counts, and wall-clock time so we can build the accuracy × cost × latency
    Pareto frontier our paper needs.

Design notes:
  - Scoring is decoupled from generation: the harness takes (qa, prediction)
    pairs that you produced however you like (RAG, MemGPT, oracle, ...).
  - The judge backend is chosen by env var or argument, NOT hard-coded — so
    H1 (controlled re-evaluation) is one judge, one retriever for all systems.
  - We never call the judge from inside the F1 path; F1 / EM are pure CPU.
"""
from __future__ import annotations

import os
import re
import string
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# F1 / EM scoring — ported from snap-research/locomo/task_eval/evaluation.py
# (PorterStemmer dependency is optional; falls back to no-stem if nltk missing)
# ---------------------------------------------------------------------------

try:
    from nltk.stem import PorterStemmer
    _PS = PorterStemmer()
    def _stem(w: str) -> str:
        return _PS.stem(w)
except Exception:  # nltk optional
    def _stem(w: str) -> str:
        return w


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization with LoCoMo's modifications:
       remove commas, lowercase, remove punctuation, drop {a, an, the, and}, collapse ws."""
    if s is None:
        return ""
    s = str(s).replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    s = unicodedata.normalize("NFD", s)
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    p = set(_normalize_answer(prediction).split())
    g = set(_normalize_answer(ground_truth).split())
    return float(p == g and len(g) > 0)


def f1_single(prediction: str, ground_truth: str) -> float:
    """Whole-answer F1 with stemming. LoCoMo's f1_score()."""
    p = [_stem(w) for w in _normalize_answer(prediction).split()]
    g = [_stem(w) for w in _normalize_answer(ground_truth).split()]
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer split F1. LoCoMo's f1() — split each side by comma,
    take mean over ground-truth slots of max-over-prediction-slots single F1."""
    preds = [p.strip() for p in str(prediction).split(",") if p.strip()]
    golds = [g.strip() for g in str(ground_truth).split(",") if g.strip()]
    if not preds or not golds:
        return 0.0
    return sum(max(f1_single(p, g) for p in preds) for g in golds) / len(golds)


def adversarial_correct(prediction: str) -> float:
    """LoCoMo adversarial scoring: model is correct iff it abstains."""
    out = str(prediction).lower()
    return float("no information available" in out or "not mentioned" in out)


def score_qa(category: int, prediction: str, ground_truth) -> float:
    """Apply the right LoCoMo metric for the given category. Returns 0..1."""
    if category == 5:  # adversarial
        return adversarial_correct(prediction)

    gt = ground_truth
    if isinstance(gt, list):
        gt = "; ".join(str(x) for x in gt)
    elif gt is None:
        gt = ""
    else:
        gt = str(gt)

    if category == 3:  # open-domain — LoCoMo splits gold on ';' and takes first
        gt = gt.split(";")[0].strip()

    if category == 1:  # multi-hop
        return f1_multi(prediction, gt)
    elif category in (2, 3, 4):  # temporal, open-domain, single-hop
        return f1_single(prediction, gt)
    else:
        raise ValueError(f"unknown category: {category}")


# ---------------------------------------------------------------------------
# LLM-judge interface — pluggable backend, single source of truth for H1
# ---------------------------------------------------------------------------

@dataclass
class JudgeCall:
    """One LLM-judge invocation, recorded for cost / latency accounting."""
    backend: str
    model: str
    prompt_chars: int
    completion_chars: int
    latency_s: float
    raw_output: str = ""


class JudgeBackend:
    name: str = "abstract"
    model: str = "abstract"

    def judge(self, question: str, prediction: str, gold: str, category: int) -> tuple[float, str]:
        """Return (score in [0,1], raw_output)."""
        raise NotImplementedError


class DummyJudge(JudgeBackend):
    """Always returns the LoCoMo F1 score, no LLM call. Useful for offline runs
    and for sanity-checking the harness without API access."""
    name = "dummy"
    model = "loCoMo-f1"

    def judge(self, question, prediction, gold, category):
        s = score_qa(category, prediction, gold if not isinstance(gold, list) else "; ".join(map(str, gold)))
        return s, f"f1={s:.3f}"


JUDGE_PROMPT = """You are an evaluator for a long-term dialogue memory benchmark.
You will see a question, a model's predicted answer, and a gold reference answer.
Decide whether the model's answer is correct relative to the gold reference.

Be lenient about wording and order — the model is correct if its answer
contains the key facts in the gold reference and adds no contradicting facts.
For temporal questions, accept any phrasing of the same date / time.
For "no information available" or abstention questions, the model is correct
only if it explicitly abstains.

Reply with exactly one token: CORRECT or INCORRECT.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}
Verdict:"""


class OpenAIJudge(JudgeBackend):
    """OpenAI / OpenAI-compatible (vLLM / Together / etc.) judge.
    Reads OPENAI_API_KEY and OPENAI_BASE_URL from env at construction time."""
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        from openai import OpenAI  # lazy import
        self.model = model
        self._client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    def judge(self, question, prediction, gold, category):
        if category == 5:
            # Adversarial questions are deterministic — no need for an LLM call.
            s = adversarial_correct(prediction)
            return s, f"adversarial(rule)={s}"
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold=gold if gold not in (None, "") else "(no answer expected)",
            prediction=prediction,
        )
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        score = 1.0 if "CORRECT" in out and "INCORRECT" not in out else 0.0
        return score, out


def get_judge(name: str | None = None, model: str | None = None) -> JudgeBackend:
    name = name or os.getenv("JUDGE_BACKEND", "dummy")
    if name == "dummy":
        return DummyJudge()
    if name == "openai":
        return OpenAIJudge(model=model or os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    raise ValueError(f"unknown judge backend: {name}")


# ---------------------------------------------------------------------------
# Scorer — aggregates per-system runs and produces the metrics our paper needs
# ---------------------------------------------------------------------------

@dataclass
class SystemRun:
    """One end-to-end run of a memory system over a slice of LoCoMo.
    Tracks accuracy by reasoning type and the cost / latency budget consumed."""
    system_name: str
    backbone: str
    judge_name: str = ""
    n_qas: int = 0
    sum_f1: float = 0.0
    sum_judge: float = 0.0
    by_label_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_label_f1: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_label_judge: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    n_llm_calls: int = 0           # generator + judge + memory writes — see add()
    n_generator_calls: int = 0
    n_judge_calls: int = 0
    n_memory_calls: int = 0
    total_prompt_chars: int = 0
    total_completion_chars: int = 0
    total_wall_time_s: float = 0.0

    def add(
        self,
        *,
        label: str,
        f1: float,
        judge_score: float | None,
        generator_calls: int = 1,
        judge_calls: int = 0,
        memory_calls: int = 0,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        wall_time_s: float = 0.0,
    ) -> None:
        self.n_qas += 1
        self.sum_f1 += f1
        self.by_label_count[label] += 1
        self.by_label_f1[label] += f1
        if judge_score is not None:
            self.sum_judge += judge_score
            self.by_label_judge[label] += judge_score
        self.n_generator_calls += generator_calls
        self.n_judge_calls += judge_calls
        self.n_memory_calls += memory_calls
        self.n_llm_calls += generator_calls + judge_calls + memory_calls
        self.total_prompt_chars += prompt_chars
        self.total_completion_chars += completion_chars
        self.total_wall_time_s += wall_time_s

    @property
    def avg_f1(self) -> float:
        return self.sum_f1 / self.n_qas if self.n_qas else 0.0

    @property
    def avg_judge(self) -> float:
        return self.sum_judge / self.n_qas if self.n_qas else 0.0

    @property
    def f1_by_label(self) -> dict[str, float]:
        return {k: self.by_label_f1[k] / self.by_label_count[k]
                for k in self.by_label_count}

    @property
    def judge_by_label(self) -> dict[str, float]:
        return {k: self.by_label_judge[k] / self.by_label_count[k]
                for k in self.by_label_count}

    @property
    def calls_per_qa(self) -> float:
        return self.n_llm_calls / self.n_qas if self.n_qas else 0.0

    @property
    def f1_per_100_calls(self) -> float:
        """Cost-normalized accuracy: H3 metric."""
        return 100 * self.sum_f1 / max(1, self.n_llm_calls)

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "backbone": self.backbone,
            "judge": self.judge_name,
            "n_qas": self.n_qas,
            "f1": round(self.avg_f1, 4),
            "judge_acc": round(self.avg_judge, 4) if self.sum_judge else None,
            "f1_by_label": {k: round(v, 4) for k, v in self.f1_by_label.items()},
            "judge_by_label": {k: round(v, 4) for k, v in self.judge_by_label.items()} if self.sum_judge else None,
            "n_llm_calls": self.n_llm_calls,
            "calls_per_qa": round(self.calls_per_qa, 3),
            "f1_per_100_calls": round(self.f1_per_100_calls, 4),
            "total_wall_time_s": round(self.total_wall_time_s, 2),
            "total_prompt_chars": self.total_prompt_chars,
            "total_completion_chars": self.total_completion_chars,
        }


def evaluate_predictions(
    qas_with_predictions: Iterable[tuple[Any, str, dict]],
    system_name: str,
    backbone: str,
    judge: JudgeBackend | None = None,
) -> SystemRun:
    """Score a list of (qa, prediction, telemetry) triples and return a SystemRun.

    `qa` is a src.locomo.QAExample.
    `telemetry` is a dict with optional keys: generator_calls, memory_calls,
        prompt_chars, completion_chars, wall_time_s.
    """
    judge = judge or DummyJudge()
    run = SystemRun(system_name=system_name, backbone=backbone, judge_name=judge.name)
    for qa, prediction, telemetry in qas_with_predictions:
        f1 = score_qa(qa.category, prediction, qa.answer)
        if judge.name == "dummy":
            judge_score = None
            judge_calls = 0
        else:
            t0 = time.perf_counter()
            judge_score, _ = judge.judge(qa.question, prediction, qa.answer, qa.category)
            telemetry = dict(telemetry)
            telemetry["wall_time_s"] = telemetry.get("wall_time_s", 0.0) + (time.perf_counter() - t0)
            judge_calls = 1 if qa.category != 5 else 0  # adversarial uses rule
        run.add(
            label=qa.label,
            f1=f1,
            judge_score=judge_score,
            generator_calls=telemetry.get("generator_calls", 1),
            judge_calls=judge_calls,
            memory_calls=telemetry.get("memory_calls", 0),
            prompt_chars=telemetry.get("prompt_chars", 0),
            completion_chars=telemetry.get("completion_chars", 0),
            wall_time_s=telemetry.get("wall_time_s", 0.0),
        )
    return run
