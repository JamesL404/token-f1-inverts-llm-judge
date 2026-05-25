"""Multi-call ReAct-lite memory architecture.

Tests whether the architecture-zero finding survives in a multi-call regime,
which is what MemGPT and similar systems use.

Mechanism (2 calls per question):
  1. CALL 1 (planner): given question and initial BM25 evidence, the model
     either emits a final ANSWER or emits a refined RETRIEVAL-QUERY.
  2. If RETRIEVAL-QUERY: re-run BM25 with the refined query, then CALL 2
     (answerer) with the new evidence.
  3. If ANSWER: that is the final answer (1-call).

The architecture lever here is the planner's *option to refine retrieval*
before answering. If the planner consistently improves over single-call
flat, the architecture-zero finding narrows to single-call settings; if
not, it generalizes.

Reuses the existing CRAG generator; only the orchestration is multi-call.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.recipe.generator import GenerationCall, Generator
from src.recipe.prompts import load_template
from src.recipe.retrieval import retrieve_evidence
from src.recipe.router import RouteResult, Router, RuleBasedRouter


PLANNER_PROMPT_TEMPLATE = """You are answering a question about a long conversation.

You will see retrieved excerpts. You have two options:

OPTION A: If the excerpts contain enough information to answer the question, output:
ANSWER: <your concise final answer>

OPTION B: If the excerpts do not contain the answer or you need different evidence, output:
SEARCH: <a refined search query that targets the missing information>

Excerpts:
{retrieved_turns_with_timestamps}

Question: {question}

Reply with exactly ONE line, starting with either ANSWER: or SEARCH:."""


ANSWERER_PROMPT_TEMPLATE = """You are answering a question about a long conversation. Use the retrieved excerpts to give a concise final answer.

Excerpts:
{retrieved_turns_with_timestamps}

Question: {question}

Final answer:"""


@dataclass
class MultiCallResult:
    question: str
    prediction: str
    routed_category: str
    rule_fired: str
    n_retrieved_turns: int
    n_llm_calls: int
    total_prompt_chars: int
    total_completion_chars: int
    wall_time_s: float
    refined_query: str | None = None
    used_search: bool = False
    raw_calls: list[GenerationCall] = field(default_factory=list)


class MultiCallReAct:
    """Two-call ReAct-lite: planner decides ANSWER-or-SEARCH; if SEARCH, re-retrieve and answer."""

    def __init__(
        self,
        benchmark: str = "longmemeval",
        router: Router | None = None,
        generator: Generator | None = None,
        top_k: int = 5,
        max_tokens: int = 96,
    ):
        if benchmark not in ("longmemeval", "locomo"):
            raise ValueError(f"unknown benchmark: {benchmark}")
        self.benchmark = benchmark
        self.router = router or RuleBasedRouter(benchmark)
        self.generator = generator
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, question: str, container: Any) -> MultiCallResult:
        t0 = time.perf_counter()
        route = self.router.route(question)

        # Initial flat BM25 retrieval
        retrieved = retrieve_evidence(container, question, top_k=self.top_k, benchmark=self.benchmark)
        speaker_a = retrieved.speaker_a or "user"
        speaker_b = retrieved.speaker_b or "assistant"

        # Call 1: planner
        planner_prompt = PLANNER_PROMPT_TEMPLATE.format(
            retrieved_turns_with_timestamps=retrieved.text_block_with_timestamps[:6000],
            question=question,
        )
        c1 = self.generator.generate(planner_prompt, max_tokens=self.max_tokens)

        # Parse planner output
        planner_text = c1.completion.strip()
        used_search = False
        refined_query = None
        prediction = ""

        # Look for ANSWER: or SEARCH: prefix (case-insensitive, robust to whitespace)
        m_ans = re.match(r"^\s*ANSWER\s*:\s*(.+?)\s*(?:$|\n)", planner_text, re.IGNORECASE | re.DOTALL)
        m_search = re.match(r"^\s*SEARCH\s*:\s*(.+?)\s*(?:$|\n)", planner_text, re.IGNORECASE | re.DOTALL)
        calls = [c1]

        if m_ans and not m_search:
            prediction = m_ans.group(1).strip().splitlines()[0]
        elif m_search:
            used_search = True
            refined_query = m_search.group(1).strip().splitlines()[0]
            # Re-retrieve with refined query
            retrieved2 = retrieve_evidence(container, refined_query, top_k=self.top_k, benchmark=self.benchmark)
            answerer_prompt = ANSWERER_PROMPT_TEMPLATE.format(
                retrieved_turns_with_timestamps=retrieved2.text_block_with_timestamps[:6000],
                question=question,
            )
            c2 = self.generator.generate(answerer_prompt, max_tokens=self.max_tokens)
            calls.append(c2)
            prediction = c2.completion.strip().splitlines()[0] if c2.completion else ""
        else:
            # Couldn't parse; fall back to using the raw output as the answer
            prediction = planner_text.splitlines()[0] if planner_text else ""

        wall = time.perf_counter() - t0
        return MultiCallResult(
            question=question, prediction=prediction,
            routed_category=route.category, rule_fired=route.rule_fired,
            n_retrieved_turns=len(retrieved.turns),
            n_llm_calls=len(calls),
            total_prompt_chars=sum(c.prompt_chars for c in calls),
            total_completion_chars=sum(c.completion_chars for c in calls),
            wall_time_s=wall,
            refined_query=refined_query, used_search=used_search,
            raw_calls=calls,
        )
