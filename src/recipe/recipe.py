"""C-RAG end-to-end orchestrator.

Per-question pipeline:
  1. Router → category
  2. Retrieve top-K turns via BM25 over the question's parent dialogue/haystack
  3. Format the prompt template with retrieved evidence
  4. Generator → answer
  5. Return CRAGResult with telemetry

Designed so the entire pipeline runs end-to-end on CPU with DummyLLM, AND
swaps in vLLM-served Qwen2.5-14B by changing one constructor argument.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.recipe.generator import Generator, DummyLLM, GenerationCall
from src.recipe.prompts import load_template
from src.recipe.retrieval import (
    retrieve_evidence,
    retrieve_evidence_amem_evolution,
    retrieve_evidence_amem_faithful,
    retrieve_evidence_bm25_rerank,
    retrieve_evidence_dense,
    retrieve_evidence_hipporag,
    retrieve_evidence_linked_notes,
    retrieve_evidence_linked_sessions,
    retrieve_evidence_session_bank,
)
from src.recipe.router import Router, RuleBasedRouter, RouteResult


@dataclass
class CRAGResult:
    question: str
    prediction: str
    routed_category: str
    rule_fired: str
    n_retrieved_turns: int
    n_llm_calls: int
    total_prompt_chars: int
    total_completion_chars: int
    wall_time_s: float
    raw_calls: list[GenerationCall] = field(default_factory=list)


class CRAG:
    """Category-Aware RAG: route → retrieve → prompt → generate."""

    def __init__(
        self,
        benchmark: str = "locomo",
        router: Router | None = None,
        generator: Generator | None = None,
        top_k: int = 5,
        max_tokens: int = 96,
        template_category_override: str | None = None,
        retrieval_mode: str = "flat",
        prompt_variant: str = "tight",
    ):
        if benchmark not in ("locomo", "longmemeval", "beam"):
            raise ValueError(f"unknown benchmark: {benchmark}")
        if prompt_variant not in ("tight", "loose", "tight-fcs", "tight-stripped"):
            raise ValueError(f"unknown prompt_variant: {prompt_variant}")
        self.benchmark = benchmark
        self.router = router or RuleBasedRouter(benchmark)
        self.generator = generator or DummyLLM("echo_question")
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.template_category_override = template_category_override
        self.retrieval_mode = retrieval_mode
        self.prompt_variant = prompt_variant

    def answer(self, question: str, dialogue_or_haystack: Any) -> CRAGResult:
        t0 = time.perf_counter()
        # 1. Route
        route = self.router.route(question)

        # 2. Retrieve top-K turns + format with timestamps if temporal
        if self.retrieval_mode == "linked-sessions":
            retrieved = retrieve_evidence_linked_sessions(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        elif self.retrieval_mode == "linked-notes":
            retrieved = retrieve_evidence_linked_notes(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        elif self.retrieval_mode == "session-bank":
            retrieved = retrieve_evidence_session_bank(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        elif self.retrieval_mode == "amem-faithful":
            retrieved = retrieve_evidence_amem_faithful(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
                note_generator=self.generator,
            )
        elif self.retrieval_mode == "amem-evolution":
            retrieved = retrieve_evidence_amem_evolution(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
                note_generator=self.generator,
            )
        elif self.retrieval_mode == "dense":
            retrieved = retrieve_evidence_dense(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        elif self.retrieval_mode == "bm25-rerank":
            retrieved = retrieve_evidence_bm25_rerank(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        elif self.retrieval_mode == "hipporag":
            retrieved = retrieve_evidence_hipporag(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        else:
            retrieved = retrieve_evidence(
                dialogue_or_haystack,
                question,
                top_k=self.top_k,
                benchmark=self.benchmark,
            )
        n_retrieved = len(retrieved.turns)

        # 3. Format prompt with the appropriate template
        if self.prompt_variant == "loose":
            template_category = "loose"
            template = load_template(self.benchmark, template_category)
        elif self.prompt_variant == "tight-fcs":
            template_category = self.template_category_override or route.category
            template = load_template(f"{self.benchmark}-fcs", template_category)
        elif self.prompt_variant == "tight-stripped":
            template_category = self.template_category_override or route.category
            template = load_template(f"{self.benchmark}-stripped", template_category)
        else:
            template_category = self.template_category_override or route.category
            template = load_template(self.benchmark, template_category)
        prompt_kwargs = {
            "question": question,
            "retrieved_turns": retrieved.text_block,
            "retrieved_turns_with_timestamps": retrieved.text_block_with_timestamps,
        }
        # LoCoMo prompts mention speakers; LME doesn't
        if self.benchmark == "locomo":
            prompt_kwargs["speaker_a"] = retrieved.speaker_a or "Speaker A"
            prompt_kwargs["speaker_b"] = retrieved.speaker_b or "Speaker B"
        try:
            prompt = template.format(**prompt_kwargs)
        except KeyError as e:
            # Some templates may not have all placeholders; fall back to safe substitution
            for k in list(prompt_kwargs):
                template = template.replace("{" + k + "}", str(prompt_kwargs[k]))
            prompt = template

        # 4. Generate
        call = self.generator.generate(prompt, max_tokens=self.max_tokens)

        # 5. Wrap result
        wall = time.perf_counter() - t0
        return CRAGResult(
            question=question,
            prediction=call.completion,
            routed_category=route.category,
            rule_fired=route.rule_fired,
            n_retrieved_turns=n_retrieved,
            n_llm_calls=1,
            total_prompt_chars=call.prompt_chars,
            total_completion_chars=call.completion_chars,
            wall_time_s=wall,
            raw_calls=[call],
        )
