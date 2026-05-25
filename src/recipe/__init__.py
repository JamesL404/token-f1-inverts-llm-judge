"""C-RAG: Category-Aware Retrieval-Augmented Generation.

Public API:
  - Router: classifies a question into a reasoning category
  - PromptTemplate: per-category prompt format
  - Generator: pluggable LLM backend (DummyLLM for CPU sanity / OpenAILLM for vLLM/API)
  - CRAG: end-to-end orchestrator
"""
from src.recipe.router import (
    Router,
    RuleBasedRouter,
    LOCOMO_CATEGORIES,
    LONGMEMEVAL_CATEGORIES,
)
from src.recipe.prompts import load_template, list_templates
from src.recipe.generator import Generator, DummyLLM, OpenAILLM
from src.recipe.recipe import CRAG, CRAGResult

__all__ = [
    "Router", "RuleBasedRouter",
    "LOCOMO_CATEGORIES", "LONGMEMEVAL_CATEGORIES",
    "load_template", "list_templates",
    "Generator", "DummyLLM", "OpenAILLM",
    "CRAG", "CRAGResult",
]
