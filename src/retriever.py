"""Retrievers for LoCoMo. Closed-book per-dialogue indexing.

Two backends:
  - BM25Retriever  : rank_bm25 over turn texts
  - DenseRetriever : sentence-transformers/all-MiniLM-L6-v2 (CPU-friendly)

Both share a common interface so the rest of the pipeline is retriever-agnostic.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from src.locomo import Dialogue, Turn

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "")]


def turn_text(t) -> str:
    """Canonical text representation of a turn used by both retrievers.
    Works on LoCoMo Turn (has .speaker) and LMETurn (has .role)."""
    speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
    text = getattr(t, "text", "") or getattr(t, "content", "")
    return f"{speaker}: {text}"


@dataclass
class RetrievalHit:
    turn: Turn
    score: float


class Retriever:
    """Abstract per-dialogue retriever."""
    name: str = "abstract"

    def index(self, dialogue: Dialogue) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalHit]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self):
        self._turns: list[Turn] = []
        self._bm25 = None

    def index(self, dialogue: Dialogue) -> None:
        from rank_bm25 import BM25Okapi  # lazy
        self._turns = list(dialogue.turns)
        tokenized = [_tokenize(turn_text(t)) for t in self._turns]
        # Avoid empty docs (rank_bm25 dislikes them)
        tokenized = [doc if doc else ["<empty>"] for doc in tokenized]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalHit]:
        if self._bm25 is None:
            raise RuntimeError("call .index(dialogue) first")
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(self._turns)), key=lambda i: -scores[i])[:k]
        return [RetrievalHit(self._turns[i], float(scores[i])) for i in ranked]


# Module-level model cache so we encode once across many dialogues.
_DENSE_MODEL = None


def _get_dense_model(model_name: str):
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy
        _DENSE_MODEL = SentenceTransformer(model_name, device="cpu")
    return _DENSE_MODEL


class DenseRetriever(Retriever):
    name = "dense-minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._turns: list[Turn] = []
        self._embeds = None  # numpy array (n_turns, dim)

    def index(self, dialogue: Dialogue) -> None:
        import numpy as np
        model = _get_dense_model(self._model_name)
        self._turns = list(dialogue.turns)
        texts = [turn_text(t) for t in self._turns]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                            batch_size=64, convert_to_numpy=True)
        self._embeds = embs.astype(np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalHit]:
        import numpy as np
        if self._embeds is None:
            raise RuntimeError("call .index(dialogue) first")
        model = _get_dense_model(self._model_name)
        q = model.encode([query], normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True)[0].astype(self._embeds.dtype)
        scores = self._embeds @ q  # cosine since both normalized
        ranked = np.argsort(-scores)[:k].tolist()
        return [RetrievalHit(self._turns[i], float(scores[i])) for i in ranked]


def get_retriever(name: str) -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    if name in ("dense", "dense-minilm"):
        return DenseRetriever()
    raise ValueError(f"unknown retriever: {name}")
