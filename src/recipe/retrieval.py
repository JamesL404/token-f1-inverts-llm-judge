"""Retrieval helper for C-RAG.

Wraps src.retriever.BM25Retriever to handle both LoCoMo dialogues and
LongMemEval per-question haystacks. Returns a structured Evidence object
with text blocks pre-formatted for the prompt templates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retriever import BM25Retriever


@dataclass
class Evidence:
    """Retrieved evidence ready to be slotted into a prompt template."""
    turns: list = field(default_factory=list)         # list of Turn-like objects
    text_block: str = ""                              # plain "Speaker: text" lines
    text_block_with_timestamps: str = ""              # "[date] Speaker: text" lines
    speaker_a: str = ""
    speaker_b: str = ""


def _format_evidence(turns: list, speaker_a: str, speaker_b: str) -> Evidence:
    def turn_line(t):
        speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
        text = getattr(t, "text", "")
        return f"{speaker}: {text}"

    def turn_line_dated(t):
        speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
        text = getattr(t, "text", "")
        date = getattr(t, "session_date_time", None) or getattr(t, "session_date", "")
        date_str = f"[{date}] " if date else ""
        return f"{date_str}{speaker}: {text}"

    return Evidence(
        turns=turns,
        text_block="\n".join(turn_line(t) for t in turns),
        text_block_with_timestamps="\n".join(turn_line_dated(t) for t in turns),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )


def retrieve_evidence(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "locomo",
) -> Evidence:
    """Retrieve top-K turns for a question from a LoCoMo Dialogue or
    LongMemEval LMEQuestion. The container exposes a `turns` attribute
    (LoCoMo Dialogue) or a list-of-sessions structure (LongMemEval).
    """
    # Build a BM25 index over the container's turns. Both Dialogue and
    # LMEQuestion expose a `turns` property; for LME the per-question
    # haystack is treated as the corpus.
    retriever = BM25Retriever()
    if benchmark == "locomo":
        # container is a src.locomo.Dialogue
        retriever.index(container)
        speaker_a = getattr(container, "speaker_a", "")
        speaker_b = getattr(container, "speaker_b", "")
    else:
        # container is a src.longmemeval.LMEQuestion; we need a Dialogue-like
        # interface for BM25Retriever.index. Build a tiny shim.
        class _Shim:
            pass
        shim = _Shim()
        shim.turns = container.turns  # property exists on LMEQuestion
        retriever.index(shim)
        speaker_a = "user"
        speaker_b = "assistant"

    hits = retriever.retrieve(question, k=top_k)
    turns = [h.turn for h in hits]

    return _format_evidence(turns, speaker_a, speaker_b)


_NOTE_CACHE: dict[str, str] = {}
_EVOLVED_CACHE: dict[str, list[str]] = {}


def _split_notes(text: str) -> list[str]:
    """Parse a numbered/bulleted note list into individual notes."""
    import re
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    notes = []
    for ln in lines:
        ln = re.sub(r"^[\-\*•–]\s*", "", ln)
        ln = re.sub(r"^\d+[\.)]\s*", "", ln)
        if ln:
            notes.append(ln)
    return notes


def retrieve_evidence_amem_evolution(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    note_generator=None,
) -> Evidence:
    """Faithful A-MEM with memory evolution + cross-session linking.

    Heavier than retrieve_evidence_amem_faithful: at index time, each session's
    notes are computed AND a second LLM pass merges/updates contradicted notes
    across all sessions in the haystack (the 'evolution' step). At query time,
    BM25 over the evolved global note set; return top-K turns from source
    sessions of the top notes.

    Closer to A-MEM's actual mechanism. Cost: ~2 LLM calls per haystack
    (1 per session for extraction + 1 global for evolution). Cached by
    haystack-content hash so re-running on the same haystack is free.
    """
    if benchmark != "longmemeval":
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)
    if note_generator is None:
        return retrieve_evidence_session_bank(container, question, top_k=top_k, benchmark=benchmark)

    sessions = list(getattr(container, "sessions", []))
    if not sessions:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    def session_text(session) -> str:
        return "\n".join(
            f"{getattr(t, 'role', '')}: {getattr(t, 'text', '')}"
            for t in session
        )

    sess_texts = [session_text(s) for s in sessions]

    # Step 1: extract notes per session (reuse note cache from amem_faithful)
    def note_text(sess_text: str) -> str:
        h = hash(sess_text)
        key = f"{benchmark}:{h}"
        if key in _NOTE_CACHE:
            return _NOTE_CACHE[key]
        prompt = (
            "Extract 3-5 short factual notes from this conversation session. "
            "Each note should be one short sentence stating a fact, preference, "
            "event, or update mentioned by the user or assistant. "
            "Output the notes as a numbered list, nothing else.\n\n"
            f"Session:\n{sess_text[:4000]}\n\nNotes:"
        )
        call = note_generator.generate(prompt, max_tokens=160)
        notes = call.completion.strip()
        _NOTE_CACHE[key] = notes
        return notes

    per_sess_notes = [note_text(t) for t in sess_texts]

    # Step 2: GLOBAL evolution pass — merge contradicted notes across all sessions
    haystack_key = f"{benchmark}_evol:" + "|".join(str(hash(t)) for t in sess_texts)
    if haystack_key in _EVOLVED_CACHE:
        evolved = _EVOLVED_CACHE[haystack_key]
        sess_index_for_note = [i for i, sess_notes in enumerate(per_sess_notes) for _ in _split_notes(sess_notes)]
        # Re-derive evolved-to-source mapping from cache (simple: round-robin distribute)
        sess_index_for_evolved = sess_index_for_note[:len(evolved)] if len(sess_index_for_note) >= len(evolved) else sess_index_for_note + [sess_index_for_note[-1]] * (len(evolved) - len(sess_index_for_note))
    else:
        all_notes_with_src: list[tuple[int, str]] = []
        for sidx, sess_notes in enumerate(per_sess_notes):
            for n in _split_notes(sess_notes):
                all_notes_with_src.append((sidx, n))
        if not all_notes_with_src:
            return retrieve_evidence_session_bank(container, question, top_k=top_k, benchmark=benchmark)
        bullet_block = "\n".join(f"- [s{sidx}] {n}" for sidx, n in all_notes_with_src)
        evolution_prompt = (
            "You are reviewing memory notes extracted from multiple conversation sessions. "
            "Some notes may be UPDATED by later notes (e.g. 'lives in Boston' then later 'moved to NYC'). "
            "Some may be DUPLICATES with slight rephrasing. "
            "Output a clean list of the FINAL state of each fact: keep the latest update, drop duplicates, mark unchanged facts. "
            "Output as a numbered list of one-sentence facts. Do not add new information.\n\n"
            f"Notes (with source session tag):\n{bullet_block[:4000]}\n\n"
            "Final consolidated notes:"
        )
        call = note_generator.generate(evolution_prompt, max_tokens=320)
        evolved_text = call.completion.strip()
        evolved = _split_notes(evolved_text)
        if not evolved:
            evolved = [n for _, n in all_notes_with_src]  # fallback to raw notes
        _EVOLVED_CACHE[haystack_key] = evolved
        # For the source-mapping, we approximate: assign each evolved note to the
        # source-session of the most-similar input note via lexical match
        sess_index_for_evolved = []
        for evol_n in evolved:
            evol_low = evol_n.lower()
            best_sidx, best_overlap = all_notes_with_src[0][0], 0
            evol_words = set(evol_low.split())
            for sidx, raw_n in all_notes_with_src:
                raw_words = set(raw_n.lower().split())
                overlap = len(evol_words & raw_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sidx = sidx
            sess_index_for_evolved.append(best_sidx)

    # Step 3: BM25-rank evolved notes against query
    note_retriever = BM25Retriever()
    note_shim = _Shim()
    note_shim.turns = [
        type("NoteDoc", (), {"role": f"evol_{i}", "text": evolved[i], "session_index": sess_index_for_evolved[i]})()
        for i in range(len(evolved))
    ]
    note_retriever.index(note_shim)
    note_hits = note_retriever.retrieve(question, k=min(3, len(evolved)))
    selected = sorted({getattr(hit.turn, "session_index") for hit in note_hits})

    # Step 4: return top-K turns from selected sessions, ranked by query-BM25
    turn_retriever = BM25Retriever()
    turn_shim = _Shim()
    turn_shim.turns = [t for idx in selected for t in sessions[idx]]
    turn_retriever.index(turn_shim)
    turns = [h.turn for h in turn_retriever.retrieve(question, k=top_k)]
    return _format_evidence(turns, "user", "assistant")


def retrieve_evidence_amem_faithful(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    note_generator=None,
) -> Evidence:
    """Faithful-ish A-MEM-style memory: LLM-extracted notes + content linking.

    Mechanism (heavier than linked-notes / session-bank):
      1. INDEX TIME: for each session in the haystack, call `note_generator`
         to extract 3-5 'memory notes' (key facts as short sentences).
         Cached by session text hash so the same session across questions
         only pays the LLM cost once.
      2. QUERY TIME: BM25-rank the notes against the query; take top-K
         notes; gather the source sessions of those notes; return the top-K
         most query-relevant TURNS within those sessions.

    Compared to session-bank, this adds an LLM-driven note-extraction step
    at index time. Compared to linked-notes, this uses LLM semantics rather
    than lexical similarity. Closer to A-MEM's actual mechanism, but does
    not include memory evolution or learned forgetting.

    Cost: ~1 LLM call per unique session (cached). For LME oracle ~150
    unique sessions across the 500 questions, this is a one-time ~3 min
    cost on Qwen-14B.
    """
    if benchmark != "longmemeval":
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    if note_generator is None:
        # Fallback: use session-bank if no note generator provided
        return retrieve_evidence_session_bank(container, question, top_k=top_k, benchmark=benchmark)

    sessions = list(getattr(container, "sessions", []))
    if not sessions:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    def session_text(session) -> str:
        return "\n".join(
            f"{getattr(t, 'role', '')}: {getattr(t, 'text', '')}"
            for t in session
        )

    def note_text(sess_idx: int, sess_text: str) -> str:
        h = hash(sess_text)
        key = f"{benchmark}:{h}"
        if key in _NOTE_CACHE:
            return _NOTE_CACHE[key]
        prompt = (
            "Extract 3-5 short factual notes from this conversation session. "
            "Each note should be one short sentence stating a fact, preference, "
            "event, or update mentioned by the user or assistant. "
            "Output the notes as a numbered list, nothing else.\n\n"
            f"Session:\n{sess_text[:4000]}\n\nNotes:"
        )
        call = note_generator.generate(prompt, max_tokens=160)
        notes = call.completion.strip()
        _NOTE_CACHE[key] = notes
        return notes

    note_strs = [note_text(i, t) for i, t in enumerate([session_text(s) for s in sessions])]

    # BM25-rank notes against the query
    note_retriever = BM25Retriever()
    note_shim = _Shim()
    note_shim.turns = [
        type("NoteDoc", (), {"role": f"note_{i}", "text": n, "session_index": i})()
        for i, n in enumerate(note_strs)
    ]
    note_retriever.index(note_shim)
    note_hits = note_retriever.retrieve(question, k=min(2, len(note_strs)))
    selected = sorted({getattr(hit.turn, "session_index") for hit in note_hits})

    # Return top-K turns from selected sessions, ranked by query-BM25
    turn_retriever = BM25Retriever()
    turn_shim = _Shim()
    turn_shim.turns = [t for idx in selected for t in sessions[idx]]
    turn_retriever.index(turn_shim)
    turns = [h.turn for h in turn_retriever.retrieve(question, k=top_k)]
    return _format_evidence(turns, "user", "assistant")


_DENSE_ENCODER = None
_RERANKER = None


def _get_dense_encoder(model_name: str = "intfloat/e5-base-v2"):
    """Lazy-load a SentenceTransformer encoder."""
    global _DENSE_ENCODER
    if _DENSE_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _DENSE_ENCODER = SentenceTransformer(model_name, device="cpu")
    return _DENSE_ENCODER


def _get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load a CrossEncoder reranker."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(model_name, device="cpu")
    return _RERANKER


def retrieve_evidence_dense(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
) -> Evidence:
    """Dense retrieval ablation: replace BM25 with E5-base-v2 cosine similarity."""
    if benchmark == "locomo":
        all_turns = list(getattr(container, "turns", []))
        speaker_a = getattr(container, "speaker_a", "")
        speaker_b = getattr(container, "speaker_b", "")
    else:
        all_turns = list(getattr(container, "turns", []))
        speaker_a, speaker_b = "user", "assistant"

    if not all_turns:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    encoder = _get_dense_encoder()
    # E5 expects 'query: ' and 'passage: ' prefixes
    q_text = f"query: {question}"
    p_texts = []
    for t in all_turns:
        speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
        text = getattr(t, "text", "")
        p_texts.append(f"passage: {speaker}: {text}")

    import numpy as np
    q_emb = encoder.encode([q_text], normalize_embeddings=True)
    p_embs = encoder.encode(p_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    scores = (q_emb @ p_embs.T).flatten()
    top_idx = np.argsort(-scores)[:top_k]
    turns = [all_turns[i] for i in top_idx]
    return _format_evidence(turns, speaker_a, speaker_b)


def retrieve_evidence_bm25_rerank(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    pool_size: int = 20,
) -> Evidence:
    """BM25 top-20 then cross-encoder rerank to top-5."""
    if benchmark == "locomo":
        all_turns = list(getattr(container, "turns", []))
        speaker_a = getattr(container, "speaker_a", "")
        speaker_b = getattr(container, "speaker_b", "")
    else:
        all_turns = list(getattr(container, "turns", []))
        speaker_a, speaker_b = "user", "assistant"

    if not all_turns:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    bm25 = BM25Retriever()
    shim = _Shim()
    shim.turns = all_turns
    bm25.index(shim)
    bm25_hits = bm25.retrieve(question, k=min(pool_size, len(all_turns)))
    pool = [h.turn for h in bm25_hits]

    if not pool:
        return _format_evidence([], speaker_a, speaker_b)

    reranker = _get_reranker()
    pairs = []
    for t in pool:
        speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
        text = getattr(t, "text", "")
        pairs.append([question, f"{speaker}: {text}"])
    scores = reranker.predict(pairs, show_progress_bar=False)
    import numpy as np
    top_idx = np.argsort(-scores)[:top_k]
    turns = [pool[i] for i in top_idx]
    return _format_evidence(turns, speaker_a, speaker_b)


_HIPPORAG_GRAPH_CACHE: dict[str, dict] = {}


def _extract_entities_simple(text: str) -> list[str]:
    """Cheap entity extraction: capitalized noun phrases + numbers + dates.

    Avoids spaCy dependency. Captures most named entities for our purposes.
    """
    import re
    # Multi-word capitalized phrases (e.g. "John Smith", "Open AI")
    cap_phrases = re.findall(r"\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", text)
    # Standalone numbers and dates
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text)
    months = re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b", text, re.IGNORECASE)
    ents = set()
    for x in cap_phrases + nums + dates + months:
        x = x.strip()
        if len(x) >= 2 and x.lower() not in {"the", "a", "an", "i", "it"}:
            ents.add(x.lower())
    return list(ents)


def retrieve_evidence_hipporag(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    n_seed_sessions: int = 2,
    ppr_alpha: float = 0.85,
) -> Evidence:
    """HippoRAG-lite: entity extraction + Personalized PageRank over session-entity graph.

    Mechanism:
      1. INDEX TIME: extract entities (capitalized phrases, numbers, dates) per
         session; build a bipartite graph where sessions are nodes connected
         to the entities they mention.
      2. QUERY TIME: extract entities from the question; run Personalized
         PageRank (PPR) seeded on those entities; rank sessions by PPR score.
         Take top n_seed_sessions sessions.
      3. ANSWER TIME: BM25 within those sessions to get top-K turns.

    This is mechanistically distinct from BM25 (lexical), linked-notes
    (turn-similarity link expansion), session-bank (session-as-doc BM25),
    and amem-faithful (LLM-extracted notes). HippoRAG-lite uses graph
    structure over entities — the closest matched-pipeline approximation
    of Gutiérrez et al. 2024 without LLM-driven entity normalization.
    """
    if benchmark != "longmemeval":
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    sessions = list(getattr(container, "sessions", []))
    if not sessions:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    def session_text(session) -> str:
        return "\n".join(
            f"{getattr(t, 'role', '')}: {getattr(t, 'text', '')}"
            for t in session
        )

    sess_texts = [session_text(s) for s in sessions]
    haystack_key = "|".join(str(hash(t)) for t in sess_texts)

    if haystack_key in _HIPPORAG_GRAPH_CACHE:
        cached = _HIPPORAG_GRAPH_CACHE[haystack_key]
        sess_to_ents = cached["sess_to_ents"]
        ent_to_sess = cached["ent_to_sess"]
    else:
        sess_to_ents: list[set[str]] = []
        ent_to_sess: dict[str, list[int]] = {}
        for sidx, st in enumerate(sess_texts):
            ents = set(_extract_entities_simple(st))
            sess_to_ents.append(ents)
            for e in ents:
                ent_to_sess.setdefault(e, []).append(sidx)
        _HIPPORAG_GRAPH_CACHE[haystack_key] = {"sess_to_ents": sess_to_ents, "ent_to_sess": ent_to_sess}

    # Build the bipartite graph
    import networkx as nx
    g = nx.Graph()
    for sidx in range(len(sessions)):
        g.add_node(("s", sidx), bipartite=0)
    for e in ent_to_sess:
        g.add_node(("e", e), bipartite=1)
        for sidx in ent_to_sess[e]:
            g.add_edge(("s", sidx), ("e", e))

    # Seed PPR on entities mentioned in the question
    q_ents = set(_extract_entities_simple(question)) & set(ent_to_sess.keys())
    if not q_ents:
        # No entity overlap — fall back to session-bank
        return retrieve_evidence_session_bank(container, question, top_k=top_k, benchmark=benchmark)

    personalization = {("e", e): 1.0 for e in q_ents}
    # Add small personalization on all session nodes to keep PPR well-defined
    for sidx in range(len(sessions)):
        personalization[("s", sidx)] = personalization.get(("s", sidx), 0.0) + 0.001

    try:
        pr = nx.pagerank(g, alpha=ppr_alpha, personalization=personalization, max_iter=50)
    except Exception:
        return retrieve_evidence_session_bank(container, question, top_k=top_k, benchmark=benchmark)

    # Rank sessions by PPR score
    sess_scores = [(sidx, pr.get(("s", sidx), 0.0)) for sidx in range(len(sessions))]
    sess_scores.sort(key=lambda x: -x[1])
    selected = sorted(sidx for sidx, _ in sess_scores[:n_seed_sessions])

    # BM25 within selected sessions
    turn_retriever = BM25Retriever()
    turn_shim = _Shim()
    turn_shim.turns = [t for idx in selected for t in sessions[idx]]
    turn_retriever.index(turn_shim)
    turns = [h.turn for h in turn_retriever.retrieve(question, k=top_k)]
    return _format_evidence(turns, "user", "assistant")


def retrieve_evidence_session_bank(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    n_sessions: int = 2,
) -> Evidence:
    """Architecture-family memory-bank / session-summary baseline.

    Mechanism:
      1. Treat each session as a document (concatenate role: text per turn).
      2. BM25-rank sessions against the query.
      3. Take the top `n_sessions`. Return turns from those sessions, capped
         at `top_k` total — picked by query-BM25 within the selected
         session set.

    Difference from `linked-sessions`:
      - linked_sessions seeds at session level then EXPANDS via lexical
        link to extra sessions. This baseline does NOT expand — it just
        retrieves sessions and reads.
      - It is the architecture-family equivalent of "memory bank" /
        "session summary" approaches (Mem0g, MemoryBank-style) without
        the LLM-summary index step. The substrate is the same: rank
        and select at session granularity, then read.

    Cost: 2 BM25 indexes (sessions + selected turns), no extra LLM calls.
    """
    if benchmark not in ("longmemeval", "locomo"):
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    if benchmark == "longmemeval":
        sessions = list(getattr(container, "sessions", []))
        speaker_a, speaker_b = "user", "assistant"
    else:
        # LoCoMo Dialogue exposes `session_turns(session_id) -> list[Turn]`
        # plus `num_sessions`. Iterate session ids in order.
        n_sess = getattr(container, "num_sessions", 0)
        if n_sess and hasattr(container, "session_turns"):
            sessions = []
            for sid in sorted(set(t.session_id for t in container.turns)):
                sess_turns = container.session_turns(sid)
                if sess_turns:
                    sessions.append(sess_turns)
        else:
            sessions = []
        speaker_a = getattr(container, "speaker_a", "")
        speaker_b = getattr(container, "speaker_b", "")
    if not sessions:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    def session_text(session) -> str:
        return "\n".join(
            f"{getattr(t, 'speaker', None) or getattr(t, 'role', '')}: {getattr(t, 'text', '')}"
            for t in session
        )

    session_texts = [session_text(s) for s in sessions]
    seed_retriever = BM25Retriever()
    seed_shim = _Shim()
    seed_shim.turns = [
        type("SessionDoc", (), {"role": f"session_{idx}", "text": text, "session_index": idx})()
        for idx, text in enumerate(session_texts)
    ]
    seed_retriever.index(seed_shim)
    seed_hits = seed_retriever.retrieve(question, k=min(n_sessions, len(sessions)))
    selected = sorted({getattr(hit.turn, "session_index") for hit in seed_hits})

    turn_retriever = BM25Retriever()
    turn_shim = _Shim()
    turn_shim.turns = [t for idx in selected for t in sessions[idx]]
    turn_retriever.index(turn_shim)
    turns = [h.turn for h in turn_retriever.retrieve(question, k=top_k)]
    return _format_evidence(turns, speaker_a, speaker_b)


def retrieve_evidence_linked_notes(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    seed_notes: int = 3,
    links_per_seed: int = 3,
) -> Evidence:
    """Architecture-family A-MEM-style linking-memory retrieval.

    The mechanism: each turn is a 'note'. At index time we precompute, for
    each seed note retrieved against the query, its lexically-similar
    neighbour notes from elsewhere in the same haystack. At query time the
    candidate pool is the union (seeds ∪ neighbours-of-seeds), scored once
    more by query-BM25.

    This is deliberately not a faithful A-MEM reproduction. Per the proposal
    (paper/proposal.md, §"Baseline taxonomy"), this is an
    architecture-family baseline whose purpose is to compare
    `with-content-linking` against `without-content-linking` (vanilla
    BM25) on the same backbone. v1 uses lexical (BM25) self-similarity for
    linking; an embedding-based v2 is on the to-do list.

    Difference from retrieve_evidence_linked_sessions:
    - linked_sessions: session-level expansion (coarse)
    - linked_notes:    turn/note-level expansion driven by note-to-note
                       content similarity (fine, content-linked)
    """
    if benchmark not in ("longmemeval", "locomo"):
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    if benchmark == "locomo":
        all_turns = list(getattr(container, "turns", []))
        speaker_a = getattr(container, "speaker_a", "")
        speaker_b = getattr(container, "speaker_b", "")
    else:
        all_turns = list(getattr(container, "turns", []))
        speaker_a = "user"
        speaker_b = "assistant"

    if not all_turns:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    seed_retriever = BM25Retriever()
    seed_shim = _Shim()
    seed_shim.turns = all_turns
    seed_retriever.index(seed_shim)
    seed_hits = seed_retriever.retrieve(
        question, k=min(seed_notes, len(all_turns))
    )

    seed_indices: list[int] = []
    seed_id = id  # use Python id as a stable handle
    seed_id_set: set[int] = set()
    seed_objs: list = []
    for hit in seed_hits:
        turn = hit.turn
        if seed_id(turn) in seed_id_set:
            continue
        seed_id_set.add(seed_id(turn))
        seed_objs.append(turn)

    def turn_text(t) -> str:
        speaker = getattr(t, "speaker", None) or getattr(t, "role", "")
        text = getattr(t, "text", "")
        return f"{speaker}: {text}".strip()

    selected: list = list(seed_objs)
    selected_ids: set[int] = set(seed_id_set)

    for seed_turn in seed_objs:
        candidate_pool = [t for t in all_turns if seed_id(t) not in selected_ids]
        if not candidate_pool:
            break
        link_retriever = BM25Retriever()
        link_shim = _Shim()
        link_shim.turns = candidate_pool
        link_retriever.index(link_shim)
        link_hits = link_retriever.retrieve(
            turn_text(seed_turn), k=min(links_per_seed, len(candidate_pool))
        )
        for hit in link_hits:
            t = hit.turn
            if seed_id(t) not in selected_ids:
                selected.append(t)
                selected_ids.add(seed_id(t))

    rerank_retriever = BM25Retriever()
    rerank_shim = _Shim()
    rerank_shim.turns = selected
    rerank_retriever.index(rerank_shim)
    final_hits = rerank_retriever.retrieve(question, k=min(top_k, len(selected)))
    final_turns = [h.turn for h in final_hits]

    return _format_evidence(final_turns, speaker_a, speaker_b)


def retrieve_evidence_linked_sessions(
    container: Any,
    question: str,
    top_k: int = 5,
    benchmark: str = "longmemeval",
    seed_sessions: int = 2,
    expansion_sessions: int = 2,
) -> Evidence:
    """Graph-like retrieval approximation for LongMemEval.

    1. Score sessions by BM25 against the question.
    2. Take the top seed sessions.
    3. Expand to linked sessions using lexical overlap between session texts.
    4. Run turn-level BM25 only within the selected sessions.

    This is a lightweight proxy for graph/linking memory systems: it explicitly
    broadens retrieval from the highest-scoring seed session(s) to connected
    sessions that mention similar entities/topics.
    """
    if benchmark != "longmemeval":
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    sessions = list(getattr(container, "sessions", []))
    if not sessions:
        return retrieve_evidence(container, question, top_k=top_k, benchmark=benchmark)

    class _Shim:
        pass

    def session_text(session) -> str:
        return "\n".join(
            f"{getattr(t, 'role', '')}: {getattr(t, 'text', '')}"
            for t in session
        )

    session_texts = [session_text(s) for s in sessions]
    seed_retriever = BM25Retriever()
    seed_shim = _Shim()
    seed_shim.turns = [
        type("SessionDoc", (), {"role": f"session_{idx}", "text": text, "session_index": idx})()
        for idx, text in enumerate(session_texts)
    ]
    seed_retriever.index(seed_shim)
    seed_hits = seed_retriever.retrieve(question, k=min(seed_sessions, len(sessions)))
    selected = {getattr(hit.turn, "session_index") for hit in seed_hits}

    def salient_tokens(text: str) -> set[str]:
        import re
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "been", "being", "it", "that",
            "this", "i", "you", "he", "she", "they", "we", "my", "your", "our",
            "from", "at", "as", "by", "about", "after", "before", "during",
        }
        toks = {t.lower() for t in re.findall(r"[A-Za-z0-9']+", text)}
        return {t for t in toks if len(t) >= 4 and t not in stop}

    session_tokens = [salient_tokens(text) for text in session_texts]
    candidate_scores: list[tuple[float, int]] = []
    for idx, toks in enumerate(session_tokens):
        if idx in selected:
            continue
        score = 0.0
        for seed_idx in selected:
            denom = max(1, len(toks | session_tokens[seed_idx]))
            score = max(score, len(toks & session_tokens[seed_idx]) / denom)
        if score > 0:
            candidate_scores.append((score, idx))
    candidate_scores.sort(reverse=True)
    for _, idx in candidate_scores[:expansion_sessions]:
        selected.add(idx)

    turn_retriever = BM25Retriever()
    turn_shim = _Shim()
    turn_shim.turns = [t for idx in sorted(selected) for t in sessions[idx]]
    turn_retriever.index(turn_shim)
    turns = [h.turn for h in turn_retriever.retrieve(question, k=top_k)]
    return _format_evidence(turns, "user", "assistant")
