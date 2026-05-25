# Experiment crag-6: A-MEM-inspired linking-memory baseline

**Status:** OPEN 2026-04-28, scaffolded but not yet run.

## Purpose

Per `paper/proposal.md` §"Ranking Shift Protocol" and §"Baseline taxonomy", this baseline is the proposal's **must-have** comparison point. Without it the paper can show "format affects scores" but not "format affects method-comparison conclusions."

`linked-notes` retrieval (architecture-family A-MEM-style; see
`literature/linking-baseline-design.md`) replaces flat BM25 with:
1. BM25-retrieve top `seed_notes=3` turns by query.
2. For each seed, BM25-retrieve top `links_per_seed=3` turns from the rest of the haystack using the seed's text as the query.
3. Re-rank the union by question-BM25 and return top `top_k=5`.

## Pre-registered locked predictions

Decision tree (decisions made before the run, per proposal §"判定标准"):

1. **If `linked-notes` matches or beats flat BM25 on aggregate F1 by ≥0.5 points on LongMemEval oracle and ≥0.5 on LoCoMo:** the linking mechanism is non-trivially helpful. Promote to a real baseline in the ranking-shift table.
2. **If it ties (within ±0.5 F1 on both) but improves multi-session by ≥3 F1 points:** still a useful contribution — partial / category-specific lift consistent with the architecture's intended target.
3. **If it underperforms flat BM25 on aggregate by ≥0.5 F1 with no category-specific upside:** record as a strengthening negative result. The next escalation is the embedding-based v2 (semantic linking) or a session-summary / memory-bank baseline.

## Comparison cells

Same matched generator (Qwen2.5-14B-Instruct), same router (rule-based), same call budget (1/Q), same `top_k=5`. The only knob that moves is `retrieval_mode`.

| Cell | Benchmark | Slice | Retrieval mode | Notes |
|---|---|---|---|---|
| 6a | LongMemEval | oracle (500 Qs) | flat | reproduces run_011 |
| 6b | LongMemEval | oracle (500 Qs) | linked-notes | new baseline |
| 6c | LongMemEval | oracle multi-session (133 Qs) | flat | sub-slice |
| 6d | LongMemEval | oracle multi-session (133 Qs) | linked-notes | sub-slice |
| 6e | LoCoMo | full (1986 Qs) | flat | reproduces run_012 |
| 6f | LoCoMo | full (1986 Qs) | linked-notes | new baseline |

Total estimated cost: ~2,500 LLM calls × ~0.2 s/Q ≈ 8 minutes per cell on Qwen2.5-14B.

## What's reported

For each cell:
- aggregate F1
- per-category F1
- containment
- completion length
- LoCoMo MC10 accuracy on cells 6e/6f (sanity check for the F1/MC tension)

## Falsification criteria

The pre-registered "the linking baseline doesn't justify its complexity" verdict triggers if:
- aggregate F1 lower by >0.5 on both LongMemEval and LoCoMo, AND
- multi-session F1 lower or unchanged.

If this triggers, the next baseline to try is session-summary / memory-bank, then graph-style. Any of those failing the same criterion would weaken H_ranking_shift's premise that **some** memory architecture is worth comparing against — it would imply that under matched conditions, no architecture-family mechanism beats flat BM25, in which case the paper's framing has to change again.

## Sanity checks before recording results

- All retrieval cells must return n_retrieved == top_k (no degenerate cases).
- Smoke test (already passing in `src/recipe/test_crag.py`) shows linked-notes diverges from flat on at least 50% of multi-session questions; this is the floor for a meaningful mechanism.
