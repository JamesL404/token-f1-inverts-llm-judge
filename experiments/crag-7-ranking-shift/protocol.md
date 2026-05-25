# Experiment crag-7: Ranking Shift Protocol — paper headline

**Status:** OPEN 2026-04-28, blocked on crag-6 results + FCS prompt rendering wired into CRAG.

## Purpose

This is the proposal&apos;s paper-headline experiment (`paper/proposal.md`
&sect;&quot;Ranking Shift Protocol&quot;). It is what upgrades the contribution
from &quot;prompt-format affects scores&quot; to &quot;prompt-format affects who
beats whom&quot;.

Per the proposal&apos;s pre-registered success criteria, the paper&apos;s main
claim is supported if any of the following hold above the noise floor on
&ge;2 benchmarks:

1. Gap shrinkage &ge;30% between a strong memory baseline and generic
   RAG, after switching from format-uncontrolled to format-controlled.
2. &ge;3 pairwise ordering flips across `(benchmark, category)` cells.
3. &ge;40% shrinkage of a claimed memory gain.

If none of these hold, the paper falls back to a benchmark-validity
finding (still publishable, but weaker).

## Method matrix

| ID | Method | Layer | Mechanism |
|---|---|---|---|
| M1 | generic RAG | minimum baseline | flat BM25 + benchmark `*_generic.txt` prompts (no category awareness) |
| M2 | C-RAG (rule router) | minimum baseline | flat BM25 + per-category prompts, rule-based router |
| M3 | C-RAG (oracle router) | minimum baseline | flat BM25 + per-category prompts, oracle router (cheating upper bound) |
| M4 | linked-notes (architecture-family) | strong baseline | linked-notes BM25 + C-RAG + rule router |
| M5 | session-summary / memory-bank (architecture-family) | strong baseline | TODO — not yet implemented |

Once crag-6 reports, we know whether M4 deserves to stay in the table.
M5 is on the to-do list and will be added in a later tick.

## Setting matrix

For every method, every benchmark, every category, run two settings:

| Setting | Prompt source | What it constrains | What it does NOT constrain |
|---|---|---|---|
| Format-Uncontrolled | `fcs.render_uncontrolled_prefix` | nothing (allows explanation, full sentences, model&apos;s natural voice) | retrieval, router, budget, backbone (all matched) |
| Format-Controlled | `fcs.render_controlled_prefix(SurfaceForm)` | length cap, answer form template, abstention string when applicable | task-solving hints (must NOT contain reasoning hints) |

The controlled prompt is **derived from gold-answer surface form**, not
hand-tuned for score. See `src/recipe/fcs.py` and
`data/fcs_surface_form_profile.json`.

## Benchmarks

| Benchmark | Slice | Total cells per backbone |
|---|---|---:|
| LongMemEval oracle | 500 Qs (4 categories) | 5 methods x 2 settings = 10 |
| LoCoMo | 1986 Qs (5 categories incl. adversarial) | 10 |
| BEAM 100K core | 160 Qs (4 categories) | 10 |

Backbones: Qwen2.5-14B-Instruct (primary). Qwen2.5-7B-Instruct as
cross-backbone control on a subset of cells if compute allows.

## Pre-registered locked predictions

Per proposal &sect;&quot;判定标准&quot;:

### Main claim succeeds (any of these):

1. (LongMemEval | LoCoMo | BEAM): gap between M4 (or M5) and M1 in
   format-controlled setting is &le;70% of the gap in format-uncontrolled
   setting on &ge;2 benchmarks. (i.e. &gt;=30% gap shrinkage)
2. Any 3 (benchmark, category) cells exhibit a pairwise ordering flip
   between the two settings (e.g. M2 &gt; M4 uncontrolled, M4 &gt; M2 controlled).
3. Any claimed memory gain (e.g. M4 over M1, or M3 over M2) shrinks by
   &ge;40% in absolute F1 after format control on &ge;2 benchmarks.

All three thresholds must exceed the noise floor measured in step 0
below.

### Weak success (degrade to &quot;findings&quot; paper):

- &ge;2 (benchmark, category) cells with gap shrinkage &ge;20% above noise floor.
- OR local pairwise instability above noise floor on &ge;2 cells.

### Negative (refutation):

- All gap shrinkages &le;10% AND no pairwise flips above noise floor.
- Conclusion: scores are sensitive but method comparison is stable
  enough that prompt format does not change leaderboards.
- Paper still publishable as a benchmark-validity finding; main claim
  is downgraded.

## Step 0: Noise floor

Before any ranking-shift claim is recorded, measure noise floor on
&ge;1 cell. Two options for our deterministic-greedy setup:

a. **Bootstrap resampling** of predictions from a single run: 1000
   bootstrap resamples on n=500 LongMemEval Qs, compute F1 each time,
   report the 5th-95th percentile spread.
b. **Sample-time variation**: switch `do_sample=True, temperature=0.7`
   and run the same cell 3 times; report the F1 std.

Pre-registered: any reported &Delta;F1 must exceed 2x the (a) bootstrap
spread to be considered above noise.

## Output format

For each (method, benchmark, setting), write
`experiments/crag-7-ranking-shift/results_{method}_{benchmark}_{setting}.json`
with the standard CRAG runner schema.

The post-processing summary (gap shrinkages, pairwise flips, ranking
table) is produced by `scripts/analyze_crag7.py` (TODO).

## Compute estimate

- 5 methods x 3 benchmarks x 2 settings = 30 cells
- Per cell: 160-1986 Qs, ~0.2 s/Q on Qwen2.5-14B = 30 s - 7 min
- Total: ~2-3 hours wall time on GPUs 3+4

This is compatible with a single overnight run if we serialize cells.

## Sanity checks before recording results

- For each benchmark, verify FCS controlled prompt sanity check
  (`fcs.fcs_sanity_check_passes`) holds: at least 2 methods retain
  &ge;80% of their uncontrolled F1. If it fails, the controlled prompt
  is bad-prompt rather than good-control; investigate before continuing.
- Verify each cell&apos;s n_questions matches the benchmark size; no
  truncation due to runner errors.
- Verify all cells use the same router and the same backbone (logged
  in the result JSON).
