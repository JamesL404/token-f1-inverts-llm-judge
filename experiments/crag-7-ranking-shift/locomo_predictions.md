# Pre-registered predictions for the LoCoMo crag-7 cells

**Date:** 2026-04-29 (logged before LA, LB, LC, LD finished)
**Logger:** autoresearch agent
**Predictions made before any LoCoMo non-LF cell completed.**

## Known cells (pre-registration baseline)

| | tight-per-cat | tight-generic | loose |
|---|---:|---:|---:|
| flat (LE) | **0.4065** | ? | ? |
| linked-notes (LF) | **0.3872** | ? | ? |

## Predictions for the 4 running cells

Reasoning: on LongMemEval, the format-tightness effect (loose -> tight-generic)
was +0.06 for both retrievals, and the category-aware effect (tight-generic ->
tight-per-category) was +0.18-0.19 for both retrievals. If LoCoMo follows the
same pattern, applying these deltas to the LoCoMo flat row gives:

- LA = LE - category_aware_effect ≈ 0.4065 - 0.184 ≈ 0.22
- LB = LA - format_tightness_effect ≈ 0.22 - 0.062 ≈ 0.16
- LC = LF - category_aware_effect ≈ 0.3872 - 0.18 ≈ 0.21
- LD = LC - format_tightness_effect ≈ 0.21 - 0.06 ≈ 0.15

| | tight-per-cat | tight-generic (predicted) | loose (predicted) |
|---|---:|---:|---:|
| flat (LE) | 0.4065 | **0.22** | **0.16** |
| linked-notes (LF) | 0.3872 | **0.21** | **0.15** |

## Expected qualitative outcomes

1. **Format-tightness effect on LoCoMo**: ~+0.06 F1 (replicates LongMemEval).
2. **Category-aware effect on LoCoMo**: ~+0.18-0.20 F1 (replicates LongMemEval).
3. **Architecture effect**: stays near 0 or slightly negative on LoCoMo across all 3 settings (consistent with LF showing -0.019). The architecture penalty might be larger or smaller under loose vs tight; either way, both are expected within a small range.

## Falsification

- If format-tightness on LoCoMo is < +0.02 or > +0.12, the prompt-engineering pattern is benchmark-specific.
- If category-aware on LoCoMo is < +0.10 or > +0.30, the pattern is benchmark-specific.
- If architecture-effect signs flip (linked > flat at any prompt setting on LoCoMo), the architecture-effect picture is more nuanced than "neutral/harmful only."

## Why this matters

If the predictions match, the cross-benchmark replication is clean and the paper's claim becomes:
> Across two benchmarks, on a matched pipeline: prompt engineering dominates by 30x while the architecture-family memory mechanism is between neutral and slightly harmful.

If they fail, the paper claim narrows to LongMemEval and we have to think harder about what LoCoMo is doing differently.
