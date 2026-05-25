# Reproducibility Checklist for Long-Horizon Dialogue Memory Claims

Companion artifact to *"Token-F1 Inverts Answer Correctness: An Audit of Long-Horizon Dialogue Memory Evaluation"* (§6, Adoption Pathway). A one-page checklist for authors, reviewers, and leaderboard maintainers proposing or evaluating memory-architecture claims on LongMemEval, LoCoMo, BEAM, or similar long-horizon dialogue QA benchmarks.

---

## A. Baseline matching (mandatory)

- [ ] **Surface-form-matched flat-RAG baseline reported on the same backbone.** Length cap, output template, and abstention string match the proposed system's prompts. *Rationale: §5.1, Finding 1; §5.4, Finding 3.*
- [ ] **Identical prompt design across architecture variants and the flat baseline.** No silent prompt re-tuning per architecture. *Rationale: §5.5, Finding 4.*
- [ ] **Single-call vs multi-call regime stated.** If multi-call, the flat baseline is also multi-call; otherwise the comparison is confounded with call budget.

## B. Statistical reporting (mandatory)

- [ ] **Per-category F1 with paired-bootstrap 90% CIs**, not aggregate F1 alone. *Rationale: faithful Mem0 (§5.7) — aggregate Δ=+0.021 (n.s.) hides per-label Δ=+0.25 / Δ=−0.08.*
- [ ] **Pre-registered minimum-detectable-effect (MDE)** for the test you run, with sample-size justification. *Rationale: App. A.10.*
- [ ] **Two-sided p-value reported alongside one-sided** for any α=0.05 claim.
- [ ] **Multiple-comparisons correction** stated when running ≥ 3 tests (Bonferroni or comparable). *Rationale: §5.7 — A-MEM and HippoRAG cross unadjusted α=0.05 in opposite directions; neither survives Bonferroni-7.*

## C. Metric reporting (mandatory)

- [ ] **Output-length statistics** (mean / median / p75 chars and words) per cell. *Rationale: §5.5–§5.6 — the +0.082 FCS gain is fully explained by 3.6× length compression.*
- [ ] **Cross-family LLM-judge accuracy** with paired McNemar tests on at least two judges from different model families (e.g., one OpenAI, one Anthropic). *Rationale: §5.2, §5.6, Finding 1.*
- [ ] **Token-F1 reported only with surface-form context** (length, format match, abstention-string match). Do not report token-F1 alone for prompt-engineering claims.

## D. Architecture/protocol fairness (mandatory)

- [ ] **Native-prompt sanity cell** alongside the matched cell. Run the proposed system under its own published answer prompt at native token budget on the same paired indices. *Rationale: §5.7 native-prompt sanity — closes the "you hobbled them" objection.*
- [ ] **Pipeline details released** (model versions, seeds, decoding params, retrieval parameters, prompt files, evaluation code).
- [ ] **Per-question prediction JSONs released** for paired re-analysis.

## E. Scope statements (mandatory)

- [ ] **Which regimes are out of scope** explicitly stated (single-call vs multi-call; oracle vs non-oracle; with/without forgetting; persistent vs per-question memory). *Rationale: §7.*
- [ ] **Which benchmarks are out of scope** stated (e.g., BEAM-style verbose-gold benchmarks where token-F1 is uninformative).
- [ ] **What would falsify the claim** stated as a pre-registered MDE × condition × judge requirement, not a vague "we expect."

## F. Optional but recommended

- [ ] FCS-style diagnostic prompt run as a robustness probe (with the disclosure that FCS is calibrated on gold and is not a leaderboard method).
- [ ] Cross-backbone replication on at least one non-primary backbone family.
- [ ] Adversarial-category and abstention-string sensitivity check (LoCoMo-style 0.20 F1 swing test).
- [ ] Held-out calibration ablation if any prompt component is derived from data (App. A.11).

---

## How to use this checklist

**Authors.** Before submitting a memory-architecture claim, mark each box. An unchecked box in §A–E should be addressed in the manuscript or in the limitations section.

**Reviewers.** Each unchecked box in §A–E is grounds for a request-revision question.

**Leaderboard maintainers.** §A–C constitute the proposed minimum reporting requirement for a memory-architecture leaderboard entry; §D items can be enforced via code-release verification.

---

*This checklist is a starting point, not a finished community standard. Suggest revisions via the public repository.*
