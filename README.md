# Token-F1 Inverts LLM-Judged Answer Correctness — Code & Data Release

Reproduces all experimental results from *"Token-F1 Inverts LLM-Judged Answer Correctness in Long-Horizon Dialogue Memory Evaluation"*. Includes the FCS rendering protocol, paired LLM-judge harness, bootstrap-noise utility, and per-cell prediction JSONs for every cell reported in the paper.

---

## Quick start

```bash
# 1. Clone and create environment
git clone <THIS_REPO> token-f1-inverts && cd token-f1-inverts
conda create -n tokenf1 python=3.11 -y && conda activate tokenf1
pip install -r requirements.txt

# 2. Set API keys — see "API keys" section below for details
cp .env.example .env
# Edit .env to fill in OPENAI_API_KEY and ANTHROPIC_API_KEY

# 3. Download benchmark data (see data/README.md)
# LongMemEval: https://github.com/xiaowu0162/LongMemEval
# LoCoMo: https://github.com/snap-research/locomo

# 4. Reproduce a key headline result (paired LLM-judge on B vs E)
python experiments/crag-7-ranking-shift/llm_judge.py \
  --cell-e experiments/crag-6-linking-baseline/results_6a_lme_flat.json \
  --cell-j experiments/crag-7-ranking-shift/results_7J_lme_flat_tight_fcs.json \
  --output results_llm_judge_e_vs_j.json
```

---

## API keys

All scripts that call LLM APIs load credentials from a `.env` file at the **repository root**.

**Setup:**

```bash
cp .env.example .env
```

Then open `.env` in your editor and fill in:

```
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

**Where each key is used:**

| Key | Used by |
|---|---|
| `OPENAI_API_KEY` | `gpt-4o-mini` for the primary LLM-judge harness (`experiments/crag-7-ranking-shift/llm_judge*.py`); Mem0 / A-MEM extraction layer (`mem0ai`, `agentic_memory`); HippoRAG OpenIE + text-embedding-3-small (`hipporag`); FCS 5-seed leakage ablation (`experiments/crag-10-fcs-heldout/`) |
| `ANTHROPIC_API_KEY` | `Claude-Sonnet-4.5` for the cross-family LLM-judge replication (`experiments/crag-7-ranking-shift/llm_judge_cross.py`, `experiments/crag-9-faithful-repro/llm_judge_faithful_vs_cellj.py`) |

**Security:**

- `.env` is in `.gitignore` — keys never leave your machine
- Never commit `.env` or paste keys into shared docs / issues / pull requests
- For shared compute (e.g. lab clusters), consider setting the keys as environment variables in your shell init (`~/.bashrc`) instead of writing them to `.env`. The scripts load via `os.environ` if the variable is already set.

**Approximate cost to reproduce all main results:** ~$15 USD total in API calls. The bulk is gpt-4o-mini judge calls (~$10) and Claude cross-family judge (~$5).

**Optional keys:**

- `HF_TOKEN`: HuggingFace access token if loading gated models (e.g. Llama-3.1-8B-Instruct). Not required for the open Qwen models used as the primary backbone.
- `HF_HOME` / `TRANSFORMERS_CACHE`: override the default HuggingFace cache location if your home directory has limited space.

---

## Repository structure

```
.
├── README.md                              # this file
├── LICENSE                                # Apache-2.0
├── requirements.txt                       # Python deps
├── .env.example                           # API key template
├── CHECKLIST.md                           # community-standard checklist (App. A.18)
├── src/                                   # core library
│   ├── recipe/
│   │   ├── fcs.py                         # FCS rendering protocol
│   │   ├── generator.py                   # HF + API generator wrappers
│   │   ├── router.py                      # per-benchmark category router
│   │   ├── prompts/                       # prompt templates (lme_*_fcs.txt, etc.)
│   │   └── recipe.py                      # full CRAG pipeline
│   ├── locomo.py                          # LoCoMo loader
│   ├── longmemeval.py                     # LongMemEval loader
│   ├── retriever.py                       # BM25 / dense / rerank substrates
│   └── eval.py                            # token-F1 + helpers
├── scripts/                               # analysis utilities
│   ├── bootstrap_noise_floor.py           # per-cell bootstrap CIs
│   ├── paired_bootstrap.py                # paired-bootstrap p-values
│   ├── inversion_synthesis.py             # 19-cell sign-agreement matrix
│   ├── audit_score.py                     # human-audit scorer
│   ├── faithful_vs_cellj_paired.py        # faithful-repro paired stats
│   └── plot_*.py                          # paper figure generators
├── experiments/
│   ├── crag-6-linking-baseline/           # 3-arch main matrix base cells
│   ├── crag-7-ranking-shift/              # FCS, full 19-cell judge matrix, truncate-rejudge
│   ├── crag-9-faithful-repro/             # Mem0 / HippoRAG / A-MEM upstream-code repros + 2-rater audit
│   │   ├── human_annotation/              # rater 1 (author) labels + sealed unblind key
│   │   └── human_annotation_rater2/       # rater 2 (outside PhD) labels + inter-rater analysis
│   └── crag-10-fcs-heldout/               # 5-seed held-out FCS leakage ablation
└── data/                                  # benchmark loaders (raw data downloaded separately)
    └── README.md                          # data acquisition + license notes
```

---

## Reproducing the four findings

### Finding 1 — Token-F1 inverts LLM-judge ranking

```bash
# Run paired LLM-judge on cells B (loose), E (tight-per-cat), J (tight-fcs):
python experiments/crag-7-ranking-shift/llm_judge.py
python experiments/crag-7-ranking-shift/llm_judge_full.py     # 19-cell matrix
python experiments/crag-7-ranking-shift/llm_judge_cross.py    # Claude-Sonnet-4.5 replication

# Inversion synthesis
python scripts/inversion_synthesis.py
```

Result JSONs:
- `experiments/crag-7-ranking-shift/results_llm_judge_e_vs_j.json`
- `experiments/crag-7-ranking-shift/results_llm_judge_full_matrix.json`
- `experiments/crag-7-ranking-shift/results_llm_judge_claude.json`
- `experiments/crag-7-ranking-shift/results_inversion_synthesis.json`

### Finding 2 — 0.20 F1 swing from LoCoMo abstention string

Computed inline from the main matrix runs (cells LA–LI). Compare loose-vs-tight-generic aggregate F1 with and without the adversarial category.

### Finding 3 — Architecture-null across 6 family-level mechanisms + 3 faithful reproductions

```bash
# Faithful upstream-code reproductions (n=200 stratified, paired against cell J)
python experiments/crag-9-faithful-repro/run_mem0.py --limit 200
python experiments/crag-9-faithful-repro/run_hipporag.py --limit 449  # max paired indices
python experiments/crag-9-faithful-repro/run_amem.py --limit 200

# Paired bootstrap vs cell J + judge cross-family
python scripts/faithful_vs_cellj_paired.py
python experiments/crag-9-faithful-repro/llm_judge_faithful_vs_cellj.py
```

### Finding 4 — Prompt engineering accounts for the entire token-F1 range; FCS gain decomposes

```bash
# FCS rendering and held-out ablation
python experiments/crag-10-fcs-heldout/run_multiseed.py    # 5-seed leakage ablation
# Stripped ablation (cell Q)
python experiments/crag-7-ranking-shift/run_tight_stripped.py
```

---

## Two-rater human audit reproduction

The audit JSONs are released in:
- `experiments/crag-9-faithful-repro/human_annotation/` (rater 1, paper author)
- `experiments/crag-9-faithful-repro/human_annotation_rater2/` (rater 2, outside PhD researcher)

Inter-rater statistics:

```bash
python experiments/crag-9-faithful-repro/human_annotation_rater2/score_inter_rater.py
# → results_inter_rater.json
# Cohen's κ_combined = 0.96; McNemar between raters p = 1.0
```

The sealed-key X/Y blinding file is `human_annotation/UNBLIND_KEY_DO_NOT_SHOW_RATERS.json`. To recruit a *third* rater, copy `human_annotation_rater2/annotation_items_rater2.json` (blank-label JSON) and have them fill it; instructions in `human_annotation_rater2/README_FOR_RATER2.md`.

---

## Community standard checklist

`CHECKLIST.md` is the App. A.18 reproducibility checklist for memory-architecture claims. Authors, reviewers, and leaderboard maintainers can use it as a self-contained reporting standard.

---

## Compute requirements

- **CPU-only paths**: BM25 retrieval, all statistical analysis (paired-bootstrap, McNemar, inter-rater κ), figure generation
- **Local inference paths**: any setup that can run the relevant HuggingFace causal LM (Qwen2.5-3B/7B/14B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct). The primary backbone (Qwen-14B) requires sufficient memory to hold a 14B-parameter model in bfloat16
- **API paths**: see "API keys" section above. Total ~$15 USD to reproduce all main results

---

## Citing this work

```bibtex
@inproceedings{anonymous2026tokenf1,
  title={Token-F1 Inverts LLM-Judged Answer Correctness in Long-Horizon Dialogue Memory Evaluation},
  author={Anonymous},
  booktitle={Anonymous Submission},
  year={2026}
}
```

---

## License

Code: Apache-2.0 (see `LICENSE`).

The released per-cell prediction JSONs and per-rater audit labels are released under CC-BY-4.0.

Benchmark data (LongMemEval, LoCoMo) is **not** redistributed here — see `data/README.md` for upstream sources and licensing.
