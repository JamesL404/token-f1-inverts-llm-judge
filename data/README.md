# Benchmark data acquisition

This release does **not** redistribute benchmark data. Download from the upstream sources below and place under `data/`.

## LongMemEval

- **Upstream:** https://github.com/xiaowu0162/LongMemEval
- **License:** check upstream (typically research-only)
- **Files needed:** `longmemeval_oracle.json` (n=500)
- **Place at:** `data/LongMemEval/data/longmemeval_oracle.json`

```bash
mkdir -p data/LongMemEval/data
# follow upstream instructions to download longmemeval_oracle.json
```

## LoCoMo

- **Upstream:** https://github.com/snap-research/locomo
- **License:** Apache-2.0 (verify upstream)
- **Files needed:** `locomo10.json` (n=1986 QA across 10 dialogues)
- **Place at:** `data/locomo/data/locomo10.json`

```bash
mkdir -p data/locomo/data
git clone https://github.com/snap-research/locomo /tmp/locomo
cp /tmp/locomo/data/locomo10.json data/locomo/data/
```

## Loader sanity check

```bash
python -c "from src.longmemeval import load_questions, iter_questions; \
qs = list(iter_questions(load_questions())); print(f'LongMemEval oracle: {len(qs)} questions')"
# Expected: LongMemEval oracle: 500 questions

python -c "from src.locomo import load_questions, iter_questions; \
qs = list(iter_questions(load_questions())); print(f'LoCoMo: {len(qs)} questions')"
# Expected: LoCoMo: 1986 questions
```

## Why we don't redistribute

LongMemEval and LoCoMo benchmark data has its own license terms; redistributing in this code repository would conflate code licensing (Apache-2.0) with data licensing. Downloading from the upstream sources also ensures you're using the canonical version.

## BEAM

Not used in matched-pipeline experiments. Cited as cross-benchmark validity context only.
