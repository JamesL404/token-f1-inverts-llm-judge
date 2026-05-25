# Human annotation task — long-horizon dialogue QA

## What you're doing

You are evaluating which of two automatic-system answers is more correct on long-horizon dialogue questions. Each item in `annotation_items.csv` shows a **question**, a **gold (reference) answer**, and **two candidate outputs** labeled `Output X` and `Output Y`. You don't know which system produced X vs Y — that's intentional, the assignment is randomized per item.

For each item, you fill in **three columns** in `annotation_items.csv`:

| Column | Values | Meaning |
|---|---|---|
| `rater_label_X` | `correct` / `partial` / `incorrect` | How well does Output X match the gold answer? |
| `rater_label_Y` | `correct` / `partial` / `incorrect` | How well does Output Y match the gold answer? |
| `rater_preference` | `X` / `Y` / `tie` | If forced to pick one, which output is more useful for answering the question? |
| `notes` | free text (optional) | Any disagreement with gold, ambiguity, or rationale you want noted |

## Labeling conventions

### `correct`
The output conveys the same factual answer as the gold, even if the phrasing differs.

- Gold: `25:50`. Output: `25 minutes and 50 seconds` → **correct** (same answer, different phrasing)
- Gold: `7 days. 8 days (including the last day) is also acceptable.` Output: `7 days` → **correct**
- Gold: `Japanese short-grain rice`. Output: `lavender shampoo from Trader Joe's` → **incorrect** (different topic; in this case the question was about shampoo so disregard this constructed example)

### `partial`
The output gets some of the answer right but is missing material the gold considers necessary, or contains the right answer wrapped in unsupported elaboration.

- Gold: `25`. Output: `there are about 25 titles, mostly from streaming` → **partial** (right number but adds unverified elaboration)
- Gold: `Instant Pot`. Output: `there is no information available about that` → **incorrect** (not partial — gold IS specific, output abstains)
- Gold: `over a year`. Output: `Over a year` → **correct** (case + leading-space differences are OK)

### `incorrect`
The output gives the wrong factual answer, or refuses/abstains when the gold has a specific answer.

- Gold: `4 years and 9 months`. Output: `no information available` → **incorrect**
- Gold: `Persistent cough`. Output: `skin tag removal` → **incorrect**

### Abstention questions
Some gold answers are themselves abstentions (e.g., `"The information provided is not enough. You did not mention X."`).

- Gold: `The information provided is not enough.` Output: `there is no info on this` → **correct**
- Gold: `The information provided is not enough.` Output: `Ferrari model` → **incorrect** (gives a specific answer when gold is "no answer")

### Preference column
After labeling each output, indicate which one is more useful for answering the question. If both are correct or both incorrect, you can mark `tie`. The preference column is the most important single output of this audit.

## Important rules

1. **Don't look at the unblind key.** A separate file `UNBLIND_KEY_DO_NOT_SHOW_RATERS.json` exists in this directory but should not be opened until after you submit your labels.
2. **Treat each item independently.** Don't try to detect a pattern across items.
3. **When in doubt, leave a note** in the `notes` column rather than guessing.
4. **Length is not correctness.** A long output and a short output can both be correct. Do not penalize length per se.
5. **The author's own audit at n=50** is in `experiments/crag-7-ranking-shift/results_blind_audit.json` — please do **not** look at this until after you submit your labels (and ideally not at all, to keep your judgment independent).

## How to submit

Save the completed `annotation_items.csv` (with your three columns filled per row) as `annotation_items_<your_name>.csv` and return it to the author.

If you have access to a colleague who can also annotate the same items independently, please ask them to do so — inter-rater agreement is part of the analysis.

## Time estimate

Roughly 30 seconds to 2 minutes per item depending on item complexity. 100 items = 1-3 hours of focused work. You can split across sittings.

## Questions

If anything is ambiguous, leave a note in the `notes` column and we'll discuss after submission. Don't email us during labeling — that breaks blinding.

Thank you.
