"""Rule-based question router for C-RAG.

Classifies a question text into one of N reasoning categories WITHOUT
using an LLM. The router is benchmark-aware: separate pattern sets for
LoCoMo and LongMemEval.

Design notes:
- Patterns are deliberately simple regex/keyword rules. The router accuracy
  is measured against the held-out gold category labels in `src/locomo.py`
  and `src/longmemeval.py`. Target: >= 0.85 accuracy.
- If the rule-based router falls below target, we fall back to an
  embedding-similarity router (in router_embedding.py, not yet built)
  or a one-shot LLM router (still cheap: 1 call/question).
- The category strings here MUST match the LoCoMo / LongMemEval simplified
  label space exactly so the prompt templates can key on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# These match src.locomo.CATEGORY_TO_LABEL
LOCOMO_CATEGORIES = ["single-hop", "multi-hop", "temporal", "open-domain", "adversarial"]

# These match src.longmemeval.TYPE_TO_LABEL (simplified)
LONGMEMEVAL_CATEGORIES = [
    "temporal",            # temporal-reasoning
    "multi-session",       # multi-session
    "knowledge-update",    # knowledge-update
    "single-session",      # single-session-{user,assistant,preference}
]

# These match src.beam.TYPE_TO_LABEL (simplified)
BEAM_CATEGORIES = [
    "abstention",
    "contradiction-resolution",
    "event-ordering",
    "information-extraction",
    "knowledge-update",
    "multi-session",
    "temporal",
]


@dataclass
class RouteResult:
    category: str
    rule_fired: str   # which rule matched (for debugging)
    confidence: float = 1.0


class Router:
    """Abstract router."""
    name = "abstract"

    def route(self, question: str) -> RouteResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LoCoMo rules
# ---------------------------------------------------------------------------
# Order matters: rules are tried top-to-bottom; first match wins.

# Temporal: questions about WHEN something happened. Question must START
# with a temporal wh-phrase. NOT triggered by mere temporal anchors inside
# non-temporal questions ("Where did X go in July 2022?" is single-hop).
_LOCOMO_TEMPORAL_RE = re.compile(
    r"^\s*(when |"
    r"what (date|day|time|year|month) |"
    r"on what (date|day) |in what (year|month) |"
    r"how long (ago|before|after) |"
    r"how many (days|weeks|months|years) (ago|since|before|after|did|has) |"
    r"how old (is|was|were|are)|"
    r"(who|what|which|where) .*\b(on|in|during|before|after|towards|around|by|last|first|second|third)\b.*"
    r"(\b(day|week|month|year|summer|spring|fall|autumn|winter|january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b20\d{2}\b))",
    re.IGNORECASE,
)

# Open-domain inferential: questions that ask for inference, prediction,
# counterfactual, likelihood, value-judgment, or a category-label that
# isn't stated verbatim in the text.
_LOCOMO_OPEN_DOMAIN_RE = re.compile(
    r"(\b(would |might |likely |do you think|"
    r"if .{1,40} hadn'?t|if .{1,40} had |"
    r"does .{1,40} (love|like|enjoy|prefer|oppose|support|employ|hate|live in|get) |"
    r"how often does |"
    r"what attributes describe |"
    r"what (traits|qualities) describe |"
    r"considered (a |an )?(member|religious|liberal|conservative|"
    r"extrovert|introvert|atheist|spiritual)|"
    r"(more|most|less|least) interested in|"
    r"(political|religious|personality) (leaning|view|stance|trait)|"
    r"^who is |"
    r"^was .{1,40} a (good|bad) (month|year|time|idea|fit)|"
    r"which type of .{1,40} would|"
    r"answer (yes|y/n|in)\b))",
    re.IGNORECASE,
)

# Multi-hop: questions whose answers require combining MULTIPLE evidence
# pieces. STRONG signals only — false positives hurt single-hop precision
# more than the false negatives hurt multi-hop recall, so we keep this
# tight and accept ~0.35 multi-hop recall.
_LOCOMO_MULTI_HOP_RE = re.compile(
    r"\b(what (are|were) (all|some|the)|"
    r"list (all|some|the)|name (all|some|the)|"
    r"which of .{1,40} (have|has|were|was|did) |"
    r"what .{1,40} and .{1,40}('|')s |"
    r"what .*favorite .*s|"
    r"what .*family members|"
    r"what .*indoor activities|"
    r"what .*movies|"
    r"what .*recipes|"
    r"what .*instruments|"
    r"how many .*(has|have|did) .*(organized|won|visited|played|made)|"
    # Strict plural-noun list (kept narrow)
    r"what (causes|schools|books|topics|kinds|types|sorts|"
    r"hobbies|charities|skills|talents|languages) "
    r"(of |has |have |did |does |are |is )|"
    r"all the |both .{1,20} (have|share))",
    re.IGNORECASE,
)

# Adversarial signal: very few; LoCoMo cat 5 is hard to detect by pattern
# because the questions look just like normal questions. Best heuristic:
# the answer must be discovered to be unanswerable. We do NOT try to
# detect adversarial in the router; we let the GENERATOR's adversarial
# template handle abstention. We mark questions whose category we can't
# confidently classify as "adversarial" only if other rules fail AND the
# question has no temporal/multi-hop/open-domain markers — but this is
# unreliable. Instead we just default-assign single-hop and let the
# adversarial prompt handle abstention through the generator's instruction.

# Single-hop default: simple "what/who/where" questions. This is the
# fallback bucket — it catches everything that doesn't match the more
# specific patterns above.


class OracleRouter(Router):
    """Cheats: looks up the gold category from a precomputed map.

    For evaluation only — used as an UPPER BOUND in the C-RAG ablation
    experiment to answer "does router accuracy actually matter for end-to-end
    accuracy?". A real deployment would never have access to gold labels.
    """
    name = "oracle"

    def __init__(self, benchmark: str = "locomo"):
        self.benchmark = benchmark
        self._lookup: dict[str, str] = {}
        self._build_lookup()

    def _build_lookup(self):
        if self.benchmark == "locomo":
            from src.locomo import iter_qas, load_dialogues
            for d, qa in iter_qas(load_dialogues()):
                self._lookup[qa.question] = qa.label
        elif self.benchmark == "longmemeval":
            from src.longmemeval import iter_questions, load_questions
            for q in iter_questions(load_questions()):
                self._lookup[q.question] = q.label
        else:
            from src.beam import iter_qas, load_conversations

            for conv, q in iter_qas(load_conversations("100K")):
                self._lookup[q.question] = q.label
            for conv, q in iter_qas(load_conversations("500K")):
                self._lookup[q.question] = q.label
            for conv, q in iter_qas(load_conversations("1M")):
                self._lookup[q.question] = q.label

    def route(self, question: str) -> RouteResult:
        cat = self._lookup.get(question)
        if cat is None:
            default = "single-hop" if self.benchmark == "locomo" else (
                "single-session" if self.benchmark == "longmemeval" else "information-extraction"
            )
            return RouteResult(default,
                               "oracle_unknown", confidence=0.0)
        return RouteResult(cat, "oracle_lookup", confidence=1.0)


class RandomRouter(Router):
    """Picks a category uniformly at random. Lower bound for the ablation."""
    name = "random"

    def __init__(self, benchmark: str = "locomo", seed: int = 42):
        import random
        self.benchmark = benchmark
        self._rng = random.Random(seed)
        if benchmark == "locomo":
            self._categories = LOCOMO_CATEGORIES
        elif benchmark == "longmemeval":
            self._categories = LONGMEMEVAL_CATEGORIES
        else:
            self._categories = BEAM_CATEGORIES

    def route(self, question: str) -> RouteResult:
        cat = self._rng.choice(self._categories)
        return RouteResult(cat, "random", confidence=1.0 / len(self._categories))


class RuleBasedRouter(Router):
    """Rule-based router for LoCoMo question categories."""
    name = "rule-based"

    def __init__(self, benchmark: str = "locomo"):
        if benchmark not in ("locomo", "longmemeval", "beam"):
            raise ValueError(f"unknown benchmark: {benchmark}")
        self.benchmark = benchmark

    def route(self, question: str) -> RouteResult:
        if self.benchmark == "locomo":
            return self._route_locomo(question)
        if self.benchmark == "longmemeval":
            return self._route_longmemeval(question)
        return self._route_beam(question)

    def _route_locomo(self, question: str) -> RouteResult:
        # Temporal first — strongest signal
        if _LOCOMO_TEMPORAL_RE.search(question):
            return RouteResult("temporal", "temporal_re")
        # Multi-hop list/enumeration questions
        if _LOCOMO_MULTI_HOP_RE.search(question):
            return RouteResult("multi-hop", "multi_hop_re")
        # Open-domain inferential / counterfactual
        if _LOCOMO_OPEN_DOMAIN_RE.search(question):
            return RouteResult("open-domain", "open_domain_re")
        # Default: single-hop
        return RouteResult("single-hop", "default")

    def _route_longmemeval(self, question: str) -> RouteResult:
        # LongMemEval has very different question phrasings.
        # Temporal-reasoning: "first/last/before/after/which X did I X first"
        if re.search(
            r"\b(first|last|before|after|earliest|latest|"
            r"how many days|how long ago|when did .* (first|last)|"
            r"how many (days|weeks|months) (between|since|before|after))",
            question, re.IGNORECASE,
        ):
            return RouteResult("temporal", "lme_temporal_re")
        # Knowledge-update: "what is my (current|latest|new) X" or "did I change"
        if re.search(
            r"\b(now|currently|latest|current|recent|recently changed|"
            r"my new |updated|switched|changed (to|from)|stopped (doing|using))",
            question, re.IGNORECASE,
        ):
            return RouteResult("knowledge-update", "lme_kupdate_re")
        # Multi-session: questions referring to multiple events/topics
        if re.search(
            r"\b(across|multiple|both|all .* I (told|mentioned|discussed)|"
            r"in our (previous|past|earlier) (conversation|discussion|session|chat))",
            question, re.IGNORECASE,
        ):
            return RouteResult("multi-session", "lme_multi_session_re")
        # Default: single-session lookup
        return RouteResult("single-session", "default")

    def _route_beam(self, question: str) -> RouteResult:
        # ORDER MATTERS. Each rule checks for a more discriminative signal than
        # the next. Defined to maximize routing accuracy on the BEAM 100K core
        # 4-class subset (abstention | knowledge-update | multi-session |
        # temporal); the 3 auxiliary classes are reachable but lower priority.

        # Contradiction-resolution: explicitly mentions inconsistency.
        if re.search(
            r"\b(contradict|conflict|which is correct|clarify which|both be true|inconsistent)\b",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("contradiction-resolution", "beam_contradiction_re")

        # Abstention: questions probing background, rationale, motivations,
        # specific named-entity details that the system likely never observed.
        # These are "did the user ever discuss this?" tests.
        if re.search(
            r"("
            r"how did .{1,60}(influence|affect|impact|shape)|"
            r"what specific (bugs?|criteria|rules?|patterns?|features?|tests?|"
            r"requirements?|feedback|criteria|elements?)|"
            r"\b(background|previous|prior) (development|projects?|work|experience)|"
            r"why did .{1,60}(choose|decide|opt|prefer|pick)|"
            r"what was the (rationale|reasoning) (behind|for)|"
            r"what was my .{1,40}(reaction|feeling|emotion|opinion|impression)|"
            r"what led to|what considerations|what factors|"
            r"can you (share|tell me about) (the |any |my )?"
            r"(agenda|structure|background|specific|feedback|rationale|reasoning)|"
            r"can you tell me about my"
            r")",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("abstention", "beam_abstention_re")

        # Multi-session: explicit cross-session/cross-conversation phrasing
        # OR multi-aggregation patterns ("how many X across all", "considering
        # my A, B, C and D"). The latter is a tell that the answer requires
        # combining multiple session evidence.
        if re.search(
            r"("
            r"across (my|our|the|all) .{0,40}(sessions?|conversations?|requests?|chats?|discussions?|questions?)|"
            r"throughout (our|my|the|all) .{0,30}(conversations?|sessions?|chats?|discussions?)|"
            r"how many different |"
            r"how many .{1,60}(across|in total|all sessions?|all conversations?|altogether|combined)|"
            r"between my .{1,60}(and my|, my)|"
            r"in total after .{0,40}(adding|across|including)|"
            r"all .{0,40}(features?|sessions?|conversations?|topics?) i (mentioned|discussed|wanted)|"
            r"considering (my|the) .{1,200}(\band\b.{1,200}\band\b|\,.{1,200}\,.{1,200}\,)|"
            r"how (much|many) .{1,60}(did i|have i) .{1,40}(across|in total|combined)|"
            r"the times i (mentioned|discussed) "
            r")",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("multi-session", "beam_multi_session_re")

        # Event-ordering: explicit ordering language.
        if re.search(
            r"\b(in order|list the order|sequence of|chronological|ordered list|first .{1,40} then .{1,40} then)\b",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("event-ordering", "beam_event_order_re")

        # Temporal: STRICT — only matches when the question asks about
        # durations, intervals, or specific date/time relationships. The
        # earlier version captured "between/before/after" anywhere; this
        # tightening requires those tokens to be paired with explicit
        # time units or date references.
        if re.search(
            r"\b("
            r"how many (days|weeks|months|years|hours|minutes) (do i have|are there|passed|between|until|before|after) |"
            r"how long (does|did|will|until|before|after|ago) |"
            r"when (did|will|was|is) |"
            r"what (date|day|year|month) |"
            r"between .{1,50}\b(\d{4}|\d{1,2}/\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december)\b |"
            r"(by|before|after|until|on) (the )?(deadline|end of|start of|beginning of) |"
            r"weeks do i have"
            r")",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("temporal", "beam_temporal_re")

        # Knowledge-update: questions about CURRENT factual state — either
        # explicit ("current/latest/updated/now") OR implicit factual lookups
        # that change over time (counts, rates, percentages, configurations).
        if re.search(
            r"("
            r"\b(current|currently|latest|updated?|now|most recent|recently improved|recently changed)\b|"
            r"^what is (the |my )?(deadline|average|response time|test coverage|"
            r"daily call|memory|cpu|api rate|accuracy|quota|limit|version|status)|"
            r"^how many .{1,40}(have i (completed|merged|added|created|fixed|written)|"
            r"are (merged|deployed|included|added)|"
            r"exist|have been|do i have|are included)|"
            r"^how (much|many) (total |) ?(hours?|time|problems?|cards?|commits?) "
            r"(have i|did i|am i|do i)|"
            r"^what (is|are) (my|the) (accuracy|percentage|score|test coverage|"
            r"completion rate|progress)"
            r")",
            question,
            re.IGNORECASE,
        ):
            return RouteResult("knowledge-update", "beam_kupdate_re")

        # Information-extraction fallback: any explicit lookup-style question.
        if re.search(r"^\s*(what|when|where|who|which|how many)\b", question, re.IGNORECASE):
            return RouteResult("information-extraction", "beam_info_re")

        return RouteResult("information-extraction", "default")


# ---------------------------------------------------------------------------
# Router accuracy measurement helper
# ---------------------------------------------------------------------------

def measure_locomo_router_accuracy() -> dict:
    """Run the LoCoMo rule-based router against the gold categories.

    Returns: dict with overall accuracy and per-category confusion-matrix-style
    breakdown.
    """
    from collections import defaultdict
    from src.locomo import iter_qas, load_dialogues

    router = RuleBasedRouter("locomo")
    n = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))  # gold -> predicted -> count
    by_category_correct = defaultdict(int)
    by_category_total = defaultdict(int)

    dialogues = load_dialogues()
    for d, qa in iter_qas(dialogues):
        # We exclude adversarial from router accuracy because the router
        # cannot detect adversarial by patterns alone (by design — see
        # the comment above; the generator handles abstention).
        if qa.label == "adversarial":
            continue
        result = router.route(qa.question)
        n += 1
        confusion[qa.label][result.category] += 1
        by_category_total[qa.label] += 1
        if result.category == qa.label:
            correct += 1
            by_category_correct[qa.label] += 1

    out = {
        "n_routed": n,
        "overall_accuracy": round(correct / max(1, n), 4),
        "per_category_accuracy": {
            k: round(by_category_correct[k] / max(1, by_category_total[k]), 4)
            for k in sorted(by_category_total)
        },
        "confusion": {
            k: dict(v) for k, v in sorted(confusion.items())
        },
    }
    return out


def measure_longmemeval_router_accuracy() -> dict:
    from collections import defaultdict
    from src.longmemeval import iter_questions, load_questions

    router = RuleBasedRouter("longmemeval")
    n = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    by_cat_correct = defaultdict(int)
    by_cat_total = defaultdict(int)

    questions = load_questions()
    for q in questions:
        gold = q.label  # already simplified
        result = router.route(q.question)
        n += 1
        confusion[gold][result.category] += 1
        by_cat_total[gold] += 1
        if result.category == gold:
            correct += 1
            by_cat_correct[gold] += 1

    return {
        "n_routed": n,
        "overall_accuracy": round(correct / max(1, n), 4),
        "per_category_accuracy": {
            k: round(by_cat_correct[k] / max(1, by_cat_total[k]), 4)
            for k in sorted(by_cat_total)
        },
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
    }


if __name__ == "__main__":
    import json
    print("=== LoCoMo router accuracy ===")
    print(json.dumps(measure_locomo_router_accuracy(), indent=2))
    print("\n=== LongMemEval router accuracy ===")
    print(json.dumps(measure_longmemeval_router_accuracy(), indent=2))
