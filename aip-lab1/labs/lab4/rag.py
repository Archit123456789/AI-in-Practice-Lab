#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, enforce_citations  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# Part A. Six required elements:
#   1. source-only, no general knowledge      -> opening line
#   2. cite by index [1] / [2][5]             -> rule 2
#   3. never cite an unsupplied number        -> rule 3
#   4. exact refusal string, given verbatim   -> rule 1
#   5. what to do when sources disagree       -> rule 4
#   6. length discipline                      -> rule 5
# Rule 6 (partial answers) is not one of the six -- I added it because the
# handout's own Q37 case (Part C2) needs it, and the reference ANSWER_SYSTEM
# in aip/rag.py does NOT have it. That's the main diff from the reference:
# the reference relies on you noticing the Q37 failure and patching it later;
# this version tries to prevent it up front. Worth checking in report.md
# whether that actually changes your Part C numbers versus the reference.
ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided below. Do not
use outside or general knowledge, even if you are confident it is correct.

1. If the sources do not contain the answer, reply with exactly this string
   and nothing else:
   "{REFUSAL}"
2. Every factual sentence must end with a citation of the source(s) that
   support it, in the form [1] or [2][5].
3. Never cite a source number that was not given to you.
4. If sources disagree, do not silently pick one -- say they disagree and
   cite both sides.
5. Be concise. Two or three sentences unless the question genuinely needs
   more; every output token costs latency and money.
6. If the sources answer only part of the question, answer the part they
   support and explicitly say what is not covered, instead of refusing the
   whole question.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """B2. Code-level guarantee, not a model behaviour.

    `aip.guards.enforce_citations` already does the index check (every [n] in
    1..n_sources); this wraps it with the truncation, empty-answer, refusal,
    and missing-citation checks the handout asks for.

    Order matters: truncation is checked FIRST, before we even look at
    citations, because a cut-off prose answer with valid-looking citations is
    exactly the dangerous case (T1 failure mode 4) -- it looks fine and is
    silently incomplete.
    """
    truncated = finish_reason == "length"
    stripped = (text or "").strip()

    if truncated:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": 0, "truncated": True,
                "reason": "truncated (finish_reason == 'length') -- a cut-off "
                          "answer can look complete and is not"}

    if not stripped:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": 0, "truncated": False, "reason": "empty answer"}

    refused = stripped.startswith(REFUSAL[:40])
    if refused:
        # A refusal needs no citation. enforce_citations() would report
        # ok=False here (no cited indices) -- that's expected, not an error,
        # so we short-circuit before calling it.
        return {"valid": True, "refused": True, "invalid_citations": [],
                "n_citations": 0, "truncated": False, "reason": "refusal"}

    ok, invalid = enforce_citations(text, n_sources)
    n_citations = len({int(m) for m in re.findall(r"\[(\d+)\]", text)})

    if invalid:
        return {"valid": False, "refused": False, "invalid_citations": invalid,
                "n_citations": n_citations, "truncated": False,
                "reason": f"citation(s) out of range: {invalid}"}

    if n_citations == 0:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": 0, "truncated": False,
                "reason": "non-refusal answer with no citation -- unauditable"}

    return {"valid": True, "refused": False, "invalid_citations": [],
            "n_citations": n_citations, "truncated": False, "reason": "ok"}


def strip_invalid_citations(text: str, n_sources: int) -> str:
    """B3 (chosen failure path, see answer_question): drop any [n] outside
    1..n_sources rather than returning it or silently keeping it. A missing
    citation is safer than a fabricated one, and the sentence is still
    auditable as long as at least one valid citation survives elsewhere in
    the answer.
    """
    def _keep(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_sources else ""
    return re.sub(r"\[(\d+)\]", _keep, text)


def _generate(question: str, hits: list[Hit], *, tier: str,
              max_tokens: int = 600) -> tuple[str, str | None]:
    """The one generation call path. answer_question() and
    answer_with_gold_context() both go through this -- E2's comparison is
    only meaningful if "same generator" is literally true, not just similar.
    """
    context = delimit_untrusted(format_context(hits, max_chars=8000))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    result = chat(prompt, system=ANSWER_SYSTEM, tier=tier,
                 temperature=0.0, max_tokens=max_tokens, return_full=True)
    return result["text"].strip(), result.get("finish_reason")


def _validate_and_repair(question: str, text: str, finish: str | None,
                         hits: list[Hit], *, tier: str) -> tuple[str, dict]:
    """B3. The repair ladder, cheapest/most-likely fix first:

      1. truncated?              -> retry once at 2x max_tokens
      2. out-of-range citation?  -> strip it, re-check
      3. no citation at all?     -> one corrective rewrite request
      4. still broken?           -> fall back to REFUSAL

    The invariant this enforces: this function never returns
    (text, check) with check["valid"]=False and check["refused"]=False.
    """
    n = len(hits)
    check = validate_answer(text, n, finish)

    if check["truncated"]:
        text, finish = _generate(question, hits, tier=tier, max_tokens=1200)
        check = validate_answer(text, n, finish)

    if not check["valid"] and check["invalid_citations"] and not check["refused"]:
        text = strip_invalid_citations(text, n)
        check = validate_answer(text, n, finish)

    if not check["valid"] and check["reason"] == (
            "non-refusal answer with no citation -- unauditable"):
        context = delimit_untrusted(format_context(hits, max_chars=8000))
        prompt = (
            f"{context}\n\nQuestion: {question}\n\n"
            f"Previous attempt (missing a citation): {text}\n\n"
            "Rewrite it, citing the source number(s) that support each "
            f'factual sentence, or reply with exactly "{REFUSAL}" if the '
            "sources do not support an answer."
        )
        result = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0,
                     max_tokens=600, return_full=True)
        text, finish = result["text"].strip(), result.get("finish_reason")
        check = validate_answer(text, n, finish)

    if not check["valid"] and not check["refused"]:
        # Last resort: an uncited or malformed claim must never reach the
        # user. This is the line that makes the invariant actually true.
        text = REFUSAL
        check = validate_answer(text, n, None)

    return text, check


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    """retrieve -> (rerank) -> generate -> validate -> maybe repair."""
    hits = retriever.search(question, k=k)
    final = reranker.rerank(question, hits, k=final_k) if reranker else list(hits)[:final_k]

    text, finish = _generate(question, final, tier=tier)
    text, check = _validate_and_repair(question, text, finish, final, tier=tier)

    return Answer(
        question=question, text=text, hits=final, refused=check["refused"],
        citations_valid=check["valid"], invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"], truncated=check["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """E2: same generator (_generate/_validate_and_repair), but the context
    is the gold documents chunked directly -- no retrieval at all. The only
    thing that differs from answer_question() is where `hits` came from,
    which is exactly what makes the A-vs-B comparison meaningful.
    """
    hits: list[Hit] = []
    for i, doc_text in enumerate(gold_docs):
        doc_id = f"gold_{i}"
        for chunk in markdown_chunks(doc_text, doc_id, size=800):
            hits.append(Hit(chunk, score=1.0, source="gold", rank=len(hits)))

    text, finish = _generate(question, hits, tier=tier)
    text, check = _validate_and_repair(question, text, finish, hits, tier=tier)

    return Answer(
        question=question, text=text, hits=hits, refused=check["refused"],
        citations_valid=check["valid"], invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"], truncated=check["truncated"],
    )