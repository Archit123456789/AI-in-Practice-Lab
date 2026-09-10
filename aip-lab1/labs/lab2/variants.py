#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import Field, create_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import StructuredOutputError, structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — chosen examples (picked from data/eval/extraction_dev.jsonl)
# ---------------------------------------------------------------------------
FEW_SHOT_IDS: list[str] = [
    "T0097",  # billing vs complaint boundary: angry + ombudsman threat, but
              # the actual ask is a refund (money) -> billing, not complaint
    "T0054",  # no policy number in the live message -> null, never invented
    "T0100",  # Hinglish (hi-en): "Kripya", "Jaldi karo please"
    "T0222",  # sentiment/urgency independence: satisfied tone, but still a
              # live request that needs a lookup -- urgency isn't just "1"
              # because the customer is happy (closest match in this dev set;
              # no satisfied+high-urgency ticket exists here -- noted in report)
    "T0025",  # quote-boundary handling: policy number is in the live text,
              # a boilerplate quoted auto-reply follows it
    "T0095",  # one Lab 1 actually got wrong (urgency)
]

# Verbatim quotes for each example's `evidence` field (must be an exact
# substring of the ticket, <=200 chars, per TicketRecord's field description).
_FEW_SHOT_EVIDENCE = {
    "T0097": "Rs 8750 taken twice, nobody has called back in 45 days.",
    "T0054": "Your agent mis-sold me this policy.",
    "T0100": "Rs 8750 is still showing as due.",
    "T0222": "Can you confirm the restoration benefit is still available this year on AUR-4940770?",
    "T0025": "This is your problem, not mine.",
    "T0095": "How many wellness points do I currently have on AUR-9746149 andwhat discount does that give me?",
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str]) -> str:
    """Render the chosen examples into the prompt.

    Built through the real TicketRecord model (not hand-typed JSON) so the
    example format can never drift from what we actually ask the model to
    produce.
    """
    blocks = []
    for row in load_examples(ids):
        gold = row["expected"]
        record = TicketRecord(
            evidence=_FEW_SHOT_EVIDENCE[row["id"]],
            category=gold["category"],
            urgency=gold["urgency"],
            sentiment=gold["sentiment"],
            product=gold["product"],
            language=gold["language"],
            policy_number=gold["policy_number"],
            contains_pii=gold["contains_pii"],
        )
        blocks.append(
            f"Ticket:\n{row['input']}\n\nOutput:\n{record.model_dump_json(indent=2)}"
        )
    return (
        "Here are worked examples. Match this exact output format.\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n---\n\nNow do the same for the next ticket.\n"
    )


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _fallback(err: Exception) -> dict:
    """Same shape as extract.py's own fallback -- fail safe, flag for a human.

    Catches ANY failure here, not just schema-validation errors: a rate-limit
    error, timeout, or connection hiccup is a different exception type than
    StructuredOutputError, and under grid.py's default 4 concurrent workers
    on 60 tickets those showed up ~50% of the time. Recording them as a
    flagged fallback (like extract.py already does for extract_b/extract_c)
    keeps the eval running and keeps the failure honest, instead of crashing
    that ticket's evaluation outright.
    """
    return {
        "evidence": "", "category": "information", "urgency": 1,
        "sentiment": "neutral", "product": "unknown", "language": "en",
        "policy_number": None, "contains_pii": False, "escalate": True,
        "needs_human_review": True, "review_reason": str(err),
    }


def _postprocess(rec, ticket: str) -> dict:
    """Deterministic fields override whatever the model guessed; escalate is
    computed by code, never asked of the model (T2 §4.3 / Lab 1 Part C)."""
    result = rec.model_dump()
    result.update(extract_deterministic(ticket))   # policy_number, contains_pii
    result = apply_business_rules(result, ticket)   # escalate
    result.setdefault("needs_human_review", False)
    result.setdefault("review_reason", "")
    return result


# ---------------------------------------------------------------------------
# Part B — the six baseline variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    try:
        rec = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier=tier)
    except Exception as e:
        return _fallback(e)
    return _postprocess(rec, ticket)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    prompt = few_shot_block(FEW_SHOT_IDS) + f"\nTicket:\n{ticket}"
    try:
        rec = structured(prompt, schema=TicketRecord, system=SYSTEM_PROMPT, tier=tier)
    except Exception as e:
        return _fallback(e)
    return _postprocess(rec, ticket)


# `reasoning` must come FIRST so it conditions the answer (T2 §3.3).
# NOTE: `class TicketRecordReasoned(TicketRecord): reasoning: str = ...` would
# put `reasoning` LAST -- Pydantic always orders inherited fields ahead of a
# subclass's own new fields, regardless of where you write it in the source.
# So this builds the schema explicitly instead of subclassing. policy_number/
# contains_pii get overwritten deterministically in `_postprocess` regardless
# of what the model outputs for them, so we don't need to carry over
# TicketRecord's validator here.
_reasoned_fields = {
    "reasoning": (str, Field(description=(
        "Think step by step: what in the ticket determines the category, "
        "what determines the urgency level, and is the tone actually a "
        "reason to change urgency (it should not be). 2-4 sentences. "
        "Write this BEFORE deciding the fields below -- it should shape "
        "your answer, not justify one you already picked."
    ))),
}
_reasoned_fields.update({
    name: (finfo.annotation, finfo) for name, finfo in TicketRecord.model_fields.items()
})
TicketRecordReasoned = create_model("TicketRecordReasoned", **_reasoned_fields)


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    prompt = few_shot_block(FEW_SHOT_IDS) + f"\nTicket:\n{ticket}"
    try:
        rec = structured(prompt, schema=TicketRecordReasoned, system=SYSTEM_PROMPT, tier=tier)
    except Exception as e:
        return _fallback(e)
    return _postprocess(rec, ticket)


# ---------------------------------------------------------------------------
# Part C — the cascade
# ---------------------------------------------------------------------------
def cascade(ticket: str) -> dict:
    """SMALL first; escalate to MAIN only when two SMALL samples disagree.

    The trap: two samples at temperature 0 are the same request, so the
    cache returns the second from the first -- they're always identical,
    escalation reads 0%, nothing errors. Fix: draw the second sample at
    temperature > 0, which changes the sampling AND the cache key.
    """
    try:
        rec_a_model = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL")
        rec_a = _postprocess(rec_a_model, ticket)
    except Exception as e:
        rec = _fallback(e)
        rec["_path"] = "large"  # couldn't even get a SMALL answer -- treat as escalated
        return rec

    try:
        rec_b_model = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT,
                                  tier="SMALL", temperature=0.7)
        rec_b = _postprocess(rec_b_model, ticket)
    except Exception:
        rec_b = rec_a  # second sample failed -- don't force escalation on that alone

    agreed = all(rec_a.get(f) == rec_b.get(f) for f in ("category", "urgency", "sentiment"))
    rec_a["_agreed"] = agreed   # for the agree-when-right vs agree-when-wrong check (3.3)

    if agreed:
        rec_a["_path"] = "small"
        return rec_a

    try:
        rec = zero_shot(ticket, "MAIN")
    except Exception as e:
        rec = _fallback(e)
    rec["_path"] = "large"
    return rec


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}