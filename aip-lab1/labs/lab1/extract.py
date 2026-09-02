#!/usr/bin/env python3
"""Lab 1, Parts B and C — ticket extraction."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS
from aip.llm import StructuredOutputError, structured


CATEGORIES = Literal[
    "billing",
    "claims",
    "policy_change",
    "technical",
    "complaint",
    "information",
]


# ===========================================================================
# PART B — full model schema
# ===========================================================================

class TicketRecord(BaseModel):
    """
    Full schema for Part B.

    Evidence is deliberately first because it gives the model the
    justification before the classification fields.
    """

    evidence: str = Field(
        max_length=200,
        description=(
            "Quote verbatim the shortest span of the ticket that determined "
            "the category. One sentence at most."
        ),
    )

    category: CATEGORIES = Field(
        description=(
            "Choose exactly one category. "
            "billing = money in, including premiums, debits, refunds, "
            "invoices, 80D tax certificates or instalments; "
            "claims = an actual or intended insurance claim such as cashless, "
            "reimbursement, settlement amount, deduction or rejection; "
            "policy_change = altering the contract, including adding or "
            "removing a member, upgrading, porting or changing contact "
            "details; "
            "technical = the app, portal, OTP, login, locator or document "
            "upload is broken; "
            "complaint = Aurora's conduct itself is the subject, such as "
            "mis-selling, being kept on hold or an ignored grievance; "
            "information = a general question with no pending transaction. "
            "If a customer is angry about a claim but still wants the claim "
            "processed, choose claims, not complaint."
        ),
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description=(
            "Use this exact urgency scale. "
            "1 = answerable from general product knowledge or self-service "
            "without opening the customer's record. "
            "2 = Aurora must look up this customer's account, act on it, "
            "fix a defect, or a transaction is in flight. "
            "3 = something has already gone wrong or is stuck and the "
            "customer is waiting. "
            "4 = repeated failure to resolve, money or access is at risk now, "
            "or the customer threatens escalation. "
            "5 = emergency in progress, formal denial demanding immediate "
            "reversal, or the customer explicitly states they are filing "
            "with the Ombudsman. "
            "Add 1 if there is a same-day or next-morning deadline, capped "
            "at 5. "
            "Do not increase urgency merely because the customer is angry."
        ),
    )

    sentiment: Literal[
        "angry",
        "frustrated",
        "neutral",
        "satisfied",
    ] = Field(
        description=(
            "Classify tone only. "
            "angry = hostile, shouting or threatening; "
            "frustrated = unhappy and refers to a prior failure, repeat "
            "attempt, unanswered request, delay or something not working; "
            "neutral = matter-of-fact first-time request without prior "
            "failure; "
            "satisfied = thanks or praise."
        ),
    )

    product: Literal[
        "bronze",
        "silver",
        "gold",
        "platinum",
        "unknown",
    ] = Field(
        description=(
            "Use bronze, silver, gold or platinum only when the plan is "
            "explicitly named in the ticket. Otherwise use unknown. "
            "Never infer the plan from the context, policy number or "
            "sum insured."
        ),
    )

    language: Literal["en", "hi-en"] = Field(
        description=(
            "Use hi-en if Hindi words are mixed with English, including "
            "transliterated Hindi in Latin script such as jaldi, kripya, "
            "bahut or turant. Otherwise use en."
        ),
    )

    policy_number: str | None = Field(
        default=None,
        description=(
            "Copy a policy number only from the live/current customer "
            "message. It must be exactly AUR- followed by 7 digits. "
            "Return null if no such number appears in the live message. "
            "Never invent, alter or reformat a number. "
            "Quoted reply history beginning with > must not be used."
        ),
    )

    contains_pii: bool = Field(
        default=False,
        description=(
            "True if the finished ticket contains a phone number or an "
            "email address other than Aurora's published "
            "support@aurorahealth.example or "
            "grievance@aurorahealth.example addresses. "
            "A person's name alone does not count as PII for this dataset."
        ),
    )

    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v):
        if v is None:
            return None

        v = str(v).strip()

        if v.lower() in {"", "null", "none", "n/a"}:
            return None

        if not re.fullmatch(r"AUR-\d{7}", v):
            raise ValueError(
                "policy_number must exactly match AUR-<7 digits>"
            )

        return v


# ===========================================================================
# PART B — system prompt
# ===========================================================================

SYSTEM_PROMPT = """\
You are a reliable support-ticket extraction system.

Extract structured information from the customer ticket.

Follow the JSON Schema field descriptions exactly.

Important rules:
1. Use only information present in the ticket.
2. Never invent a policy number or product.
3. Category must follow the distinction between claims and complaint.
4. Urgency must follow the exact 1-5 scale in the schema.
5. Sentiment is tone only and must not be confused with urgency.
6. Policy numbers must come from the live message, not quoted history.
7. Return only the structured object required by the schema.

The ticket text follows.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides all extraction fields."""

    try:
        return structured(
            ticket,
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
        )

    except StructuredOutputError as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=str(e),
        )


# ===========================================================================
# PART C — deterministic extraction
# ===========================================================================

POLICY_RE = re.compile(r"\bAUR-\d{7}\b")
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """
    Compute policy_number and contains_pii without an LLM.

    Policy-number rule:
    Only use a policy number from the live/current message. Once a line
    beginning with '>' is encountered, everything after it is treated as
    quoted conversation history and ignored for policy extraction.

    PII rule:
    Scan the complete finished ticket, including quoted material. This is
    intentional because the dataset's contains_pii label is based on the
    final text, not only the live body.
    """

    # ---------------------------------------------------------------
    # POLICY NUMBER
    # ---------------------------------------------------------------

    quote_match = QUOTE_MARKER.search(ticket)

    if quote_match:
        live_text = ticket[:quote_match.start()]
    else:
        live_text = ticket

    policy_matches = POLICY_RE.findall(live_text)

    policy_number = (
        policy_matches[0]
        if policy_matches
        else None
    )

    # ---------------------------------------------------------------
    # PII
    # ---------------------------------------------------------------

    contains_pii = False

    for pattern in _PII_PATTERNS.values():
        if pattern.search(ticket):
            # Aurora's own published support addresses do NOT count.
            if pattern.pattern == _PII_PATTERNS["EMAIL"].pattern:
                emails = pattern.findall(ticket)

                for email in emails:
                    email_lower = email.lower()

                    if email_lower not in {
                        "support@aurorahealth.example",
                        "grievance@aurorahealth.example",
                    }:
                        contains_pii = True
                        break

            else:
                contains_pii = True

        if contains_pii:
            break

    return {
        "policy_number": policy_number,
        "contains_pii": contains_pii,
    }


# ===========================================================================
# PART C — business rules
# ===========================================================================

def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """
    Compute deterministic business rules.

    escalate = urgency >= 4 OR 'ombudsman' appears anywhere in the ticket.
    """

    result = dict(rec_fields)

    urgency = result.get("urgency", 1)

    result["escalate"] = (
        urgency >= 4
        or "ombudsman" in ticket.lower()
    )

    return result


# ===========================================================================
# PART C — reduced model schema
# ===========================================================================

class TicketRecordC(BaseModel):
    """
    Reduced schema.

    policy_number, contains_pii and escalate are deliberately removed
    because they are deterministic and should be computed by code.
    """

    evidence: str = Field(
        max_length=200,
        description=(
            "Quote verbatim the shortest span of the ticket that determined "
            "the category. One sentence at most."
        ),
    )

    category: CATEGORIES = Field(
        description=(
            "Choose exactly one. "
            "billing = money in, including premiums, debits, refunds, "
            "invoices, 80D tax certificates or instalments; "
            "claims = an actual or intended claim such as cashless, "
            "reimbursement, settlement amount, deduction or rejection; "
            "policy_change = altering the contract, including members, "
            "upgrades, porting or contact details; "
            "technical = the app, portal, OTP, login, locator or document "
            "upload is broken; "
            "complaint = Aurora's conduct itself is the subject; "
            "information = a question with no pending transaction. "
            "An angry claim remains claims if the customer wants the claim "
            "processed."
        ),
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description=(
            "Use this exact urgency scale. "
            "1 = general product knowledge or self-service how-to with no "
            "customer lookup. "
            "2 = Aurora must look up this customer's account, act on it, "
            "fix a defect, or a transaction is in flight. "
            "3 = something has gone wrong or is stuck and the customer is "
            "waiting. "
            "4 = repeated failure to resolve, money/access at risk now, "
            "or an explicit escalation threat. "
            "5 = emergency in progress, formal denial demanding immediate "
            "reversal, or customer explicitly says they are filing with "
            "the Ombudsman. "
            "Add 1 for a same-day or next-morning deadline, capped at 5. "
            "Anger alone does not increase urgency."
        ),
    )

    sentiment: Literal[
        "angry",
        "frustrated",
        "neutral",
        "satisfied",
    ] = Field(
        description=(
            "Tone only. "
            "angry = hostile, shouting or threatening; "
            "frustrated = unhappy and refers to a prior failure, repeat "
            "attempt, unanswered request, delay or something not working; "
            "neutral = matter-of-fact first-time request; "
            "satisfied = thanks or praise."
        ),
    )

    product: Literal[
        "bronze",
        "silver",
        "gold",
        "platinum",
        "unknown",
    ] = Field(
        description=(
            "Use bronze, silver, gold or platinum only when explicitly "
            "named in the ticket. Otherwise use unknown. Never infer it."
        ),
    )

    language: Literal["en", "hi-en"] = Field(
        description=(
            "Use hi-en when Hindi words are mixed with English, including "
            "transliterated Hindi such as jaldi, kripya, bahut or turant. "
            "Otherwise use en."
        ),
    )


SYSTEM_PROMPT_C = """\
You are a reliable support-ticket classification system.

Classify the customer ticket using only information present in the ticket.

Make judgement calls only for:
- category
- urgency
- sentiment
- product
- language
- evidence

Follow the JSON Schema descriptions exactly.

Important:
- Do not invent information.
- Distinguish complaint from the category of the underlying request.
- Use the exact urgency scale provided.
- Sentiment is tone only.
- Product must be explicitly named.
- Return only the JSON object required by the schema.

The ticket text follows.
"""


def extract_c(ticket: str) -> dict:
    """
    Part C.

    LLM handles judgement.
    Python handles deterministic extraction and business rules.
    """

    try:
        model_record = structured(
            ticket,
            schema=TicketRecordC,
            system=SYSTEM_PROMPT_C,
        )

    except StructuredOutputError as e:
        return {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "escalate": True,
            "needs_human_review": True,
            "review_reason": str(e),
        }

    # Model judgement fields
    result = model_record.model_dump()

    # Deterministic fields
    deterministic = extract_deterministic(ticket)
    result.update(deterministic)

    # Business rule
    result = apply_business_rules(result, ticket)

    # Defaults
    result.setdefault("needs_human_review", False)
    result.setdefault("review_reason", "")

    return result


# ===========================================================================
# MANUAL TEST
# ===========================================================================

if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]

    sample = json.loads(
        (
            root / "data/eval/extraction_dev.jsonl"
        ).open(encoding="utf-8").readline()
    )

    print("--- ticket ---")
    print(sample["input"][:600])

    print("\n--- gold ---")
    print(sample["expected"])

    print("\n--- yours ---")
    print(extract_c(sample["input"]))